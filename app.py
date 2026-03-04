"""
app.py  ─  FastAPI 後端主程式
功能：
  1. 在背景持續訂閱 MQTT，快取最新的溫溼度數據
  2. GET  /api/sensor   → 回傳最新溫溼度
  3. POST /api/predict  → 接收 storage_time，執行 AI 推理，回傳腐敗值與折扣
  4. GET  /api/image    → 代理 M5Stack 即時圖像（解決 CORS）
  5. 伺服前端靜態網頁
"""

import io
import time
import threading
import logging
import joblib
import numpy as np
import httpx
import torch
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import paho.mqtt.client as mqtt

from hybrid_model import HybridModel

# ─── 基本設定 ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH  = "./spoilage_model_rnn.pth"
SCALER_PATH = "./sensor_scaler.pkl"
CAMERA_URL  = "https://ezdata2.m5stack.com/9888E0031824/captured.jpg"

MQTT_BROKER    = "broker.emqx.io"
MQTT_PORT      = 1883
MQTT_TOPIC     = "m5go/ISM/env"
MQTT_CLIENT_ID = "FreshServer_ISM_001"

IMG_HEIGHT = 128
IMG_WIDTH  = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ─── 折扣設定（可依需求調整）────────────────────────────────────────────────
# spoilage_level 預設為 0~100 的數值；若你的訓練資料是 0~1，請把下表除以 100
DISCOUNT_TABLE = [
    (20,  0),    # 0~20   → 無折扣
    (40,  10),   # 20~40  → 9折
    (60,  25),   # 40~60  → 75折
    (80,  40),   # 60~80  → 6折
    (100, 60),   # 80~100 → 4折
]

def calc_discount(spoilage: float) -> int:
    """依腐敗值回傳折扣百分比（整數，例如 25 代表打75折）"""
    for threshold, discount in DISCOUNT_TABLE:
        if spoilage <= threshold:
            return discount
    return DISCOUNT_TABLE[-1][1]

# ─── 全域狀態 ────────────────────────────────────────────────────────────────
latest_sensor: dict = {"temperature": None, "humidity": None, "timestamp": None}
model: HybridModel  = None
scaler              = None

# ─── 圖像前處理 ──────────────────────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

# ─── MQTT 背景訂閱 ───────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("MQTT connected, subscribing to %s", MQTT_TOPIC)
        client.subscribe(MQTT_TOPIC)
    else:
        logger.warning("MQTT connect failed, rc=%s", rc)

def on_message(client, userdata, msg):
    import json
    global latest_sensor
    try:
        data = json.loads(msg.payload.decode())
        latest_sensor = {
            "temperature": float(data.get("temp", data.get("temperature", 0))),
            "humidity":    float(data.get("hum",  data.get("humidity", 0))),
            "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        logger.debug("Sensor updated: %s", latest_sensor)
    except Exception as e:
        logger.error("MQTT parse error: %s", e)

def start_mqtt():
    client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            client.loop_forever()
        except Exception as e:
            logger.warning("MQTT error: %s, retrying in 5s...", e)
            time.sleep(5)

# ─── FastAPI 應用程式 ─────────────────────────────────────────────────────────
app = FastAPI(title="🍌 Fruit Freshness API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    global model, scaler

    # 載入模型
    logger.info("Loading model...")
    model = HybridModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    logger.info("Model loaded ✓")

    # 載入 Scaler
    scaler = joblib.load(SCALER_PATH)
    logger.info("Scaler loaded ✓")

    # 啟動 MQTT 背景執行緒
    t = threading.Thread(target=start_mqtt, daemon=True)
    t.start()
    logger.info("MQTT thread started ✓")

# ─── API 端點 ─────────────────────────────────────────────────────────────────

@app.get("/api/sensor")
async def get_sensor():
    """回傳最新感測器數據"""
    return JSONResponse(content=latest_sensor)


@app.get("/api/image")
async def proxy_image():
    """代理 M5Stack 即時圖像，解決瀏覽器 CORS 限制"""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(CAMERA_URL)
            resp.raise_for_status()
        return StreamingResponse(
            io.BytesIO(resp.content),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Camera fetch failed: {e}")


class PredictRequest(BaseModel):
    storage_time: float        # 儲存天數
    base_price: float = 100.0  # 原始售價（預設 100 元）
    # 可選：若前端已取得感測器數據直接帶入（否則用伺服器快取值）
    temperature: float | None = None
    humidity:    float | None = None


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """執行 AI 推理，回傳腐敗值與定價"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    # 決定溫溼度來源
    temp = req.temperature if req.temperature is not None else latest_sensor.get("temperature")
    hum  = req.humidity    if req.humidity    is not None else latest_sensor.get("humidity")

    if temp is None or hum is None:
        raise HTTPException(
            status_code=422,
            detail="尚未收到感測器數據，請確認 M5GO 裝置已開機並連接 MQTT。"
        )

    # ── 1. 取得即時圖像 ──
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(CAMERA_URL)
            resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Camera fetch failed: {e}")

    img_tensor = infer_transform(image).unsqueeze(0).to(DEVICE)  # (1,3,128,128)

    # ── 2. 感測器正規化 ──
    raw_sensor = np.array([[temp, hum, req.storage_time]], dtype=np.float32)
    scaled_sensor = scaler.transform(raw_sensor)
    sensor_tensor = torch.tensor(scaled_sensor, dtype=torch.float32).to(DEVICE)  # (1,3)

    # ── 3. 模型推理 ──
    with torch.no_grad():
        prediction = model(img_tensor, sensor_tensor).item()

    # 限制在合理範圍（依訓練資料決定；預設 0~100）
    spoilage = float(np.clip(prediction, 0, 100))

    # ── 4. 計算折扣與售價 ──
    discount_pct = calc_discount(spoilage)
    final_price  = round(req.base_price * (1 - discount_pct / 100), 2)

    # ── 5. 新鮮程度標籤 ──
    if spoilage < 20:
        freshness_label = "非常新鮮 🟢"
        freshness_color = "#22c55e"
    elif spoilage < 40:
        freshness_label = "新鮮 🟡"
        freshness_color = "#eab308"
    elif spoilage < 60:
        freshness_label = "輕微腐敗 🟠"
        freshness_color = "#f97316"
    elif spoilage < 80:
        freshness_label = "中度腐敗 🔴"
        freshness_color = "#ef4444"
    else:
        freshness_label = "嚴重腐敗 ⚫"
        freshness_color = "#7f1d1d"

    return {
        "spoilage_level":  round(spoilage, 2),
        "freshness_label": freshness_label,
        "freshness_color": freshness_color,
        "discount_pct":    discount_pct,
        "base_price":      req.base_price,
        "final_price":     final_price,
        "sensor": {
            "temperature": round(temp, 1),
            "humidity":    round(hum, 1),
            "storage_time": req.storage_time,
        },
        "model_input_shape": {
            "image": list(img_tensor.shape),
            "sensor": list(sensor_tensor.shape),
        }
    }


# ─── 靜態網頁 ─────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
