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
# spoilage_level 範圍：1 ~ 20 (1=最新鮮, 20=最腐敗)
# 修正後的閾值表，對應 1-20 的範圍
DISCOUNT_TABLE = [
    (4,   0),    # 1~4    → 無折扣 (對應原 0~20%)
    (8,   10),   # 4~8    → 9折   (對應原 20~40%)
    (12,  25),   # 8~12   → 75折  (對應原 40~60%)
    (16,  40),   # 12~16  → 6折   (對應原 60~80%)
    (20,  60),   # 16~20  → 4折   (對應原 80~100%)
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
auto_predict_interval = 0  # 0 = 關閉, 單位: 分鐘
auto_predict_thread = None
stop_auto_predict = threading.Event()

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
    # [FIX] 檢查並修正濕度範圍 (若 > 1 則除以 100轉為 0~1)
    model_hum = hum
    if model_hum is not None and model_hum > 1.0:
        model_hum = model_hum / 100.0
    
    # [FIX] 檢查並修正儲存時間 (假設輸入為天數，模型訓練為小時)
    # Scaler mean for storage_time is ~102.5, suggesting hours.
    # Input is labeled "儲存天數" (days).
    storage_hours = req.storage_time * 24.0

    logger.info(f"Raw sensor input (adjusted): temp={temp}, hum={model_hum}, storage_hours={storage_hours}")
    
    raw_sensor = np.array([[temp, model_hum, storage_hours]], dtype=np.float32)
    scaled_sensor = scaler.transform(raw_sensor)
    logger.info(f"Scaled sensor: {scaled_sensor}")

    sensor_tensor = torch.tensor(scaled_sensor, dtype=torch.float32).to(DEVICE)  # (1,3)

    # ── 3. 模型推理 ──
    with torch.no_grad():
        prediction = model(img_tensor, sensor_tensor).item()
        logger.info(f"Model raw prediction: {prediction}")

    # [NEW] 應用動態定價公式: Q_pricing(t) = Q_human(t) * (1 - r_pricing)
    r_pricing = 0.21
    spoilage = prediction * (1 - r_pricing)

    # [NEW] 限制在合理範圍 1-20
    spoilage = float(np.clip(spoilage, 1.0, 20.0))
    
    # ── 4. 計算折扣與售價 ──
    # 使用修正後的 DISCOUNT_TABLE (範圍 1-20) 直接查表
    discount_pct = calc_discount(spoilage)
    final_price  = round(req.base_price * (1 - discount_pct / 100), 2)

    # ── 5. 新鮮程度標籤 (範圍 1-20) ──
    if spoilage < 4:
        freshness_label = "非常新鮮 🟢"
        freshness_color = "#22c55e"
    elif spoilage < 8:
        freshness_label = "新鮮 🟡"
        freshness_color = "#eab308"
    elif spoilage < 12:
        freshness_label = "輕微腐敗 🟠"
        freshness_color = "#f97316"
    elif spoilage < 16:
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


class SettingsRequest(BaseModel):
    interval_minutes: int

@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    """更新自動預測的時間間隔"""
    global auto_predict_interval, stop_auto_predict, auto_predict_thread
    
    new_interval = req.interval_minutes
    if new_interval < 0:
        raise HTTPException(status_code=400, detail="Invalid interval")
        
    logger.info(f"Updating auto-predict interval to {new_interval} minutes")
    
    # 如果原本有執行中的執行緒，先通知停止
    if auto_predict_thread and auto_predict_thread.is_alive():
        stop_auto_predict.set()
        # 不等待 join，避免卡住 API 回應，讓舊執行緒自然結束
    
    auto_predict_interval = new_interval
    
    if auto_predict_interval > 0:
        stop_auto_predict.clear()
        auto_predict_thread = threading.Thread(target=auto_predict_loop, args=(auto_predict_interval,), daemon=True)
        auto_predict_thread.start()
        logger.info("Auto-predict thread started")
    
    return {"status": "ok", "interval_minutes": auto_predict_interval}

def auto_predict_loop(interval_minutes):
    """背景執行緒：定時執行預測 (模擬)"""
    logger.info(f"Auto-predict loop started. Interval: {interval_minutes}m")
    while not stop_auto_predict.is_set():
        # 這裡僅作示範：每隔 interval_minutes 執行一次
        # 實際上自動預測的結果通常要寫入資料庫或推送到前端 (WebSocket)
        # 由於目前架構是前端 polling 或主動請求，這裡我們先記錄 Log 代表執行了預測
        logger.info(f"⏰ Auto-predict triggered! (Interval: {interval_minutes}m)")
        
        # 可以在這裡呼叫內部的預測邏輯並儲存結果
        # ... predict logic ...
        
        # 等待下一次執行 (分段 sleep 以便能快速回應 stop)
        for _ in range(interval_minutes * 60):
            if stop_auto_predict.is_set():
                logger.info("Auto-predict loop stopping...")
                return
            time.sleep(1)

# ─── 靜態網頁 ─────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
