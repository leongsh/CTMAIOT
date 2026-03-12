"""
app.py — AIoT 智慧貨架系統後端 v4.0
三層架構：
  Layer 1 (前端)  : 登入頁 / 地圖應用層 / 後台管理 — 靜態 HTML
  Layer 2 (應用層): 節點地圖、品質評估、動態定價 API
  Layer 3 (後端)  : 用戶認證、節點管理、數據記錄 API

API 路由：
  POST /api/auth/login          — 登入取得 JWT Token
  GET  /api/auth/me             — 取得當前用戶資訊
  POST /api/auth/users          — 新增用戶（admin only）
  GET  /api/auth/users          — 列出所有用戶（admin only）
  DELETE /api/auth/users/{id}   — 刪除用戶（admin only）

  GET  /api/nodes               — 列出所有節點
  POST /api/nodes               — 新增節點（admin only）
  PUT  /api/nodes/{node_id}     — 更新節點（admin only）
  DELETE /api/nodes/{node_id}   — 刪除節點（admin only）
  GET  /api/nodes/{node_id}/readings    — 節點感測器歷史
  GET  /api/nodes/{node_id}/predictions — 節點評估歷史

  GET  /api/sensor              — 最新感測器數據（MQTT）
  GET  /api/image               — 代理相機圖像
  POST /api/predict             — AI + 論文公式複合評估（記錄到 DB）
  POST /api/quality             — 純論文公式評估（記錄到 DB）
  GET  /api/quality/params      — 論文模型參數

  GET  /api/admin/dashboard     — 後台儀表板統計（admin only）
"""

import io
import time
import threading
import logging
import os
import hashlib
import base64
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone, timedelta
from typing import Optional

# 香港時區 UTC+8
_HKT = timezone(timedelta(hours=8))

def now_hkt() -> str:
    """返回 UTC+8 香港時間的格式化字串"""
    return datetime.now(_HKT).strftime("%Y-%m-%d %H:%M:%S")

import requests
import joblib
import numpy as np
import httpx
import torch
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import paho.mqtt.client as mqtt

from hybrid_model import HybridModel
from quality_model import (
    calculate_quality,
    quality_result_to_dict,
    PRODUCT_PARAMS,
    DISCOUNT_RULES,
    Q0,
    Q_THRESHOLD,
    R,
    T_REF_K,
    ALPHA_FORMULA,
    ALPHA_AI,
    calc_k_comp_ref,
)
from database import (
    init_db, get_user, get_all_nodes, get_node, upsert_node,
    insert_reading, insert_prediction,
    get_node_readings, get_node_predictions,
    get_dashboard_stats, verify_password, _hash_password,
    get_db, DATABASE_URL, update_node_settings,
)
from auth import (
    create_access_token, decode_token, authenticate_user,
    get_current_user, require_admin,
)

# ─── 基本設定 ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH  = "./spoilage_model_rnn.pth"
SCALER_PATH = "./sensor_scaler.pkl"
CAMERA_URL  = "https://ezdata2.m5stack.com/9888E0031824/captured.jpg"

MQTT_BROKER    = "broker.emqx.io"
MQTT_PORT      = 1883
MQTT_TOPIC     = "m5go/ISM/env"
# 使用動態唯一 ID 防止多個實例互踢（固定 ID 會導致連線循環斷線）
import uuid as _uuid
MQTT_CLIENT_ID = f"FreshServer_{_uuid.uuid4().hex[:8]}"

IMG_HEIGHT = 128
IMG_WIDTH  = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)

# ─── 全域狀態 ────────────────────────────────────────────────────────────────
# 多節點感測器快取：{node_id: {temperature, humidity, timestamp}}
sensor_cache: dict = {
    "A01": {"temperature": None, "humidity": None, "timestamp": None}
}

# MQTT 主題 → node_id 的記憶體快取（避免每次收到訊息都查 DB）
# 格式：{"m5go/ISM/env": "A01", ...}
_mqtt_topic_map: dict = {}
model_ai: HybridModel = None
scaler = None

# AI 推理結果快取：{node_id: {spoilage, label, color, timestamp, quality_data}}
ai_cache: dict = {}

# 相機圖片快取：{node_id: {data: bytes, timestamp: float, content_type: str}}
# 後端定期抓取圖片並快取，前端請求時直接回傳快取內容（不需再等待 M5Stack）
CAMERA_PREFETCH_INTERVAL = 4   # 預取間隔（秒）：背景執行緒每 X 秒向相機抓取一次
CAMERA_CACHE_TTL = 10          # 快取有效期（秒）：必須 > CAMERA_PREFETCH_INTERVAL，避免空窗期
camera_cache: dict = {}  # {node_id: {"data": bytes, "ts": float}}

# DATABASE_URL 已在上方 import 中引入

# 模型版本管理
MODEL_BACKUP_DIR = "./model_versions"
model_registry: dict = {
    "current": {
        "model_path":  MODEL_PATH,
        "scaler_path": SCALER_PATH,
        "version":     "v1.0-original",
        "uploaded_at": None,
        "uploaded_by": "system",
        "description": "原始訓練模型",
        "inference_count": 0,
    },
    "history": [],  # 最多保留 5 個歷史版本
}
os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)

# ─── 圖像前處理 ──────────────────────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

# ─── MQTT 背景訂閱 ───────────────────────────────────────────────────────────
def _refresh_mqtt_topic_map():
    """從資料庫更新 MQTT 主題→node_id 對應表（只在連線時或節點變更時呼叫）"""
    global _mqtt_topic_map
    try:
        nodes = get_all_nodes()
        new_map = {}
        for n in nodes:
            topic = n.get("mqtt_topic", "").strip()
            if topic:
                new_map[topic] = n["node_id"]
        _mqtt_topic_map = new_map
        logger.info("MQTT topic map refreshed: %s", _mqtt_topic_map)
    except Exception as e:
        logger.warning("_refresh_mqtt_topic_map error: %s", e)


def on_message(client, userdata, msg):
    import json
    global sensor_cache
    try:
        data = json.loads(msg.payload.decode())
        ts = now_hkt()
        temp  = float(data.get("temp", data.get("temperature", 0)))
        hum   = float(data.get("hum",  data.get("humidity", 0)))
        light_raw = float(data.get("light", data.get("light_lux", 375)))
        # M5GO 光照感測器輸出為反向 ADC（0-65535），數値越大越暗，需反轉
        light = max(0.0, 65535.0 - light_raw)
        pres  = float(data.get("pres",  data.get("pressure", 1013.0)))
        air   = float(data.get("air_velocity", 0.22))

        # 從記憶體快取查找對應 node_id（不查 DB，避免 1.5 秒延遲）
        node_id = _mqtt_topic_map.get(msg.topic, "A01")

        sensor_cache[node_id] = {
            "temperature": temp,
            "humidity":    hum,
            "light_lux":   light,
            "pressure":    pres,
            "air_velocity": air,
            "timestamp":   ts,
        }
        # 自動記錄到資料庫（使用連線池，快速完成）
        try:
            insert_reading(node_id, temp, hum, light, air)
        except Exception as e:
            logger.debug("Insert reading error: %s", e)

        mqtt_status["last_message"] = ts
        mqtt_status["messages_received"] += 1
        logger.info("Sensor[%s] T=%.1f H=%.1f (total=%d)", node_id, temp, hum, mqtt_status["messages_received"])
    except Exception as e:
        logger.error("MQTT parse error: %s", e)


# MQTT 連線狀態診斷
mqtt_status = {
    "connected": False,
    "last_connect": None,
    "last_message": None,
    "client_id": MQTT_CLIENT_ID,
    "reconnect_count": 0,
    "messages_received": 0,
}


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        mqtt_status["connected"] = True
        mqtt_status["last_connect"] = now_hkt()
        logger.info("MQTT connected (id=%s), subscribing to %s", MQTT_CLIENT_ID, MQTT_TOPIC)
        client.subscribe(MQTT_TOPIC)
        # 更新主題對應表並訂閱所有節點主題
        _refresh_mqtt_topic_map()
        for topic, node_id in _mqtt_topic_map.items():
            if topic != MQTT_TOPIC:
                client.subscribe(topic)
                logger.info("Also subscribing: %s -> %s", topic, node_id)
    else:
        mqtt_status["connected"] = False
        logger.warning("MQTT connect failed, rc=%s", rc)


def on_disconnect(client, userdata, rc):
    mqtt_status["connected"] = False
    mqtt_status["reconnect_count"] += 1
    logger.warning("MQTT disconnected rc=%s, reconnect_count=%d", rc, mqtt_status["reconnect_count"])


def start_mqtt():
    import uuid
    # 每次啟動都使用全新的唯一 ID，防止跟舊連線互踢
    unique_id = f"FreshServer_{uuid.uuid4().hex[:8]}"
    mqtt_status["client_id"] = unique_id
    client = mqtt.Client(client_id=unique_id, clean_session=True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    while True:
        try:
            logger.info("MQTT connecting with id=%s", unique_id)
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
            client.loop_forever()
        except Exception as e:
            logger.warning("MQTT error: %s, retrying in 5s...", e)
            mqtt_status["connected"] = False
            time.sleep(5)


# ─── AI 自動推理背景任務 ─────────────────────────────────────────────────────
AI_INFER_INTERVAL = 7200  # 秒（2 小時）


def run_ai_inference_for_node(node_id: str, node: dict) -> dict | None:
    """
    對單一節點執行 AI 推理：
    1. 從感測器快取取得溫濕度
    2. 從相機 URL 抓取圖像
    3. 執行 HybridModel 推理
    4. 計算複合品質評分
    回傳推理結果字典，失敗時回傳 None
    """
    global model_ai, scaler, sensor_cache, ai_cache

    if model_ai is None or scaler is None:
        return None

    sensor = sensor_cache.get(node_id, {})
    temp = sensor.get("temperature")
    hum  = sensor.get("humidity")
    # 若無即時感測器數據，使用預設環境值（室溫 25°C、濕度 65%）
    # 確保所有節點都能顯示基本品質評估
    using_default_sensor = False
    if temp is None or hum is None:
        temp = 25.0
        hum  = 65.0
        using_default_sensor = True
        logger.debug("AI inference [%s]: using default sensor values (25°C, 65%%)", node_id)

    # 取得相機 URL
    cam_url = node.get("camera_url") or CAMERA_URL
    camera_snapshot_url = None
    camera_image_base64 = None  # 儲存圖片的 Base64 字串

    try:
        resp = requests.get(cam_url, timeout=8)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        # 快照 URL 加入時間戳以區分不同時間的快照
        ts_param = int(time.time())
        camera_snapshot_url = f"{cam_url}?_snap={ts_param}" if cam_url else None
        # 將圖片壓縮後轉為 Base64 字串儲存（JPEG 品質 70％，約 30-80KB）
        try:
            buf = io.BytesIO()
            # 先將圖片縮放到 640x480 以減少儲存空間
            img_save = image.copy()
            img_save.thumbnail((640, 480), Image.LANCZOS)
            img_save.save(buf, format="JPEG", quality=70, optimize=True)
            camera_image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            logger.info("AI inference [%s] camera image captured: %d bytes (base64)", node_id, len(camera_image_base64))
        except Exception as b64_err:
            logger.warning("AI inference [%s] base64 encode error: %s", node_id, b64_err)
            camera_image_base64 = None
    except Exception as e:
        logger.warning("AI inference [%s] camera fetch failed: %s — using blank image fallback", node_id, e)
        # 相機失敗時使用空白圖片繼續推理（不中斷評估）
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        camera_snapshot_url = None
        camera_image_base64 = None

    try:
        img_tensor = infer_transform(image).unsqueeze(0).to(DEVICE)

        model_hum     = hum / 100.0 if hum > 1.0 else hum
        # 優先從節點設定讀取已儲存天數，其次從 ai_cache，預設 1.0
        storage_days  = float(node.get("days_stored") or ai_cache.get(node_id, {}).get("storage_days") or 1.0)
        storage_hours = storage_days * 24.0
        raw_sensor    = np.array([[temp, model_hum, storage_hours]], dtype=np.float32)
        scaled_sensor = scaler.transform(raw_sensor)
        sensor_tensor = torch.tensor(scaled_sensor, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            prediction = model_ai(img_tensor, sensor_tensor).item()
        # 訓練標籤範圍是 1–20，縮放到 0–100
        # 公式：spoilage = (raw_output - 1) / (20 - 1) * 100
        raw_clipped = float(np.clip(prediction, 1.0, 20.0))
        spoilage = (raw_clipped - 1.0) / 19.0 * 100.0
        logger.info("AI inference [%s] raw_output=%.4f -> spoilage=%.2f%%", node_id, prediction, spoilage)
    except Exception as e:
        logger.warning("AI inference [%s] model error: %s", node_id, e)
        return None

    # AI 標籤
    if spoilage < 20:   ai_label, ai_color = "非常新鮮 🟢", "#22c55e"
    elif spoilage < 40: ai_label, ai_color = "新鮮 🟡",     "#eab308"
    elif spoilage < 60: ai_label, ai_color = "輕微腐敗 🟠", "#f97316"
    elif spoilage < 80: ai_label, ai_color = "中度腐敗 🔴", "#ef4444"
    else:               ai_label, ai_color = "嚴重腐敗 ⚫", "#7f1d1d"

    # 計算複合品質評分
    # 注意：資料庫欄位名稱是 product（不是 product_type）
    product     = node.get("product") or node.get("product_type", "banana")
    initial_dsl = node.get("initial_dsl") or node.get("initial_dsl_days")
    base_price  = node.get("base_price", 100.0)
    if initial_dsl is None:
        params_p = PRODUCT_PARAMS.get(product, PRODUCT_PARAMS["banana"])
        initial_dsl = float(params_p["initial_dsl_days"])

    try:
        quality_result = calculate_quality(
            temperature=temp, humidity=hum,
            storage_days=storage_days,
            product=product,
            ai_spoilage=spoilage,
            initial_dsl=initial_dsl,
        )
        quality_data = quality_result_to_dict(quality_result, base_price=base_price)
    except Exception as e:
        logger.warning("AI inference [%s] quality calc error: %s", node_id, e)
        quality_data = None

    ts = now_hkt()
    result = {
        "spoilage":              round(spoilage, 2),
        "ai_label":              ai_label,
        "ai_color":              ai_color,
        "quality_data":          quality_data,
        "storage_days":          storage_days,
        "timestamp":             ts,
        "using_default_sensor":  using_default_sensor,
    }

    # 更新快取
    ai_cache[node_id] = result
    model_registry["current"]["inference_count"] += 1
    logger.info("AI inference [%s] done: spoilage=%.1f%% label=%s", node_id, spoilage, ai_label)

    # 持久化到 DB
    save_ai_cache_to_db(node_id, result)

    # 自動記錄到資料庫（每 2 小時推理時記錄，包含相機圖片 Base64）
    if quality_data:
        try:
            insert_prediction(node_id, {
                "storage_days":          storage_days,
                "temperature":           temp,
                "humidity":              hum,
                "ai_spoilage":           spoilage,
                "quality_ai":            quality_data.get("quality_ai"),
                "quality_formula":       quality_data["quality_formula"],
                "quality_combined":      quality_data["quality_score"],
                "dsl_combined":          quality_data["dsl_days"],
                "discount_pct":          quality_data["discount_pct"],
                "base_price":            base_price,
                "final_price":           quality_data["final_price"],
                "freshness_label":       quality_data["freshness_label"],
                "product":               product,
                "camera_snapshot_url":   camera_snapshot_url,
                "camera_image_base64":   camera_image_base64,
            })
            logger.info("Auto-save prediction [%s] OK (image=%s)", node_id,
                        "yes" if camera_image_base64 else "no")
        except Exception as e:
            logger.debug("Auto-save prediction error: %s", e)

    return result


def auto_ai_inference_loop():
    """背景執行緒：每 2 小時對所有節點執行 AI 推理並儲存相機圖片 Base64"""
    # 等待系統完全啟動（模型載入、DB 初始化）
    time.sleep(15)
    logger.info("Auto AI inference loop started")

    while True:
        try:
            nodes = get_all_nodes()
            for node in nodes:
                node_id = node.get("node_id")
                if node_id:
                    run_ai_inference_for_node(node_id, node)
        except Exception as e:
            logger.warning("Auto AI inference loop error: %s", e)

        time.sleep(AI_INFER_INTERVAL)


# ─── FastAPI 應用程式 ─────────────────────────────────────────────────────────
app = FastAPI(title="AIoT Smart Shelf API", version="4.1.1")


# ── ai_cache 持久化函數（PostgreSQL，使用連線池）──────────────────────────────
def _init_ai_cache_db():
    """初始化 ai_cache 持久化資料表（PostgreSQL）"""
    import json
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_cache (
                    node_id    TEXT PRIMARY KEY,
                    data       TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
        logger.info("ai_cache table initialized in PostgreSQL")
    except Exception as e:
        logger.warning("_init_ai_cache_db error: %s", e)


def save_ai_cache_to_db(node_id: str, result: dict):
    """將單一節點的 ai_cache 寫入持久化 PostgreSQL（使用連線池）"""
    import json
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ai_cache (node_id, data, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (node_id) DO UPDATE SET
                    data = EXCLUDED.data,
                    updated_at = NOW()
            """, (node_id, json.dumps(result, ensure_ascii=False)))
    except Exception as e:
        logger.debug("save_ai_cache_to_db error: %s", e)


def load_ai_cache_from_db():
    """從 PostgreSQL 載入 ai_cache（使用連線池）"""
    import json
    global ai_cache
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT node_id, data FROM ai_cache")
            rows = cur.fetchall()
        for row in rows:
            try:
                ai_cache[row[0]] = json.loads(row[1])
            except Exception:
                pass
        logger.info("Loaded %d ai_cache entries from PostgreSQL", len(rows))
    except Exception as e:
        logger.warning("load_ai_cache_from_db error: %s", e)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global model_ai, scaler
    # 初始化資料庫
    init_db()
    logger.info("Database initialized")

    # 初始化 ai_cache 持久化表並載入上次的推理結果
    _init_ai_cache_db()
    load_ai_cache_from_db()
    logger.info("ai_cache loaded from PostgreSQL: %d entries", len(ai_cache))

    # 載入 AI 模型
    try:
        model_ai = HybridModel().to(DEVICE)
        model_ai.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model_ai.eval()
        scaler = joblib.load(SCALER_PATH)
        logger.info("AI Model loaded")
    except Exception as e:
        logger.warning("AI Model load failed: %s", e)

    # 啟動 MQTT 背景執行緒
    t = threading.Thread(target=start_mqtt, daemon=True)
    t.start()
    logger.info("MQTT thread started")

    # 啟動 AI 自動推理背景執行緒（每 30 分鐘）
    ai_thread = threading.Thread(target=auto_ai_inference_loop, daemon=True)
    ai_thread.start()
    logger.info("Auto AI inference thread started (interval=1800s / 30min)")

    # 啟動 Keep-Alive 執行緒（每 14 分鐘 ping 自己，防止 Render 免費方案冷啟動）
    ka_thread = threading.Thread(target=_keep_alive_loop, daemon=True)
    ka_thread.start()
    logger.info("Keep-alive thread started")

    # 啟動相機圖片背景預取執行緒（每 5 秒預取一次，讓前端請求時直接命中快取）
    cam_prefetch_thread = threading.Thread(target=_camera_prefetch_loop, daemon=True)
    cam_prefetch_thread.start()
    logger.info("Camera prefetch thread started (interval=5s)")


# ─── Keep-Alive（防止 Render 免費方案 15 分鐘冷啟動）────────────────────────────
def _keep_alive_loop():
    """每 14 分鐘 ping 自己的健康檢查端點，防止 Render 休眠"""
    import requests as _req
    # 等待服務完全啟動
    time.sleep(60)
    # 支援 Fly.io（FLY_APP_NAME）和 Render（RENDER_EXTERNAL_URL）兩個平台
    fly_app = os.environ.get("FLY_APP_NAME")
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if fly_app:
        base_url = f"https://{fly_app}.fly.dev"
    elif render_url:
        base_url = render_url
    else:
        base_url = "http://localhost:8080"
    ping_url = f"{base_url}/api/health"
    while True:
        try:
            r = _req.get(ping_url, timeout=10)
            logger.debug("Keep-alive ping: %s %s", ping_url, r.status_code)
        except Exception as e:
            logger.debug("Keep-alive ping failed: %s", e)
        time.sleep(14 * 60)  # 每 14 分鐘


def _fetch_single_camera(nid: str, cam_url: str):
    """抓取單一節點相機圖片，結果存入 camera_cache（由 ThreadPoolExecutor 並行呼叫）"""
    import requests as _req
    try:
        resp = _req.get(cam_url, timeout=5)  # 超時設為 5 秒，不造成其他節點阻塞
        if resp.status_code == 200 and resp.content:
            camera_cache[nid] = {"data": resp.content, "ts": time.time()}
            logger.debug("Camera prefetch [%s] OK, %d bytes", nid, len(resp.content))
    except Exception as e:
        logger.debug("Camera prefetch [%s] failed: %s", nid, e)


def _camera_prefetch_loop():
    """
    背景執行緒：每 CAMERA_PREFETCH_INTERVAL 秒並行抓取所有節點相機圖片
    - 使用 ThreadPoolExecutor 並行，單節點超時不會阻塞其他節點
    - TTL > 預取間隔，確保快取永遠有效，消除空窗期
    """
    from concurrent.futures import ThreadPoolExecutor
    from database import get_all_nodes as _get_all_nodes
    # 等待服務完全啟動並載入節點資料
    time.sleep(15)
    logger.info("Camera prefetch loop started (interval=%ds, TTL=%ds)",
                CAMERA_PREFETCH_INTERVAL, CAMERA_CACHE_TTL)
    while True:
        try:
            all_nodes = _get_all_nodes()
            tasks = []
            for node in all_nodes:
                nid = node.get("node_id")
                cam_url = node.get("camera_url") or CAMERA_URL
                if cam_url:
                    tasks.append((nid, cam_url))
            if tasks:
                # 並行抓取，最多 8 個執行緒同時執行
                with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
                    pool.map(lambda t: _fetch_single_camera(*t), tasks)
        except Exception as e:
            logger.debug("Camera prefetch loop error: %s", e)
        time.sleep(CAMERA_PREFETCH_INTERVAL)  # 每 CAMERA_PREFETCH_INTERVAL 秒預取一次


@app.get("/api/health")
async def health_check():
    """健康檢查端點（Keep-Alive 使用）"""
    return {"status": "ok", "timestamp": now_hkt()}


@app.get("/api/mqtt/status")
async def mqtt_status_api():
    """返回 MQTT 連線狀態與最新感測器數據（免登入）"""
    # 每次訪問自動刷新 topic map，確保資料庫變更能即時生效
    _refresh_mqtt_topic_map()
    return {
        "mqtt": mqtt_status,
        "sensor_cache": sensor_cache,
        "topic_map": _mqtt_topic_map,
        "timestamp": now_hkt(),
    }


@app.post("/api/mqtt/refresh")
async def mqtt_refresh_api(current_user: dict = Depends(get_current_user)):
    """手動刷新 MQTT topic map（需登入）"""
    _refresh_mqtt_topic_map()
    return {
        "status": "refreshed",
        "topic_map": _mqtt_topic_map,
        "timestamp": now_hkt(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3 — 認證 API
# ═══════════════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"  # admin | user
    display_name: str = ""


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="帳號或密碼錯誤",
        )
    # 更新最後登入時間（使用連線池）
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE users SET last_login=NOW() WHERE username=%s", (req.username,))

    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "username":     user["username"],
            "role":         user["role"],
            "display_name": user["display_name"],
        }
    }


@app.get("/api/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "username":     current_user["username"],
        "role":         current_user["role"],
        "display_name": current_user["display_name"],
        "last_login":   current_user.get("last_login"),
    }


@app.get("/api/auth/users")
async def list_users(admin: dict = Depends(require_admin)):
    from database import _to_hkt_str as _to_hkt
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT id, username, role, display_name, created_at, last_login FROM users ORDER BY id")
        rows = cur.fetchall()
    result = []
    for r in rows:
        d = dict(r)
        for ts_col in ('last_login', 'created_at'):
            if d.get(ts_col) is not None and not isinstance(d[ts_col], str):
                d[ts_col] = _to_hkt(d[ts_col])
        result.append(d)
    return result


@app.post("/api/auth/users")
async def create_user(req: CreateUserRequest, admin: dict = Depends(require_admin)):
    # 角色驗證：只允許 admin 或 user
    if req.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="角色必須為 admin 或 user")
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (username, password, role, display_name) VALUES (%s, %s, %s, %s)",
                (req.username, _hash_password(req.password), req.role, req.display_name)
            )
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(status_code=400, detail="用戶名稱已存在")
    return {"message": f"用戶 {req.username} 已建立，角色：{req.role}"}


@app.delete("/api/auth/users/{user_id}")
async def delete_user(user_id: int, admin: dict = Depends(require_admin)):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE id=%s AND username != 'admin'", (user_id,))
    return {"message": "用戶已刪除"}


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3 — 節點管理 API
# ═══════════════════════════════════════════════════════════════════════════════

class NodeRequest(BaseModel):
    node_id: str
    name: str
    location_name: str = ""
    lat: float = 22.3193
    lng: float = 114.1694
    floor: str = ""
    product: str = "banana"
    initial_dsl: float = 10.0
    storage_date: Optional[str] = None   # 'YYYY-MM-DD' 入庫日期
    days_stored: float = 1.0             # 由後端依 storage_date 自動計算
    base_price: float = 100.0
    camera_url: str = ""
    mqtt_topic: str = ""
    blynk_token: str = ""
    status: str = "active"


@app.get("/api/nodes")
async def list_nodes(current_user: dict = Depends(get_current_user)):
    nodes = get_all_nodes()
    # 為每個節點附加最新感測器數據
    for n in nodes:
        nid = n["node_id"]
        n["latest_sensor"] = sensor_cache.get(nid, {"temperature": None, "humidity": None})
    return nodes


def _calc_days_stored(storage_date_str: Optional[str]) -> float:
    """依入庫日期計算已存放天數（整數天）"""
    if not storage_date_str:
        return 1.0
    try:
        from datetime import date
        sd = date.fromisoformat(storage_date_str)
        diff = (date.today() - sd).days
        return float(max(0, diff))
    except Exception:
        return 1.0


@app.post("/api/nodes")
async def create_node(req: NodeRequest, admin: dict = Depends(require_admin)):
    data = req.dict()
    # 自動依入庫日期計算已存放天數
    if data.get("storage_date"):
        data["days_stored"] = _calc_days_stored(data["storage_date"])
    upsert_node(data)
    # 初始化感測器快取
    sensor_cache[req.node_id] = {"temperature": None, "humidity": None, "timestamp": None}
    # 更新 MQTT 主題對應表
    _refresh_mqtt_topic_map()
    return {"message": f"節點 {req.node_id} 已建立/更新"}


@app.put("/api/nodes/{node_id}")
async def update_node(node_id: str, req: NodeRequest, admin: dict = Depends(require_admin)):
    data = req.dict()
    data["node_id"] = node_id
    # 自動依入庫日期計算已存放天數
    if data.get("storage_date"):
        data["days_stored"] = _calc_days_stored(data["storage_date"])
    upsert_node(data)
    # 更新 MQTT 主題對應表
    _refresh_mqtt_topic_map()
    return {"message": f"節點 {node_id} 已更新"}


@app.delete("/api/nodes/{node_id}")
async def delete_node(node_id: str, admin: dict = Depends(require_admin)):
    from database import delete_node as db_delete_node
    db_delete_node(node_id)
    sensor_cache.pop(node_id, None)
    # 更新 MQTT 主題對應表
    _refresh_mqtt_topic_map()
    return {"message": f"節點 {node_id} 已刪除"}


@app.get("/api/nodes/{node_id}/readings")
async def node_readings(node_id: str, limit: int = 100,
                        current_user: dict = Depends(get_current_user)):
    return get_node_readings(node_id, limit)


class NodeSettingsRequest(BaseModel):
    """PATCH /api/nodes/{node_id}/settings 的請求體，只更新評估設定欄位"""
    initial_dsl:  float = 10.0
    storage_date: Optional[str] = None   # 'YYYY-MM-DD'
    base_price:   float = 100.0
    product:      str = "banana"


@app.patch("/api/nodes/{node_id}/settings")
async def patch_node_settings(
    node_id: str,
    req: NodeSettingsRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    儲存節點的評估設定（初始 DSL、入庫日期、原始售價、產品類型）。
    所有登入用戶均可呼叫（不需要 admin 權限）。
    """
    node = get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"節點 {node_id} 不存在")
    update_node_settings(
        node_id=node_id,
        initial_dsl=req.initial_dsl,
        storage_date=req.storage_date,
        base_price=req.base_price,
        product=req.product,
    )
    # 同步更新 AI 快取中的 storage_days（讓下次自動推理使用新天數）
    if node_id in ai_cache:
        from datetime import date
        try:
            if req.storage_date:
                sd = date.fromisoformat(req.storage_date)
                new_days = float(max(0, (date.today() - sd).days))
            else:
                new_days = ai_cache[node_id].get("storage_days", 1.0)
            ai_cache[node_id]["storage_days"] = new_days
        except Exception:
            pass
    return {"message": f"節點 {node_id} 評估設定已儲存",
            "initial_dsl": req.initial_dsl,
            "storage_date": req.storage_date,
            "base_price": req.base_price,
            "product": req.product}


@app.get("/api/nodes/{node_id}/predictions")
async def node_predictions(node_id: str, limit: int = 50,
                           current_user: dict = Depends(get_current_user)):
    return get_node_predictions(node_id, limit)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3 — 後台儀表板
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/admin/dashboard")
async def admin_dashboard(admin: dict = Depends(require_admin)):
    return get_dashboard_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3 — AI 模型熱更新 API
# ═══════════════════════════════════════════════════════════════════════════════

def _reload_model(model_path: str, scaler_path: str) -> tuple:
    """即時重載 AI 模型與 scaler，回傳 (model, scaler)"""
    new_model = HybridModel().to(DEVICE)
    new_model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    new_model.eval()
    new_scaler = joblib.load(scaler_path)
    return new_model, new_scaler


@app.get("/api/model/status")
async def model_status(admin: dict = Depends(require_admin)):
    """查看目前 AI 模型版本與推理狀態"""
    cur = model_registry["current"]
    history = model_registry["history"]
    return {
        "current": {
            "version":         cur["version"],
            "uploaded_at":     cur["uploaded_at"],
            "uploaded_by":     cur["uploaded_by"],
            "description":     cur["description"],
            "model_path":      cur["model_path"],
            "scaler_path":     cur["scaler_path"],
            "inference_count": cur["inference_count"],
            "model_loaded":    model_ai is not None,
            "scaler_loaded":   scaler is not None,
        },
        "history_count": len(history),
        "history": [
            {
                "version":     h["version"],
                "uploaded_at": h["uploaded_at"],
                "uploaded_by": h["uploaded_by"],
                "description": h["description"],
            }
            for h in history
        ],
        "can_rollback": len(history) > 0,
    }


@app.post("/api/model/upload")
async def upload_model(
    admin: dict = Depends(require_admin),
    model_file: Optional[bytes] = None,
    scaler_file: Optional[bytes] = None,
):
    """
    上傳新模型（支援僅上傳 .pth 或僅上傳 .pkl 或兩者同時上傳）
    上傳後系統即時重載模型，不需重新部署
    """
    from fastapi import UploadFile, File, Form
    raise HTTPException(status_code=400, detail="請使用 multipart/form-data 上傳")


from fastapi import UploadFile, File, Form


@app.post("/api/model/upload-form")
async def upload_model_form(
    admin: dict = Depends(require_admin),
    version: str = Form(default=""),
    description: str = Form(default=""),
    model_file: Optional[UploadFile] = File(default=None),
    scaler_file: Optional[UploadFile] = File(default=None),
):
    """
    上傳新 AI 模型（multipart/form-data）
    - model_file: .pth 模型檔案（可選）
    - scaler_file: .pkl scaler 檔案（可選）
    - version: 版本號（如 v2.0）
    - description: 版本說明
    """
    global model_ai, scaler, model_registry

    if model_file is None and scaler_file is None:
        raise HTTPException(status_code=400, detail="請至少上傳 model_file (.pth) 或 scaler_file (.pkl) 其中一個")

    ts_str = datetime.now(_HKT).strftime("%Y%m%d_%H%M%S")
    new_version = version.strip() or f"v-{ts_str}"
    new_model_path  = model_registry["current"]["model_path"]
    new_scaler_path = model_registry["current"]["scaler_path"]

    # 備份目前版本到 history
    old_entry = dict(model_registry["current"])
    model_registry["history"].insert(0, old_entry)
    if len(model_registry["history"]) > 5:
        model_registry["history"] = model_registry["history"][:5]

    # 儲存新模型檔案
    if model_file is not None:
        content = await model_file.read()
        if len(content) < 1000:
            raise HTTPException(status_code=400, detail="模型檔案過小，請確認檔案格式正確 (.pth)")
        new_model_path = os.path.join(MODEL_BACKUP_DIR, f"model_{ts_str}.pth")
        with open(new_model_path, "wb") as f:
            f.write(content)
        logger.info("New model file saved: %s (%d bytes)", new_model_path, len(content))

    if scaler_file is not None:
        content = await scaler_file.read()
        if len(content) < 100:
            raise HTTPException(status_code=400, detail="Scaler 檔案過小，請確認檔案格式正確 (.pkl)")
        new_scaler_path = os.path.join(MODEL_BACKUP_DIR, f"scaler_{ts_str}.pkl")
        with open(new_scaler_path, "wb") as f:
            f.write(content)
        logger.info("New scaler file saved: %s (%d bytes)", new_scaler_path, len(content))

    # 即時重載模型
    try:
        new_model, new_sc = _reload_model(new_model_path, new_scaler_path)
        model_ai = new_model
        scaler   = new_sc
        logger.info("Model hot-reloaded successfully: %s", new_version)
    except Exception as e:
        # 重載失敗：回滚到舊版本
        model_registry["history"].pop(0)
        logger.error("Model reload failed: %s", e)
        raise HTTPException(status_code=500, detail=f"模型重載失敗：{e}")

    # 更新版本記錄
    model_registry["current"] = {
        "model_path":      new_model_path,
        "scaler_path":     new_scaler_path,
        "version":         new_version,
        "uploaded_at":     now_hkt(),
        "uploaded_by":     admin["username"],
        "description":     description or f"上傳於 {ts_str}",
        "inference_count": 0,
    }

    # 清空 AI 快取，下一輪推理將使用新模型
    ai_cache.clear()

    return {
        "message":     f"模型已成功更新並重載！版本：{new_version}",
        "version":     new_version,
        "uploaded_at": model_registry["current"]["uploaded_at"],
        "model_file":  model_file.filename if model_file else "未更新",
        "scaler_file": scaler_file.filename if scaler_file else "未更新",
    }


@app.post("/api/model/rollback")
async def rollback_model(admin: dict = Depends(require_admin)):
    """回滚到上一個模型版本"""
    global model_ai, scaler, model_registry

    if not model_registry["history"]:
        raise HTTPException(status_code=400, detail="沒有可回滚的歷史版本")

    # 取出上一個版本
    prev = model_registry["history"].pop(0)

    # 尝試重載
    try:
        new_model, new_sc = _reload_model(prev["model_path"], prev["scaler_path"])
        model_ai = new_model
        scaler   = new_sc
        logger.info("Model rolled back to: %s", prev["version"])
    except Exception as e:
        # 回滚失敗：把舊版本放回
        model_registry["history"].insert(0, prev)
        raise HTTPException(status_code=500, detail=f"回滚失敗：{e}")

    # 將目前版本備份到 history
    cur = dict(model_registry["current"])
    model_registry["history"].insert(0, cur)
    if len(model_registry["history"]) > 5:
        model_registry["history"] = model_registry["history"][:5]

    # 更新版本記錄
    model_registry["current"] = prev
    ai_cache.clear()

    return {
        "message":     f"已回滚到版本：{prev['version']}",
        "version":     prev["version"],
        "uploaded_at": prev["uploaded_at"],
        "description": prev["description"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2 — 感測器 & 相機 API（保持向下相容）
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/sensor")
async def get_sensor(node_id: str = "A01"):
    return JSONResponse(content=sensor_cache.get(node_id, {
        "temperature": None, "humidity": None, "timestamp": None
    }))


@app.get("/api/image")
async def proxy_image(node_id: str = "A01"):
    # 取得節點相機 URL
    node = get_node(node_id)
    cam_url = (node.get("camera_url") or CAMERA_URL) if node else CAMERA_URL
    if not cam_url:
        cam_url = CAMERA_URL

    # 檢查記憶體快取（TTL = 3 秒）
    cached = camera_cache.get(node_id)
    if cached and (time.time() - cached["ts"]) < CAMERA_CACHE_TTL:
        logger.debug("/api/image [%s] cache HIT (age=%.1fs)", node_id, time.time() - cached["ts"])
        return StreamingResponse(
            io.BytesIO(cached["data"]),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Cache": "HIT"},
        )

    # 快取未命中，向 M5Stack 抓取新圖片
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(cam_url)
            resp.raise_for_status()
        img_data = resp.content
        # 儲存到記憶體快取
        camera_cache[node_id] = {"data": img_data, "ts": time.time()}
        logger.debug("/api/image [%s] cache MISS, fetched %d bytes", node_id, len(img_data))
        return StreamingResponse(
            io.BytesIO(img_data),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Cache": "MISS"},
        )
    except Exception as e:
        # 抓取失敗時，若有舊快取則回傳舊圖片（不要讓前端看到空白）
        if cached:
            logger.warning("/api/image [%s] fetch failed, returning stale cache: %s", node_id, e)
            return StreamingResponse(
                io.BytesIO(cached["data"]),
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store", "X-Cache": "STALE"},
            )
        raise HTTPException(status_code=502, detail="Camera fetch failed: {}".format(e))


@app.get("/api/image/stream")
async def stream_camera(node_id: str = "A01"):
    """
    MJPEG 即時影像串流端點：持續推送相機畫面（每 2 秒一幀）
    前端只需把 <img src="/api/image/stream?node_id=XXX"> 即可得到持續更新的影像
    """
    import asyncio

    async def generate():
        boundary = b"--frame"
        while True:
            # 從快取取得最新圖片
            cached = camera_cache.get(node_id)
            if cached and cached.get("data"):
                frame = cached["data"]
                yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(2)  # 每 2 秒推送一幀

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/image/prefetch")
async def prefetch_image(node_id: str = "A01"):
    """後端主動預取相機圖片到快取，前端呼叫後立即再呼叫 /api/image 即可得到快取圖片"""
    node = get_node(node_id)
    cam_url = (node.get("camera_url") or CAMERA_URL) if node else CAMERA_URL
    if not cam_url:
        cam_url = CAMERA_URL
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(cam_url)
            resp.raise_for_status()
        camera_cache[node_id] = {"data": resp.content, "ts": time.time()}
        logger.info("/api/image/prefetch [%s] OK, %d bytes cached", node_id, len(resp.content))
        return JSONResponse({"status": "ok", "size": len(resp.content), "node_id": node_id})
    except Exception as e:
        logger.warning("/api/image/prefetch [%s] failed: %s", node_id, e)
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=502)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2 — 品質評估 API
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/quality/params")
async def get_quality_params():
    products_with_q0 = {}
    for pid, p in PRODUCT_PARAMS.items():
        k_ref_val = calc_k_comp_ref(p["Ea"], p["k_ref"], p["optimal_rh_min"], p["optimal_rh_max"])
        q0_val = Q_THRESHOLD + k_ref_val * p["initial_dsl_days"]
        products_with_q0[pid] = {
            **p,
            "k_comp_ref": round(k_ref_val, 4),
            "q0_from_initial_dsl": round(q0_val, 2),
        }
    return JSONResponse(content={
        "constants": {
            "R": R, "T_ref_K": T_REF_K, "T_ref_C": T_REF_K - 273.15,
            "Q0": Q0, "Q_threshold": Q_THRESHOLD,
            "alpha_formula": ALPHA_FORMULA, "alpha_ai": ALPHA_AI,
        },
        "products": products_with_q0,
        "discount_rules": [
            {"dsl_threshold_days": t, "discount_pct": p, "reason": r}
            for t, p, r in DISCOUNT_RULES
        ],
        "formulas": {
            "zero_order":   "Q(t) = Q0 - k_comp * t",
            "arrhenius":    "k(T) = k_ref * exp( Ea/R * (1/T_ref - 1/T) )",
            "comprehensive":"k_comp = k(T) * f_H(H) * f_L(L) * f_A(A)",
            "dsl":          "DSL = (Q(t) - Q_threshold) / k_comp",
            "ai_quality":   "Q_AI = 100 - S",
            "combined":     "Q_combined = alpha * Q(t) + (1-alpha) * Q_AI",
            "combined_dsl": "DSL_combined = (Q_combined - Q_threshold) / k_comp",
        },
    })


class QualityRequest(BaseModel):
    storage_days: float
    base_price: float = 100.0
    product: str = "banana"
    light_lux: float = 500.0
    air_velocity: float = 0.3
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    ai_spoilage: Optional[float] = None
    initial_dsl: Optional[float] = None
    node_id: str = "A01"
    save_record: bool = True


@app.post("/api/quality")
async def calc_quality(req: QualityRequest):
    node_sensor = sensor_cache.get(req.node_id, {})
    # 檢查 MQTT 數據是否真實有效（必須有 timestamp，且在 10 分鐘內）
    _mqtt_ts = node_sensor.get("timestamp")
    _mqtt_fresh = False
    if _mqtt_ts:
        try:
            _age = (datetime.now(_HKT) - datetime.strptime(_mqtt_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_HKT)).total_seconds()
            _mqtt_fresh = _age < 600  # 10 分鐘內視為有效
        except Exception:
            _mqtt_fresh = False

    temp = req.temperature if req.temperature is not None else (node_sensor.get("temperature") if _mqtt_fresh else None)
    hum  = req.humidity    if req.humidity    is not None else (node_sensor.get("humidity")    if _mqtt_fresh else None)

    # 無 MQTT 數據時使用預設環境值（室溫 25°C、濕度 65%），不中斷評估
    using_default_sensor = False
    if temp is None or hum is None:
        temp = 25.0
        hum  = 65.0
        using_default_sensor = True
        logger.info("/api/quality [%s]: no MQTT data, using default sensor (25°C, 65%%)", req.node_id)
    elif req.temperature is None and req.humidity is None and _mqtt_fresh:
        # 明確標記使用的是 MQTT 即時數據（非手動輸入）
        using_default_sensor = False

    initial_dsl = req.initial_dsl
    if initial_dsl is None:
        params = PRODUCT_PARAMS.get(req.product, PRODUCT_PARAMS["banana"])
        initial_dsl = float(params["initial_dsl_days"])

    result = calculate_quality(
        temperature=temp, humidity=hum, storage_days=req.storage_days,
        light_lux=req.light_lux, air_velocity=req.air_velocity,
        product=req.product, ai_spoilage=req.ai_spoilage, initial_dsl=initial_dsl,
    )
    data = quality_result_to_dict(result, base_price=req.base_price)

    # 記錄到資料庫（含相機圖片 Base64）
    if req.save_record:
        # 嘗試抓取相機圖片
        _cam_snapshot_url = None
        _cam_image_base64 = None
        try:
            _node = get_node(req.node_id)
            _cam_url = (_node.get("camera_url") or CAMERA_URL) if _node else CAMERA_URL
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=8.0) as _client:
                _resp = await _client.get(_cam_url)
                _resp.raise_for_status()
            _img = Image.open(io.BytesIO(_resp.content)).convert("RGB")
            _ts = int(time.time())
            _cam_snapshot_url = f"{_cam_url}?_snap={_ts}"
            _buf = io.BytesIO()
            _img_save = _img.copy()
            _img_save.thumbnail((640, 480), Image.LANCZOS)
            _img_save.save(_buf, format="JPEG", quality=70, optimize=True)
            _cam_image_base64 = base64.b64encode(_buf.getvalue()).decode("utf-8")
        except Exception as _cam_err:
            logger.warning("/api/quality [%s] camera fetch failed: %s", req.node_id, _cam_err)
        try:
            insert_prediction(req.node_id, {
                "storage_days": req.storage_days, "temperature": temp, "humidity": hum,
                "ai_spoilage": req.ai_spoilage, "quality_ai": data.get("quality_ai"),
                "quality_formula": data["quality_formula"],
                "quality_combined": data["quality_score"],
                "dsl_combined": data["dsl_days"],
                "discount_pct": data["discount_pct"],
                "base_price": req.base_price, "final_price": data["final_price"],
                "freshness_label": data["freshness_label"], "product": req.product,
                "camera_snapshot_url": _cam_snapshot_url,
                "camera_image_base64": _cam_image_base64,
            })
            logger.info("/api/quality [%s] saved with image=%s", req.node_id, "yes" if _cam_image_base64 else "no")
        except Exception as e:
            logger.warning("Save prediction failed: %s", e)

    # ── 同步更新 ai_cache，讓電子報價牌即時反映最新評估結果 ──────────────
    # 手動評估的優先級高於自動推理（因為包含更完整的參數設定）
    ai_spoilage_sync = req.ai_spoilage if req.ai_spoilage is not None else (
        ai_cache.get(req.node_id, {}).get("spoilage")
    )
    freshness_label = data.get("freshness_label", "")
    # 依新鮮度標籤決定顏色
    if "非常新鮮" in freshness_label:
        ai_color = "#22c55e"
        ai_label = f"非常新鮮 🟢"
    elif "良好新鮮" in freshness_label:
        ai_color = "#84cc16"
        ai_label = f"良好新鮮 🟡"
    elif "輕微老化" in freshness_label:
        ai_color = "#f59e0b"
        ai_label = f"輕微老化 🟠"
    elif "明顯老化" in freshness_label:
        ai_color = "#ef4444"
        ai_label = f"明顯老化 🔴"
    else:
        ai_color = "#ef4444"
        ai_label = freshness_label

    manual_result = {
        "spoilage":     ai_spoilage_sync,
        "ai_label":     ai_label,
        "ai_color":     ai_color,
        "timestamp":    now_hkt(),
        "quality_data": data,
        "storage_days": req.storage_days,
        "source":       "manual",   # 標記來源為手動評估
    }
    ai_cache[req.node_id] = manual_result
    save_ai_cache_to_db(req.node_id, manual_result)  # 持久化手動評估結果
    logger.info("ai_cache updated from manual assessment: node=%s Q=%.1f discount=%s%%",
                req.node_id, data.get("quality_score", 0), data.get("discount_pct", 0))

    data["using_default_sensor"] = using_default_sensor
    data["sensor_source"] = ("default (25°C/65%)" if using_default_sensor
                             else ("manual" if req.temperature is not None or req.humidity is not None else "mqtt"))
    return data


class PredictRequest(BaseModel):
    storage_time: float
    base_price: float = 100.0
    product: str = "banana"
    light_lux: float = 500.0
    air_velocity: float = 0.3
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    initial_dsl: Optional[float] = None
    node_id: str = "A01"
    save_record: bool = True


@app.post("/api/predict")
async def predict(req: PredictRequest):
    try:
        return await _predict_impl(req)
    except HTTPException:
        raise
    except Exception as _e:
        import traceback
        logger.error("/api/predict unhandled error: %s\n%s", _e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"predict error: {_e}")

async def _predict_impl(req: PredictRequest):
    if model_ai is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    node_sensor = sensor_cache.get(req.node_id, {})
    # 檢查 MQTT 數據是否真實有效（必須有 timestamp，且在 10 分鐘內）
    _mqtt_ts2 = node_sensor.get("timestamp")
    _mqtt_fresh2 = False
    if _mqtt_ts2:
        try:
            _age2 = (datetime.now(_HKT) - datetime.strptime(_mqtt_ts2, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_HKT)).total_seconds()
            _mqtt_fresh2 = _age2 < 600  # 10 分鐘內視為有效
        except Exception:
            _mqtt_fresh2 = False

    temp = req.temperature if req.temperature is not None else (node_sensor.get("temperature") if _mqtt_fresh2 else None)
    hum  = req.humidity    if req.humidity    is not None else (node_sensor.get("humidity")    if _mqtt_fresh2 else None)
    # 無 MQTT 數據時使用預設環境值（室溫 25°C、濕度 65%），不中斷評估
    using_default_sensor = False
    if temp is None or hum is None:
        temp = 25.0
        hum  = 65.0
        using_default_sensor = True
        logger.info("/api/predict [%s]: no MQTT data, using default sensor (25°C, 65%%)", req.node_id)

    # 取得節點相機 URL
    node = get_node(req.node_id)
    cam_url = (node.get("camera_url") or CAMERA_URL) if node else CAMERA_URL

    camera_snapshot_url = None
    camera_image_base64 = None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(cam_url)
            resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        # 記錄快照 URL（帶時間戳）
        ts_param = int(time.time())
        camera_snapshot_url = f"{cam_url}?_snap={ts_param}" if cam_url else None
        # 將圖片壓縮後轉為 Base64 字串儲存
        try:
            buf = io.BytesIO()
            img_save = image.copy()
            img_save.thumbnail((640, 480), Image.LANCZOS)
            img_save.save(buf, format="JPEG", quality=70, optimize=True)
            camera_image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as b64_err:
            logger.warning("/api/predict [%s] base64 encode error: %s", req.node_id, b64_err)
            camera_image_base64 = None
    except Exception as e:
        logger.warning("/api/predict [%s] camera fetch failed: %s — using blank image fallback", req.node_id, e)
        # 相機失敗時使用空白圖片繼續推理（不中斷評估）
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    img_tensor = infer_transform(image).unsqueeze(0).to(DEVICE)

    model_hum     = hum / 100.0 if hum > 1.0 else hum
    storage_hours = req.storage_time * 24.0
    raw_sensor    = np.array([[temp, model_hum, storage_hours]], dtype=np.float32)
    scaled_sensor = scaler.transform(raw_sensor)
    sensor_tensor = torch.tensor(scaled_sensor, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        prediction = model_ai(img_tensor, sensor_tensor).item()
    # 訓練標籤範圍是 1–20，縮放到 0–100
    # 公式：spoilage = (raw_output - 1) / (20 - 1) * 100
    raw_clipped = float(np.clip(prediction, 1.0, 20.0))
    spoilage = (raw_clipped - 1.0) / 19.0 * 100.0
    logger.info("/api/predict raw_output=%.4f -> spoilage=%.2f%%", prediction, spoilage)

    initial_dsl = req.initial_dsl
    if initial_dsl is None:
        params_p = PRODUCT_PARAMS.get(req.product, PRODUCT_PARAMS["banana"])
        initial_dsl = float(params_p["initial_dsl_days"])

    quality_result = calculate_quality(
        temperature=temp, humidity=hum, storage_days=req.storage_time,
        light_lux=req.light_lux, air_velocity=req.air_velocity,
        product=req.product, ai_spoilage=spoilage, initial_dsl=initial_dsl,
    )
    quality_data = quality_result_to_dict(quality_result, base_price=req.base_price)

    if spoilage < 20:   ai_label, ai_color = "非常新鮮 🟢", "#22c55e"
    elif spoilage < 40: ai_label, ai_color = "新鮮 🟡", "#eab308"
    elif spoilage < 60: ai_label, ai_color = "輕微腐敗 🟠", "#f97316"
    elif spoilage < 80: ai_label, ai_color = "中度腐敗 🔴", "#ef4444"
    else:               ai_label, ai_color = "嚴重腐敗 ⚫", "#7f1d1d"

    # 記錄到資料庫（含相機圖片 Base64）
    if req.save_record:
        try:
            insert_prediction(req.node_id, {
                "storage_days": req.storage_time, "temperature": temp, "humidity": hum,
                "ai_spoilage": spoilage, "quality_ai": quality_data.get("quality_ai"),
                "quality_formula": quality_data["quality_formula"],
                "quality_combined": quality_data["quality_score"],
                "dsl_combined": quality_data["dsl_days"],
                "discount_pct": quality_data["discount_pct"],
                "base_price": req.base_price, "final_price": quality_data["final_price"],
                "freshness_label": quality_data["freshness_label"], "product": req.product,
                "camera_snapshot_url": camera_snapshot_url,
                "camera_image_base64": camera_image_base64,
            })
            logger.info("/api/predict [%s] saved with image=%s", req.node_id, "yes" if camera_image_base64 else "no")
        except Exception as e:
            logger.warning("Save prediction failed: %s", e)
    # ── 同步更新 ai_cache，讓電子報價牌即時反映最新完整推理結果 ─────────────────
    predict_result = {
        "spoilage":     spoilage,
        "ai_label":     ai_label,
        "ai_color":     ai_color,
        "timestamp":    now_hkt(),
        "quality_data": quality_data,
        "storage_days": req.storage_time,
        "source":       "manual_predict",
    }
    ai_cache[req.node_id] = predict_result
    save_ai_cache_to_db(req.node_id, predict_result)  # 持久化到 DB
    logger.info("ai_cache updated from /api/predict: node=%s spoilage=%.1f Q=%.1f discount=%s%%",
                req.node_id, spoilage, quality_data.get("quality_score", 0), quality_data.get("discount_pct", 0))

    return {
        "spoilage_level":   round(spoilage, 2),
        "ai_label":         ai_label,
        "ai_color":         ai_color,
        "quality_ai":       quality_data["quality_ai"],
        "quality_formula":  quality_data["quality_formula"],
        "dsl_formula":      quality_data["dsl_formula"],
        "quality_combined": quality_data["quality_score"],
        "freshness_label":  quality_data["freshness_label"],
        "freshness_color":  quality_data["freshness_color"],
        "dsl_combined":     quality_data["dsl_days"],
        "alpha_formula":    quality_data["alpha_formula"],
        "alpha_ai":         quality_data["alpha_ai"],
        "initial_dsl":      quality_data["initial_dsl"],
        "discount_pct":     quality_data["discount_pct"],
        "discount_reason":  quality_data["discount_reason"],
        "base_price":       req.base_price,
        "final_price":      quality_data["final_price"],
        "formula":          quality_data["formula"],
        "sensor": {
            "temperature":  round(temp, 1),
            "humidity":     round(hum, 1),
            "storage_time": req.storage_time,
        },
        "using_default_sensor": using_default_sensor,
        "sensor_source": "default (25°C/65%)" if using_default_sensor
                         else ("manual" if req.temperature is not None or req.humidity is not None else "mqtt"),
    }
# ═══════════════════════════════════════════════════════════════════════════════
# 公開報價牌 API（免登入）
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/display/all")
async def display_all_nodes_v2():
    """
    公開銷售版面 API — 免登入
    回傳所有節點的最新品質評估與定價資訊，供銷售排行版面使用
    """
    nodes_list = get_all_nodes()
    result = []

    for node in nodes_list:
        node_id = node["node_id"]
        sensor = sensor_cache.get(node_id, {})
        ai_result = ai_cache.get(node_id)

        quality_data = None
        ai_spoilage_val = None
        ai_label_val = None
        ai_color_val = None
        ai_infer_ts = None
        source = None

        if ai_result:
            quality_data    = ai_result.get("quality_data")
            ai_spoilage_val = ai_result.get("spoilage")
            ai_label_val    = ai_result.get("ai_label")
            ai_color_val    = ai_result.get("ai_color")
            ai_infer_ts     = ai_result.get("timestamp")
            source          = ai_result.get("source", "auto")

        product_type = node.get("product") or node.get("product_type", "banana")
        product_names = {"banana": "香蕉", "apple": "蘋果", "tomato": "番茄", "lettuce": "萵苣"}

        result.append({
            "node_id":       node_id,
            "name":          node.get("name"),
            "location_name": node.get("location_name"),
            "status":        node.get("status", "active"),
            "product":       product_type,
            "product_name":  product_names.get(product_type, product_type),
            "base_price":    node.get("base_price", 100.0),
            "sensor": {
                "temperature": sensor.get("temperature"),
                "humidity":    sensor.get("humidity"),
                "online":      sensor.get("temperature") is not None,
            },
            "ai": {
                "spoilage":    ai_spoilage_val,
                "label":       ai_label_val,
                "color":       ai_color_val,
                "inferred_at": ai_infer_ts,
                "source":      source,
            },
            "quality": quality_data,
            "updated_at": now_hkt(),
        })

    return {"nodes": result, "total": len(result), "updated_at": now_hkt()}


@app.get("/api/display/{node_id}")
async def display_data(node_id: str):
    """
    公開報價牌 API — 免登入
    回傳：節點資訊、即時感測器數據、最新品質評估結果
    """
    # 取得節點資訊
    node = get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"節點 {node_id} 不存在")

    # 取得即時感測器數據（MQTT 快取）
    sensor = sensor_cache.get(node_id, {})
    temperature = sensor.get("temperature")
    humidity    = sensor.get("humidity")
    sensor_ts   = sensor.get("timestamp")

    # 取得最新 AI 推理結果（自動推理快取）
    ai_result = ai_cache.get(node_id)

    # 取得最新評估記錄（用於備用）
    predictions = get_node_predictions(node_id, limit=1)
    latest = predictions[0] if predictions else None

    # 品質數據優先使用 ai_cache（自動推理結果）
    quality_data = None
    ai_spoilage_val = None
    ai_label_val    = None
    ai_color_val    = None
    ai_infer_ts     = None

    if ai_result:
        # 自動推理快取有效
        quality_data    = ai_result.get("quality_data")
        ai_spoilage_val = ai_result.get("spoilage")
        ai_label_val    = ai_result.get("ai_label")
        ai_color_val    = ai_result.get("ai_color")
        ai_infer_ts     = ai_result.get("timestamp")
    elif temperature is not None and humidity is not None:
        # 備用：若 AI 快取尚空（第一次啟動），用純公式計算
        try:
            product     = node.get("product_type", "banana")
            initial_dsl = node.get("initial_dsl", None)
            storage_days = latest.get("storage_days", 1) if latest else 1
            base_price   = node.get("base_price", 100.0)
            result = calculate_quality(
                temperature=temperature, humidity=humidity,
                storage_days=storage_days, product=product,
                initial_dsl=initial_dsl, ai_spoilage=None,
            )
            quality_data = quality_result_to_dict(result, base_price=base_price)
        except Exception as e:
            logger.warning("Display quality fallback error: %s", e)

    # 取得最近感測器歷史（用於圖表，最近 20 筆）
    readings = get_node_readings(node_id, limit=20)

    return {
        "node": {
            "node_id":       node.get("node_id"),
            "name":          node.get("name"),
            "location_name": node.get("location_name"),
            "product_type":  node.get("product_type", "banana"),
            "base_price":    node.get("base_price", 100.0),
            "initial_dsl":   node.get("initial_dsl"),
            "camera_url":    node.get("camera_url") or CAMERA_URL,
        },
        "sensor": {
            "temperature":  temperature,
            "humidity":     humidity,
            "light_lux":    sensor.get("light_lux") if sensor.get("light_lux") is not None else node.get("light_lux", 375),
            "air_velocity": sensor.get("air_velocity") if sensor.get("air_velocity") is not None else node.get("air_velocity", 0.22),
            "pressure":     sensor.get("pressure"),
            "timestamp":    sensor_ts,
            "online":       temperature is not None,
        },
        "ai": {
            "spoilage":   ai_spoilage_val,
            "label":      ai_label_val,
            "color":      ai_color_val,
            "inferred_at": ai_infer_ts,
            "available":  ai_result is not None,
            "source":     ai_result.get("source", "auto") if ai_result else None,
        },
        "quality":           quality_data,
        "latest_prediction": latest,
        "readings_history":  readings,
        "updated_at":        now_hkt(),
    }


@app.get("/api/ai-status")
async def ai_status():
    """查看 AI 自動推理快取狀態（免登入，用於診斷）"""
    result = {}
    for nid, data in ai_cache.items():
        result[nid] = {
            "spoilage":    data.get("spoilage"),
            "ai_label":    data.get("ai_label"),
            "timestamp":   data.get("timestamp"),
            "storage_days": data.get("storage_days"),
        }
    return {
        "ai_cache_count": len(ai_cache),
        "model_loaded":   model_ai is not None,
        "scaler_loaded":  scaler is not None,
        "interval_sec":   AI_INFER_INTERVAL,
        "nodes":          result,
    }


@app.get("/api/display/{node_id}/sensor")
async def display_sensor_only(node_id: str):
    """輕量感測器快照 API（免登入，用於高頻輪詢）"""
    sensor = sensor_cache.get(node_id, {})
    node   = get_node(node_id)
    # 優先使用 MQTT 即時數據，如果尚未收到則用節點設定的預設值
    light_val = sensor.get("light_lux")
    air_val   = sensor.get("air_velocity")
    return {
        "temperature":  sensor.get("temperature"),
        "humidity":     sensor.get("humidity"),
        "light_lux":    light_val if light_val is not None else (node.get("light_lux", 375) if node else 375),
        "air_velocity": air_val   if air_val   is not None else (node.get("air_velocity", 0.22) if node else 0.22),
        "pressure":     sensor.get("pressure"),
        "timestamp":    sensor.get("timestamp"),
        "online":       sensor.get("temperature") is not None,
    }


# ─── 數據匯出 API ─────────────────────────────────────────────────────────────────

@app.get("/api/export/predictions")
async def export_predictions(
    format: str = "csv",
    node_id: Optional[str] = None,
    limit: int = 1000,
    admin: dict = Depends(require_admin),
):
    """
    匯出歷史評估記錄（用於模型重訓練）
    - format: csv 或 json
    - node_id: 指定節點（空白則匯出所有節點）
    - limit: 最多筆數（預設 1000）
    """
    import csv
    import json as _json
    from io import StringIO

    try:
        with get_db() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            if node_id:
                cur.execute("""
                    SELECT p.*, n.name as node_name
                    FROM predictions p
                    LEFT JOIN nodes n ON p.node_id = n.node_id
                    WHERE p.node_id = %s
                    ORDER BY p.recorded_at DESC
                    LIMIT %s
                """, (node_id, limit))
            else:
                cur.execute("""
                    SELECT p.*, n.name as node_name
                    FROM predictions p
                    LEFT JOIN nodes n ON p.node_id = n.node_id
                    ORDER BY p.recorded_at DESC
                    LIMIT %s
                """, (limit,))
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"資料庫查詢失敗: {e}")

    # 轉換時間格式
    records = []
    for r in rows:
        d = dict(r)
        if d.get('recorded_at'):
            try:
                from datetime import timezone, timedelta
                _HKT = timezone(timedelta(hours=8))
                dt = d['recorded_at']
                if hasattr(dt, 'astimezone'):
                    d['recorded_at'] = dt.astimezone(_HKT).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    d['recorded_at'] = str(dt)
            except Exception:
                d['recorded_at'] = str(d['recorded_at'])
        records.append(d)

    if format.lower() == "json":
        content = _json.dumps({
            "exported_at": now_hkt(),
            "total": len(records),
            "node_id": node_id or "all",
            "data": records
        }, ensure_ascii=False, indent=2)
        filename = f"predictions_{node_id or 'all'}_{datetime.now(_HKT).strftime('%Y%m%d_%H%M%S')}.json"
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    else:
        # CSV 格式（加入 UTF-8 BOM，讓 Excel 正確識別中文編碼）
        from io import BytesIO
        output = StringIO()
        if records:
            fieldnames = list(records[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        # 加 UTF-8 BOM (\xef\xbb\xbf) 解決 Excel 開啟中文亂碼
        csv_bytes = b"\xef\xbb\xbf" + output.getvalue().encode("utf-8")
        filename = f"predictions_{node_id or 'all'}_{datetime.now(_HKT).strftime('%Y%m%d_%H%M%S')}.csv"
        return StreamingResponse(
            iter([csv_bytes]),
            media_type="text/csv; charset=utf-8-sig",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "text/csv; charset=utf-8-sig",
            }
        )


@app.get("/api/export/readings")
async def export_readings(
    format: str = "csv",
    node_id: Optional[str] = None,
    limit: int = 1000,
    admin: dict = Depends(require_admin),
):
    """
    匯出感測器讀數記錄
    - format: csv 或 json
    - node_id: 指定節點（空白則匯出所有節點）
    - limit: 最多筆數（預設 1000）
    """
    import csv
    import json as _json
    from io import StringIO

    try:
        with get_db() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            if node_id:
                cur.execute("""
                    SELECT r.*, n.name as node_name
                    FROM readings r
                    LEFT JOIN nodes n ON r.node_id = n.node_id
                    WHERE r.node_id = %s
                    ORDER BY r.recorded_at DESC
                    LIMIT %s
                """, (node_id, limit))
            else:
                cur.execute("""
                    SELECT r.*, n.name as node_name
                    FROM readings r
                    LEFT JOIN nodes n ON r.node_id = n.node_id
                    ORDER BY r.recorded_at DESC
                    LIMIT %s
                """, (limit,))
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"資料庫查詢失敗: {e}")

    records = []
    for r in rows:
        d = dict(r)
        if d.get('recorded_at'):
            try:
                from datetime import timezone, timedelta
                _HKT = timezone(timedelta(hours=8))
                dt = d['recorded_at']
                if hasattr(dt, 'astimezone'):
                    d['recorded_at'] = dt.astimezone(_HKT).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    d['recorded_at'] = str(dt)
            except Exception:
                d['recorded_at'] = str(d['recorded_at'])
        records.append(d)

    if format.lower() == "json":
        content = _json.dumps({
            "exported_at": now_hkt(),
            "total": len(records),
            "node_id": node_id or "all",
            "data": records
        }, ensure_ascii=False, indent=2)
        filename = f"readings_{node_id or 'all'}_{datetime.now(_HKT).strftime('%Y%m%d_%H%M%S')}.json"
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    else:
        # CSV 格式（加入 UTF-8 BOM，讓 Excel 正確識別中文編碼）
        output = StringIO()
        if records:
            fieldnames = list(records[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        # 加 UTF-8 BOM (小\xef\xbb\xbf) 解決 Excel 開啟中文亂碼
        csv_bytes = b"\xef\xbb\xbf" + output.getvalue().encode("utf-8")
        filename = f"readings_{node_id or 'all'}_{datetime.now(_HKT).strftime('%Y%m%d_%H%M%S')}.csv"
        return StreamingResponse(
            iter([csv_bytes]),
            media_type="text/csv; charset=utf-8-sig",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "text/csv; charset=utf-8-sig",
            }
        )


# ─── 靜態網頁 ─────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/login.html")


@app.get("/login")
async def login_page():
    return FileResponse("static/login.html")


@app.get("/app")
async def app_page():
    return FileResponse(
        "static/app.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )
@app.get("/admin")
async def admin_page():
    return FileResponse(
        "static/admin.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )


@app.get("/display/all")
async def display_all_page():
    """公開銷售排行版面（免登入，顯示所有節點）"""
    return FileResponse("static/display_all.html", headers={"Cache-Control": "no-store, no-cache, must-revalidate"})
@app.get("/display/{node_id}")
async def display_page(node_id: str):
    """公開報價牌頁面（免登入）"""
    return FileResponse("static/display.html", headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
