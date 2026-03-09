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
import sqlite3
from datetime import datetime
from typing import Optional

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
    get_db,
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
MQTT_CLIENT_ID = "FreshServer_ISM_001"

IMG_HEIGHT = 128
IMG_WIDTH  = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)

# ─── 全域狀態 ────────────────────────────────────────────────────────────────
# 多節點感測器快取：{node_id: {temperature, humidity, timestamp}}
sensor_cache: dict = {
    "NODE_ISM_001": {"temperature": None, "humidity": None, "timestamp": None}
}
model_ai: HybridModel = None
scaler = None

# AI 推理結果快取：{node_id: {spoilage, label, color, timestamp, quality_data}}
ai_cache: dict = {}

# ai_cache 持久化路徑（與 SQLite DB 同目錄）
AI_CACHE_DB = os.environ.get("DB_PATH", "./data/ctmaiot.db").replace("ctmaiot.db", "ai_cache.db")

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
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("MQTT connected, subscribing to %s", MQTT_TOPIC)
        client.subscribe(MQTT_TOPIC)
        # 訂閱所有節點的 MQTT 主題
        try:
            nodes = get_all_nodes()
            for n in nodes:
                if n.get("mqtt_topic") and n["mqtt_topic"] != MQTT_TOPIC:
                    client.subscribe(n["mqtt_topic"])
                    logger.info("Also subscribing: %s", n["mqtt_topic"])
        except Exception:
            pass
    else:
        logger.warning("MQTT connect failed, rc=%s", rc)


def on_message(client, userdata, msg):
    import json
    global sensor_cache
    try:
        data = json.loads(msg.payload.decode())
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        temp = float(data.get("temp", data.get("temperature", 0)))
        hum  = float(data.get("hum",  data.get("humidity", 0)))

        # 找到對應節點
        node_id = "NODE_ISM_001"
        try:
            nodes = get_all_nodes()
            for n in nodes:
                if n.get("mqtt_topic") == msg.topic:
                    node_id = n["node_id"]
                    break
        except Exception:
            pass

        sensor_cache[node_id] = {
            "temperature": temp,
            "humidity":    hum,
            "timestamp":   ts,
        }
        # 自動記錄到資料庫
        try:
            insert_reading(node_id, temp, hum)
        except Exception as e:
            logger.debug("Insert reading error: %s", e)

        logger.debug("Sensor[%s] updated: T=%.1f H=%.1f", node_id, temp, hum)
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


# ─── AI 自動推理背景任務 ─────────────────────────────────────────────────────
AI_INFER_INTERVAL = 10  # 秒


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
    if temp is None or hum is None:
        logger.debug("AI inference skip [%s]: no sensor data", node_id)
        return None

    # 取得相機 URL
    cam_url = node.get("camera_url") or CAMERA_URL

    try:
        resp = requests.get(cam_url, timeout=8)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.warning("AI inference [%s] camera fetch failed: %s", node_id, e)
        return None

    try:
        img_tensor = infer_transform(image).unsqueeze(0).to(DEVICE)

        model_hum     = hum / 100.0 if hum > 1.0 else hum
        storage_days  = ai_cache.get(node_id, {}).get("storage_days", 1.0)
        # storage_hours = storage_days * 24.0 # No longer used in model
        raw_sensor    = np.array([[temp, model_hum]], dtype=np.float32)
        scaled_sensor = scaler.transform(raw_sensor)
        sensor_tensor = torch.tensor(scaled_sensor, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            prediction = model_ai(img_tensor, sensor_tensor).item()
        spoilage = float(np.clip(prediction, 0, 100))
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
    product     = node.get("product_type", "banana")
    initial_dsl = node.get("initial_dsl")
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

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = {
        "spoilage":     round(spoilage, 2),
        "ai_label":     ai_label,
        "ai_color":     ai_color,
        "quality_data": quality_data,
        "storage_days": storage_days,
        "timestamp":    ts,
    }

    # 更新快取
    ai_cache[node_id] = result
    model_registry["current"]["inference_count"] += 1
    logger.info("AI inference [%s] done: spoilage=%.1f%% label=%s", node_id, spoilage, ai_label)

    # 持久化到 DB
    save_ai_cache_to_db(node_id, result)

    # 自動記錄到資料庫（每次推理都記錄）
    if quality_data:
        try:
            insert_prediction(node_id, {
                "storage_days":     storage_days,
                "temperature":      temp,
                "humidity":         hum,
                "ai_spoilage":      spoilage,
                "quality_ai":       quality_data.get("quality_ai"),
                "quality_formula":  quality_data["quality_formula"],
                "quality_combined": quality_data["quality_score"],
                "dsl_combined":     quality_data["dsl_days"],
                "discount_pct":     quality_data["discount_pct"],
                "base_price":       base_price,
                "final_price":      quality_data["final_price"],
                "freshness_label":  quality_data["freshness_label"],
                "product":          product,
            })
        except Exception as e:
            logger.debug("Auto-save prediction error: %s", e)

    return result


def auto_ai_inference_loop():
    """背景執行緒：每 10 秒對所有節點執行 AI 推理"""
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
app = FastAPI(title="AIoT Smart Shelf API", version="4.1.0")


# ── ai_cache 持久化函數 ───────────────────────────────────────────────────────────────────────────────
def _init_ai_cache_db():
    """初始化 ai_cache 持久化資料庫"""
    import json
    os.makedirs(os.path.dirname(AI_CACHE_DB) if os.path.dirname(AI_CACHE_DB) else '.', exist_ok=True)
    conn = sqlite3.connect(AI_CACHE_DB, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_cache (
            node_id TEXT PRIMARY KEY,
            data    TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info("ai_cache DB initialized: %s", AI_CACHE_DB)


def save_ai_cache_to_db(node_id: str, result: dict):
    """\u5c07單一節點的 ai_cache 寫入持久化 DB"""
    import json
    try:
        conn = sqlite3.connect(AI_CACHE_DB, check_same_thread=False)
        conn.execute(
            "INSERT OR REPLACE INTO ai_cache (node_id, data, updated_at) VALUES (?, ?, ?)",
            (node_id, json.dumps(result, ensure_ascii=False), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug("save_ai_cache_to_db error: %s", e)


def load_ai_cache_from_db():
    """\u5f9e持久化 DB \u8f09入 ai_cache"""
    import json
    global ai_cache
    try:
        if not os.path.exists(AI_CACHE_DB):
            return
        conn = sqlite3.connect(AI_CACHE_DB, check_same_thread=False)
        rows = conn.execute("SELECT node_id, data FROM ai_cache").fetchall()
        conn.close()
        for row in rows:
            try:
                ai_cache[row[0]] = json.loads(row[1])
            except Exception:
                pass
        logger.info("Loaded %d ai_cache entries from DB", len(rows))
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

    # 初始化 ai_cache 持久化 DB 並載入上次的推理結果
    _init_ai_cache_db()
    load_ai_cache_from_db()
    logger.info("ai_cache loaded from persistent DB: %d entries", len(ai_cache))

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

    # 啟動 AI 自動推理背景執行緒（每 10 秒）
    ai_thread = threading.Thread(target=auto_ai_inference_loop, daemon=True)
    ai_thread.start()
    logger.info("Auto AI inference thread started (interval=10s)")


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
    # 更新最後登入時間
    conn = get_db()
    conn.execute("UPDATE users SET last_login=datetime('now') WHERE username=?", (req.username,))
    conn.commit()
    conn.close()

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
    conn = get_db()
    rows = conn.execute(
        "SELECT id, username, role, display_name, created_at, last_login FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/auth/users")
async def create_user(req: CreateUserRequest, admin: dict = Depends(require_admin)):
    # 角色驗證：只允許 admin 或 user
    if req.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="角色必須為 admin 或 user")
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password, role, display_name) VALUES (?,?,?,?)",
            (req.username, _hash_password(req.password), req.role, req.display_name)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="用戶名稱已存在")
    finally:
        conn.close()
    return {"message": f"用戶 {req.username} 已建立，角色：{req.role}"}


@app.delete("/api/auth/users/{user_id}")
async def delete_user(user_id: int, admin: dict = Depends(require_admin)):
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id=? AND username != 'admin'", (user_id,))
    conn.commit()
    conn.close()
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
    base_price: float = 100.0
    camera_url: str = ""
    mqtt_topic: str = ""
    status: str = "active"


@app.get("/api/nodes")
async def list_nodes(current_user: dict = Depends(get_current_user)):
    nodes = get_all_nodes()
    # 為每個節點附加最新感測器數據
    for n in nodes:
        nid = n["node_id"]
        n["latest_sensor"] = sensor_cache.get(nid, {"temperature": None, "humidity": None})
    return nodes


@app.post("/api/nodes")
async def create_node(req: NodeRequest, admin: dict = Depends(require_admin)):
    upsert_node(req.dict())
    # 初始化感測器快取
    sensor_cache[req.node_id] = {"temperature": None, "humidity": None, "timestamp": None}
    return {"message": f"節點 {req.node_id} 已建立/更新"}


@app.put("/api/nodes/{node_id}")
async def update_node(node_id: str, req: NodeRequest, admin: dict = Depends(require_admin)):
    req_dict = req.dict()
    req_dict["node_id"] = node_id
    upsert_node(req_dict)
    return {"message": f"節點 {node_id} 已更新"}


@app.delete("/api/nodes/{node_id}")
async def delete_node(node_id: str, admin: dict = Depends(require_admin)):
    conn = get_db()
    conn.execute("DELETE FROM nodes WHERE node_id=?", (node_id,))
    conn.commit()
    conn.close()
    sensor_cache.pop(node_id, None)
    return {"message": f"節點 {node_id} 已刪除"}


@app.get("/api/nodes/{node_id}/readings")
async def node_readings(node_id: str, limit: int = 100,
                        current_user: dict = Depends(get_current_user)):
    return get_node_readings(node_id, limit)


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

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        "uploaded_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
async def get_sensor(node_id: str = "NODE_ISM_001"):
    return JSONResponse(content=sensor_cache.get(node_id, {
        "temperature": None, "humidity": None, "timestamp": None
    }))


@app.get("/api/image")
async def proxy_image(node_id: str = "NODE_ISM_001"):
    # 取得節點相機 URL
    node = get_node(node_id)
    cam_url = (node.get("camera_url") or CAMERA_URL) if node else CAMERA_URL
    if not cam_url:
        cam_url = CAMERA_URL
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(cam_url)
            resp.raise_for_status()
        return StreamingResponse(
            io.BytesIO(resp.content),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail="Camera fetch failed: {}".format(e))


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
    node_id: str = "NODE_ISM_001"
    save_record: bool = True


@app.post("/api/quality")
async def calc_quality(req: QualityRequest):
    node_sensor = sensor_cache.get(req.node_id, {})
    temp = req.temperature if req.temperature is not None else node_sensor.get("temperature")
    hum  = req.humidity    if req.humidity    is not None else node_sensor.get("humidity")

    if temp is None or hum is None:
        raise HTTPException(
            status_code=422,
            detail="尚未收到感測器數據，請確認 M5GO 裝置已開機並連接 MQTT，或手動輸入溫溼度。"
        )

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

    # 記錄到資料庫
    if req.save_record:
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
            })
        except Exception as e:
            logger.warning("Save prediction failed: %s", e)

    # ── 同步更新 ai_cache，讓電子報價牌即時反映最新評估結果 ──────────────────
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
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "quality_data": data,
        "storage_days": req.storage_days,
        "source":       "manual",   # 標記來源為手動評估
    }
    ai_cache[req.node_id] = manual_result
    save_ai_cache_to_db(req.node_id, manual_result)  # 持久化手動評估結果
    logger.info("ai_cache updated from manual assessment: node=%s Q=%.1f discount=%s%%",
                req.node_id, data.get("quality_score", 0), data.get("discount_pct", 0))

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
    node_id: str = "NODE_ISM_001"
    save_record: bool = True


@app.post("/api/predict")
async def predict(req: PredictRequest):
    if model_ai is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    node_sensor = sensor_cache.get(req.node_id, {})
    temp = req.temperature if req.temperature is not None else node_sensor.get("temperature")
    hum  = req.humidity    if req.humidity    is not None else node_sensor.get("humidity")

    if temp is None or hum is None:
        raise HTTPException(
            status_code=422,
            detail="尚未收到感測器數據，請確認 M5GO 裝置已開機並連接 MQTT。"
        )

    # 取得節點相機 URL
    node = get_node(req.node_id)
    cam_url = (node.get("camera_url") or CAMERA_URL) if node else CAMERA_URL

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(cam_url)
            resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=502, detail="Camera fetch failed: {}".format(e))

    img_tensor = infer_transform(image).unsqueeze(0).to(DEVICE)

    model_hum     = hum / 100.0 if hum > 1.0 else hum
    storage_hours = req.storage_time * 24.0
    raw_sensor    = np.array([[temp, model_hum]], dtype=np.float32)
    scaled_sensor = scaler.transform(raw_sensor)
    sensor_tensor = torch.tensor(scaled_sensor, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        prediction = model_ai(img_tensor, sensor_tensor).item()
    spoilage = float(np.clip(prediction, 0, 100))

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

    # 記錄到資料庫
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
            })
        except Exception as e:
            logger.warning("Save prediction failed: %s", e)
    # ── 同步更新 ai_cache，讓電子報價牌即時反映最新完整推理結果 ─────────────────
    predict_result = {
        "spoilage":     spoilage,
        "ai_label":     ai_label,
        "ai_color":     ai_color,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    return {"nodes": result, "total": len(result), "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


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
            "light_lux":    node.get("light_lux", 375),
            "air_velocity": node.get("air_velocity", 0.22),
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
        "updated_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    return {
        "temperature":  sensor.get("temperature"),
        "humidity":     sensor.get("humidity"),
        "light_lux":    node.get("light_lux", 375) if node else 375,
        "air_velocity": node.get("air_velocity", 0.22) if node else 0.22,
        "timestamp":    sensor.get("timestamp"),
        "online":       sensor.get("temperature") is not None,
    }


# ─── 靜態網頁 ─────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/login.html")


@app.get("/login")
async def login_page():
    return FileResponse("static/login.html")


@app.get("/app")
async def app_page():
    return FileResponse("static/app.html")


@app.get("/admin")
async def admin_page():
    return FileResponse("static/admin.html")


@app.get("/display/all")
async def display_all_page():
    """公開銷售排行版面（免登入，顯示所有節點）"""
    return FileResponse("static/display_all.html")


@app.get("/display/{node_id}")
async def display_page(node_id: str):
    """公開報價牌頁面（免登入）"""
    return FileResponse("static/display.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
