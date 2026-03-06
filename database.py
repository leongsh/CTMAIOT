"""
database.py — SQLite 資料庫模型與初始化
表格：
  users       — 用戶帳號（admin / operator）
  nodes       — 貨架節點（位置、設備資訊）
  readings    — 感測器讀數記錄
  predictions — AI 品質評估記錄
"""

import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = os.environ.get("DB_PATH", "./data/ctmaiot.db")


def get_db():
    """取得資料庫連線（每次請求建立新連線）"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """初始化資料庫，建立所有表格與預設管理員帳號"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    cur = conn.cursor()

    # ── users 表 ──────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            password    TEXT    NOT NULL,
            role        TEXT    NOT NULL DEFAULT 'user',  -- admin | user
            display_name TEXT   DEFAULT '',
            created_at  TEXT    DEFAULT (datetime('now')),
            last_login  TEXT
        )
    """)

    # ── nodes 表（貨架節點）────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id      TEXT    UNIQUE NOT NULL,   -- 唯一識別碼，如 NODE_001
            name         TEXT    NOT NULL,
            location_name TEXT   DEFAULT '',        -- 位置名稱（如「倉庫A-第3排」）
            lat          REAL    DEFAULT 22.3193,   -- 緯度（預設香港）
            lng          REAL    DEFAULT 114.1694,  -- 經度
            floor        TEXT    DEFAULT '',        -- 樓層/區域
            product      TEXT    DEFAULT 'banana',  -- 當前監測產品
            initial_dsl  REAL    DEFAULT 10.0,      -- 初始 DSL（天）
            base_price   REAL    DEFAULT 100.0,     -- 商品原始售價
            camera_url   TEXT    DEFAULT '',        -- 相機 URL
            mqtt_topic   TEXT    DEFAULT '',        -- MQTT 主題
            status       TEXT    DEFAULT 'active',  -- active | inactive | maintenance
            created_at   TEXT    DEFAULT (datetime('now')),
            updated_at   TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ── readings 表（感測器讀數記錄）──────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id     TEXT    NOT NULL REFERENCES nodes(node_id),
            temperature REAL,
            humidity    REAL,
            light_lux   REAL    DEFAULT 500,
            air_velocity REAL   DEFAULT 0.3,
            recorded_at TEXT    DEFAULT (datetime('now'))
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_readings_node ON readings(node_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_readings_time ON readings(recorded_at)")

    # ── predictions 表（AI 品質評估記錄）─────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id         TEXT    NOT NULL REFERENCES nodes(node_id),
            storage_days    REAL,
            temperature     REAL,
            humidity        REAL,
            ai_spoilage     REAL,
            quality_ai      REAL,
            quality_formula REAL,
            quality_combined REAL,
            dsl_combined    REAL,
            discount_pct    REAL,
            base_price      REAL,
            final_price     REAL,
            freshness_label TEXT,
            product         TEXT,
            recorded_at     TEXT    DEFAULT (datetime('now'))
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_node ON predictions(node_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions(recorded_at)")

    # ── 預設帳號 ───────────────────────────────────────────────────────────────
    admin_pw = _hash_password("admin123")
    cur.execute("""
        INSERT OR IGNORE INTO users (username, password, role, display_name)
        VALUES (?, ?, 'admin', '系統管理員')
    """, ("admin", admin_pw))

    user_pw = _hash_password("user123")
    cur.execute("""
        INSERT OR IGNORE INTO users (username, password, role, display_name)
        VALUES (?, ?, 'user', '普通用戶')
    """, ("user", user_pw))

    # ── 預設節點（現有 ISM 貨架）──────────────────────────────────────────────
    cur.execute("""
        INSERT OR IGNORE INTO nodes
            (node_id, name, location_name, lat, lng, product, initial_dsl,
             base_price, camera_url, mqtt_topic, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "NODE_ISM_001",
        "ISM 智慧貨架 #1",
        "實驗室 A 區",
        22.3193, 114.1694,
        "banana", 10.0, 100.0,
        "https://ezdata2.m5stack.com/9888E0031824/captured.jpg",
        "m5go/ISM/env",
        "active"
    ))

    conn.commit()
    conn.close()
    print(f"[DB] Initialized: {DB_PATH}")


def _hash_password(password: str) -> str:
    """SHA-256 密碼雜湊（生產環境建議改用 bcrypt）"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    return _hash_password(plain) == hashed


# ── CRUD 輔助函數 ──────────────────────────────────────────────────────────────

def get_user(username: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_nodes():
    conn = get_db()
    rows = conn.execute("SELECT * FROM nodes ORDER BY created_at").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_node(node_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_node(data: dict):
    conn = get_db()
    conn.execute("""
        INSERT INTO nodes (node_id, name, location_name, lat, lng, floor, product,
                           initial_dsl, base_price, camera_url, mqtt_topic, status)
        VALUES (:node_id, :name, :location_name, :lat, :lng, :floor, :product,
                :initial_dsl, :base_price, :camera_url, :mqtt_topic, :status)
        ON CONFLICT(node_id) DO UPDATE SET
            name=excluded.name,
            location_name=excluded.location_name,
            lat=excluded.lat,
            lng=excluded.lng,
            floor=excluded.floor,
            product=excluded.product,
            initial_dsl=excluded.initial_dsl,
            base_price=excluded.base_price,
            camera_url=excluded.camera_url,
            mqtt_topic=excluded.mqtt_topic,
            status=excluded.status,
            updated_at=datetime('now')
    """, data)
    conn.commit()
    conn.close()


def insert_reading(node_id: str, temperature: float, humidity: float,
                   light_lux: float = 500, air_velocity: float = 0.3):
    conn = get_db()
    conn.execute("""
        INSERT INTO readings (node_id, temperature, humidity, light_lux, air_velocity)
        VALUES (?, ?, ?, ?, ?)
    """, (node_id, temperature, humidity, light_lux, air_velocity))
    conn.commit()
    conn.close()


def insert_prediction(node_id: str, data: dict):
    conn = get_db()
    conn.execute("""
        INSERT INTO predictions
            (node_id, storage_days, temperature, humidity, ai_spoilage, quality_ai,
             quality_formula, quality_combined, dsl_combined, discount_pct,
             base_price, final_price, freshness_label, product)
        VALUES
            (:node_id, :storage_days, :temperature, :humidity, :ai_spoilage, :quality_ai,
             :quality_formula, :quality_combined, :dsl_combined, :discount_pct,
             :base_price, :final_price, :freshness_label, :product)
    """, {"node_id": node_id, **data})
    conn.commit()
    conn.close()


def get_node_readings(node_id: str, limit: int = 100):
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM readings WHERE node_id=?
        ORDER BY recorded_at DESC LIMIT ?
    """, (node_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_node_predictions(node_id: str, limit: int = 50):
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM predictions WHERE node_id=?
        ORDER BY recorded_at DESC LIMIT ?
    """, (node_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_dashboard_stats():
    """後台儀表板統計數據"""
    conn = get_db()
    stats = {}
    stats["total_nodes"]   = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    stats["active_nodes"]  = conn.execute("SELECT COUNT(*) FROM nodes WHERE status='active'").fetchone()[0]
    stats["total_readings"] = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    stats["total_predictions"] = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    stats["avg_quality"]   = conn.execute(
        "SELECT ROUND(AVG(quality_combined),2) FROM predictions WHERE recorded_at > datetime('now','-24 hours')"
    ).fetchone()[0]
    stats["avg_discount"]  = conn.execute(
        "SELECT ROUND(AVG(discount_pct),2) FROM predictions WHERE recorded_at > datetime('now','-24 hours')"
    ).fetchone()[0]
    # 最近 10 筆預測
    rows = conn.execute("""
        SELECT p.*, n.name as node_name FROM predictions p
        JOIN nodes n ON p.node_id = n.node_id
        ORDER BY p.recorded_at DESC LIMIT 10
    """).fetchall()
    stats["recent_predictions"] = [dict(r) for r in rows]
    conn.close()
    return stats


if __name__ == "__main__":
    init_db()
    print("DB init complete")
