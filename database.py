"""
database.py — PostgreSQL 資料庫模型與初始化（Supabase）
使用連線池（ThreadedConnectionPool）消除每次請求重建連線的 1.5 秒延遲。

表格：
  users       — 用戶帳號（admin / operator）
  nodes       — 貨架節點（位置、設備資訊）
  readings    — 感測器讀數記錄
  predictions — AI 品質評估記錄
"""

import psycopg2
import psycopg2.extras
import psycopg2.pool
import hashlib
import os
import threading
import logging
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# 優先使用 DATABASE_URL 環境變數（Render 設定），否則使用 Supabase 預設
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres.gcoowcpdzalfbvurvibf:TFy032nWioYRKw52@aws-1-ap-southeast-2.pooler.supabase.com:5432/postgres"
)

# ─── 連線池（全域單例）────────────────────────────────────────────────────────
# minconn=2: 預先建立 2 條連線（消除冷啟動延遲）
# maxconn=10: 最多 10 條並發連線
_pool: psycopg2.pool.ThreadedConnectionPool = None
_pool_lock = threading.Lock()


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """取得或初始化連線池（執行緒安全）"""
    global _pool
    if _pool is None or _pool.closed:
        with _pool_lock:
            if _pool is None or _pool.closed:
                logger.info("Initializing DB connection pool (min=2, max=10)...")
                _pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=10,
                    dsn=DATABASE_URL,
                    connect_timeout=15,
                )
                logger.info("DB connection pool ready")
    return _pool


@contextmanager
def get_db():
    """
    從連線池取得連線，用完自動歸還。
    用法：
        with get_db() as conn:
            cur = conn.cursor()
            ...
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        conn.autocommit = False
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def get_db_legacy():
    """
    向後相容的舊式 get_db()，回傳連線物件（需手動 close → 歸還到池）。
    新代碼請改用 with get_db() as conn。
    """
    pool = _get_pool()
    conn = pool.getconn()
    conn.autocommit = False
    # 包裝 close() 使其歸還到池而非真正關閉
    original_close = conn.close

    def _return_to_pool():
        try:
            conn.rollback()
        except Exception:
            pass
        pool.putconn(conn)

    conn.close = _return_to_pool
    return conn


def init_db():
    """初始化資料庫，建立所有表格與預設管理員帳號"""
    with get_db() as conn:
        cur = conn.cursor()

        # ── users 表 ──────────────────────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          SERIAL PRIMARY KEY,
                username    TEXT    UNIQUE NOT NULL,
                password    TEXT    NOT NULL,
                role        TEXT    NOT NULL DEFAULT 'user',
                display_name TEXT   DEFAULT '',
                created_at  TIMESTAMP DEFAULT NOW(),
                last_login  TIMESTAMP
            )
        """)

        # ── nodes 表（貨架節點）────────────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id           SERIAL PRIMARY KEY,
                node_id      TEXT    UNIQUE NOT NULL,
                name         TEXT    NOT NULL,
                location_name TEXT   DEFAULT '',
                lat          REAL    DEFAULT 22.3193,
                lng          REAL    DEFAULT 114.1694,
                floor        TEXT    DEFAULT '',
                product      TEXT    DEFAULT 'banana',
                initial_dsl  REAL    DEFAULT 10.0,
                base_price   REAL    DEFAULT 100.0,
                camera_url   TEXT    DEFAULT '',
                mqtt_topic   TEXT    DEFAULT '',
                status       TEXT    DEFAULT 'active',
                created_at   TIMESTAMP DEFAULT NOW(),
                updated_at   TIMESTAMP DEFAULT NOW()
            )
        """)

        # ── readings 表（感測器讀數記錄）──────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id          SERIAL PRIMARY KEY,
                node_id     TEXT    NOT NULL REFERENCES nodes(node_id),
                temperature REAL,
                humidity    REAL,
                light_lux   REAL    DEFAULT 500,
                air_velocity REAL   DEFAULT 0.3,
                recorded_at TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_readings_node ON readings(node_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_readings_time ON readings(recorded_at)")

        # ── predictions 表（AI 品質評估記錄）─────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              SERIAL PRIMARY KEY,
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
                recorded_at     TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_node ON predictions(node_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions(recorded_at)")

        # ── 預設帳號 ───────────────────────────────────────────────────────────────
        admin_pw = _hash_password("admin123")
        cur.execute("""
            INSERT INTO users (username, password, role, display_name)
            VALUES (%s, %s, 'admin', '系統管理員')
            ON CONFLICT (username) DO NOTHING
        """, ("admin", admin_pw))

        user_pw = _hash_password("user123")
        cur.execute("""
            INSERT INTO users (username, password, role, display_name)
            VALUES (%s, %s, 'user', '普通用戶')
            ON CONFLICT (username) DO NOTHING
        """, ("user", user_pw))

        # ── 預設節點（現有 ISM 貨架）──────────────────────────────────────────────
        cur.execute("""
            INSERT INTO nodes
                (node_id, name, location_name, lat, lng, product, initial_dsl,
                 base_price, camera_url, mqtt_topic, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (node_id) DO NOTHING
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

    logger.info("Database initialized (PostgreSQL / Supabase)")
    print("[DB] Initialized: PostgreSQL (Supabase) with connection pool")


def _hash_password(password: str) -> str:
    """SHA-256 密碼雜湊"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    return _hash_password(plain) == hashed


# ── CRUD 輔助函數 ──────────────────────────────────────────────────────────────

def get_user(username: str):
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        row = cur.fetchone()
    return dict(row) if row else None


def get_all_nodes():
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM nodes ORDER BY created_at")
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_node(node_id: str):
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM nodes WHERE node_id=%s", (node_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def upsert_node(data: dict):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO nodes (node_id, name, location_name, lat, lng, floor, product,
                               initial_dsl, base_price, camera_url, mqtt_topic, status)
            VALUES (%(node_id)s, %(name)s, %(location_name)s, %(lat)s, %(lng)s, %(floor)s,
                    %(product)s, %(initial_dsl)s, %(base_price)s, %(camera_url)s,
                    %(mqtt_topic)s, %(status)s)
            ON CONFLICT (node_id) DO UPDATE SET
                name=EXCLUDED.name,
                location_name=EXCLUDED.location_name,
                lat=EXCLUDED.lat,
                lng=EXCLUDED.lng,
                floor=EXCLUDED.floor,
                product=EXCLUDED.product,
                initial_dsl=EXCLUDED.initial_dsl,
                base_price=EXCLUDED.base_price,
                camera_url=EXCLUDED.camera_url,
                mqtt_topic=EXCLUDED.mqtt_topic,
                status=EXCLUDED.status,
                updated_at=NOW()
        """, data)


def delete_node(node_id: str):
    with get_db() as conn:
        cur = conn.cursor()
        # 先刪除外鍵關聯的子記錄，再刪除節點
        cur.execute("DELETE FROM predictions WHERE node_id=%s", (node_id,))
        cur.execute("DELETE FROM readings WHERE node_id=%s", (node_id,))
        cur.execute("DELETE FROM nodes WHERE node_id=%s", (node_id,))


def insert_reading(node_id: str, temperature: float, humidity: float,
                   light_lux: float = 500, air_velocity: float = 0.3):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO readings (node_id, temperature, humidity, light_lux, air_velocity)
            VALUES (%s, %s, %s, %s, %s)
        """, (node_id, temperature, humidity, light_lux, air_velocity))


def insert_prediction(node_id: str, data: dict):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions
                (node_id, storage_days, temperature, humidity, ai_spoilage, quality_ai,
                 quality_formula, quality_combined, dsl_combined, discount_pct,
                 base_price, final_price, freshness_label, product)
            VALUES
                (%(node_id)s, %(storage_days)s, %(temperature)s, %(humidity)s,
                 %(ai_spoilage)s, %(quality_ai)s, %(quality_formula)s, %(quality_combined)s,
                 %(dsl_combined)s, %(discount_pct)s, %(base_price)s, %(final_price)s,
                 %(freshness_label)s, %(product)s)
        """, {"node_id": node_id, **data})


def get_node_readings(node_id: str, limit: int = 100):
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT * FROM readings WHERE node_id=%s
            ORDER BY recorded_at DESC LIMIT %s
        """, (node_id, limit))
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_node_predictions(node_id: str, limit: int = 50):
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT * FROM predictions WHERE node_id=%s
            ORDER BY recorded_at DESC LIMIT %s
        """, (node_id, limit))
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_dashboard_stats():
    """後台儀表板統計數據"""
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        stats = {}

        cur.execute("SELECT COUNT(*) as cnt FROM nodes")
        stats["total_nodes"] = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(*) as cnt FROM nodes WHERE status='active'")
        stats["active_nodes"] = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(*) as cnt FROM readings")
        stats["total_readings"] = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(*) as cnt FROM predictions")
        stats["total_predictions"] = cur.fetchone()["cnt"]

        cur.execute("""
            SELECT ROUND(AVG(quality_combined)::numeric, 2) as avg_q
            FROM predictions WHERE recorded_at > NOW() - INTERVAL '24 hours'
        """)
        stats["avg_quality"] = cur.fetchone()["avg_q"]

        cur.execute("""
            SELECT ROUND(AVG(discount_pct)::numeric, 2) as avg_d
            FROM predictions WHERE recorded_at > NOW() - INTERVAL '24 hours'
        """)
        stats["avg_discount"] = cur.fetchone()["avg_d"]

        cur.execute("""
            SELECT p.*, n.name as node_name FROM predictions p
            JOIN nodes n ON p.node_id = n.node_id
            ORDER BY p.recorded_at DESC LIMIT 10
        """)
        stats["recent_predictions"] = [dict(r) for r in cur.fetchall()]

    return stats


if __name__ == "__main__":
    init_db()
    print("DB init complete")
