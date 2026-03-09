"""
quality_model.py — 品質降解模型 + AI 複合評分（依論文公式）v4.0
=============================================================
論文：AIoT-Based Smart Fresh Produce Shelf System:
      Quality Degradation Modeling and Dynamic Pricing Optimization

公式來源：Chapter 3, Section 3.2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【論文品質公式】

公式(1) 零階動力學基礎模型：
    Q(t) = Q₀ - k · t

公式(2) Arrhenius 溫度修正：
    k(T) = k_ref · exp( Eₐ/R · (1/T_ref - 1/T) )

公式(3) 綜合降解常數（濕度、光照、氣流修正）：
    k_comp = k(T) · f_H(H) · f_L(L) · f_A(A)

動態剩餘貨架壽命（DSL）：
    DSL = (Q(t) - Q_threshold) / k_comp

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【初始 DSL 設計（v4.0 新增）】

使用者可自訂初始 DSL（initial_dsl），系統自動反推 Q₀：

    Q₀_effective = Q_threshold + k_comp_ref × initial_dsl
                 = 70 + k_comp_ref × initial_dsl

其中 k_comp_ref 為標準環境（20°C、最佳濕度）下的降解常數，
確保初始 DSL 的定義與標準儲存條件一致。

若使用者未指定 initial_dsl，則使用論文預設 Q₀ = 97。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【AI 複合品質評分（本系統擴充）】

AI 模型輸出腐敗指數 S（0~100，越高越腐敗），
先轉換為 AI 品質分數：
    Q_AI = 100 - S

與論文公式品質分數 Q(t) 加權融合：
    Q_combined = α · Q(t) + (1-α) · Q_AI
    α = 0.5（預設，可調整）

複合 DSL：
    DSL_combined = (Q_combined - Q_threshold) / k_comp

折扣依 DSL_combined 觸發（論文 Table 5）：
    DSL > 7 天     → 無折扣
    3 < DSL ≤ 7    → 10%（12% 消費者動機）
    1 < DSL ≤ 3    → 25%（38% 消費者動機，最大區間）
    0.5 < DSL ≤ 1  → 35%（累積 85% 消費者動機）
    DSL ≤ 0.5      → 50%（最終清倉）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
產品參數（論文 Table 1）：
    產品      Eₐ(kJ/mol)  k_ref(at 20°C)  最佳濕度(%RH)  預設初始DSL(天)
    蘋果        55           0.8            90-95           14
    番茄        62           1.1            90-95           10
    萵苣        48           1.5            95-98            7
    香蕉(預設)  52           1.2            85-90           10
"""

import math
from dataclasses import dataclass, field
from typing import Optional

# ─── 通用常數 ────────────────────────────────────────────────────────────────
R = 8.314          # 通用氣體常數 (J/mol·K)
T_REF_K = 293.15   # 參考溫度 20°C → Kelvin
Q0 = 97.0          # 論文預設初始品質分數（未指定 initial_dsl 時使用）
Q_THRESHOLD = 70.0 # 品質閾值（低於此值視為不可銷售）

# AI 與論文公式的加權係數
ALPHA_FORMULA = 0.6   # 論文公式加權値
ALPHA_AI      = 0.4   # AI 加權値

# ─── 產品參數（依論文 Table 1）────────────────────────────────────────────────
# initial_dsl_days：各產品在標準環境（20°C、最佳濕度）下的預設初始 DSL
PRODUCT_PARAMS = {
    "apple": {
        "name": "蘋果",
        "Ea": 55_000,
        "k_ref": 0.8,
        "optimal_rh_min": 90.0,
        "optimal_rh_max": 95.0,
        "shelf_life_days": 14,
        "initial_dsl_days": 14,   # 預設初始 DSL（天）
    },
    "tomato": {
        "name": "番茄",
        "Ea": 62_000,
        "k_ref": 1.1,
        "optimal_rh_min": 90.0,
        "optimal_rh_max": 95.0,
        "shelf_life_days": 10,
        "initial_dsl_days": 10,
    },
    "lettuce": {
        "name": "萵苣",
        "Ea": 48_000,
        "k_ref": 1.5,
        "optimal_rh_min": 95.0,
        "optimal_rh_max": 98.0,
        "shelf_life_days": 7,
        "initial_dsl_days": 7,
    },
    "banana": {
        "name": "香蕉（預設）",
        "Ea": 52_000,
        "k_ref": 1.2,
        "optimal_rh_min": 85.0,
        "optimal_rh_max": 90.0,
        "shelf_life_days": 10,
        "initial_dsl_days": 10,
    },
}

# ─── 折扣規則（依論文 Table 5 消費者價格敏感度）────────────────────────────────
DISCOUNT_RULES = [
    (7.0,   0,  "新鮮，無折扣"),
    (3.0,  10,  "輕微老化，9折（12% 消費者動機）"),
    (1.0,  25,  "近效期，75折（38% 消費者動機，最大區間）"),
    (0.5,  35,  "急需銷售，65折（累積 85% 消費者動機）"),
    (0.0,  50,  "最終清倉，5折"),
]


@dataclass
class QualityResult:
    """品質評估結果（含 AI 複合評分與初始 DSL）"""
    # ── 輸入參數 ──
    temperature: float
    humidity: float
    storage_days: float
    light_lux: float = 500.0
    air_velocity: float = 0.3
    product: str = "banana"
    ai_spoilage: Optional[float] = None
    initial_dsl: Optional[float] = None   # 使用者自訂初始 DSL（None = 使用論文預設 Q₀=97）

    # ── 初始 DSL 計算結果 ──
    q0_effective: float = Q0              # 實際使用的 Q₀（由 initial_dsl 反推或使用預設）
    initial_dsl_used: float = 0.0         # 實際使用的初始 DSL 天數
    k_comp_ref: float = 0.0              # 標準環境（20°C、最佳濕度）下的 k_comp（反推用）

    # ── 論文公式計算結果 ──
    k_T: float = 0.0
    f_H: float = 1.0
    f_L: float = 1.0
    f_A: float = 1.0
    k_comp: float = 0.0
    quality_formula: float = 0.0
    dsl_formula: float = 0.0

    # ── AI 品質分數 ──
    quality_ai: Optional[float] = None

    # ── 複合評分（最終決策依據）──
    quality_combined: float = 0.0
    dsl_combined: float = 0.0
    alpha_used: float = ALPHA_FORMULA

    # ── 最終輸出 ──
    discount_pct: int = 0
    discount_reason: str = ""
    freshness_label: str = ""
    freshness_color: str = ""


# ─── 各影響函數 ──────────────────────────────────────────────────────────────

def calc_k_T(Ea: float, k_ref: float, T_celsius: float) -> float:
    """公式(2) Arrhenius 方程：k(T) = k_ref · exp( Eₐ/R · (1/T_ref - 1/T) )"""
    T_K = T_celsius + 273.15
    exponent = (Ea / R) * (1.0 / T_REF_K - 1.0 / T_K)
    return k_ref * math.exp(exponent)


def calc_f_H(humidity: float, rh_min: float, rh_max: float) -> float:
    """濕度影響函數 f_H(H)"""
    if humidity < rh_min:
        deficit = rh_min - humidity
        return 1.0 + 0.02 * deficit
    elif humidity > rh_max:
        excess = humidity - rh_max
        return 1.0 + 0.01 * excess
    return 1.0


def calc_f_L(light_lux: float) -> float:
    """光照影響函數 f_L(L)"""
    baseline_lux = 500.0
    if light_lux <= 0:
        return 0.85
    f = 1.0 + 0.0002 * (light_lux - baseline_lux)
    return max(0.85, min(f, 1.5))


def calc_f_A(air_velocity: float) -> float:
    """氣流影響函數 f_A(A)"""
    if air_velocity < 0.1:
        return 1.0 + 0.5 * (0.1 - air_velocity) / 0.1
    elif air_velocity > 1.0:
        return 1.0 + 0.1 * (air_velocity - 1.0)
    return 1.0


def calc_k_comp_ref(Ea: float, k_ref: float,
                    rh_min: float, rh_max: float) -> float:
    """
    計算標準環境（20°C、最佳濕度中間值）下的 k_comp_ref。
    用於從 initial_dsl 反推 Q₀_effective。
    標準環境：T=20°C、H=(rh_min+rh_max)/2、L=500lux、A=0.3m/s
    """
    k_T_ref  = calc_k_T(Ea, k_ref, 20.0)         # 20°C 下的 k(T) = k_ref（Arrhenius 基準點）
    rh_mid   = (rh_min + rh_max) / 2.0
    f_H_ref  = calc_f_H(rh_mid, rh_min, rh_max)  # 最佳濕度 → f_H = 1.0
    f_L_ref  = calc_f_L(500.0)                    # 標準光照 → f_L ≈ 1.0
    f_A_ref  = calc_f_A(0.3)                      # 標準氣流 → f_A = 1.0
    return k_T_ref * f_H_ref * f_L_ref * f_A_ref


def calc_dsl(quality_score: float, k_comp: float) -> float:
    """DSL = (Q - Q_threshold) / k_comp"""
    if k_comp <= 0:
        return 999.0
    return max(0.0, (quality_score - Q_THRESHOLD) / k_comp)


def calc_discount_from_dsl(dsl_days: float):
    """依 DSL 計算折扣（論文 Table 5）"""
    for threshold, pct, reason in DISCOUNT_RULES:
        if dsl_days > threshold:
            return pct, reason
    return DISCOUNT_RULES[-1][1], DISCOUNT_RULES[-1][2]


def get_freshness_label(quality_score: float):
    """依品質分數回傳新鮮度標籤與顏色"""
    if quality_score >= 85:
        return "優質新鮮 🟢", "#22c55e"
    elif quality_score >= 70:
        return "良好新鮮 🟡", "#eab308"
    elif quality_score >= 55:
        return "輕微老化 🟠", "#f97316"
    elif quality_score >= 40:
        return "中度老化 🔴", "#ef4444"
    return "嚴重老化 ⚫", "#7f1d1d"


# ─── 主計算函數 ──────────────────────────────────────────────────────────────

def calculate_quality(
    temperature: float,
    humidity: float,
    storage_days: float,
    light_lux: float = 500.0,
    air_velocity: float = 0.3,
    product: str = "banana",
    ai_spoilage: Optional[float] = None,
    initial_dsl: Optional[float] = None,
) -> QualityResult:
    """
    主計算函數：依論文公式計算品質評分，可整合 AI 腐敗指數與自訂初始 DSL。

    Args:
        temperature:  溫度 (°C)
        humidity:     相對濕度 (%)
        storage_days: 儲存天數
        light_lux:    光照強度 (lux)
        air_velocity: 氣流速度 (m/s)
        product:      產品類型
        ai_spoilage:  AI 腐敗指數（0~100），None 表示 AI 不可用
        initial_dsl:  使用者自訂初始 DSL（天），None = 使用論文預設 Q₀=97

    Returns:
        QualityResult
    """
    result = QualityResult(
        temperature=temperature,
        humidity=humidity,
        storage_days=storage_days,
        light_lux=light_lux,
        air_velocity=air_velocity,
        product=product,
        ai_spoilage=ai_spoilage,
        initial_dsl=initial_dsl,
    )

    params = PRODUCT_PARAMS.get(product, PRODUCT_PARAMS["banana"])

    # ── Step 0：決定 Q₀（初始品質分數）──────────────────────────────────────
    # 若使用者指定 initial_dsl，反推 Q₀_effective；否則使用論文預設值 97
    result.k_comp_ref = calc_k_comp_ref(
        params["Ea"], params["k_ref"],
        params["optimal_rh_min"], params["optimal_rh_max"]
    )

    if initial_dsl is not None and initial_dsl > 0:
        # 反推公式：Q₀ = Q_threshold + k_comp_ref × initial_dsl
        result.initial_dsl_used = float(initial_dsl)
        result.q0_effective = Q_THRESHOLD + result.k_comp_ref * result.initial_dsl_used
        # 限制在合理範圍（70~100）
        result.q0_effective = max(Q_THRESHOLD + 1.0, min(100.0, result.q0_effective))
    else:
        # 使用論文預設 Q₀ = 97，反算對應的初始 DSL（供展示用）
        result.q0_effective = Q0
        result.initial_dsl_used = (Q0 - Q_THRESHOLD) / result.k_comp_ref if result.k_comp_ref > 0 else 0.0

    # ── Step 1：論文公式計算 Q(t) ──────────────────────────────────────────
    result.k_T    = calc_k_T(params["Ea"], params["k_ref"], temperature)
    result.f_H    = calc_f_H(humidity, params["optimal_rh_min"], params["optimal_rh_max"])
    result.f_L    = calc_f_L(light_lux)
    result.f_A    = calc_f_A(air_velocity)
    result.k_comp = result.k_T * result.f_H * result.f_L * result.f_A

    # Q(t) = Q₀_effective - k_comp × t
    result.quality_formula = max(0.0, result.q0_effective - result.k_comp * storage_days)
    result.dsl_formula     = calc_dsl(result.quality_formula, result.k_comp)

    # ── Step 2：AI 品質分數（若 AI 可用）──────────────────────────────────
    if ai_spoilage is not None:
        result.quality_ai = max(0.0, min(100.0, 100.0 - ai_spoilage))
        result.alpha_used = ALPHA_FORMULA
    else:
        result.quality_ai = None
        result.alpha_used = 1.0

    # ── Step 3：複合品質評分 ──────────────────────────────────────────────
    if result.quality_ai is not None:
        result.quality_combined = (
            result.alpha_used * result.quality_formula +
            (1.0 - result.alpha_used) * result.quality_ai
        )
    else:
        result.quality_combined = result.quality_formula

    # ── Step 4：複合 DSL ──────────────────────────────────────────────────
    result.dsl_combined = calc_dsl(result.quality_combined, result.k_comp)

    # ── Step 5：折扣決策（依複合 DSL）────────────────────────────────────
    result.discount_pct, result.discount_reason = calc_discount_from_dsl(result.dsl_combined)

    # ── Step 6：新鮮度標籤 ────────────────────────────────────────────────
    result.freshness_label, result.freshness_color = get_freshness_label(result.quality_combined)

    return result


def quality_result_to_dict(r: QualityResult, base_price: float = 100.0) -> dict:
    """將 QualityResult 轉為 API 回應格式（含完整公式推導數據）"""
    final_price = round(base_price * (1 - r.discount_pct / 100), 2)
    params = PRODUCT_PARAMS.get(r.product, PRODUCT_PARAMS["banana"])
    has_ai = r.quality_ai is not None

    # is_custom：使用者明確傳入 initial_dsl 且與產品預設值不同時為 True
    _default_dsl = float(params.get("initial_dsl_days", 10))
    _is_custom = (
        r.initial_dsl is not None and
        abs(float(r.initial_dsl) - _default_dsl) > 0.001
    )

    return {
        # ── 最終複合品質評分 ──
        "quality_score":    round(r.quality_combined, 2),
        "freshness_label":  r.freshness_label,
        "freshness_color":  r.freshness_color,
        "dsl_days":         round(r.dsl_combined, 2),

        # ── 各分項評分 ──
        "quality_formula":  round(r.quality_formula, 2),
        "quality_ai":       round(r.quality_ai, 2) if has_ai else None,
        "ai_spoilage":      round(r.ai_spoilage, 2) if r.ai_spoilage is not None else None,
        "dsl_formula":      round(r.dsl_formula, 2),
        "alpha_formula":    round(r.alpha_used, 2),
        "alpha_ai":         round(1.0 - r.alpha_used, 2),
        "has_ai":           has_ai,

        # ── 初始 DSL 資訊（v4.0 新增）──
        "initial_dsl": {
            "used_days":    round(r.initial_dsl_used, 2),
            "q0_effective": round(r.q0_effective, 2),
            "k_comp_ref":   round(r.k_comp_ref, 4),
            "is_custom":    _is_custom,
            "formula":      "Q₀ = Q_threshold + k_comp_ref × initial_dsl = {:.1f} + {:.4f} × {:.1f} = {:.2f}".format(
                Q_THRESHOLD, r.k_comp_ref, r.initial_dsl_used, r.q0_effective
            ),
        },

        # ── 折扣定價 ──
        "discount_pct":     r.discount_pct,
        "discount_reason":  r.discount_reason,
        "base_price":       base_price,
        "final_price":      final_price,

        # ── 論文公式中間值（計算推導）──
        "formula": {
            "Q0":           round(r.q0_effective, 2),   # 使用實際 Q₀（可能由 initial_dsl 反推）
            "Q0_original":  Q0,                          # 論文原始 Q₀ = 97
            "k_ref":        params["k_ref"],
            "Ea_kJ":        params["Ea"] / 1000,
            "k_T":          round(r.k_T, 4),
            "f_H":          round(r.f_H, 4),
            "f_L":          round(r.f_L, 4),
            "f_A":          round(r.f_A, 4),
            "k_comp":       round(r.k_comp, 4),
            "k_comp_ref":   round(r.k_comp_ref, 4),
            "storage_days": r.storage_days,
            "Q_threshold":  Q_THRESHOLD,
        },

        # ── 感測器輸入 ──
        "sensor": {
            "temperature":  round(r.temperature, 1),
            "humidity":     round(r.humidity, 1),
            "light_lux":    round(r.light_lux, 0),
            "air_velocity": round(r.air_velocity, 2),
            "storage_days": r.storage_days,
        },

        # ── 產品資訊 ──
        "product": {
            "id":                r.product,
            "name":              params["name"],
            "shelf_life_days":   params["shelf_life_days"],
            "initial_dsl_days":  params["initial_dsl_days"],  # 產品預設初始 DSL
            "optimal_rh":        "{}-{}%".format(params["optimal_rh_min"], params["optimal_rh_max"]),
        },
    }
