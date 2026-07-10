"""零股（odd-lot）交易成本模型——實證校準版。

背景（docs/prereg_odd_lot_arm_20260711.md）：
    personal-baseline 整股口徑 FAIL 後的唯一開放問題——零股交易可解除 33 元價格上限
    （3.3 萬部位買得起任何價位的零股），但零股簿的 spread/流動性成本比整股簿差，
    需要實證校準的成本模型，不可拍腦袋。

校準方法（scripts/calibrate_odd_lot_costs.py，樣本窗見 CALIBRATION_SOURCE）：
    - 資料：TWSE 盤中零股交易行情單（TWTC7U）+ TPEx 盤中零股每日收盤行情（oddQuote），
      官方免費日資料，含每檔收盤最後揭示買/賣價。
    - 估計量：per-side premium = 各 amt_20 流動性層「收盤零股報價半價差
      (ask-bid)/2/mid」的 P75（取 P75 而非中位數：偏保守，涵蓋較差簿況；
      悲觀臂另乘 1.5）。VWAP 偏離與零股/整股收盤基差作為佐證分佈（量級一致），不進模型。
    - 分層：amt_20（整股 20 日均成交金額）固定門檻 0.1 / 0.3 / 1 / 5 億，
      低流動性端加密（個人口徑臂主要交易 <1 億微型股）。

成本組成（round trip）：
    2 × premium_per_side(amt_20) × premium_mult × era_mult(trade_date)
  + 2 × min_fee / position_size
    - premium：跨越零股簿半價差的執行成本（每邊一次）
    - min_fee：券商零股手續費低消（常見 1~20 元/筆，保守取 20 元；
      對 33,333 元部位 = 0.06%/邊。註：backtest 的 transaction_cost 已含整股
      0.1425% 手續費，此處低消為**額外疊加的保守 buffer**，偏悲觀方向）
    - era_mult：盤中零股 2020-10-26 上線；之前只有盤後零股（一天一撮、流動性更差），
      該時代 premium ×2 近似（僅全窗參考臂會觸發，標注 reference-only）

時代限制（誠實處理）：
    主臂窗口 = 2020-11-01 起（盤中零股上線後）；2016~2020-10 只能以 era_mult 近似，
    對應臂僅供參考、不進裁決。
"""

from __future__ import annotations

import math
from datetime import date
from typing import Dict, Tuple

# ── 制度常數 ─────────────────────────────────────────────────────────────────
# 盤中零股交易上線日（TWSE/TPEx 同日）
INTRADAY_ODD_LOT_START = date(2020, 10, 26)
# 上線前（盤後零股一天一撮時代）premium 近似倍率（全窗參考臂用）
AFTER_HOURS_ERA_PREMIUM_MULT = 2.0

# ── 個人口徑參數（prereg：資金 100 萬 / 30 檔）──────────────────────────────
DEFAULT_POSITION_SIZE_TWD = 33_333.0
DEFAULT_MIN_FEE_TWD = 20.0  # 券商零股低消，保守端（常見 1~20 元）

# ── 實證校準分層 premium 表 ──────────────────────────────────────────────────
# 來源：artifacts/odd_lot/calibration.json（單一真相源為本常數，JSON 為 provenance；
# tests/test_odd_lot_costs.py 鎖定兩者一致）。
# per-side premium = 該層收盤零股報價半價差的 P75。
CALIBRATION_SOURCE = (
    "artifacts/odd_lot/calibration.json"
    "（TWSE TWTC7U + TPEx oddQuote，2025-07-01 ~ 2026-06-30，"
    "estimator=P75 closing quoted half-spread per tier）"
)
# amt_20 門檻（元）：<0.1 億 / 0.1~0.3 億 / 0.3~1 億 / 1~5 億 / >=5 億
ODD_LOT_TIER_BOUNDS: Tuple[float, ...] = (1e7, 3e7, 1e8, 5e8)
ODD_LOT_TIER_LABELS: Tuple[str, ...] = (
    "lt_0.1yi", "0.1_0.3yi", "0.3_1yi", "1_5yi", "ge_5yi",
)
# 全量校準值（calibration.json generated_at 2026-07-10T22:39 CST，
# 樣本 465,537 檔-日 / 241 交易日 / 2025-07-01 ~ 2026-06-30）。
# 佐證分佈（微型股層）：half-spread median 0.45% / P90 2.15%；|basis| median 1.05%；
# 無零股成交率 3.4%（其餘層 ≈ 0）——詳見 calibration.json。
ODD_LOT_PREMIUM_PER_SIDE: Dict[str, float] = {
    "lt_0.1yi": 0.00990,
    "0.1_0.3yi": 0.00437,
    "0.3_1yi": 0.00328,
    "1_5yi": 0.00233,
    "ge_5yi": 0.00186,
}


def tier_label(amt_20: float) -> str:
    """amt_20（元）→ 流動性層標籤。NaN / <=0 視為最差層（保守）。"""
    if amt_20 is None or not math.isfinite(amt_20) or amt_20 <= 0:
        return ODD_LOT_TIER_LABELS[0]
    for bound, label in zip(ODD_LOT_TIER_BOUNDS, ODD_LOT_TIER_LABELS[:-1]):
        if amt_20 < bound:
            return label
    return ODD_LOT_TIER_LABELS[-1]


def premium_per_side(amt_20: float) -> float:
    """查該流動性層的單邊零股 premium（未乘任何倍率）。"""
    return ODD_LOT_PREMIUM_PER_SIDE[tier_label(amt_20)]


def era_premium_mult(trade_date: date) -> float:
    """盤中零股上線前 premium ×2（盤後零股時代近似）；上線後 ×1。"""
    return AFTER_HOURS_ERA_PREMIUM_MULT if trade_date < INTRADAY_ODD_LOT_START else 1.0


def odd_lot_round_trip_cost(
    amt_20: float,
    trade_date: date,
    premium_mult: float = 1.0,
    position_size_twd: float = DEFAULT_POSITION_SIZE_TWD,
    min_fee_twd: float = DEFAULT_MIN_FEE_TWD,
) -> float:
    """單筆持倉的零股來回執行成本（比例）。

    = 2 × premium_per_side × premium_mult × era_mult + 2 × min_fee / position_size

    Args:
        amt_20: 整股 20 日均成交金額（元）；NaN/<=0 落最差層（保守）
        trade_date: 進場（再平衡）日，決定 era multiplier
        premium_mult: 悲觀敏感度倍率（只縮放 premium，不縮放低消；預登記悲觀臂 1.5）
        position_size_twd: 每檔部位金額（低消換算用）
        min_fee_twd: 券商零股手續費低消（每邊）

    Returns:
        來回成本比例（>= 0），供 backtest tiered_slippage_map 路徑直接消費。
    """
    if premium_mult <= 0:
        raise ValueError(f"premium_mult 必須 > 0，收到 {premium_mult}")
    if position_size_twd <= 0:
        raise ValueError(f"position_size_twd 必須 > 0，收到 {position_size_twd}")
    per_side = premium_per_side(amt_20) * premium_mult * era_premium_mult(trade_date)
    min_fee_pct_per_side = min_fee_twd / position_size_twd
    return 2.0 * per_side + 2.0 * min_fee_pct_per_side
