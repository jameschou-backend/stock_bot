"""零股（odd-lot）成本模型測試。

鎖定：
1. 分層查表（含 NaN/0/負值落最差層的保守行為）
2. premium 隨流動性單調不增
3. era multiplier（盤中零股 2020-10-26 上線前 ×2）
4. round-trip 成本公式精確性（premium_mult 只縮放 premium、不縮放低消）
5. 模組常數與 artifacts/odd_lot/calibration.json 一致（provenance 鎖定）
6. WalkForwardConfig 預設關閉（不改變既有基準行為）
7. summary 記錄區塊 _odd_lot_config_block
"""

import json
import math
from datetime import date
from pathlib import Path

import pytest

from skills.odd_lot_costs import (
    AFTER_HOURS_ERA_PREMIUM_MULT,
    DEFAULT_MIN_FEE_TWD,
    DEFAULT_POSITION_SIZE_TWD,
    INTRADAY_ODD_LOT_START,
    ODD_LOT_PREMIUM_PER_SIDE,
    ODD_LOT_TIER_BOUNDS,
    ODD_LOT_TIER_LABELS,
    era_premium_mult,
    odd_lot_round_trip_cost,
    premium_per_side,
    tier_label,
)

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = ROOT / "artifacts" / "odd_lot" / "calibration.json"

AFTER = date(2021, 6, 1)   # 盤中零股時代
BEFORE = date(2019, 6, 1)  # 盤後零股時代


# ── 1. 分層查表 ──────────────────────────────────────────────────────────────

def test_tier_label_boundaries():
    assert tier_label(5e6) == "lt_0.1yi"
    assert tier_label(1e7) == "0.1_0.3yi"      # 邊界含於上層
    assert tier_label(2e7) == "0.1_0.3yi"
    assert tier_label(3e7) == "0.3_1yi"
    assert tier_label(9.9e7) == "0.3_1yi"
    assert tier_label(1e8) == "1_5yi"
    assert tier_label(4.9e8) == "1_5yi"
    assert tier_label(5e8) == "ge_5yi"
    assert tier_label(1e10) == "ge_5yi"


def test_tier_label_invalid_falls_to_worst_tier():
    """amt_20 缺失/無效 → 最差流動性層（保守）。"""
    assert tier_label(float("nan")) == "lt_0.1yi"
    assert tier_label(0.0) == "lt_0.1yi"
    assert tier_label(-1.0) == "lt_0.1yi"
    assert tier_label(None) == "lt_0.1yi"
    assert tier_label(float("inf")) == "lt_0.1yi"


def test_tier_structure_consistent():
    assert len(ODD_LOT_TIER_LABELS) == len(ODD_LOT_TIER_BOUNDS) + 1
    assert set(ODD_LOT_PREMIUM_PER_SIDE.keys()) == set(ODD_LOT_TIER_LABELS)


# ── 2. premium 單調性 ────────────────────────────────────────────────────────

def test_premium_monotonic_nonincreasing_with_liquidity():
    """流動性越好，premium 不得更貴。"""
    values = [ODD_LOT_PREMIUM_PER_SIDE[t] for t in ODD_LOT_TIER_LABELS]
    for worse, better in zip(values, values[1:]):
        assert worse >= better
    # 全部為正、且量級 sane（單邊 premium 不可能 > 10%）
    for v in values:
        assert 0 < v < 0.10


def test_premium_per_side_lookup():
    assert premium_per_side(5e6) == ODD_LOT_PREMIUM_PER_SIDE["lt_0.1yi"]
    assert premium_per_side(2e8) == ODD_LOT_PREMIUM_PER_SIDE["1_5yi"]


# ── 3. era multiplier ────────────────────────────────────────────────────────

def test_era_multiplier():
    assert INTRADAY_ODD_LOT_START == date(2020, 10, 26)
    assert era_premium_mult(BEFORE) == AFTER_HOURS_ERA_PREMIUM_MULT == 2.0
    assert era_premium_mult(date(2020, 10, 25)) == 2.0
    assert era_premium_mult(date(2020, 10, 26)) == 1.0
    assert era_premium_mult(AFTER) == 1.0


# ── 4. round-trip 成本公式 ───────────────────────────────────────────────────

def test_round_trip_cost_formula_exact():
    amt = 5e6  # lt_0.1yi
    p = ODD_LOT_PREMIUM_PER_SIDE["lt_0.1yi"]
    fee = DEFAULT_MIN_FEE_TWD / DEFAULT_POSITION_SIZE_TWD
    expected = 2 * p + 2 * fee
    assert odd_lot_round_trip_cost(amt, AFTER) == pytest.approx(expected, abs=1e-12)


def test_round_trip_premium_mult_scales_premium_only():
    """悲觀倍率只縮放 premium，不縮放低消。"""
    amt = 2e7  # 0.1_0.3yi
    p = ODD_LOT_PREMIUM_PER_SIDE["0.1_0.3yi"]
    fee = DEFAULT_MIN_FEE_TWD / DEFAULT_POSITION_SIZE_TWD
    base = odd_lot_round_trip_cost(amt, AFTER, premium_mult=1.0)
    pess = odd_lot_round_trip_cost(amt, AFTER, premium_mult=1.5)
    assert pess - base == pytest.approx(2 * p * 0.5, abs=1e-12)
    assert base == pytest.approx(2 * p + 2 * fee, abs=1e-12)


def test_round_trip_era_multiplier_applies_to_premium_only():
    amt = 5e6
    p = ODD_LOT_PREMIUM_PER_SIDE["lt_0.1yi"]
    fee = DEFAULT_MIN_FEE_TWD / DEFAULT_POSITION_SIZE_TWD
    old_era = odd_lot_round_trip_cost(amt, BEFORE)
    assert old_era == pytest.approx(2 * p * 2.0 + 2 * fee, abs=1e-12)


def test_round_trip_nan_amt20_uses_worst_tier():
    assert odd_lot_round_trip_cost(float("nan"), AFTER) == odd_lot_round_trip_cost(5e6, AFTER)


def test_round_trip_invalid_args_raise():
    with pytest.raises(ValueError):
        odd_lot_round_trip_cost(5e6, AFTER, premium_mult=0.0)
    with pytest.raises(ValueError):
        odd_lot_round_trip_cost(5e6, AFTER, position_size_twd=0.0)


def test_min_fee_pct_magnitude():
    """低消 20 元 @ 33,333 部位 = 0.06%/邊（prereg 口徑）。"""
    fee = DEFAULT_MIN_FEE_TWD / DEFAULT_POSITION_SIZE_TWD
    assert fee == pytest.approx(0.0006, rel=0.01)


# ── 5. calibration.json 一致性（provenance 鎖定）────────────────────────────

@pytest.mark.skipif(not CALIBRATION_PATH.exists(), reason="calibration.json 不在本機")
def test_module_constants_match_calibration_json():
    with open(CALIBRATION_PATH, encoding="utf-8") as f:
        cal = json.load(f)
    assert cal["tier_labels"] == list(ODD_LOT_TIER_LABELS)
    assert [b * 1e8 for b in cal["tier_bounds_yi"]] == pytest.approx(list(ODD_LOT_TIER_BOUNDS))
    for label in ODD_LOT_TIER_LABELS:
        assert cal["tiers"][label]["premium_per_side"] == pytest.approx(
            ODD_LOT_PREMIUM_PER_SIDE[label], abs=1e-9,
        ), f"tier {label} 常數與 calibration.json 不一致——重跑校準後須同步更新模組常數"


# ── 6/7. backtest 整合 ───────────────────────────────────────────────────────

def test_walkforward_config_defaults_off():
    """新參數預設值不改變既有基準行為。"""
    from skills.backtest import WalkForwardConfig
    cfg = WalkForwardConfig()
    assert cfg.enable_odd_lot_costs is False
    assert cfg.odd_lot_premium_mult == 1.0


def test_odd_lot_config_block():
    from skills.backtest import _odd_lot_config_block
    assert _odd_lot_config_block(False, 1.0) is None
    block = _odd_lot_config_block(True, 1.5)
    assert block["enabled"] is True
    assert block["premium_mult"] == 1.5
    assert block["premium_per_side_by_tier"] == ODD_LOT_PREMIUM_PER_SIDE
    assert block["intraday_odd_lot_start"] == "2020-10-26"
    assert block["after_hours_era_premium_mult"] == 2.0


def test_odd_lot_cost_vs_tiered_slippage_magnitude():
    """微型股層零股成本應高於既有整股 tiered slippage 假設（1.0% 來回），
    大型股層應低於 1.0%——校準若倒掛代表資料或估計量出錯。"""
    micro = odd_lot_round_trip_cost(5e6, AFTER)
    large = odd_lot_round_trip_cost(1e9, AFTER)
    assert micro > 0.010
    assert large < 0.010
