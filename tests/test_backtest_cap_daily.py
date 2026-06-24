"""cap-daily-return 診斷旗標的核心邏輯測試。

背景（2026-06-22）：survivorship 回補的下市股價含未還原的減資/停牌假跳動
（單日 close-to-close >±10%，物理不可能），被回測誤判為真實 -50% 虧損。
cap_daily_return_pct 把持有期每日報酬 winsorize 到 ±cap，中和這些假動作，
用來量化「現行基準 0.805 有多少是 artifact」。此測試鎖定 _calc_stock_return 的
gross_ret_override 分支與 winsorize 數學。
"""
import numpy as np
import pandas as pd

from skills.backtest import _calc_stock_return


def test_default_path_unchanged():
    """gross_ret_override=None → 維持 exit/entry-1 行為（byte-identical 保護）。"""
    r = _calc_stock_return(100.0, 110.0, 0.0, 0.0, -0.50)
    assert abs(r - 0.10) < 1e-12


def test_override_replaces_price_ratio():
    """提供 override → 用 override 當毛報酬，忽略 exit/entry。"""
    # exit/entry-1 = -0.50（會觸 clip），但 override=+0.03 → 應回 +0.03
    r = _calc_stock_return(100.0, 50.0, 0.0, 0.0, -0.50, gross_ret_override=0.03)
    assert abs(r - 0.03) < 1e-12


def test_override_still_applies_costs_and_clip():
    """override 仍扣成本/滑價並套 clip。"""
    r = _calc_stock_return(100.0, 100.0, 0.005, 0.002, -0.50, gross_ret_override=0.10)
    assert abs(r - (0.10 - 0.005 - 0.002)) < 1e-12
    # override 極負 → 仍被 clip 到 -0.50
    r2 = _calc_stock_return(100.0, 100.0, 0.0, 0.0, -0.50, gross_ret_override=-0.90)
    assert abs(r2 - (-0.50)) < 1e-12


def test_winsorize_math_neutralizes_impossible_single_day():
    """winsorize 數學：單日 -40% 假跳動（減資）被削到 -10%，真實連續跌停序列不受影響。"""
    cap = 0.10
    # 序列：100 → 60（-40%，物理不可能）→ 66（+10%）
    closes = pd.Series([100.0, 60.0, 66.0])
    capped = float((1.0 + closes.pct_change().dropna().clip(-cap, cap)).prod() - 1.0)
    # -40% 被削到 -10%，+10% 保留 → (1-0.10)*(1+0.10)-1 = -0.01
    assert abs(capped - (-0.01)) < 1e-9
    # 對照：未 cap 的真實序列毛報酬
    raw = float(closes.iloc[-1] / closes.iloc[0] - 1.0)
    assert abs(raw - (-0.34)) < 1e-9

    # 真實連續跌停（每日 -10%）不被 cap 改變
    limit_seq = pd.Series([100.0, 90.0, 81.0, 72.9])
    capped_limit = float((1.0 + limit_seq.pct_change().dropna().clip(-cap, cap)).prod() - 1.0)
    raw_limit = float(limit_seq.iloc[-1] / limit_seq.iloc[0] - 1.0)
    assert abs(capped_limit - raw_limit) < 1e-9  # 每日恰 -10%，cap 不動它
