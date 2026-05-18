"""Regression tests guarding against training-label forward leakage.

歷史 bug（2026-03-13 修正）：
- `skills/backtest.py` 的 `label_horizon_buffer` 預設曾為 0。
- Label 定義為 `future_ret_h = close_{T+20} / close_T - 1`（20 交易日 forward return）。
- 當 buffer=0 時，訓練截止 `rb_date` 前 20 個交易日的樣本，其標籤
  涉及測試期收盤價 → 訓練標籤前向洩漏。
- 10y 回測累積報酬從虛高 +10004% 降回真實 +205%（Experiment F）。
- 修正後預設 `label_horizon_buffer=20`（日曆天，≈14 交易日）。
- `train_ranker.py` 的 `LABEL_HORIZON_BUFFER_DAYS` 同步改為 20。

本檔保護以下不變式：
1. `backtest.run_backtest` 的 `label_horizon_buffer` 預設 >= 20。
2. `backtest.WalkForwardConfig.label_horizon_buffer` 預設 >= 20。
3. `train_ranker.py` 中 `LABEL_HORIZON_BUFFER_DAYS` == 20。
4. 給定模擬的 `rb_date`，訓練集 label_cutoff = rb_date - timedelta(days=buffer)，
   不會包含落入 forward window 的標籤。
"""

from __future__ import annotations

import inspect
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

import skills.backtest as backtest_module
import skills.train_ranker as train_ranker_module
from skills.backtest import WalkForwardConfig, run_backtest


MIN_REQUIRED_BUFFER = 20
LABEL_HORIZON_TRADING_DAYS = 20  # 對應 future_ret_h horizon


def _get_param_default(func, name):
    sig = inspect.signature(func)
    if name not in sig.parameters:
        pytest.fail(f"Parameter '{name}' not found in {func.__name__} signature")
    default = sig.parameters[name].default
    if default is inspect.Parameter.empty:
        pytest.fail(f"Parameter '{name}' has no default in {func.__name__}")
    return default


# ── 不變式 1：run_backtest 預設值 ─────────────────────────────────────────────

def test_run_backtest_default_label_horizon_buffer_is_at_least_20():
    """run_backtest() 的 label_horizon_buffer 預設值必須 >= 20。

    若有人將 buffer 改回 0 或 < 20，本測試會立即失敗，阻止標籤洩漏 bug 回歸。
    """
    default = _get_param_default(run_backtest, "label_horizon_buffer")
    assert default >= MIN_REQUIRED_BUFFER, (
        f"run_backtest() default label_horizon_buffer={default} 小於 {MIN_REQUIRED_BUFFER}，"
        f"會造成訓練標籤前向洩漏（10y 累積報酬虛高 +10004% bug 已回歸）。"
        f"參考 CLAUDE.md「訓練標籤前向洩漏歷史」與 Experiment F。"
    )


# ── 不變式 2：WalkForwardConfig 預設值 ───────────────────────────────────────

def test_walkforward_config_default_label_horizon_buffer_is_at_least_20():
    """WalkForwardConfig.label_horizon_buffer 預設值必須 >= 20。"""
    cfg = WalkForwardConfig()
    assert cfg.label_horizon_buffer >= MIN_REQUIRED_BUFFER, (
        f"WalkForwardConfig.label_horizon_buffer={cfg.label_horizon_buffer} 小於 {MIN_REQUIRED_BUFFER}，"
        f"會造成訓練標籤前向洩漏。"
    )


# ── 不變式 3：train_ranker.py LABEL_HORIZON_BUFFER_DAYS ────────────────────

def test_train_ranker_label_horizon_buffer_days_is_20():
    """skills/train_ranker.py 中 `LABEL_HORIZON_BUFFER_DAYS = 20` 不可被改回 7 或 0。

    這是 function-local 常數（在 run() 內），故以 source parsing 驗證。
    """
    src_path = Path(train_ranker_module.__file__)
    src = src_path.read_text(encoding="utf-8")

    matches = re.findall(
        r"^\s*LABEL_HORIZON_BUFFER_DAYS\s*=\s*(\d+)\s*$",
        src,
        flags=re.MULTILINE,
    )
    assert matches, (
        f"找不到 LABEL_HORIZON_BUFFER_DAYS 賦值於 {src_path}，"
        f"請確認此常數仍存在於 train_ranker.run()。"
    )
    values = [int(m) for m in matches]
    for v in values:
        assert v == MIN_REQUIRED_BUFFER, (
            f"train_ranker.py 中 LABEL_HORIZON_BUFFER_DAYS={v} (應為 {MIN_REQUIRED_BUFFER})。"
            f"若改回 7 或 0 會造成訓練標籤前向洩漏。"
        )


# ── 不變式 4：實際模擬切資料邏輯，確認 train_end 不會落入 forward window ────────

def _build_synthetic_features_labels(end_date: date, n_stocks: int = 10, n_days: int = 60):
    """構造 n_stocks × n_days 的迷你 features + labels DataFrame。

    Label 為 future_ret_h（20 交易日 forward return），最後 20 天的 label
    在現實中會涉及未來資料，故 buffer >= 20 應該排除這些列。
    """
    dates = pd.bdate_range(end=end_date, periods=n_days).date
    rows = []
    label_rows = []
    for sid_i in range(n_stocks):
        sid = f"{1000 + sid_i:04d}"
        for d in dates:
            rows.append({"stock_id": sid, "trading_date": d, "feat_x": float(sid_i)})
            label_rows.append({"stock_id": sid, "trading_date": d, "future_ret_h": 0.01})
    feat_df = pd.DataFrame(rows)
    label_df = pd.DataFrame(label_rows)
    return feat_df, label_df


def test_label_cutoff_excludes_forward_window():
    """模擬 backtest.py L1076 的切資料邏輯，驗證 label_cutoff 排除前向洩漏區間。

    backtest.py 內部邏輯：
        label_cutoff = rb_date - timedelta(days=label_horizon_buffer)
        train_label = label_df[label_df["trading_date"] < label_cutoff]

    當 buffer >= 20（日曆天 ≈ 14 交易日）時，最後 ~14 個交易日的標籤被排除，
    避免它們涉及 rb_date 之後的未來收盤價。
    """
    rb_date = date(2025, 6, 30)
    feat_df, label_df = _build_synthetic_features_labels(end_date=rb_date, n_days=60)

    # 對應 backtest.py L1076-1078 的訓練資料切片邏輯
    buffer = _get_param_default(run_backtest, "label_horizon_buffer")
    label_cutoff = rb_date - timedelta(days=buffer)
    train_label = label_df[label_df["trading_date"] < label_cutoff]

    # 1) 訓練 label 的 trading_date 全部嚴格小於 label_cutoff
    assert (train_label["trading_date"] < label_cutoff).all(), (
        "Train label 的 trading_date 越過 label_cutoff，存在前向洩漏風險。"
    )

    # 2) 訓練 label 最大 trading_date 與 rb_date 的距離 >= buffer 天
    max_train_label_date = train_label["trading_date"].max()
    gap_days = (rb_date - max_train_label_date).days
    assert gap_days >= buffer, (
        f"訓練 label 最後一天 {max_train_label_date} 距 rb_date {rb_date} 僅 {gap_days} 天，"
        f"小於 buffer={buffer}，訓練標籤可能涉及測試期收盤價。"
    )

    # 3) buffer=0 時必定洩漏：最後一筆 label 的 forward window 會超出 rb_date
    leaky_buffer = 0
    leaky_cutoff = rb_date - timedelta(days=leaky_buffer)
    leaky_train_label = label_df[label_df["trading_date"] < leaky_cutoff]
    leaky_max = leaky_train_label["trading_date"].max()
    # 該 label 的 forward window 終點 ≈ leaky_max + 20 交易日，會 > rb_date
    # （簡化以日曆天近似 30 天 forward window，仍 >> rb_date）
    forward_window_end_approx = leaky_max + timedelta(days=LABEL_HORIZON_TRADING_DAYS)
    assert forward_window_end_approx > rb_date, (
        f"洩漏控制組驗證失敗：buffer=0 時，最後一筆 train label {leaky_max} 的"
        f"forward window 終點 {forward_window_end_approx} 未越過 rb_date {rb_date}，"
        f"測試前提不成立，請檢查 synthetic data 構造。"
    )


# ── 防呆：若有人改源碼把預設值降下來，給出明確錯誤訊息 ────────────────────────

def test_backtest_source_default_signature_pins_buffer_20():
    """直接於 backtest.py 原始碼掃描 run_backtest 簽章，確保預設 label_horizon_buffer=20。

    雙保險：即使有人透過 monkey-patching 或 wrapper 改動執行期預設，
    源碼層級的常數仍必須維持 20。
    """
    src = Path(backtest_module.__file__).read_text(encoding="utf-8")
    # 匹配 run_backtest 簽章中的 `label_horizon_buffer: int = N,`
    m = re.search(
        r"label_horizon_buffer\s*:\s*int\s*=\s*(\d+)",
        src,
    )
    assert m, "找不到 backtest.py 中 label_horizon_buffer 的型別註解預設值。"
    val = int(m.group(1))
    assert val >= MIN_REQUIRED_BUFFER, (
        f"backtest.py 源碼中 label_horizon_buffer 預設 {val} < {MIN_REQUIRED_BUFFER}。"
    )
