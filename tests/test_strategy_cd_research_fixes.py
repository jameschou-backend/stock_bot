"""Strategy C/D 研究線 2026-07-03 健檢修正的回歸測試。

涵蓋三項修正：
- P1-2：rotation 訓練標籤 cutoff 改「交易日」制（compute_trading_day_cutoff），
        舊日曆天制（20 日曆天 ≈ 14 交易日）蓋不住 20 交易日 label horizon。
- P1-3：C/D label 與 rotation 模擬報酬改用還原價 adj_close
        （raw close 會把配息股除權息跌價誤標「未來差」）。
- P2-7b：rotation 交易成本預設 0.003 → 0.00585（台股真實來回成本）。
"""
from __future__ import annotations

import inspect
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_rotation import (  # noqa: E402
    DEFAULT_TRANSACTION_COST,
    compute_trading_day_cutoff,
    run_rotation,
)
from skills.build_features import apply_adj_factors  # noqa: E402


# ─────────────────────────────────────────────
# 工具：產生連續平日（模擬交易日序列）
# ─────────────────────────────────────────────
def _weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:  # 週一~週五
            out.append(d)
        d += timedelta(days=1)
    return out


# ─────────────────────────────────────────────
# P1-2：交易日制 cutoff
# ─────────────────────────────────────────────
class TestComputeTradingDayCutoff:
    def test_cutoff_is_trading_days_not_calendar_days(self):
        """buffer=20 時 cutoff = ref 前第 20 個「交易日」，而非 20 日曆天。"""
        tds = _weekdays(date(2025, 1, 6), 60)
        ref = tds[50]
        cutoff = compute_trading_day_cutoff(tds, ref, 20)
        # 交易日制：往前退恰好 20 個交易日
        assert cutoff == tds[30]
        # 舊日曆天制的 cutoff（ref - 20 日曆天 ≈ 只退 14 個交易日）比交易日制晚，
        # 代表舊制多納入了 ~6 個交易日的洩漏樣本
        legacy_cutoff = ref - timedelta(days=20)
        assert cutoff < legacy_cutoff
        # 20 個交易日在平日序列上至少橫跨 28 個日曆天
        assert (ref - cutoff).days >= 28

    def test_ref_date_not_in_sequence_uses_left_position(self):
        """ref 不在序列內（如週末/資料缺日）時，以 searchsorted-left 定位。"""
        tds = _weekdays(date(2025, 1, 6), 40)
        ref = tds[-1] + timedelta(days=1)  # 序列外的下一天
        cutoff = compute_trading_day_cutoff(tds, ref, 5)
        assert cutoff == tds[len(tds) - 5]

    def test_insufficient_history_falls_back_to_calendar(self):
        """交易日不足時回退日曆天（保守 fallback，與主回測一致）。"""
        tds = _weekdays(date(2025, 1, 6), 10)
        ref = tds[5]
        cutoff = compute_trading_day_cutoff(tds, ref, 20)
        assert cutoff == ref - timedelta(days=20)

    def test_zero_buffer_returns_ref_date(self):
        tds = _weekdays(date(2025, 1, 6), 10)
        assert compute_trading_day_cutoff(tds, tds[5], 0) == tds[5]

    def test_cutoff_strictly_covers_horizon(self):
        """cutoff 前（不含）任一樣本 T，其 T+20 交易日 label 皆早於 ref。

        訓練取 trading_date < cutoff：最後一個可用樣本是 tds[idx(cutoff)-1]，
        其 label 用到 close[T + 20 交易日] = tds[idx(cutoff)+19] < ref = tds[idx(cutoff)+20]。
        """
        tds = _weekdays(date(2024, 1, 1), 120)
        ref = tds[100]
        buffer_days = 20
        cutoff = compute_trading_day_cutoff(tds, ref, buffer_days)
        cut_idx = tds.index(cutoff)
        last_train_sample = tds[cut_idx - 1]  # < cutoff 的最後一個交易日
        label_end = tds[tds.index(last_train_sample) + buffer_days]
        assert label_end < ref


# ─────────────────────────────────────────────
# P1-3：還原價 label 語義
# ─────────────────────────────────────────────
class TestAdjClosLabel:
    def _make_dividend_stock(self):
        """配息股：raw close 100 → 除息日跌至 95（配 5 元），股價實質未跌。"""
        tds = _weekdays(date(2025, 3, 3), 10)
        ex_div_idx = 5
        rows = []
        for i, d in enumerate(tds):
            close = 100.0 if i < ex_div_idx else 95.0
            rows.append({"stock_id": "1234", "trading_date": d,
                         "close": close, "volume": 1000})
        price_df = pd.DataFrame(rows)
        # 累積還原因子：除息日前 0.95（adj_close = 100×0.95 = 95），之後 1.0
        factor_df = pd.DataFrame([
            {"stock_id": "1234", "trading_date": d,
             "adj_factor": 0.95 if i < ex_div_idx else 1.0}
            for i, d in enumerate(tds)
        ])
        return price_df, factor_df, tds

    def test_adj_label_neutralizes_ex_dividend_drop(self):
        """還原後 label：跨除息日 forward return 應為 0（含息），raw 為 -5%。"""
        price_df, factor_df, tds = self._make_dividend_stock()
        horizon = 3
        adj_df = apply_adj_factors(price_df, factor_df)

        def _fwd(df, col):
            pp = df.pivot_table(index="trading_date", columns="stock_id",
                                values=col, aggfunc="last").sort_index()
            return pp.shift(-horizon) / pp - 1

        raw_ret = _fwd(adj_df, "close")
        adj_ret = _fwd(adj_df, "adj_close")
        # 樣本日 tds[3]，label 用 tds[6]（跨除息日 tds[5]）
        t = pd.Timestamp(tds[3])
        assert raw_ret.loc[t, "1234"] == pytest.approx(-0.05)   # raw 被誤標 -5%
        assert adj_ret.loc[t, "1234"] == pytest.approx(0.0)     # adj 正確為 0%

    def test_empty_factor_table_falls_back_to_raw(self):
        """factor 表為空（測試環境）：adj_close == close，行為不變。"""
        price_df, _, _ = self._make_dividend_stock()
        adj_df = apply_adj_factors(price_df, None)
        assert (adj_df["adj_factor"] == 1.0).all()
        pd.testing.assert_series_equal(
            adj_df["adj_close"], adj_df["close"].astype(float),
            check_names=False,
        )


# ─────────────────────────────────────────────
# P2-7b：交易成本預設值
# ─────────────────────────────────────────────
class TestTransactionCostDefault:
    def test_default_constant_is_realistic_round_trip_cost(self):
        assert DEFAULT_TRANSACTION_COST == pytest.approx(0.00585)

    def test_run_rotation_signature_default(self):
        sig = inspect.signature(run_rotation)
        assert sig.parameters["transaction_cost_pct"].default == pytest.approx(0.00585)

    def test_cli_default_matches_constant(self):
        """argparse 預設須引用 DEFAULT_TRANSACTION_COST，避免兩處 drift。"""
        src = (ROOT / "scripts" / "backtest_rotation.py").read_text(encoding="utf-8")
        assert 'default=DEFAULT_TRANSACTION_COST' in src
        assert 'default=0.003' not in src


# ─────────────────────────────────────────────
# 原始碼不變量：三個檔案的 label / cutoff 接線
# ─────────────────────────────────────────────
class TestSourceWiring:
    """輕量原始碼檢查：鎖定 adj_close label 與交易日制 cutoff 不被 revert。"""

    @pytest.mark.parametrize("fname", [
        "backtest_rotation.py", "strategy_c_pick.py", "strategy_d_pick.py",
    ])
    def test_label_pivot_uses_adj_close(self, fname):
        src = (ROOT / "scripts" / fname).read_text(encoding="utf-8")
        assert 'values="adj_close"' in src, f"{fname} 的 label pivot 必須用還原價"
        assert 'values="close"' not in src, f"{fname} 不應殘留 raw close 的 label pivot"

    def test_rotation_uses_trading_day_cutoff(self):
        src = (ROOT / "scripts" / "backtest_rotation.py").read_text(encoding="utf-8")
        # retrain 與 fixed-window 兩處都必須走 compute_trading_day_cutoff
        assert src.count("compute_trading_day_cutoff(all_dates") >= 2
        # 不可殘留日曆天制 cutoff
        assert "timedelta(days=_eff_buffer)" not in src


def test_rotation_max_price_wiring():
    """--max-price 口徑對齊（2026-07-10 D 重驗前置）：CLI → run_rotation → 進場過濾接線。

    實盤 daily_run.sh 傳 --max-price 250，重驗回測必須支援同口徑，
    否則回測含買不起的高價股，結果對使用者不可執行。
    """
    import inspect
    import scripts.backtest_rotation as rot

    sig = inspect.signature(rot.run_rotation)
    assert "max_stock_price" in sig.parameters
    assert sig.parameters["max_stock_price"].default == 0.0  # 預設關閉（向後相容）

    src = Path(rot.__file__).read_text()
    assert '"--max-price"' in src, "CLI 旗標必須存在"
    assert "max_stock_price=args.max_price" in src, "CLI 必須接進 run_rotation"
    # 過濾必須套在進場候選（candidates）而非全宇宙排名
    assert "raw_price_map.get(_s, 0.0)" in src


def test_rotation_signal_lag_wiring():
    """--signal-lag 口徑（2026-07-10 D 重驗誠實臂）：T-1 特徵決策、T 日執行。

    lag=0 用 T 日特徵（含 16-17 點才公佈的法人資料）在 T 日收盤進場，
    是資訊上不可能的口徑；實盤時序 = T-1 晚間訊號 → T 日下單。
    """
    import inspect
    import scripts.backtest_rotation as rot

    sig = inspect.signature(rot.run_rotation)
    assert "signal_lag" in sig.parameters
    assert sig.parameters["signal_lag"].default == 0  # 預設不變（向後相容 + 對照舊數字）

    src = Path(rot.__file__).read_text()
    assert '"--signal-lag"' in src
    assert "signal_lag=args.signal_lag" in src
    # 決策特徵日必須依 lag 從 bt_dates 回退
    assert "bt_dates[day_idx - signal_lag]" in src
