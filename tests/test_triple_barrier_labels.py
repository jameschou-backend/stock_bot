"""Triple-Barrier label 純函式測試（無 DB）。

驗證 López de Prado 三 barrier 邏輯：
- profit-take 先觸 → tb_label=+1
- stop-loss 先觸 → tb_label=-1
- 兩者都沒觸到 max_horizon → tb_label=0（time barrier）
- forward window 不足 → drop
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from skills.build_labels import triple_barrier_labels


def _build_prices(stock_id: str, close_seq: list[float]) -> pd.DataFrame:
    """從一個 close 序列建一個股的 panel；trading_date 從 2026-01-02 起逐日。"""
    start = date(2026, 1, 2)
    rows = []
    for i, c in enumerate(close_seq):
        rows.append({
            "stock_id": stock_id,
            "trading_date": start + timedelta(days=i),
            "close": c,
        })
    return pd.DataFrame(rows)


class TestTripleBarrierBasic:
    def test_profit_take_first(self):
        """從 100 漲到 120（+20%）→ +15% PT 應在 day 2 觸發。"""
        # entry=100, day1=110, day2=120 → day1 是 +10% 還沒 PT；day2 +20% 觸 PT
        df = _build_prices("2330", [100, 110, 120, 130, 140])
        out = triple_barrier_labels(df, upper_pt=0.15, lower_sl=-0.07, max_horizon=4)
        # entry day = day0 (100)，PT 在 day2
        first = out[out["trading_date"] == date(2026, 1, 2)].iloc[0]
        assert first["tb_label"] == 1
        assert first["tb_exit_type"] == "pt"
        assert first["tb_exit_day_offset"] == 2
        assert abs(first["tb_return"] - 0.20) < 1e-9

    def test_stop_loss_first(self):
        """從 100 跌到 92（-8%）→ -7% SL 應在 day 2 觸發。"""
        df = _build_prices("2330", [100, 95, 92, 95, 100])
        out = triple_barrier_labels(df, upper_pt=0.15, lower_sl=-0.07, max_horizon=4)
        first = out[out["trading_date"] == date(2026, 1, 2)].iloc[0]
        assert first["tb_label"] == -1
        assert first["tb_exit_type"] == "sl"
        assert first["tb_exit_day_offset"] == 2
        assert abs(first["tb_return"] - (-0.08)) < 1e-9

    def test_time_barrier(self):
        """都沒觸到 → time barrier，return = 第 max_horizon 天的 cumulative。"""
        # 全程在 ±5% 內震盪，max_horizon=4
        df = _build_prices("2330", [100, 102, 98, 103, 104, 100, 99])
        out = triple_barrier_labels(df, upper_pt=0.15, lower_sl=-0.07, max_horizon=4)
        first = out[out["trading_date"] == date(2026, 1, 2)].iloc[0]
        assert first["tb_label"] == 0
        assert first["tb_exit_type"] == "time"
        assert first["tb_exit_day_offset"] == 4
        # entry=100, day4 close=104 → +4%
        assert abs(first["tb_return"] - 0.04) < 1e-9

    def test_pt_priority_over_sl_when_same_day(self):
        """同一天若 PT 跟 SL 同時可能（不太可能但防禦），程式碼會先檢查 PT 再 SL，PT 優先。

        實際上單日內 close 只有一個值，不會同時 >= upper 與 <= lower。
        但若 upper_pt + lower_sl 設小，可能同一天 close 同時滿足兩個門檻
        （e.g. upper_pt=0.01, lower_sl=-0.01 + 開盤跌停又拉漲停）。
        我們不模擬日內，只看每日 close，所以這個 case 用「假裝 close=99.5」表示
        若同時，PT 不會觸（99.5 < 100×1.01=101），SL 觸發。
        """
        # 設超嚴 barrier：±0.5%，entry=100，day1 close=99.5 → SL 觸發
        df = _build_prices("2330", [100, 99.5, 100, 100])
        out = triple_barrier_labels(df, upper_pt=0.005, lower_sl=-0.005, max_horizon=3)
        first = out[out["trading_date"] == date(2026, 1, 2)].iloc[0]
        assert first["tb_label"] == -1

    def test_forward_window_insufficient_dropped(self):
        """最後 max_horizon 列 forward window 不足 → 整列 drop。"""
        df = _build_prices("2330", [100, 105, 110, 115])
        # max_horizon=4 但只有 4 個 sample，entry=day0 可看 day1~3（3 天）
        # 沒一天 hit PT/SL，且 forward 不足 max_horizon → 應被 drop
        out = triple_barrier_labels(df, upper_pt=0.20, lower_sl=-0.20, max_horizon=4)
        # 4 個 sample 都不該有 valid label
        assert out.empty or len(out) < len(df)
        # day0 不該存在（因為 forward window=3 < max_horizon=4）
        assert date(2026, 1, 2) not in out["trading_date"].values

    def test_multi_stock_independence(self):
        """多股不應互相影響。"""
        df_a = _build_prices("2330", [100, 120, 130])  # PT day1
        df_b = _build_prices("1101", [100, 90, 80])    # SL day1
        df = pd.concat([df_a, df_b], ignore_index=True)
        out = triple_barrier_labels(df, upper_pt=0.15, lower_sl=-0.07, max_horizon=2)
        a = out[out["stock_id"] == "2330"].iloc[0]
        b = out[out["stock_id"] == "1101"].iloc[0]
        assert a["tb_label"] == 1
        assert b["tb_label"] == -1


class TestTripleBarrierEdgeCases:
    def test_handles_nan_close(self):
        df = _build_prices("2330", [100, np.nan, 120, 130, 140])
        out = triple_barrier_labels(df, upper_pt=0.15, lower_sl=-0.07, max_horizon=3)
        # NaN 應該被 dropna 掉，剩下 100/120/130/140 連續
        # day0 entry=100，day1 (原 day2)=120 觸 PT
        first = out[out["trading_date"] == date(2026, 1, 2)].iloc[0]
        assert first["tb_label"] == 1

    def test_zero_close_skipped(self):
        df = _build_prices("2330", [0, 100, 120, 130, 140])
        out = triple_barrier_labels(df, upper_pt=0.15, lower_sl=-0.07, max_horizon=3)
        # entry=0 整列應跳過
        assert date(2026, 1, 2) not in out["trading_date"].values

    def test_single_row_per_stock(self):
        """只一個 row 無法計算 forward。"""
        df = _build_prices("2330", [100])
        out = triple_barrier_labels(df, upper_pt=0.15, lower_sl=-0.07, max_horizon=20)
        assert out.empty

    def test_empty_input(self):
        df = pd.DataFrame(columns=["stock_id", "trading_date", "close"])
        out = triple_barrier_labels(df)
        assert out.empty
        assert list(out.columns) == [
            "stock_id", "trading_date", "tb_label",
            "tb_return", "tb_exit_type", "tb_exit_day_offset",
        ]


class TestTripleBarrierValidation:
    def test_rejects_non_positive_pt(self):
        df = _build_prices("2330", [100, 110, 120])
        with pytest.raises(ValueError, match="upper_pt"):
            triple_barrier_labels(df, upper_pt=0.0)
        with pytest.raises(ValueError, match="upper_pt"):
            triple_barrier_labels(df, upper_pt=-0.1)

    def test_rejects_non_negative_sl(self):
        df = _build_prices("2330", [100, 110, 120])
        with pytest.raises(ValueError, match="lower_sl"):
            triple_barrier_labels(df, lower_sl=0.0)
        with pytest.raises(ValueError, match="lower_sl"):
            triple_barrier_labels(df, lower_sl=0.05)

    def test_rejects_zero_max_horizon(self):
        df = _build_prices("2330", [100, 110, 120])
        with pytest.raises(ValueError, match="max_horizon"):
            triple_barrier_labels(df, max_horizon=0)

    def test_rejects_missing_columns(self):
        df = pd.DataFrame({"stock_id": ["2330"], "trading_date": [date(2026, 1, 2)]})
        with pytest.raises(ValueError, match="缺欄位"):
            triple_barrier_labels(df)


class TestTripleBarrierDistribution:
    """合成資料驗證 label 分佈合理。"""

    def test_distribution_on_random_walk(self):
        """隨機漫步（無 drift）→ +1/-1/0 三類大致平衡（PT 略多 = 因為 +15% 比 -7% 距離更遠但時間 horizon 內較易回升）。"""
        rng = np.random.default_rng(42)
        # 200 天 daily log-return ~ N(0, 0.02)，cumulative close
        log_ret = rng.normal(0, 0.02, 200)
        close = 100 * np.exp(np.cumsum(log_ret))
        df = pd.DataFrame({
            "stock_id": "TEST",
            "trading_date": [date(2026, 1, 2) + timedelta(days=i) for i in range(200)],
            "close": close,
        })
        out = triple_barrier_labels(df, upper_pt=0.15, lower_sl=-0.07, max_horizon=20)
        counts = out["tb_label"].value_counts().to_dict()
        # 三類至少都有出現
        assert -1 in counts
        assert 0 in counts
        assert 1 in counts
        # 沒有任何一類佔超過 90%（明顯失衡才會這樣）
        total = len(out)
        for v in counts.values():
            assert v / total < 0.90, f"label 分佈失衡: {counts}"

    def test_pt_return_at_or_above_threshold(self):
        """tb_exit_type='pt' 的 row 必有 tb_return >= upper_pt（漲到才觸）。"""
        rng = np.random.default_rng(7)
        log_ret = rng.normal(0.001, 0.02, 300)
        close = 100 * np.exp(np.cumsum(log_ret))
        df = pd.DataFrame({
            "stock_id": "TEST",
            "trading_date": [date(2026, 1, 2) + timedelta(days=i) for i in range(300)],
            "close": close,
        })
        out = triple_barrier_labels(df, upper_pt=0.15, lower_sl=-0.07, max_horizon=20)
        pt_rows = out[out["tb_exit_type"] == "pt"]
        assert (pt_rows["tb_return"] >= 0.15 - 1e-9).all()
        sl_rows = out[out["tb_exit_type"] == "sl"]
        assert (sl_rows["tb_return"] <= -0.07 + 1e-9).all()
