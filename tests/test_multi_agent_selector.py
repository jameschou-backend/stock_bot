from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

from skills.multi_agent_selector import AGENT_NAMES, run_multi_agent_selection


def _make_feature_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ret_20": [0.12, 0.03, -0.02, 0.08, -0.10, 0.01],
            "breakout_20": [0.02, 0.00, -0.01, 0.03, -0.04, 0.00],
            "rsi_14": [70, 55, 40, 68, 30, 50],
            "macd_hist": [0.8, 0.2, -0.1, 0.5, -0.8, 0.0],
            "drawdown_60": [-0.08, -0.12, -0.18, -0.10, -0.32, -0.15],
            "vol_ratio_20": [1.3, 1.0, 0.9, 1.2, 0.8, 1.0],
            "foreign_net_20": [1000, 200, -200, 800, -1200, 100],
            "trust_net_20": [100, 50, -20, 80, -150, 10],
            "dealer_net_20": [20, 5, -10, 10, -30, 2],
            "chip_flow_intensity_20": [0.005, 0.001, -0.001, 0.003, -0.006, 0.0],
            "margin_balance_chg_20": [0.08, 0.02, -0.03, 0.05, -0.06, 0.01],
            "short_balance_chg_20": [-0.03, -0.01, 0.02, -0.02, 0.08, 0.0],
            "margin_short_ratio": [0.18, 0.22, 0.28, 0.20, 0.45, 0.24],
            "fund_revenue_yoy": [0.2, 0.05, -0.02, 0.16, -0.2, 0.01],
            "fund_revenue_mom": [0.03, 0.01, -0.01, 0.02, -0.04, 0.0],
            "fund_revenue_trend_3m": [0.12, 0.02, -0.01, 0.09, -0.15, 0.01],
            "theme_hot_score": [1.2, 0.8, 0.6, 1.1, 1.3, 0.7],
            "theme_return_20": [0.05, 0.01, -0.02, 0.04, -0.08, 0.0],
            "theme_turnover_ratio": [0.15, 0.11, 0.08, 0.14, 0.16, 0.1],
            "amt_ratio_20": [1.3, 1.0, 0.9, 1.2, 0.6, 1.0],
        }
    )


def test_multi_agent_selector_schema_and_weights():
    feature_df = _make_feature_df()
    stock_ids = pd.Series(["1101", "1216", "1301", "2303", "2317", "2330"])
    cfg = SimpleNamespace(multi_agent_weights={"tech": 0.35, "flow": 0.30, "margin": 0.10, "fund": 0.15, "theme": 0.10})
    dq_ctx = {"degraded_mode": False, "degraded_datasets": []}
    selection_meta = {"tradability": {}, "liquidity": {}}

    picks_df, dump = run_multi_agent_selection(
        feature_df=feature_df,
        stock_ids=stock_ids,
        pick_date=date(2026, 2, 28),
        topn=3,
        config=cfg,
        dq_ctx=dq_ctx,
        selection_meta=selection_meta,
    )

    assert len(picks_df) == 3
    assert {"stock_id", "score", "reason_json"}.issubset(picks_df.columns)
    first_reason = picks_df.iloc[0]["reason_json"]
    assert "_selection_meta" in first_reason
    assert "agents" in first_reason
    for agent in AGENT_NAMES:
        output = first_reason["agents"][agent]
        assert {"ticker", "agent", "signal", "confidence", "reasons", "risk_flags", "unavailable"}.issubset(output.keys())
    weights_used = first_reason["_selection_meta"]["weights_used"]
    assert abs(sum(weights_used.values()) - 1.0) < 1e-9
    assert "summary" in dump


def test_multi_agent_degraded_unavailable_and_renormalized_weights():
    feature_df = _make_feature_df()
    stock_ids = pd.Series(["1101", "1216", "1301", "2303", "2317", "2330"])
    cfg = SimpleNamespace(multi_agent_weights={"tech": 0.4, "flow": 0.3, "margin": 0.1, "fund": 0.1, "theme": 0.1})
    dq_ctx = {"degraded_mode": True, "degraded_datasets": ["raw_institutional", "raw_margin_short"]}
    selection_meta = {"tradability": {}, "liquidity": {}}

    picks_df, _ = run_multi_agent_selection(
        feature_df=feature_df,
        stock_ids=stock_ids,
        pick_date=date(2026, 2, 28),
        topn=4,
        config=cfg,
        dq_ctx=dq_ctx,
        selection_meta=selection_meta,
    )
    reason = picks_df.iloc[0]["reason_json"]
    assert reason["agents"]["flow"]["unavailable"] is True
    assert reason["agents"]["margin"]["unavailable"] is True
    weights_used = reason["_selection_meta"]["weights_used"]
    assert "flow" not in weights_used
    assert "margin" not in weights_used
    assert abs(sum(weights_used.values()) - 1.0) < 1e-9


def test_model_alignment_weight_boosts_overlap():
    """model_alignment_weight > 0 時，與 model 高分股的 final_score 應提升。"""
    feature_df = _make_feature_df()
    stock_ids = pd.Series(["1101", "1216", "1301", "2303", "2317", "2330"])
    dq_ctx = {"degraded_mode": False, "degraded_datasets": []}
    selection_meta = {"tradability": {}, "liquidity": {}}

    # 無 alignment 時的 topn=2 結果
    cfg_no_align = SimpleNamespace(
        multi_agent_weights={"tech": 0.35, "flow": 0.30, "margin": 0.10, "fund": 0.15, "theme": 0.10},
        ma_model_alignment_weight=0.0,
    )
    picks_no, _ = run_multi_agent_selection(
        feature_df=feature_df,
        stock_ids=stock_ids,
        pick_date=date(2026, 2, 28),
        topn=2,
        config=cfg_no_align,
        dq_ctx=dq_ctx,
        selection_meta=selection_meta,
    )

    # 給予 model_score_map，讓 "2303" 擁有最高 model 分（原本不在前2）
    model_score_map = {"1101": 0.1, "1216": 0.1, "1301": 0.1, "2303": 10.0, "2317": 0.1, "2330": 0.1}
    cfg_align = SimpleNamespace(
        multi_agent_weights={"tech": 0.35, "flow": 0.30, "margin": 0.10, "fund": 0.15, "theme": 0.10},
        ma_model_alignment_weight=1.0,
    )
    picks_align, _ = run_multi_agent_selection(
        feature_df=feature_df,
        stock_ids=stock_ids,
        pick_date=date(2026, 2, 28),
        topn=2,
        config=cfg_align,
        dq_ctx=dq_ctx,
        selection_meta=selection_meta,
        model_score_map=model_score_map,
    )

    # alignment 模式下 "2303" 應出現在選股結果中
    assert "2303" in picks_align["stock_id"].tolist()
    # meta 中應記錄 model_alignment_weight
    meta = picks_align.iloc[0]["reason_json"]["_selection_meta"]
    assert meta.get("model_alignment_weight") == 1.0


def test_model_alignment_weight_zero_no_effect():
    """model_alignment_weight=0 時，傳入 model_score_map 不影響結果。"""
    feature_df = _make_feature_df()
    stock_ids = pd.Series(["1101", "1216", "1301", "2303", "2317", "2330"])
    dq_ctx = {"degraded_mode": False, "degraded_datasets": []}
    selection_meta = {"tradability": {}, "liquidity": {}}
    cfg = SimpleNamespace(
        multi_agent_weights={"tech": 0.35, "flow": 0.30, "margin": 0.10, "fund": 0.15, "theme": 0.10},
        ma_model_alignment_weight=0.0,
    )
    model_score_map = {"1101": 999.0, "1216": 0.0, "1301": 0.0, "2303": 0.0, "2317": 0.0, "2330": 0.0}

    picks_no_map, _ = run_multi_agent_selection(
        feature_df=feature_df,
        stock_ids=stock_ids,
        pick_date=date(2026, 2, 28),
        topn=3,
        config=cfg,
        dq_ctx=dq_ctx,
        selection_meta=selection_meta,
    )
    picks_with_map, _ = run_multi_agent_selection(
        feature_df=feature_df,
        stock_ids=stock_ids,
        pick_date=date(2026, 2, 28),
        topn=3,
        config=cfg,
        dq_ctx=dq_ctx,
        selection_meta=selection_meta,
        model_score_map=model_score_map,
    )
    # weight=0 時結果應完全相同
    assert picks_no_map["stock_id"].tolist() == picks_with_map["stock_id"].tolist()

