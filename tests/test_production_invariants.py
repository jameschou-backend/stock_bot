"""鎖住生產配置不變量，防止無意中漂移。

背景：本專案績效宣告高度依賴特定配置組合，過去發生過 CLAUDE.md 與程式碼數字
漂移（48 vs 58 特徵、TOPN 20 vs 30、API host 0.0.0.0 vs 127.0.0.1）。這些測試
讓任何配置漂移在 CI 立即現形——若為「有意變更」，必須同步更新本測試 + CLAUDE.md
才會綠，文件漂移從此有機械防線。
"""
from __future__ import annotations

import inspect


def test_feature_column_counts():
    from skills.build_features import FEATURE_COLUMNS, PRUNED_FEATURE_COLS

    assert len(FEATURE_COLUMNS) == 87, (
        f"FEATURE_COLUMNS={len(FEATURE_COLUMNS)}（預期 87）。"
        "若有意變更特徵集，請同步更新 CLAUDE.md 與本測試。"
    )
    assert len(PRUNED_FEATURE_COLS) == 58, (
        f"PRUNED_FEATURE_COLS={len(PRUNED_FEATURE_COLS)}（預期 58）。"
        "若有意變更 SHAP 剪枝集，請同步更新 CLAUDE.md 與本測試。"
    )


def test_pruned_features_subset_except_enriched():
    """PRUNED 應為 FEATURE_COLUMNS 子集，唯一例外是 ENRICHED post-process 特徵。

    close_fracdiff_0_50 由 scripts/enrich_features_stage5_4.py 後處理寫入 parquet，
    不在 FEATURE_COLUMNS 主清單。此測試確保「例外只有它」，若未來新增其他不在
    FEATURE_COLUMNS 的 PRUNED 特徵會被擋下（避免 daily_pick 讀不到的特徵進訓練）。
    """
    from skills.build_features import FEATURE_COLUMNS, PRUNED_FEATURE_COLS

    extra = set(PRUNED_FEATURE_COLS) - set(FEATURE_COLUMNS)
    assert extra == {"close_fracdiff_0_50"}, (
        f"PRUNED 中不屬於 FEATURE_COLUMNS 的特徵：{sorted(extra)}。"
        "預期僅 close_fracdiff_0_50（ENRICHED）。"
    )


def test_backtest_walkforward_defaults():
    from skills.backtest import WalkForwardConfig

    wf = WalkForwardConfig()
    assert wf.topn == 30, f"WalkForwardConfig.topn={wf.topn}（預期 30）"
    assert wf.label_horizon_buffer == 20, (
        f"WalkForwardConfig.label_horizon_buffer={wf.label_horizon_buffer}（預期 20，"
        "對齊 20 交易日 label horizon 的去洩漏緩衝）"
    )


def test_config_default_literals():
    """檢查 config.py 的「程式碼預設值」（非 env override 後的執行值）。

    用 source 檢查而非 load_config()，因 load_config 會 load_dotenv 讀本機 .env，
    執行值取決於使用者環境。此處鎖定的是「未設 env 時的程式碼預設」。
    """
    import app.config as cfg_mod

    src = inspect.getsource(cfg_mod)
    assert 'pick("TOPN", 30)' in src, "config.py TOPN 程式碼預設應為 30"
    assert 'pick("API_HOST", "127.0.0.1")' in src, (
        "config.py API_HOST 預設應為 127.0.0.1（綁本機，避免 LAN 暴露）"
    )


def test_train_ranker_label_buffer_aligned():
    """train_ranker 的 label horizon buffer 應與 backtest 一致（=20 日曆天）。"""
    import skills.train_ranker as tr

    src = inspect.getsource(tr)
    assert "LABEL_HORIZON_BUFFER_DAYS = 20" in src, (
        "train_ranker 的 LABEL_HORIZON_BUFFER_DAYS 應為 20，與 backtest 對齊"
    )


# ── P1-5 回測=部署對齊（2026-07-03）─────────────────────────────────────────


def test_ranker_prod_params_locked():
    """RANKER_PROD_PARAMS 內容鎖定：以 backtest 現行驗證過的參數為準（500 樹、無 ES）。"""
    from skills.model_params import LGBM_BASE_PARAMS, RANKER_PROD_PARAMS

    assert RANKER_PROD_PARAMS == {**LGBM_BASE_PARAMS, "min_child_samples": 50}, (
        "RANKER_PROD_PARAMS 應 = LGBM_BASE_PARAMS + min_child_samples=50。"
        "若有意變更生產模型超參數，請同步更新本測試 + 重立回測基準。"
    )
    assert RANKER_PROD_PARAMS["n_estimators"] == 500
    assert RANKER_PROD_PARAMS["min_child_samples"] == 50


def test_backtest_and_train_ranker_share_prod_params():
    """backtest 與 train_ranker 的 LGBM regression 參數必須是同一個 dict（P1-5 核心不變量）。

    直接以小樣本各訓一個模型，斷言 get_params() 完全相等——防止未來任何一邊
    改回 hard-coded 參數（800 樹 / early stopping）造成部署 ≠ 回測。
    """
    import pytest

    pytest.importorskip("lightgbm")
    import numpy as np

    from skills import backtest as bt
    from skills import train_ranker as tr
    from skills.model_params import RANKER_PROD_PARAMS

    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 4))
    y = rng.normal(size=120)

    m_bt = bt._train_model(X, y, fast_mode=False)
    m_tr, engine = tr._build_model(X, y)

    assert engine == "lightgbm"
    assert m_bt.get_params() == m_tr.get_params(), (
        "backtest._train_model 與 train_ranker._build_model 的 LGBM 參數不一致"
    )
    for k, v in RANKER_PROD_PARAMS.items():
        assert m_tr.get_params()[k] == v, f"train_ranker 參數 {k} 偏離 RANKER_PROD_PARAMS"
    # 無 early stopping（否則部署模型樹數 ≠ 回測模型）
    assert getattr(m_tr, "best_iteration_", None) in (None, 0), (
        "train_ranker 模型不應有 early stopping（best_iteration_）"
    )


def test_no_early_stopping_and_shared_params_in_source():
    """來源層防線：兩邊都引用 RANKER_PROD_PARAMS；train_ranker 不得再出現 early_stopping。"""
    import skills.backtest as bt
    import skills.train_ranker as tr

    tr_src = inspect.getsource(tr._build_model)
    assert "RANKER_PROD_PARAMS" in tr_src
    # 檢查實際 API 呼叫（lgb.early_stopping / eval_set callbacks），docstring 提及不算
    assert "lgb.early_stopping" not in tr_src, "train_ranker._build_model 不應使用 early stopping"
    assert "eval_set" not in tr_src, "train_ranker._build_model 不應再傳 eval_set（無 early stopping）"
    assert "n_estimators=800" not in inspect.getsource(tr)

    bt_src = inspect.getsource(bt._train_model)
    assert "RANKER_PROD_PARAMS" in bt_src


def test_train_ranker_default_feature_set_is_pruned():
    """train_ranker 預設特徵集 = PRUNED_FEATURE_COLS（與生產回測 --pruned-features 一致）。"""
    from types import SimpleNamespace

    from skills.build_features import FEATURE_COLUMNS, PRUNED_FEATURE_COLS
    from skills.train_ranker import _resolve_feature_columns

    # 預設（config 無此欄 / 預設 True）→ PRUNED
    assert _resolve_feature_columns(SimpleNamespace()) == list(PRUNED_FEATURE_COLS)
    assert _resolve_feature_columns(
        SimpleNamespace(train_use_pruned_features=True)
    ) == list(PRUNED_FEATURE_COLS)
    # 顯式關閉 → 全 FEATURE_COLUMNS（診斷用）
    assert _resolve_feature_columns(
        SimpleNamespace(train_use_pruned_features=False)
    ) == list(FEATURE_COLUMNS)


def test_config_alignment_defaults():
    """config.py 對齊欄位的程式碼預設：pruned=true、liq_weighting=true、seasonal floor=5。"""
    import app.config as cfg_mod

    src = inspect.getsource(cfg_mod)
    assert 'pick("TRAIN_USE_PRUNED_FEATURES", "true")' in src
    assert 'pick("TRAIN_LIQ_WEIGHTING", "true")' in src
    assert 'pick("SEASONAL_TOPN_FLOOR", 5)' in src
    assert 'pick("TRANSACTION_COST_PCT", 0.001425)' in src, (
        "transaction_cost_pct（單邊）預設 0.001425 不可漂移"
    )


# ── 配置漂移防線：--production-baseline preset（2026-07-03）──────────────────


def test_production_baseline_preset_locked():
    """--production-baseline preset 內容鎖定（對應 CLAUDE.md 生產 CLI 指令）。

    若生產配置有意變更：同步更新 production_baseline_overrides() + 本測試 + CLAUDE.md。
    """
    from scripts.run_backtest import production_baseline_overrides

    assert production_baseline_overrides() == {
        "topn": 30,
        "enable_seasonal_filter": True,
        "no_stoploss": True,
        "market_filter_tiers": "-0.05:0.5,-0.10:0.25,-0.15:0.10",
        "market_filter_min_positions": 2,
        "liquidity_weighting": True,
        "pruned_features": True,
        "min_avg_turnover": 0.5,
    }


# ── P2-7a 交易成本顯式解析（2026-07-03）─────────────────────────────────────


def test_round_trip_cost_default_matches_legacy():
    """生產 CLI 不帶 --transaction-cost 時，有效來回成本必須維持 0.001425×4.1=0.0058425。"""
    from types import SimpleNamespace

    import pytest

    from scripts.run_backtest import resolve_round_trip_cost

    cfg = SimpleNamespace(transaction_cost_pct=0.001425, transaction_cost_round_trip=None)
    cost, source = resolve_round_trip_cost(None, cfg)
    assert cost == pytest.approx(0.0058425)
    assert source == "config.transaction_cost_pct*4.1"


def test_round_trip_cost_precedence():
    """優先序：CLI > config.transaction_cost_round_trip > 單邊 ×4.1（無 <0.005 猜單位）。"""
    from types import SimpleNamespace

    import pytest

    from scripts.run_backtest import resolve_round_trip_cost

    # CLI 最優先
    cfg = SimpleNamespace(transaction_cost_pct=0.001425, transaction_cost_round_trip=0.007)
    assert resolve_round_trip_cost(0.006, cfg) == (0.006, "cli")
    # 顯式 round_trip 次之
    assert resolve_round_trip_cost(None, cfg) == (0.007, "config.transaction_cost_round_trip")
    # 舊 silent fallback 已移除：單邊值 >= 0.005 也一樣走 ×4.1（不再被當成來回值直接用）
    cfg_big = SimpleNamespace(transaction_cost_pct=0.006, transaction_cost_round_trip=None)
    cost, source = resolve_round_trip_cost(None, cfg_big)
    assert cost == pytest.approx(0.006 * 4.1)
    assert source == "config.transaction_cost_pct*4.1"
