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
