# scripts/archive — 已歸檔腳本

> 2026-07-03 健檢清理：以下腳本經機械化驗證**零引用**（Makefile / app / skills /
> pipelines / scripts / tests / experiments 均無引用，僅 docs/memory 有歷史提及），
> 且對應實驗已結案。`git mv` 保留完整歷史，需要時可直接執行或搬回 `scripts/`。

## 分類

| 類別 | 腳本 | 結案依據 |
|------|------|----------|
| 一次性診斷 | diag_*, compare_periods, _rebuild_features | 2026-05-26 reproducibility 調查用完即棄 |
| Stage 5-7 實驗（NEGATIVE） | backtest_*_quick, backtest_stage5_4_10y, eval_*_quick, evaluate_meta_filter_effect, train_meta_label, build_triple_barrier_labels, build_fracdiff_features, enrich_features_stage5_4, ic_analysis_stage5/combo/pruned | memory/decisions.md 各 Stage 條目 |
| News/情緒研究（未進生產） | backfill_news_5y, ic_analysis_news/sentiment, news_sentiment_llm/ollama, run_news_validation.sh, ic_decay_analysis | lookahead 修正後未重驗、未進模型 |
| 一次性分析 | analyze_market_regime, oracle_analysis, dd_attribution, grid_search_dims, multi_strategy_ensemble, cpcv_validation, beta_hedge_analysis, limit_down_bounce, backtest_breakout, run_strategy_backtest, update_promotion_tracker | 對應實驗結案（多為 NEGATIVE） |
| 已完成的一次性遷移/回補 | backfill_via_twse, backfill_prices_single_stock, compare_twse_vs_finmind, backfill_delisted_prices, backfill_adj_prices | 資料已落地 DB / parquet |

## ⚠️ 特別標注

- **backfill_adj_prices.py**：FinMind sponsor 已於 2026-06-24 過期，此腳本**不可重跑**
  （TaiwanStockPriceAdj 為 sponsor 專屬）。其產出
  `artifacts/adj_prices/adj_prices_10y.parquet`（2450 檔 / 5.09M 列）是
  adj_factor 的唯一來源，**僅存檔勿刪**。
- **backfill_delisted_prices.py**：survivorship 回補工具，若未來需要補新的下市股
  區間可搬回使用（依賴 FinMind 免費 API，仍可運作）。
- **compare_twse_vs_finmind.py**：資料源切換時的對照驗證工具，之後若再換源可搬回。

注意：diag_* 系列內含**過時的 LGBM 參數副本**，不可作為「生產參數是什麼」的參考——
生產參數唯一出處是 `skills/model_params.py`。
