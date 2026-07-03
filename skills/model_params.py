"""共用模型超參數。

避免 strategy_c_pick / strategy_d_pick / backtest_rotation 三處複製貼上 LightGBM
超參數——改超參數時若漏改任一處，回測（rotation）與生產（C/D pick）會用不同模型，
正是本專案最忌諱的 backtest ≠ production 不一致。

min_child_samples 不放進 base：Ranker 用 20、Regressor 用 50（兩者不同）。
rotation 的 n_estimators 為可變（fast_mode 用較小值），引用時 override 即可。
"""
from __future__ import annotations

# C/D pick 與 rotation 共用的 LightGBM base 超參數（n_estimators 可被 override）
LGBM_BASE_PARAMS: dict = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# ── 生產排名模型共用超參數（2026-07-03 P1-5：回測=部署對齊）──
# skills/backtest.py `_train_model`（regression 路徑）與 skills/train_ranker.py `_build_model`
# 必須引用同一組參數，否則「部署模型 ≠ 回測驗證過的模型」。
# 以 backtest 現行（被 10y walk-forward 驗證過）的參數為準：
#   500 樹、無 early stopping、min_child_samples=50（Regressor 用 50，與 LGBM_BASE_PARAMS 註解一致）。
# train_ranker 舊行為（800 樹 + early_stopping(50)）已向回測收斂。
# 注意：本常數為「新增」，LGBM_BASE_PARAMS 既有值不可更動（C/D pick / rotation 仍引用）。
RANKER_PROD_PARAMS: dict = {**LGBM_BASE_PARAMS, "min_child_samples": 50}
