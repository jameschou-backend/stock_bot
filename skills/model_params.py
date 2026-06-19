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
