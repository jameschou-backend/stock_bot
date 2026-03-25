"""特徵工程共用工具模組。

提供跨模組（backtest、train_ranker、daily_pick、data_store）共用的特徵解析
與缺值填補函式，避免重複實作造成邏輯不一致。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── JSON 解析（優先 orjson，fallback 標準 json）──────────────────────────────

try:
    import orjson as _orjson

    def _loads(v: object) -> dict:
        if isinstance(v, str):
            return _orjson.loads(v)
        if isinstance(v, dict):
            return v
        return {}

except ImportError:
    import json as _json

    def _loads(v: object) -> dict:  # type: ignore[misc]
        if isinstance(v, str):
            return _json.loads(v)
        if isinstance(v, dict):
            return v
        return {}


def parse_features_json(series: pd.Series) -> pd.DataFrame:
    """解析 features_json Series，回傳展開的特徵 DataFrame。

    優先使用 orjson（快 3-5x），fallback 至標準 json。
    支援 str、dict、None 三種輸入型別。

    Args:
        series: pd.Series，每個元素為 JSON 字串或 dict（features_json 欄位）。

    Returns:
        pd.DataFrame，欄位為所有特徵，index 與輸入一致。
    """
    parsed = [_loads(v) for v in series]
    return pd.json_normalize(parsed)


# ── 特徵語義填補映射 ────────────────────────────────────────────────────────
# 說明：
#   - 大多數特徵缺值填 0（動量=無動量、籌碼=無買賣）
#   - 部分特徵有特定中性預設值：
#     - boll_pct：布林帶位置 0~1，中性值 0.5（在帶中間）
#     - rsi_14：相對強弱指數，中性值 50
#     - bias_20：乖離率（%），中性值 0.0（即使填 0 語義正確）
#     - market_above_200ma：0/1 布林值，中性值 0（預設不認定多頭）
#
# SENTINEL_FILL_MAP：欄位名稱 → 當**全欄 NaN** 時使用的填補值（非中位數）
SENTINEL_FILL_MAP: Dict[str, float] = {
    "boll_pct": 0.5,
    "rsi_14": 50.0,
    "market_above_200ma": 0.0,
}


def impute_features(
    feature_df: pd.DataFrame,
    sentinel_map: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """特徵缺值填補（語義導向）。

    填補策略：
    1. 無窮大 → NaN
    2. 各欄中位數填補（skipna=True）
    3. 全欄 NaN 的欄位：優先查 sentinel_map，找不到預設填 0

    相較於原本「全欄 NaN → 0」的簡單策略，此函式針對 boll_pct、rsi_14
    等特徵使用語義正確的中性值，避免誤導模型。

    Args:
        feature_df: 特徵 DataFrame（純數值欄位）。
        sentinel_map: 自訂 {欄名: 全NaN時填補值}，會覆蓋 SENTINEL_FILL_MAP。

    Returns:
        (filled_df, stats_dict)
        stats_dict 含 filled_cells / total_cells / fill_ratio / all_nan_cols
    """
    effective_sentinel = {**SENTINEL_FILL_MAP, **(sentinel_map or {})}

    df = feature_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    nan_mask = df.isna()
    total_cells = int(nan_mask.size)
    filled_cells = int(nan_mask.sum().sum())
    all_nan_cols: List[str] = [col for col in df.columns if df[col].isna().all()]

    medians = df.median(skipna=True)

    for col in df.columns:
        if col in all_nan_cols:
            fill_val = float(effective_sentinel.get(col, 0.0))
            # 先轉 numeric，再填補，確保型別一致（避免 pandas FutureWarning object-dtype downcasting）
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_val)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(float(medians[col]))

    fill_ratio = filled_cells / total_cells if total_cells else 0.0
    stats: Dict[str, object] = {
        "filled_cells": filled_cells,
        "total_cells": total_cells,
        "fill_ratio": round(fill_ratio, 6),
        "all_nan_cols": all_nan_cols,
    }
    return df, stats


def filter_schema_valid_rows(
    feature_matrix: pd.DataFrame,
    coverage_threshold: float = 0.50,
) -> Tuple[pd.DataFrame, int]:
    """過濾 canonical feature 覆蓋率不足的舊版資料（schema 遷移保護）。

    說明：新增特徵後，DB 中舊日期資料的 features_json 不含新特徵欄位，
    覆蓋率低於門檻的行視為舊 schema 資料，訓練時應過濾以避免污染。

    Args:
        feature_matrix: 已展開的特徵 DataFrame（純數值欄位）。
        coverage_threshold: 覆蓋率門檻，預設 0.50（50%）。

    Returns:
        (filtered_df, n_dropped)
    """
    coverage = feature_matrix.notna().mean(axis=1)
    valid_mask = coverage >= coverage_threshold
    n_dropped = int((~valid_mask).sum())
    return feature_matrix.loc[valid_mask].copy(), n_dropped
