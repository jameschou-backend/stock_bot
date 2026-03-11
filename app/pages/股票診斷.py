"""股票診斷 Dashboard – 分析指定股票在指定日期的選股排名與過濾原因。"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from app.models import Feature, ModelVersion, RawPrice, Stock
from skills import risk, tradability_filter
from skills.build_features import FEATURE_COLUMNS

st.set_page_config(page_title="股票診斷", page_icon="🔍", layout="wide")
st.title("🔍 股票診斷")
st.caption("分析指定股票在指定日期的選股排名、過濾原因與關鍵特徵值")

KEY_FEATURES = [
    "foreign_buy_streak",
    "volume_surge_ratio",
    "foreign_buy_intensity",
    "ret_20",
    "amt_ratio_20",
    "rsi_14",
    "foreign_buy_ratio_5",
    "foreign_buy_consecutive_days",
]

# ── 輸入區 ──────────────────────────────────────────
col_input1, col_input2, col_input3 = st.columns([1, 1.5, 1])
with col_input1:
    stock_id = st.text_input("股票代號", value="2408", max_chars=6).strip()
with col_input2:
    query_date = st.date_input(
        "診斷日期",
        value=date.today() - timedelta(days=1),
        min_value=date(2016, 1, 1),
        max_value=date.today(),
    )
with col_input3:
    st.write("")
    st.write("")
    show_all = st.checkbox("顯示完整排名表")

run_btn = st.button("🔍 開始診斷", type="primary", use_container_width=False)

if not run_btn:
    st.info("輸入股票代號與日期後，點擊「開始診斷」。")
    st.stop()


# ── 核心函式 ──────────────────────────────────────────

@st.cache_resource
def _load_model():
    import joblib
    with get_session() as session:
        mv = (
            session.query(ModelVersion)
            .order_by(ModelVersion.created_at.desc())
            .limit(1)
            .one_or_none()
        )
        if mv is None:
            return None, None, None
        artifact = joblib.load(mv.artifact_path)
        return artifact["model"], artifact["feature_names"], mv.model_id


def _find_feature_date(session, target: date) -> date | None:
    row = (
        session.query(Feature.trading_date)
        .filter(Feature.trading_date <= target)
        .order_by(Feature.trading_date.desc())
        .limit(1)
        .one_or_none()
    )
    return row[0] if row else None


def _load_features_on_date(session, feature_date: date) -> pd.DataFrame:
    from sqlalchemy import select
    stmt = (
        select(Feature.stock_id, Feature.features_json)
        .where(Feature.trading_date == feature_date)
    )
    df = pd.read_sql(stmt, session.get_bind())
    return df[df["stock_id"].str.fullmatch(r"\d{4}")]


def _parse_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    parsed = [
        json.loads(v) if isinstance(v, str) else v
        for v in df["features_json"]
    ]
    feat_df = pd.json_normalize(parsed)
    for col in feature_names:
        if col not in feat_df.columns:
            feat_df[col] = np.nan
    return feat_df[feature_names].copy()


def _load_price_window(session, feature_date: date, stock_ids: list) -> pd.DataFrame:
    from sqlalchemy import select
    start = feature_date - timedelta(days=40)
    stmt = (
        select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close, RawPrice.volume)
        .where(RawPrice.trading_date.between(start, feature_date))
        .where(RawPrice.stock_id.in_(stock_ids))
        .order_by(RawPrice.stock_id, RawPrice.trading_date)
    )
    df = pd.read_sql(stmt, session.get_bind())
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    return df.dropna(subset=["close", "volume"])


def _impute(feat_df: pd.DataFrame) -> pd.DataFrame:
    feat_df = feat_df.copy().replace([np.inf, -np.inf], np.nan)
    medians = feat_df.median(skipna=True)
    for col in feat_df.columns:
        if feat_df[col].isna().all():
            feat_df[col] = feat_df[col].fillna(0)
        else:
            feat_df[col] = feat_df[col].fillna(medians[col])
    return feat_df


def _effective_topn(config, feature_date: date) -> int:
    topn = config.topn
    weak_months = tuple(getattr(config, "seasonal_weak_months", (3, 10)))
    mult = float(getattr(config, "seasonal_topn_multiplier", 0.5))
    if feature_date.month in weak_months:
        topn = max(5, int(topn * mult))
    return topn


def _get_stock_name(session, sid: str) -> str:
    row = session.query(Stock.name).filter(Stock.stock_id == sid).one_or_none()
    return row[0] if row else ""


# ── 執行診斷 ──────────────────────────────────────────

config = load_config()
model, feature_names, model_id = _load_model()

if model is None:
    st.error("找不到已訓練的模型，請先執行 `make pipeline`。")
    st.stop()

with st.spinner("載入資料中…"):
    with get_session() as session:

        # 1. 找特徵日期
        feature_date = _find_feature_date(session, query_date)
        if feature_date is None:
            st.error(f"找不到 ≤ {query_date} 的特徵資料。")
            st.stop()

        if feature_date != query_date:
            st.warning(f"⚠️ 指定日期 {query_date} 無特徵，改用最近特徵日：**{feature_date}**")

        # 2. 載入特徵
        raw_df = _load_features_on_date(session, feature_date)
        if raw_df.empty:
            st.error(f"{feature_date} 無特徵資料。")
            st.stop()

        # 3. Universe
        universe_df = risk.get_universe(session, feature_date, config)
        tradability_logs: dict = {}
        if getattr(config, "enable_tradability_filter", True):
            universe_df, tradability_logs = tradability_filter.filter_universe(
                session, universe_df, feature_date, return_stats=True
            )
        valid_stocks = set(universe_df["stock_id"].astype(str).tolist())
        if valid_stocks:
            raw_df = raw_df[raw_df["stock_id"].isin(valid_stocks)]

        # 4. 流動性
        price_df = _load_price_window(session, feature_date, raw_df["stock_id"].tolist())
        avg_turnover_df = risk.apply_liquidity_filter(price_df, config)
        liquid_stocks = set(avg_turnover_df["stock_id"].tolist())

        # 目標股資訊
        target_turnover_row = avg_turnover_df[avg_turnover_df["stock_id"] == stock_id]
        target_avg_turnover = (
            float(target_turnover_row["avg_turnover"].iloc[0])
            if not target_turnover_row.empty else None
        )
        liquidity_threshold = (
            float(getattr(config, "min_amt_20", 0.0) or 0.0)
            or float(getattr(config, "min_avg_turnover", 0.0)) * 1e8
        )

        # 目標股股價
        target_price_rows = price_df[price_df["stock_id"] == stock_id].sort_values("trading_date")
        target_close = float(target_price_rows["close"].iloc[-1]) if not target_price_rows.empty else None

        # 5. 打分
        scored_df = raw_df[raw_df["stock_id"].isin(liquid_stocks)].copy().reset_index(drop=True)
        feat_df = _parse_features(scored_df, feature_names)
        feat_df_imputed = _impute(feat_df)
        scores = model.predict(feat_df_imputed.values)
        scored_df["score"] = scores
        scored_df = scored_df.sort_values("score", ascending=False).reset_index(drop=True)
        scored_df["rank"] = scored_df.index + 1

        # 有效 topN
        eff_topn = _effective_topn(config, feature_date)
        threshold_score = (
            float(scored_df.iloc[eff_topn - 1]["score"])
            if len(scored_df) >= eff_topn else float(scored_df["score"].min())
        )

        # 目標股
        stock_name = _get_stock_name(session, stock_id)
        target_row = scored_df[scored_df["stock_id"] == stock_id]
        in_universe = stock_id in valid_stocks if valid_stocks else True
        in_raw = stock_id in set(raw_df["stock_id"].tolist())
        in_liquid = stock_id in liquid_stocks

        # 目標股原始特徵
        target_raw_features: dict = {}
        if in_raw:
            raw_feat_df = _parse_features(
                raw_df[raw_df["stock_id"] == stock_id].reset_index(drop=True),
                FEATURE_COLUMNS,
            )
            if not raw_feat_df.empty:
                target_raw_features = raw_feat_df.iloc[0].to_dict()

        n_valid = int(sum(
            1 for col in FEATURE_COLUMNS
            if col in target_raw_features and not pd.isna(target_raw_features.get(col))
        ))
        n_total = len(FEATURE_COLUMNS)

        # 取前 N 名股票名稱（批次查詢）
        top_n_display = len(scored_df) if show_all else min(20, len(scored_df))
        top_sids = scored_df.head(top_n_display)["stock_id"].tolist()
        from sqlalchemy import select as sa_select
        from app.models import Stock as StockModel
        name_rows = session.query(StockModel.stock_id, StockModel.name).filter(
            StockModel.stock_id.in_(top_sids + [stock_id])
        ).all()
        name_map = {r.stock_id: r.name or "" for r in name_rows}


# ── 結果顯示 ─────────────────────────────────────────
st.divider()

title_str = f"{stock_id} {stock_name}" if stock_name else stock_id
st.subheader(f"📋 {title_str}　診斷日期：{feature_date}")

# 結果 badge
if not in_raw:
    st.error("❌ 無特徵資料（該日無記錄）")
elif not in_universe and valid_stocks:
    st.error("❌ 不在可交易 Universe（ETF / 非普通股 / 已下市）")
elif not in_liquid:
    st.error("❌ 流動性不足（未通過過濾）")
elif target_row.empty:
    st.warning("⚠️ 特徵存在但打分後無記錄（資料異常）")
else:
    rank = int(target_row.iloc[0]["rank"])
    score_val = float(target_row.iloc[0]["score"])
    if rank <= eff_topn:
        st.success(f"✅ 入選第 **{rank}** 名　（有效 TopN = {eff_topn}）")
    else:
        st.warning(f"⚠️ 排名第 **{rank}** 名　（未進前 {eff_topn}，有效 TopN = {eff_topn}）")
    month = feature_date.month
    weak_months = tuple(getattr(config, "seasonal_weak_months", (3, 10)))
    if month in weak_months:
        st.caption(f"ℹ️ {month} 月季節性降倉：topN {config.topn} → {eff_topn}")

st.write("")

# ── Row 1: 模型評分 ──
st.markdown("#### 📊 模型評分")
if not target_row.empty:
    rank = int(target_row.iloc[0]["rank"])
    score_val = float(target_row.iloc[0]["score"])
    diff = score_val - threshold_score
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("本股得分", f"{score_val:.4f}")
    c2.metric("排名", f"{rank} / {len(scored_df)}")
    c3.metric(f"第 {eff_topn} 名門檻", f"{threshold_score:.4f}")
    c4.metric("差距門檻", f"{diff:+.4f}", delta=f"{diff:+.4f}",
              delta_color="normal")
else:
    st.info("無法取得本股分數（未通過前置過濾）")

st.write("")

# ── Row 2: 過濾關卡 + 關鍵特徵 ──
col_filter, col_feat = st.columns([1, 1.4])

with col_filter:
    st.markdown("#### 🚦 過濾關卡")

    def _row(ok: bool, label: str, detail: str = ""):
        icon = "✅" if ok else "❌"
        if detail:
            st.markdown(f"{icon} **{label}**　{detail}")
        else:
            st.markdown(f"{icon} **{label}**")

    if valid_stocks:
        _row(in_universe, "Universe", "上市普通股" if in_universe else "ETF/非普通股/已下市")

    if target_avg_turnover is not None:
        liq_detail = (
            f"近20日均成交額 {target_avg_turnover/1e4:.0f} 萬元"
            f"（門檻 {liquidity_threshold/1e4:.0f} 萬元）"
        )
        _row(in_liquid, "流動性", liq_detail)
    else:
        _row(False, "流動性", f"無成交資料（門檻 {liquidity_threshold/1e4:.0f} 萬元）")

    if target_close is not None:
        _row(target_close >= 5, "股價", f"{target_close:.2f} 元（參考門檻 > 5 元）")
    else:
        st.markdown("⚠️ **股價** 無資料")

    cov_ok = n_valid >= int(n_total * 0.8)
    _row(cov_ok, "特徵覆蓋率", f"{n_valid} / {n_total} 個特徵完整")

with col_feat:
    st.markdown("#### 🔑 關鍵特徵值")
    feat_rows = []
    for feat in KEY_FEATURES:
        val = target_raw_features.get(feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            display = "N/A"
        elif isinstance(val, float):
            display = f"{val:.4f}"
        else:
            display = str(val)
        feat_rows.append({"特徵名稱": feat, "數值": display})
    st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

st.write("")

# ── Row 3: 排名表 ──
st.markdown(f"#### 🏆 {'完整排名' if show_all else '前 20 名'}（共 {len(scored_df)} 支候選）")

rank_data = []
for _, row in scored_df.head(top_n_display).iterrows():
    sid = str(row["stock_id"])
    nm = name_map.get(sid, "")
    is_target = sid == stock_id
    rank_data.append({
        "排名": int(row["rank"]),
        "代號": sid,
        "名稱": nm,
        "得分": round(float(row["score"]), 4),
        "入選": "✅" if int(row["rank"]) <= eff_topn else "",
        "目標股": "← 目標" if is_target else "",
    })

# 若目標股在前 20 名以外，且非 show_all，補在最後顯示
if not show_all and not target_row.empty:
    target_rank = int(target_row.iloc[0]["rank"])
    if target_rank > top_n_display:
        rank_data.append({
            "排名": target_rank,
            "代號": stock_id,
            "名稱": name_map.get(stock_id, ""),
            "得分": round(float(target_row.iloc[0]["score"]), 4),
            "入選": "",
            "目標股": "← 目標",
        })

rank_df = pd.DataFrame(rank_data)

def _highlight_target(row):
    if row["目標股"]:
        return ["background-color: #fff3cd"] * len(row)
    if row["入選"] == "✅":
        return ["background-color: #d4edda"] * len(row)
    return [""] * len(row)

styled = rank_df.style.apply(_highlight_target, axis=1)
st.dataframe(styled, use_container_width=True, hide_index=True)
