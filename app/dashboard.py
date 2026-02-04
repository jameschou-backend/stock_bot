from __future__ import annotations

from datetime import date
import json

import pandas as pd
import streamlit as st
from sqlalchemy import text

from app.config import load_config
from app.db import get_engine


st.set_page_config(page_title="台股 ML 選股 Dashboard", layout="wide")


def _parse_json(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:
        return {}


def _to_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def fetch_latest_pick_date(engine) -> date | None:
    query = text("SELECT MAX(pick_date) AS pick_date FROM picks")
    df = pd.read_sql(query, engine)
    return _to_date(df.loc[0, "pick_date"]) if not df.empty else None


def fetch_picks(engine, pick_date: date) -> pd.DataFrame:
    query = text(
        """
        SELECT pick_date, stock_id, score, model_id, reason_json
        FROM picks
        WHERE pick_date = :pick_date
        ORDER BY score DESC
        """
    )
    df = pd.read_sql(query, engine, params={"pick_date": pick_date})
    if not df.empty:
        df["reason_json"] = df["reason_json"].apply(_parse_json)
    return df


def fetch_model(engine, model_id: str) -> pd.DataFrame:
    query = text(
        """
        SELECT model_id, train_start, train_end, feature_set_hash, metrics_json, created_at
        FROM model_versions
        WHERE model_id = :model_id
        """
    )
    df = pd.read_sql(query, engine, params={"model_id": model_id})
    if not df.empty:
        df["metrics_json"] = df["metrics_json"].apply(_parse_json)
    return df


def fetch_stock_detail(engine, stock_id: str, trading_date: date) -> dict:
    price_q = text(
        """
        SELECT stock_id, trading_date, open, high, low, close, volume
        FROM raw_prices
        WHERE stock_id = :stock_id AND trading_date = :trading_date
        """
    )
    inst_q = text(
        """
        SELECT stock_id, trading_date,
               foreign_buy, foreign_sell, foreign_net,
               trust_buy, trust_sell, trust_net,
               dealer_buy, dealer_sell, dealer_net
        FROM raw_institutional
        WHERE stock_id = :stock_id AND trading_date = :trading_date
        """
    )
    feat_q = text(
        """
        SELECT stock_id, trading_date, features_json
        FROM features
        WHERE stock_id = :stock_id AND trading_date = :trading_date
        """
    )

    price_df = pd.read_sql(price_q, engine, params={"stock_id": stock_id, "trading_date": trading_date})
    inst_df = pd.read_sql(inst_q, engine, params={"stock_id": stock_id, "trading_date": trading_date})
    feat_df = pd.read_sql(feat_q, engine, params={"stock_id": stock_id, "trading_date": trading_date})

    detail = {
        "price": price_df.iloc[0].to_dict() if not price_df.empty else None,
        "institutional": inst_df.iloc[0].to_dict() if not inst_df.empty else None,
        "features": None,
    }
    if not feat_df.empty:
        row = feat_df.iloc[0].to_dict()
        row["features_json"] = _parse_json(row.get("features_json"))
        detail["features"] = row
    return detail


def fetch_price_history(engine, stock_id: str, trading_date: date, lookback_days: int = 120) -> pd.DataFrame:
    query = text(
        """
        SELECT trading_date, close, volume
        FROM raw_prices
        WHERE stock_id = :stock_id AND trading_date <= :trading_date
        ORDER BY trading_date DESC
        LIMIT :limit
        """
    )
    df = pd.read_sql(
        query,
        engine,
        params={"stock_id": stock_id, "trading_date": trading_date, "limit": lookback_days},
    )
    if df.empty:
        return df
    df = df.sort_values("trading_date")
    return df


def fetch_jobs(engine, limit: int = 30) -> pd.DataFrame:
    query = text(
        """
        SELECT job_id, job_name, status, started_at, ended_at, error_text
        FROM jobs
        ORDER BY started_at DESC
        LIMIT :limit
        """
    )
    return pd.read_sql(query, engine, params={"limit": limit})


st.title("台股 ML 選股 Dashboard")

config = load_config()
engine = get_engine()

latest_pick_date = fetch_latest_pick_date(engine)
if latest_pick_date is None:
    st.warning("尚未有 picks 資料，請先執行 pipeline。")
    st.stop()

pick_date = st.date_input("選擇日期", value=latest_pick_date)

picks_df = fetch_picks(engine, pick_date)
if picks_df.empty:
    st.warning("當日無 picks 資料。")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("TopN 選股清單")
    st.dataframe(
        picks_df[["stock_id", "score", "model_id"]].reset_index(drop=True),
        use_container_width=True,
    )

    st.subheader("分數 Bar Chart")
    chart_df = picks_df[["stock_id", "score"]].set_index("stock_id")
    st.bar_chart(chart_df)

with col2:
    st.subheader("模型資訊")
    model_id = picks_df["model_id"].iloc[0]
    model_df = fetch_model(engine, model_id)
    if model_df.empty:
        st.write("找不到模型版本資料")
    else:
        model_row = model_df.iloc[0]
        st.write({
            "model_id": model_row["model_id"],
            "train_start": model_row["train_start"],
            "train_end": model_row["train_end"],
            "feature_set_hash": model_row["feature_set_hash"],
        })
        st.write("metrics", model_row["metrics_json"])

st.divider()

st.subheader("個股詳情")
stock_id = st.selectbox("選擇股票", picks_df["stock_id"].tolist())

stock_detail = fetch_stock_detail(engine, stock_id, pick_date)
price_hist = fetch_price_history(engine, stock_id, pick_date)

if stock_detail["price"]:
    price_col, inst_col = st.columns(2)
    with price_col:
        st.write("價格")
        st.json(stock_detail["price"])
    with inst_col:
        st.write("法人")
        st.json(stock_detail["institutional"] or {})

    if not price_hist.empty:
        st.write("近 120 日收盤價")
        st.line_chart(price_hist.set_index("trading_date")["close"])

if stock_detail["features"]:
    st.write("特徵")
    st.json(stock_detail["features"]["features_json"])

st.divider()

st.subheader("Job Logs")
jobs_df = fetch_jobs(engine, limit=20)
if jobs_df.empty:
    st.write("尚無 job logs")
else:
    st.dataframe(jobs_df, use_container_width=True)
