from __future__ import annotations

from datetime import date
import json
import uuid
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import psutil
import streamlit as st
from sqlalchemy import text

from app.config import load_config
from app.db import get_engine, get_session
from app.job_utils import cleanup_stale_running_jobs
from app.strategy_doc import get_selection_logic
from skills import regime as regime_module

_FEATURE_PERF_PATH = PROJECT_ROOT / "artifacts" / "feature_perf.csv"
_SLOW_QUERIES_PATH = PROJECT_ROOT / "artifacts" / "slow_queries.jsonl"

st.set_page_config(page_title="台股 ML 選股 Dashboard", page_icon="🏠", layout="wide")


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


def fetch_latest_job(engine) -> dict | None:
    query = text(
        """
        SELECT job_name, status, started_at, ended_at, error_text
        FROM jobs
        ORDER BY ended_at DESC
        LIMIT 1
        """
    )
    df = pd.read_sql(query, engine)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def fetch_running_jobs(engine, limit: int = 5) -> list[dict]:
    # 每次載入 dashboard 時自動收斂 stale running jobs，避免畫面長期假卡住。
    with get_session() as session:
        cleanup_stale_running_jobs(session, stale_minutes=120, commit=False)

    query = text(
        """
        SELECT job_id, job_name, status, started_at, logs_json
        FROM jobs
        WHERE status = 'running'
        ORDER BY started_at DESC
        LIMIT :limit
        """
    )
    df = pd.read_sql(query, engine, params={"limit": limit})
    if df.empty:
        return []
    df["logs_json"] = df["logs_json"].apply(_parse_json)
    return df.to_dict(orient="records")


def fetch_raw_price_freshness(engine) -> dict:
    max_q = text("SELECT MAX(trading_date) AS trading_date FROM raw_prices")
    max_df = pd.read_sql(max_q, engine)
    latest_date = _to_date(max_df.loc[0, "trading_date"]) if not max_df.empty else None
    if latest_date is None:
        return {"latest_date": None, "rows": 0}
    count_q = text("SELECT COUNT(*) AS rows FROM raw_prices WHERE trading_date = :date")
    count_df = pd.read_sql(count_q, engine, params={"date": latest_date})
    rows = int(count_df.loc[0, "rows"]) if not count_df.empty else 0
    return {"latest_date": latest_date, "rows": rows}


def fetch_data_freshness(engine, table_name: str) -> dict:
    """通用資料新鮮度查詢"""
    try:
        max_q = text(f"SELECT MAX(trading_date) AS trading_date FROM {table_name}")
        max_df = pd.read_sql(max_q, engine)
        latest_date = _to_date(max_df.loc[0, "trading_date"]) if not max_df.empty else None
        if latest_date is None:
            return {"latest_date": None, "rows": 0, "lag_days": None}
        
        count_q = text(f"SELECT COUNT(*) AS rows FROM {table_name} WHERE trading_date = :date")
        count_df = pd.read_sql(count_q, engine, params={"date": latest_date})
        rows = int(count_df.loc[0, "rows"]) if not count_df.empty else 0
        
        lag_days = (date.today() - latest_date).days
        return {"latest_date": latest_date, "rows": rows, "lag_days": lag_days}
    except Exception:
        return {"latest_date": None, "rows": 0, "lag_days": None, "error": "table not found"}


def fetch_recent_coverage(engine, table_name: str, days: int = 10) -> pd.DataFrame:
    """查詢最近 N 天的每日股票數覆蓋"""
    try:
        query = text(f"""
            SELECT trading_date, COUNT(DISTINCT stock_id) AS stock_count
            FROM {table_name}
            GROUP BY trading_date
            ORDER BY trading_date DESC
            LIMIT :days
        """)
        df = pd.read_sql(query, engine, params={"days": days})
        return df.sort_values("trading_date")
    except Exception:
        return pd.DataFrame()


def fetch_stock_universe_summary(engine) -> dict:
    """查詢股票 universe 統計"""
    try:
        query = text("""
            SELECT 
                COUNT(*) AS total,
                SUM(CASE WHEN security_type = 'stock' THEN 1 ELSE 0 END) AS stocks,
                SUM(CASE WHEN security_type = 'etf' THEN 1 ELSE 0 END) AS etfs,
                SUM(CASE WHEN is_listed = 1 THEN 1 ELSE 0 END) AS listed,
                SUM(CASE WHEN is_listed = 0 THEN 1 ELSE 0 END) AS delisted
            FROM stocks
        """)
        df = pd.read_sql(query, engine)
        if df.empty:
            return {"total": 0, "stocks": 0, "etfs": 0, "listed": 0, "delisted": 0}
        return df.iloc[0].to_dict()
    except Exception:
        return {"total": 0, "stocks": 0, "etfs": 0, "listed": 0, "delisted": 0, "error": "table not found"}


def fetch_data_quality_reports(engine, days: int = 30) -> pd.DataFrame:
    try:
        query = text(
            """
            SELECT
                report_date,
                table_name,
                missing_ratio,
                max_trading_date,
                notes
            FROM data_quality_reports
            WHERE report_date >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
              AND report_date <= CURDATE()
            ORDER BY report_date DESC, table_name ASC
            """
        )
        df = pd.read_sql(query, engine, params={"days": days})
        return df
    except Exception:
        # SQLite 測試環境不支援 DATE_SUB，改走可攜語法
        try:
            query = text(
                """
                SELECT
                    report_date,
                    table_name,
                    missing_ratio,
                    max_trading_date,
                    notes
                FROM data_quality_reports
                WHERE report_date >= date('now', '-' || :days || ' day')
                  AND report_date <= date('now')
                ORDER BY report_date DESC, table_name ASC
                """
            )
            return pd.read_sql(query, engine, params={"days": days})
        except Exception:
            return pd.DataFrame()


def _data_quality_light(missing_ratio: float | None) -> str:
    if missing_ratio is None or pd.isna(missing_ratio):
        return "GRAY"
    value = float(missing_ratio)
    if value < 0.05:
        return "GREEN"
    if value <= 0.20:
        return "YELLOW"
    return "RED"


def _format_model_metrics(metrics: dict) -> dict:
    if not isinstance(metrics, dict):
        return {}
    if metrics.get("v") == 1:
        return {
            "v": metrics.get("v"),
            "ic_spearman": metrics.get("ic_spearman"),
            "topk": metrics.get("topk", {}),
            "hitrate": metrics.get("hitrate", {}),
            "pred_stats": metrics.get("pred_stats", {}),
            "train_rows": metrics.get("train_rows"),
            "val_rows": metrics.get("val_rows"),
            "engine": metrics.get("engine"),
            "feature_count": metrics.get("feature_count"),
        }

    # 舊版 metrics_json 相容顯示
    return {
        "v": 0,
        "ic_spearman": metrics.get("ic_spearman"),
        "topn_mean_future_ret": metrics.get("topn_mean_future_ret"),
        "train_rows": metrics.get("train_rows"),
        "val_rows": metrics.get("val_rows"),
        "engine": metrics.get("engine"),
        "feature_count": metrics.get("feature_count"),
        "raw_metrics": metrics,
    }


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


def fetch_table_count_max(engine, table_name: str, date_col: str = "trading_date") -> dict:
    try:
        q = text(f"SELECT COUNT(*) AS cnt, MAX({date_col}) AS max_date FROM {table_name}")
        df = pd.read_sql(q, engine)
        if df.empty:
            return {"count": 0, "max_date": None}
        return {
            "count": int(df.loc[0, "cnt"] or 0),
            "max_date": _to_date(df.loc[0, "max_date"]) if df.loc[0, "max_date"] is not None else None,
        }
    except Exception:
        return {"count": 0, "max_date": None}


def fetch_latest_job_by_name(engine, job_name: str) -> dict | None:
    q = text(
        """
        SELECT job_id, job_name, status, started_at, ended_at, error_text
        FROM jobs
        WHERE job_name = :job_name
        ORDER BY started_at DESC
        LIMIT 1
        """
    )
    df = pd.read_sql(q, engine, params={"job_name": job_name})
    if df.empty:
        return None
    return df.iloc[0].to_dict()


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


def fetch_strategy_configs(engine) -> pd.DataFrame:
    try:
        return pd.read_sql("SELECT * FROM strategy_configs ORDER BY created_at DESC", engine)
    except Exception:
        return pd.DataFrame()


def fetch_strategy_runs(engine) -> pd.DataFrame:
    try:
        return pd.read_sql("SELECT * FROM strategy_runs ORDER BY created_at DESC", engine)
    except Exception:
        return pd.DataFrame()


def fetch_strategy_trades(engine, run_id: str) -> pd.DataFrame:
    try:
        return pd.read_sql(
            "SELECT * FROM strategy_trades WHERE run_id = :run_id ORDER BY trading_date ASC",
            engine,
            params={"run_id": run_id},
        )
    except Exception:
        return pd.DataFrame()


def fetch_strategy_positions(engine, run_id: str) -> pd.DataFrame:
    try:
        return pd.read_sql(
            "SELECT * FROM strategy_positions WHERE run_id = :run_id ORDER BY trading_date ASC",
            engine,
            params={"run_id": run_id},
        )
    except Exception:
        return pd.DataFrame()


def fetch_market_regime(engine, config) -> dict:
    try:
        ma_days = int(getattr(config, "market_filter_ma_days", 60))
        query = text(
            """
            SELECT trading_date, AVG(close) AS avg_close
            FROM raw_prices
            GROUP BY trading_date
            ORDER BY trading_date DESC
            LIMIT :n
            """
        )
        df = pd.read_sql(query, engine, params={"n": ma_days * 2}).sort_values("trading_date")
        detector = regime_module.get_regime_detector(config)
        detected = detector.detect(df, config)
        latest_date = _to_date(df.iloc[-1]["trading_date"]) if not df.empty else None
        detected["trading_date"] = latest_date
        return detected
    except Exception:
        return {"regime": "unknown", "score": 0.0, "meta": {}}


def fetch_hot_themes(engine, limit: int = 5) -> pd.DataFrame:
    try:
        latest_q = text("SELECT MAX(trading_date) AS trading_date FROM raw_theme_flow")
        latest_df = pd.read_sql(latest_q, engine)
        if latest_df.empty or latest_df.loc[0, "trading_date"] is None:
            return pd.DataFrame()
        latest_date = _to_date(latest_df.loc[0, "trading_date"])
        q = text(
            """
            SELECT theme_id, turnover_ratio, theme_return_20, hot_score
            FROM raw_theme_flow
            WHERE trading_date = :trading_date
            ORDER BY hot_score DESC
            LIMIT :limit
            """
        )
        return pd.read_sql(q, engine, params={"trading_date": latest_date, "limit": limit})
    except Exception:
        return pd.DataFrame()


def load_latest_backtest_summary() -> dict | None:
    backtest_dir = PROJECT_ROOT / "artifacts" / "backtest"
    if not backtest_dir.exists():
        return None
    files = sorted(backtest_dir.glob("backtest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        summary = payload.get("summary") or {}
        summary["source_file"] = str(files[0].name)
        return summary
    except Exception:
        return None


config = load_config()
engine = get_engine()
show_ml = getattr(config, "dashboard_show_ml", False)

st.title("台股 ML 選股 Dashboard" if show_ml else "台股 Dashboard")

tab_main, tab_perf = st.tabs(["🏠 主看板", "⚡ 效能監控"])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: 效能監控
# ══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.subheader("⚡ 效能監控")

    # ── 重新整理按鈕 ───────────────────────────────────────────────────────────
    if st.button("🔄 重新整理", key="perf_refresh"):
        st.rerun()

    # ── 1. 系統資源（即時） ────────────────────────────────────────────────────
    st.subheader("系統資源（即時）")
    _cpu = psutil.cpu_percent(interval=0.5)
    _mem = psutil.virtual_memory()
    _disk = psutil.disk_usage("/")
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("CPU 使用率", f"{_cpu:.1f}%")
    pc2.metric(
        "RAM 使用",
        f"{_mem.used / 1e9:.1f} GB",
        delta=f"/ {_mem.total / 1e9:.1f} GB",
    )
    pc3.metric("RAM 使用率", f"{_mem.percent:.1f}%")
    pc4.metric("磁碟剩餘", f"{_disk.free / 1e9:.1f} GB")

    st.progress(min(_cpu / 100.0, 1.0), text=f"CPU {_cpu:.1f}%")
    st.progress(min(_mem.percent / 100.0, 1.0), text=f"RAM {_mem.percent:.1f}%")

    st.divider()

    # ── 2. 特徵建置耗時（feature_perf.csv） ────────────────────────────────────
    st.subheader("特徵建置耗時（最近一次，Top 20）")
    if _FEATURE_PERF_PATH.exists():
        _perf_df = pd.read_csv(_FEATURE_PERF_PATH)
        _top20 = _perf_df.head(20).copy()
        _top20_chart = _top20.set_index("feature")["total_ms"]
        st.bar_chart(_top20_chart)
        st.dataframe(
            _top20[["feature", "total_ms", "n_stocks", "avg_ms_per_stock"]],
            use_container_width=True,
        )
    else:
        st.info("尚未產生 feature_perf.csv，請先執行 make pipeline 或 make pipeline-build")

    st.divider()

    # ── 3. Pipeline 耗時歷史 ───────────────────────────────────────────────────
    st.subheader("Pipeline 步驟耗時歷史（最近 200 筆）")
    try:
        _hist_q = text("""
            SELECT job_name, started_at,
                   TIMESTAMPDIFF(SECOND, started_at, ended_at) AS duration_s
            FROM jobs
            WHERE ended_at IS NOT NULL AND status = 'success'
            ORDER BY started_at DESC
            LIMIT 200
        """)
        _hist_df = pd.read_sql(_hist_q, engine)
        if not _hist_df.empty:
            _hist_df["started_at"] = pd.to_datetime(_hist_df["started_at"])
            _pivot = (
                _hist_df.pivot_table(
                    index="started_at",
                    columns="job_name",
                    values="duration_s",
                    aggfunc="first",
                )
                .sort_index()
            )
            st.line_chart(_pivot)
            # Summary table: avg duration per job_name
            _avg = (
                _hist_df.groupby("job_name")["duration_s"]
                .agg(["mean", "min", "max", "count"])
                .rename(columns={"mean": "avg_s", "min": "min_s", "max": "max_s", "count": "runs"})
                .sort_values("avg_s", ascending=False)
            )
            st.dataframe(_avg.reset_index(), use_container_width=True)
        else:
            st.info("無 pipeline 耗時歷史（需完整跑過 make pipeline）")
    except Exception as _e:
        st.warning(f"無法載入 pipeline 耗時歷史：{_e}")

    st.divider()

    # ── 4. Slow Queries ────────────────────────────────────────────────────────
    st.subheader("Slow Queries（最近 50 筆，閾值 1 s）")
    if _SLOW_QUERIES_PATH.exists():
        _sq_records: list[dict] = []
        try:
            with open(_SLOW_QUERIES_PATH, encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line:
                        _sq_records.append(json.loads(_line))
        except Exception:
            pass
        if _sq_records:
            _sq_df = pd.DataFrame(_sq_records[-50:])
            st.dataframe(_sq_df, use_container_width=True)
        else:
            st.info("slow_queries.jsonl 為空")
    else:
        st.info("尚無 slow query 記錄（需啟動 API 並實際執行查詢）")

    st.divider()

    # ── 5. API Metrics 端點說明 ────────────────────────────────────────────────
    st.subheader("API 效能端點")
    st.markdown("""
    | 端點 | 說明 |
    |------|------|
    | `GET /metrics/system` | 即時 CPU / RAM 快照 |
    | `GET /metrics/latest` | 系統資源 + 最新 job 狀態 |
    | `GET /metrics/history?limit=100` | 歷史 job 耗時紀錄 |
    | `GET /metrics/features` | `feature_perf.csv` 資料 |
    | `GET /metrics/slow-queries?limit=50` | Slow query 記錄 |
    | `WS  /ws/metrics` | WebSocket 即時串流（每 2 秒） |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: 主看板（原有所有內容）
# ══════════════════════════════════════════════════════════════════════════════
with tab_main:

    # ========== 選股邏輯 ==========
    if show_ml:
        with st.expander("📋 選股邏輯說明", expanded=False):
            st.markdown(get_selection_logic(config))

    # ========== 資料層狀態總覽 ==========
    st.subheader("資料層狀態")

    prices_freshness = fetch_data_freshness(engine, "raw_prices")
    inst_freshness = fetch_data_freshness(engine, "raw_institutional")
    margin_freshness = fetch_data_freshness(engine, "raw_margin_short")
    stock_universe = fetch_stock_universe_summary(engine)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="raw_prices",
            value=str(prices_freshness.get("latest_date") or "無資料"),
            delta=f"落後 {prices_freshness.get('lag_days', '?')} 天" if prices_freshness.get("lag_days") else None,
        )
        st.caption(f"最新筆數: {prices_freshness.get('rows', 0):,}")

    with col2:
        st.metric(
            label="raw_institutional",
            value=str(inst_freshness.get("latest_date") or "無資料"),
            delta=f"落後 {inst_freshness.get('lag_days', '?')} 天" if inst_freshness.get("lag_days") else None,
        )
        st.caption(f"最新筆數: {inst_freshness.get('rows', 0):,}")

    with col3:
        st.metric(
            label="raw_margin_short",
            value=str(margin_freshness.get("latest_date") or "無資料"),
            delta=f"落後 {margin_freshness.get('lag_days', '?')} 天" if margin_freshness.get("lag_days") else None,
        )
        st.caption(f"最新筆數: {margin_freshness.get('rows', 0):,}")

    with col4:
        st.metric(
            label="股票 Universe",
            value=f"{stock_universe.get('listed', 0):,} 上市",
        )
        st.caption(f"股票: {stock_universe.get('stocks', 0)}, ETF: {stock_universe.get('etfs', 0)}")

    # 近 10 日覆蓋率趨勢
    st.subheader("近 10 交易日覆蓋率")
    prices_coverage = fetch_recent_coverage(engine, "raw_prices", 10)
    inst_coverage = fetch_recent_coverage(engine, "raw_institutional", 10)
    margin_coverage = fetch_recent_coverage(engine, "raw_margin_short", 10)

    if not prices_coverage.empty or not inst_coverage.empty:
        coverage_dfs = []
        if not prices_coverage.empty:
            prices_coverage = prices_coverage.rename(columns={"stock_count": "prices"})
            coverage_dfs.append(prices_coverage.set_index("trading_date")["prices"])
        if not inst_coverage.empty:
            inst_coverage = inst_coverage.rename(columns={"stock_count": "institutional"})
            coverage_dfs.append(inst_coverage.set_index("trading_date")["institutional"])
        if not margin_coverage.empty:
            margin_coverage = margin_coverage.rename(columns={"stock_count": "margin"})
            coverage_dfs.append(margin_coverage.set_index("trading_date")["margin"])
        if coverage_dfs:
            combined = pd.concat(coverage_dfs, axis=1)
            st.line_chart(combined)
    else:
        st.info("無覆蓋率資料")

    st.subheader("Data Quality")
    st.caption("最近 30 天資料品質報表（缺漏率紅黃綠燈）")
    quality_df = fetch_data_quality_reports(engine, days=30)
    if quality_df.empty:
        st.info("尚無 data_quality_reports 資料")
    else:
        quality_df = quality_df.copy()
        quality_df["missing_ratio"] = pd.to_numeric(quality_df["missing_ratio"], errors="coerce")
        quality_df["light"] = quality_df["missing_ratio"].apply(_data_quality_light)
        st.dataframe(
            quality_df[["report_date", "table_name", "light", "missing_ratio", "max_trading_date", "notes"]],
            use_container_width=True,
        )

    st.divider()

    # ========== Pipeline 狀態 ==========
    st.subheader("最近 Job 狀態")
    latest_job = fetch_latest_job(engine)
    running_jobs = fetch_running_jobs(engine)

    if running_jobs:
        st.markdown("**正在執行中**")
        for job in running_jobs:
            logs = job.get("logs_json") or {}
            progress = logs.get("progress") if isinstance(logs, dict) else None
            st.write(f"{job.get('job_name')} (started: {job.get('started_at')})")
            if isinstance(progress, dict) and progress.get("total_chunks"):
                total = int(progress.get("total_chunks") or 0)
                current = int(progress.get("current_chunk") or 0)
                ratio = current / total if total else 0.0
                st.progress(min(max(ratio, 0.0), 1.0))
                current_stock = progress.get("current_stock")
                total_stocks = progress.get("total_stocks")
                stock_id = progress.get("stock_id")
                if current_stock is not None and total_stocks is not None:
                    stock_part = f"stock {current_stock}/{total_stocks}"
                    if stock_id:
                        stock_part += f" ({stock_id})"
                    st.caption(
                        f"{stock_part} | {progress.get('chunk_start')} ~ {progress.get('chunk_end')} | rows={progress.get('rows')}"
                    )
                else:
                    st.caption(
                        f"chunk {current}/{total} | {progress.get('chunk_start')} ~ {progress.get('chunk_end')} | rows={progress.get('rows')}"
                    )
            else:
                st.caption("無進度資訊")
    else:
        st.caption("目前沒有執行中的 job")

    if latest_job:
        job_status = latest_job.get("status", "unknown")
        status_color = "green" if job_status == "success" else "red" if job_status == "failed" else "orange"
        st.markdown(f"**{latest_job.get('job_name')}**: :{status_color}[{job_status}]")
        st.write(
            {
                "ended_at": latest_job.get("ended_at"),
                "error_text": (latest_job.get("error_text") or "")[:200] or None,
            }
        )
    else:
        st.write("尚無 job records")

    st.divider()

    # ========== 策略版本與風險監控 ==========
    if show_ml:
        st.subheader("策略版本與風險監控")

        regime = fetch_market_regime(engine, config)
        latest_backtest = load_latest_backtest_summary()
        hot_themes = fetch_hot_themes(engine, limit=5)

        v1, v2, v3, v4 = st.columns(4)
        with v1:
            st.metric("策略版本", "vNext-research")
            st.caption(f"TopN={config.topn}, 停損={config.stoploss_pct:.1%}")
        with v2:
            regime_label = regime.get("regime", "unknown")
            st.metric("市場 Regime", regime_label)
            meta = regime.get("meta") or {}
            current_price = meta.get("current_price")
            ma_value = meta.get("ma_value")
            diff_pct = meta.get("diff_pct")
            ma_days = meta.get("ma_days", config.market_filter_ma_days)
            if current_price is not None and ma_value is not None:
                caption = f"mkt={float(current_price):.2f}, MA{ma_days}={float(ma_value):.2f}"
                if diff_pct is not None:
                    caption += f", diff={float(diff_pct):.2%}"
                st.caption(caption)
        with v3:
            if latest_backtest:
                mdd = float(latest_backtest.get("max_drawdown", 0.0))
                st.metric("最新回測 MDD", f"{mdd:.2%}")
            else:
                st.metric("最新回測 MDD", "N/A")
        with v4:
            if latest_backtest:
                sharpe = float(latest_backtest.get("sharpe_ratio", 0.0))
                st.metric("最新回測 Sharpe", f"{sharpe:.2f}")
            else:
                st.metric("最新回測 Sharpe", "N/A")

        if latest_backtest:
            dd = float(latest_backtest.get("max_drawdown", 0.0))
            sharpe = float(latest_backtest.get("sharpe_ratio", 0.0))
            stoploss_count = int(latest_backtest.get("stoploss_triggered", 0))
            total_trades = max(int(latest_backtest.get("total_trades", 1)), 1)
            stoploss_rate = stoploss_count / total_trades
            lights = {
                "drawdown_light": "RED" if dd <= -0.35 else ("YELLOW" if dd <= -0.25 else "GREEN"),
                "sharpe_light": "RED" if sharpe < 0.2 else ("YELLOW" if sharpe < 0.6 else "GREEN"),
                "stoploss_light": "RED" if stoploss_rate > 0.45 else ("YELLOW" if stoploss_rate > 0.30 else "GREEN"),
            }
            st.write(
                {
                    "backtest_file": latest_backtest.get("source_file"),
                    "annualized_return": latest_backtest.get("annualized_return"),
                    "excess_return": latest_backtest.get("excess_return"),
                    "risk_lights": lights,
                }
            )

        if not hot_themes.empty:
            st.caption("目前資金較熱題材（產業代理）")
            st.dataframe(hot_themes, use_container_width=True)

        st.divider()

    # ========== Picks 區塊（永遠顯示，不受 show_ml 影響） ==========
    st.subheader("選股結果（Picks）")
    latest_pick_date = fetch_latest_pick_date(engine)
    today = date.today()
    if latest_pick_date is None:
        picks_stat = fetch_table_count_max(engine, "picks", "pick_date")
        feat_stat = fetch_table_count_max(engine, "features", "trading_date")
        latest_daily_pick = fetch_latest_job_by_name(engine, "daily_pick")
        latest_build_features = fetch_latest_job_by_name(engine, "build_features")
        st.error("picks table is empty")
        st.write(
            {
                "possible_reasons": [
                    "features not built yet",
                    "daily_pick job has not run successfully",
                    "pipeline has not completed",
                ],
                "picks_count": picks_stat.get("count", 0),
                "features_count": feat_stat.get("count", 0),
                "features_max_trading_date": feat_stat.get("max_date"),
                "latest_daily_pick_job": latest_daily_pick,
                "latest_build_features_job": latest_build_features,
            }
        )
    else:
        pick_date = st.date_input("選擇日期", value=latest_pick_date, key="dashboard_pick_date")
        display_date = pick_date
        picks_df = fetch_picks(engine, pick_date)
        if picks_df.empty:
            fallback_df = fetch_picks(engine, latest_pick_date)
            if not fallback_df.empty:
                picks_df = fallback_df
                display_date = latest_pick_date
                st.warning(
                    f"所選日期 {pick_date} 無 picks，已回退至最新可用選股日期：{latest_pick_date}"
                )
            else:
                st.error("picks table has rows but selected/latest date has no valid picks")
                st.stop()

        if display_date != today:
            st.info(f"目前顯示最新可用選股日期：{display_date}（非今日）")
        else:
            st.caption(f"目前顯示日期：{display_date}")

        if not quality_df.empty:
            red_rows = quality_df[quality_df["light"] == "RED"]
            if not red_rows.empty:
                red_tables = sorted(red_rows["table_name"].astype(str).unique().tolist())
                st.warning(f"Data Quality 存在 RED 項目（{', '.join(red_tables)}），但 picks 仍可供檢視。")

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
                st.write("metrics")
                st.json(_format_model_metrics(model_row["metrics_json"]))

        st.divider()

        st.subheader("個股詳情")
        stock_id = st.selectbox("選擇股票", picks_df["stock_id"].tolist())

        stock_detail = fetch_stock_detail(engine, stock_id, display_date)
        price_hist = fetch_price_history(engine, stock_id, display_date)

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

    st.subheader("Walkforward 回測交易明細")
    trades_csv_path = PROJECT_ROOT / "artifacts" / "ai_answers" / "walkforward_trades.csv"
    if trades_csv_path.exists():
        try:
            wf_trades_df = pd.read_csv(trades_csv_path)
            wf_trades_df["entry_date"] = pd.to_datetime(wf_trades_df["entry_date"]).dt.date
            wf_trades_df["exit_date"] = pd.to_datetime(wf_trades_df["exit_date"]).dt.date
            wf_trades_df["realized_pnl_pct"] = wf_trades_df["realized_pnl_pct"].map("{:.2%}".format)
            wf_trades_df["score (Logic)"] = wf_trades_df["score"].map("{:.4f}".format)
            st.dataframe(
                wf_trades_df[["stock_id", "entry_date", "exit_date", "entry_price", "exit_price",
                               "realized_pnl_pct", "stoploss_triggered", "exit_reason", "score (Logic)"]],
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"無法讀取回測交易明細: {e}")
    else:
        st.info("尚未產生 Walkforward 回測交易明細。請執行 python scripts/run_walkforward.py 產生。")

    st.divider()

    st.subheader("Job Logs")
    jobs_df = fetch_jobs(engine, limit=20)
    if jobs_df.empty:
        st.write("尚無 job logs")
    else:
        st.dataframe(jobs_df, use_container_width=True)

    st.divider()
