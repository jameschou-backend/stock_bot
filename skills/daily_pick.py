"""每日選股模組：讀取最新特徵，套用流動性/可交易性過濾，使用 ML 模型或 multi-agent 選出 TopN 標的，
寫入 picks 表並輸出 run_manifest JSON。

支援 selection_mode: model（LightGBM）或 multi_agent，以及 research 降級模式。
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import joblib
except ModuleNotFoundError as exc:
    raise RuntimeError("Missing dependency 'joblib'. Install with `pip install -r requirements.txt`.") from exc
import numpy as np
import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import Feature, Job, ModelVersion, Pick, RawPrice, Stock
from skills.build_features import FEATURE_COLUMNS
from skills import breakthrough as _bt
from skills import regime, risk
from skills import tradability_filter
from skills import multi_agent_selector
from skills import position_sizing as _pos_sizing
from skills.feature_utils import (
    parse_features_json as _parse_features_json_shared,
    impute_features as _impute_features_shared,
)


# 選取前 8 個特徵作為 reason 說明
REASON_FEATURES = FEATURE_COLUMNS[:8]
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def _load_market_price_df(db_session: Session, target_date: date, ma_days: int) -> pd.DataFrame:
    start = target_date - timedelta(days=ma_days * 2)
    stmt = (
        select(RawPrice.trading_date, func.avg(RawPrice.close).label("avg_close"))
        .where(RawPrice.trading_date.between(start, target_date))
        .group_by(RawPrice.trading_date)
        .order_by(RawPrice.trading_date)
    )
    return pd.read_sql(stmt, db_session.get_bind())


def _load_latest_model(session: Session) -> ModelVersion | None:
    return (
        session.query(ModelVersion)
        .order_by(ModelVersion.created_at.desc())
        .limit(1)
        .one_or_none()
    )


def _parse_features(series: pd.Series) -> pd.DataFrame:
    """委派給 feature_utils.parse_features_json（統一實作，含 orjson 加速）。"""
    return _parse_features_json_shared(series)


def _impute_features(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """委派給 feature_utils.impute_features（語義導向填補，修正 bias_20/boll_pct 等特徵）。"""
    return _impute_features_shared(feature_df)


def _load_price_universe(
    db_session: Session,
    target_date,
    stock_ids: List[str],
    lookback_days: int = 60,
) -> pd.DataFrame:
    if not stock_ids:
        return pd.DataFrame()

    start_date = target_date - timedelta(days=lookback_days)
    stmt = (
        select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close, RawPrice.volume)
        .where(RawPrice.trading_date.between(start_date, target_date))
        .where(RawPrice.stock_id.in_(stock_ids))
        .order_by(RawPrice.stock_id, RawPrice.trading_date)
    )
    df = pd.read_sql(stmt, db_session.get_bind())
    if df.empty:
        return df

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["close", "volume"])
    return df


def _choose_pick_date(
    candidate_dates: List[date],
    feature_df: pd.DataFrame,
    price_df: pd.DataFrame,
    topn: int,
    config,
    fallback_days: int,
) -> Tuple[date | None, pd.DataFrame, Dict[str, object]]:
    """選擇最佳選股日期。

    20 日平均成交值門檻由 risk.apply_liquidity_filter 控制。
    """
    best_date = None
    best_df = pd.DataFrame()
    best_valid = 0
    fallback_used = 0
    best_meta: Dict[str, object] = {}

    for idx, target_date in enumerate(candidate_dates[: fallback_days + 1]):
        target = target_date
        date_features = feature_df[feature_df["trading_date"] == target].copy()
        if date_features.empty:
            continue

        date_features = date_features[date_features["stock_id"].str.fullmatch(r"\d{4}")]
        if date_features.empty:
            continue
        date_features = date_features.drop_duplicates(subset=["stock_id", "trading_date"])
        pre_liquidity_count = len(date_features)

        date_prices = price_df[price_df["trading_date"] == target].copy()
        if date_prices.empty:
            continue

        latest_price = date_prices.dropna(subset=["close", "volume"])[["stock_id", "close", "volume"]]
        if latest_price.empty:
            continue

        eligible_turnover = risk.apply_liquidity_filter(
            price_df[price_df["trading_date"] <= target],
            config,
        )
        eligible = latest_price.merge(eligible_turnover[["stock_id"]], on="stock_id", how="inner")
        eligible = eligible.drop_duplicates(subset=["stock_id"])
        if eligible.empty:
            continue

        date_features = date_features.merge(eligible[["stock_id"]], on="stock_id", how="inner")
        valid_count = len(date_features)
        meta = {
            "candidate_count_before_liquidity": pre_liquidity_count,
            "candidate_count_after_liquidity": valid_count,
            "liquidity_excluded_count": max(pre_liquidity_count - valid_count, 0),
            "liquidity_excluded_ratio": (
                max(pre_liquidity_count - valid_count, 0) / pre_liquidity_count
                if pre_liquidity_count
                else 0.0
            ),
        }
        if valid_count > 0 and best_date is None:
            best_date = target_date
            best_df = date_features
            best_valid = valid_count
            fallback_used = idx
            best_meta = meta

        if valid_count >= topn:
            return target_date, date_features, {
                "fallback_days": idx,
                "valid_candidates": valid_count,
                "topn_returned": min(valid_count, topn),
                **meta,
            }

    if best_date is None:
        return None, pd.DataFrame(), {
            "fallback_days": None,
            "valid_candidates": 0,
            "topn_returned": 0,
        }

    return best_date, best_df, {
        "fallback_days": fallback_used,
        "valid_candidates": best_valid,
        "topn_returned": min(best_valid, topn),
        **best_meta,
    }


def _load_data_quality_degraded_context(session: Session) -> Dict[str, object]:
    latest = (
        session.query(Job)
        .filter(Job.job_name == "data_quality_check")
        .order_by(Job.started_at.desc())
        .limit(1)
        .one_or_none()
    )
    if latest is None or not latest.logs_json:
        return {"degraded_mode": False, "degraded_datasets": []}
    logs = latest.logs_json if isinstance(latest.logs_json, dict) else {}
    return {
        "degraded_mode": bool(logs.get("degraded_mode", False)),
        "degraded_datasets": list(logs.get("degraded_datasets", [])),
        "dq_mode": logs.get("data_quality_mode"),
    }


def _research_score_candidates(feature_df: pd.DataFrame) -> pd.Series:
    # research 降級模式：只使用技術+流動性特徵做啟發式排序，避免依賴缺失的 institutional 資料
    keys = ["ret_20", "breakout_20", "amt_ratio_20"]
    data = feature_df.copy()
    z = pd.DataFrame(index=data.index)
    for col in keys:
        vals = pd.to_numeric(data.get(col), errors="coerce").fillna(0.0)
        std = float(vals.std())
        if std == 0:
            z[col] = 0.0
        else:
            z[col] = (vals - vals.mean()) / std
    return 0.5 * z["ret_20"] + 0.3 * z["breakout_20"] + 0.2 * z["amt_ratio_20"]


def _should_use_research_fallback(config, dq_ctx: Dict[str, object]) -> bool:
    degraded_datasets = set(str(x) for x in dq_ctx.get("degraded_datasets", []))
    return (
        str(getattr(config, "data_quality_mode", "strict")).lower() == "research"
        and bool(dq_ctx.get("degraded_mode", False))
        and "raw_institutional" in degraded_datasets
    )


def _filter_overheated(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """過濾短線過熱股票，避免在 RSI 過高或趨勢已過度延伸時追高買入。

    過濾條件（任一成立即排除）：
    1. RSI_14 > overheat_rsi_threshold（預設 75）
    2. market_rel_ret_20 > 0.20 且 macd_hist <= 0（相對強勢已減弱）
    """
    if not getattr(config, "enable_overheat_filter", False):
        return df

    rsi_threshold = float(getattr(config, "overheat_rsi_threshold", 75))
    mask_overheat = pd.Series(False, index=feature_df.index)

    if "rsi_14" in feature_df.columns:
        rsi = pd.to_numeric(feature_df["rsi_14"], errors="coerce").fillna(0)
        mask_overheat |= rsi > rsi_threshold

    if "market_rel_ret_20" in feature_df.columns and "macd_hist" in feature_df.columns:
        rel_ret = pd.to_numeric(feature_df["market_rel_ret_20"], errors="coerce").fillna(0)
        macd = pd.to_numeric(feature_df["macd_hist"], errors="coerce").fillna(0)
        mask_overheat |= (rel_ret > 0.20) & (macd <= 0)

    keep = ~mask_overheat.values
    filtered = df[keep].copy()

    n_removed = len(df) - len(filtered)
    if n_removed > 0:
        import logging
        logging.getLogger(__name__).debug("overheat_filter: removed %d stocks", n_removed)

    return filtered


def _build_agent_dump_summary(agent_dump: Dict[str, object]) -> Dict[str, object]:
    if not agent_dump:
        return {}
    return dict(agent_dump.get("summary", {}))


def _write_run_manifest(
    job_id: str,
    chosen_date: date,
    rows_df: pd.DataFrame,
    logs: Dict[str, object],
    selection_mode: str,
    score_mode: str,
    config,
    selection_meta: Dict[str, object],
    dq_ctx: Dict[str, object],
    weights_used: Dict[str, float] | None = None,
    agent_dump: Dict[str, object] | None = None,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    picks_payload = []
    sorted_df = rows_df.sort_values("score", ascending=False).reset_index(drop=True)
    for i, row in sorted_df.iterrows():
        picks_payload.append({"stock_id": str(row["stock_id"]), "rank": int(i + 1), "score": float(row["score"])})

    universe = {
        "valid_stock_universe_count": logs.get("valid_stock_universe_count"),
        "liquidity_excluded_ratio": logs.get("liquidity_excluded_ratio"),
        "tradability": selection_meta.get("tradability", {}),
        "missing_feature_columns": logs.get("missing_feature_columns", []),
    }
    manifest = {
        "job_id": job_id,
        "pick_date": chosen_date.isoformat(),
        "selection_mode": selection_mode,
        "score_mode": score_mode,
        "data_quality_mode": str(getattr(config, "data_quality_mode", "strict")),
        "degraded_mode": bool(dq_ctx.get("degraded_mode", False)),
        "degraded_datasets": list(dq_ctx.get("degraded_datasets", [])),
        "effective_topn": int(logs.get("effective_topn", getattr(config, "topn", len(picks_payload)))),
        "universe": universe,
        "weights_requested": getattr(config, "multi_agent_weights", None) if selection_mode == "multi_agent" else None,
        "weights_used": weights_used if selection_mode == "multi_agent" else None,
        "picks": picks_payload,
    }
    if selection_mode == "multi_agent":
        manifest["agent_dump_summary"] = _build_agent_dump_summary(agent_dump or {})

    manifest_path = ARTIFACTS_DIR / f"run_manifest_daily_pick_{job_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_stock_names(db_session: Session, stock_ids: List[str]) -> Dict[str, str]:
    """查詢股票中文名稱，回傳 {stock_id: name}，找不到時以 stock_id 替代。"""
    if not stock_ids:
        return {}
    rows = (
        db_session.query(Stock.stock_id, Stock.name)
        .filter(Stock.stock_id.in_(stock_ids))
        .all()
    )
    result = {str(r.stock_id): str(r.name or r.stock_id) for r in rows}
    for sid in stock_ids:
        result.setdefault(str(sid), str(sid))
    return result


def _print_breakthrough_lists(
    picks_df: pd.DataFrame,
    backup_df: pd.DataFrame,
    bt_status: Dict[str, Dict],
    stock_names: Dict[str, str],
    pick_date,
) -> None:
    """輸出突破確認三清單：可進場 / 等待突破中 / 候補股。"""
    print()
    print("=" * 62)
    print(f"  今日選股突破狀態 ({pick_date})")
    print("=" * 62)

    ready_rows = picks_df[picks_df.get("breakthrough_ready", False) == True] if "breakthrough_ready" in picks_df.columns else pd.DataFrame()
    waiting_rows = picks_df[picks_df.get("breakthrough_ready", True) == False] if "breakthrough_ready" in picks_df.columns else picks_df

    # ── 今日可進場（已突破）──
    n_ready = len(ready_rows)
    print(f"\n=== 今日可進場（已突破）=== ({n_ready} 檔)")
    if n_ready > 0:
        print(f"  {'#':>3}  {'代號':6}  {'名稱':<12}  {'突破類型':<10}  {'模型分數':>8}")
        print(f"  {'─'*3}  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*8}")
        for rank, (_, row) in enumerate(ready_rows.sort_values("score", ascending=False).iterrows(), 1):
            sid = str(row["stock_id"])
            name = stock_names.get(sid, sid)[:10]
            bt_type = bt_status.get(sid, {}).get("type") or "-"
            bt_label = "價格突破" if bt_type == "price" else ("外資籌碼" if bt_type == "institutional" else bt_type)
            score = float(row.get("score", 0))
            print(f"  {rank:>3}  {sid:6}  {name:<12}  {bt_label:<10}  {score:>8.4f}")
    else:
        print("  （今日無符合突破條件的股票）")

    # ── 等待突破中 ──
    n_waiting = len(waiting_rows)
    print(f"\n=== 等待突破中 === ({n_waiting} 檔)")
    if n_waiting > 0:
        print(f"  {'#':>3}  {'代號':6}  {'名稱':<12}  {'等待':>4}  {'距20日高點':>10}  {'量比':>10}")
        print(f"  {'─'*3}  {'─'*6}  {'─'*12}  {'─'*4}  {'─'*10}  {'─'*10}")
        for rank, (_, row) in enumerate(waiting_rows.sort_values("score", ascending=False).iterrows(), 1):
            sid = str(row["stock_id"])
            name = stock_names.get(sid, sid)[:10]
            days = int(row.get("days_waiting", 0))
            info = bt_status.get(sid, {})
            pct = info.get("pct_to_price_bt", float("nan"))
            vol_r = info.get("vol_ratio", float("nan"))
            pct_str = f"+{pct*100:.1f}%" if pct == pct and pct > 0 else (f"{pct*100:.1f}%" if pct == pct else "─")
            vol_str = f"{vol_r:.2f}x/1.5x" if vol_r == vol_r else "─"
            print(f"  {rank:>3}  {sid:6}  {name:<12}  {'第'+str(days)+'天':>4}  {pct_str:>10}  {vol_str:>10}")
    else:
        print("  （所有選股今日均已突破）")

    # ── 候補股（備用）──
    n_backup = len(backup_df)
    print(f"\n=== 候補股（備用）=== ({n_backup} 檔)")
    if n_backup > 0:
        print(f"  {'#':>3}  {'代號':6}  {'名稱':<12}  {'模型分數':>8}")
        print(f"  {'─'*3}  {'─'*6}  {'─'*12}  {'─'*8}")
        for rank, (_, row) in enumerate(backup_df.sort_values("score", ascending=False).iterrows(), 1):
            sid = str(row["stock_id"])
            name = stock_names.get(sid, sid)[:10]
            score = float(row.get("score", 0))
            print(f"  {rank:>3}  {sid:6}  {name:<12}  {score:>8.4f}")
    else:
        print("  （無候補股）")

    print()


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "daily_pick")
    coverage_stats: Dict[str, object] = {}
    
    try:
        candidate_dates = (
            db_session.query(Feature.trading_date)
            .distinct()
            .order_by(Feature.trading_date.desc())
            .limit(config.fallback_days + 1)
            .all()
        )
        candidate_dates = [row[0] for row in candidate_dates]
        coverage_stats["candidate_dates_count"] = len(candidate_dates)
        
        if not candidate_dates:
            finish_job(db_session, job_id, "success", logs={
                "rows": 0, 
                "reason": "no feature dates",
                **coverage_stats,
            })
            return {"rows": 0}
        
        coverage_stats["latest_feature_date"] = candidate_dates[0].isoformat()
        coverage_stats["oldest_candidate_date"] = candidate_dates[-1].isoformat()

        selection_mode = str(getattr(config, "selection_mode", "model")).lower()
        if selection_mode not in {"model", "multi_agent"}:
            selection_mode = "model"

        model_version = _load_latest_model(db_session)
        model = None
        feature_names = list(FEATURE_COLUMNS)
        if selection_mode == "model":
            if model_version is None:
                raise ValueError("No trained model found")
            artifact = joblib.load(model_version.artifact_path)
            model = artifact["model"]
            feature_names = artifact["feature_names"]
        
        # 取得有效股票 universe（排除 ETF、權證等）
        valid_universe_df = risk.get_universe(db_session, max(candidate_dates), config)
        tradability_logs: Dict[str, object] = {}
        if getattr(config, "enable_tradability_filter", True):
            valid_universe_df, tradability_logs = tradability_filter.filter_universe(
                db_session,
                valid_universe_df,
                max(candidate_dates),
                return_stats=True,
            )
            if tradability_logs.get("missing_status_count", 0) > 0:
                tradability_logs["warning"] = (
                    f"tradability status missing for {tradability_logs['missing_status_count']} stocks; "
                    "kept as tradable by policy"
                )
        else:
            tradability_logs = {"tradability_filter": "disabled"}
        valid_stocks = set(valid_universe_df["stock_id"].astype(str).tolist())
        coverage_stats["valid_stock_universe_count"] = len(valid_stocks)
        coverage_stats["tradability"] = tradability_logs
        
        if not valid_stocks:
            # stocks 表為空時，不過濾（向後相容）
            coverage_stats["stock_universe_filter"] = "disabled (stocks table empty)"
            stmt = (
                select(Feature.stock_id, Feature.trading_date, Feature.features_json)
                .where(Feature.trading_date.in_(candidate_dates))
                .order_by(Feature.stock_id, Feature.trading_date)
            )
        else:
            coverage_stats["stock_universe_filter"] = "enabled"
            stmt = (
                select(Feature.stock_id, Feature.trading_date, Feature.features_json)
                .where(Feature.trading_date.in_(candidate_dates))
                .where(Feature.stock_id.in_(valid_stocks))
                .order_by(Feature.stock_id, Feature.trading_date)
            )
        
        df = pd.read_sql(stmt, db_session.get_bind())
        
        coverage_stats["total_feature_rows"] = len(df)
        coverage_stats["unique_stocks_with_features"] = df["stock_id"].nunique() if not df.empty else 0
        
        if df.empty:
            finish_job(db_session, job_id, "success", logs={
                "rows": 0,
                "reason": "no features for candidate dates",
                **coverage_stats,
            })
            return {"rows": 0}

        feature_df = _parse_features(df["features_json"])
        
        # 記錄缺失欄位
        missing_cols = [col for col in feature_names if col not in feature_df.columns]
        coverage_stats["missing_feature_columns"] = missing_cols
        
        for col in feature_names:
            if col not in feature_df.columns:
                feature_df[col] = np.nan
        feature_df = feature_df[feature_names]

        price_df = _load_price_universe(
            db_session,
            max(candidate_dates),
            df["stock_id"].astype(str).unique().tolist(),
        )
        
        coverage_stats["price_universe_rows"] = len(price_df)
        coverage_stats["price_universe_stocks"] = price_df["stock_id"].nunique() if not price_df.empty else 0

        # ── 大盤過濾器：空頭市場減碼 ──
        effective_topn = config.topn
        bear_market = False
        weekly_drop: float | None = None   # 在 if 外初始化，供後續 topN floor 使用
        market_below_200ma = False
        if getattr(config, "market_filter_enabled", False):
            ma_days = getattr(config, "market_filter_ma_days", 60)
            bear_topn = getattr(config, "market_filter_bear_topn", config.topn // 2)
            crisis_topn = getattr(config, "market_filter_crisis_topn", max(2, config.topn // 4))
            crisis_threshold = float(getattr(config, "market_filter_weekly_drop_threshold", -0.03))
            detector = regime.get_regime_detector(config)
            # 載入足夠 200 日資料（用於 200MA 現金保留判斷）
            _load_days = max(ma_days, 200)
            market_df = _load_market_price_df(db_session, max(candidate_dates), _load_days)
            regime_result = detector.detect(market_df, config)
            bear_market = regime_result.get("regime") == "BEAR"
            coverage_stats["regime_detector"] = getattr(config, "regime_detector", "ma")
            coverage_stats["regime_meta"] = regime_result.get("meta", {})

            # 週跌幅危機偵測（最近 5 個交易日平均收盤價變化）
            if "avg_close" in market_df.columns and len(market_df) >= 6:
                mdf = market_df.sort_values("trading_date")
                latest_close = float(mdf["avg_close"].iloc[-1])
                prev_close = float(mdf["avg_close"].iloc[-6])
                if prev_close > 0:
                    weekly_drop = (latest_close - prev_close) / prev_close
            coverage_stats["weekly_drop"] = weekly_drop

            # 200 日均線判斷（供現金保留機制使用）
            if "avg_close" in market_df.columns and len(market_df) >= 40:
                _mdf = market_df.sort_values("trading_date")
                _mdf_close = pd.to_numeric(_mdf["avg_close"], errors="coerce").dropna()
                if len(_mdf_close) >= 40:
                    _ma200 = float(_mdf_close.rolling(200, min_periods=40).mean().iloc[-1])
                    _cur = float(_mdf_close.iloc[-1])
                    market_below_200ma = not pd.isna(_ma200) and _cur < _ma200
            coverage_stats["market_below_200ma"] = market_below_200ma

            # 動態現金水位：200MA 以下+反彈 10%；200MA 以下+未反彈 30%；200MA 以上 0%
            if market_below_200ma:
                if isinstance(weekly_drop, float) and weekly_drop > 0.03:
                    _dp_cash_ratio = 0.10  # 反彈期
                else:
                    _dp_cash_ratio = 0.30  # 下跌或橫盤
            else:
                _dp_cash_ratio = 0.0
            coverage_stats["cash_ratio"] = _dp_cash_ratio

            if weekly_drop is not None and weekly_drop < crisis_threshold:
                effective_topn = crisis_topn
                coverage_stats["market_filter"] = "CRISIS"
                coverage_stats["effective_topn"] = effective_topn
            elif bear_market:
                effective_topn = bear_topn
                coverage_stats["market_filter"] = "BEAR"
                coverage_stats["effective_topn"] = effective_topn
            else:
                coverage_stats["market_filter"] = "BULL"
        else:
            coverage_stats["market_filter"] = "disabled"
            coverage_stats["effective_topn"] = effective_topn

        # ── 季節性降倉（弱勢月份）──
        # 委派給 risk.apply_seasonal_topn_reduction（與 backtest.py 統一邏輯）
        _s_weak = tuple(getattr(config, "seasonal_weak_months", (3, 10)))
        _s_mult = float(getattr(config, "seasonal_topn_multiplier", 0.5))
        today_month = max(candidate_dates).month
        # topN 絕對下限保護（防止危機+弱勢月份疊加造成 topN=1~2 的極端集中風險）
        # 極端空頭（週跌>10%）最低 3 支；一般情況最低 5 支
        _dp_extreme_bear = isinstance(weekly_drop, float) and weekly_drop < -0.10
        _dp_min_topn = 3 if _dp_extreme_bear else 5
        effective_topn, _reduced = risk.apply_seasonal_topn_reduction(
            effective_topn, today_month, weak_months=_s_weak, multiplier=_s_mult, topn_floor=_dp_min_topn
        )
        if _reduced:
            coverage_stats["seasonal_filter"] = f"month_{today_month}"
        elif effective_topn < _dp_min_topn:
            # topN floor 觸發（非季節性，但仍需保護）
            coverage_stats["topn_floor_applied"] = f"{effective_topn}→{_dp_min_topn}"
            effective_topn = _dp_min_topn
        coverage_stats["effective_topn"] = effective_topn

        chosen_date, chosen_df, fallback_logs = _choose_pick_date(
            candidate_dates,
            df,
            price_df,
            effective_topn,
            config,
            config.fallback_days,
        )
        if chosen_date is None:
            reason = "no valid candidates after fallback"
            finish_job(db_session, job_id, "failed", error_text=reason, logs={
                "error": reason, 
                **fallback_logs,
                **coverage_stats,
            })
            raise ValueError(reason)

        selected_idx = chosen_df.index
        df = chosen_df.reset_index(drop=True)
        feature_df = feature_df.loc[selected_idx].reset_index(drop=True)
        feature_df, impute_stats = _impute_features(feature_df)

        dq_ctx = _load_data_quality_degraded_context(db_session)
        degraded_datasets = set(str(x) for x in dq_ctx.get("degraded_datasets", []))
        use_research_fallback = _should_use_research_fallback(config, dq_ctx)
        coverage_stats["degraded_mode"] = bool(dq_ctx.get("degraded_mode", False))
        coverage_stats["degraded_datasets"] = sorted(degraded_datasets)

        records: List[Dict] = []
        selection_meta = {
            "tradability": tradability_logs,
            "liquidity": {
                "excluded_ratio": fallback_logs.get("liquidity_excluded_ratio", 0.0),
                "excluded_count": fallback_logs.get("liquidity_excluded_count", 0),
                "before_count": fallback_logs.get("candidate_count_before_liquidity", 0),
                "after_count": fallback_logs.get("candidate_count_after_liquidity", 0),
            },
        }
        weights_used_manifest: Dict[str, float] | None = None
        agent_dump: Dict[str, object] | None = None
        if selection_mode == "multi_agent":
            ma_topn = int(getattr(config, "multi_agent_topn", effective_topn) or effective_topn)
            # 若啟用 model alignment，預先計算 model 預測分數傳給 MA
            ma_model_score_map: Dict[str, float] | None = None
            ma_alignment_weight = float(getattr(config, "ma_model_alignment_weight", 0.0))
            if ma_alignment_weight > 0.0 and model_version is not None:
                try:
                    artifact = joblib.load(model_version.artifact_path)
                    ma_model = artifact["model"]
                    ma_feat_names = artifact["feature_names"]
                    ma_feat_df = feature_df.copy()
                    for col in ma_feat_names:
                        if col not in ma_feat_df.columns:
                            ma_feat_df[col] = np.nan
                    ma_feat_df = ma_feat_df[ma_feat_names]
                    ma_feat_df = ma_feat_df.replace([np.inf, -np.inf], np.nan).fillna(ma_feat_df.median(skipna=True).fillna(0))
                    ma_raw_scores = ma_model.predict(ma_feat_df.values)
                    ma_model_score_map = {
                        str(sid): float(s)
                        for sid, s in zip(df["stock_id"].tolist(), ma_raw_scores)
                    }
                except Exception:
                    ma_model_score_map = None
            picks_df, agent_dump = multi_agent_selector.run_multi_agent_selection(
                feature_df=feature_df,
                stock_ids=df["stock_id"],
                pick_date=chosen_date,
                topn=min(ma_topn, effective_topn),
                config=config,
                dq_ctx=dq_ctx,
                selection_meta=selection_meta,
                model_score_map=ma_model_score_map,
            )
            coverage_stats["score_mode"] = (
                "multi_agent_degraded" if bool(dq_ctx.get("degraded_mode", False)) else "multi_agent"
            )
            df = picks_df.copy()
            if not df.empty:
                first_meta = df.iloc[0]["reason_json"].get("_selection_meta", {})
                weights_used_manifest = dict(first_meta.get("weights_used", {}))
        else:
            if use_research_fallback:
                scores = _research_score_candidates(feature_df).values
                coverage_stats["score_mode"] = "research_tech_liquidity_fallback"
            else:
                scores = model.predict(feature_df.values)
                coverage_stats["score_mode"] = "model"
            df["score"] = scores

            # ── 空頭防禦②：RSI 自適應過濾（依空頭深度分級，反彈期取消）──
            # ‧ 反彈期（5日漲 >+3%）        → 不過濾（跟上反彈領頭羊）
            # ‧ 空頭深度（20d 中位數 <-15%）→ RSI > 75
            # ‧ 空頭初期（5~15%）           → RSI > 80
            if bear_market and "rsi_14" in feature_df.columns:
                _dp_5d = weekly_drop if isinstance(weekly_drop, float) else 0.0
                if _dp_5d > 0.03:
                    _dp_rsi_thr = None   # 反彈期：完全不過濾
                else:
                    # 用 feature_df ret_20 欄位估算市場 20d 深度
                    _dp_ret20 = pd.to_numeric(
                        feature_df["ret_20"] if "ret_20" in feature_df.columns
                        else pd.Series(dtype=float), errors="coerce"
                    ).dropna()
                    _dp_median20 = float(_dp_ret20.median()) if not _dp_ret20.empty else 0.0
                    _dp_rsi_thr = 75 if _dp_median20 < -0.15 else 80
                if _dp_rsi_thr is not None:
                    _rsi_ok = feature_df["rsi_14"].fillna(100).values <= _dp_rsi_thr
                    _rsi_removed = int((~_rsi_ok).sum())
                    if _rsi_removed > 0:
                        df = df.loc[_rsi_ok].reset_index(drop=True)
                        feature_df = feature_df.loc[_rsi_ok].reset_index(drop=True)
                        coverage_stats["bear_rsi_filtered"] = _rsi_removed
                        coverage_stats["bear_rsi_threshold"] = _dp_rsi_thr
                # df 為空時自然產生 0 picks，由後續流程統一處理

            # ── 空頭防禦③：低波動加權（atr_inv z-score 加分 0.3 倍）──
            if bear_market and "atr_inv" in feature_df.columns:
                _atr_inv = pd.to_numeric(feature_df["atr_inv"], errors="coerce").fillna(0)
                _atr_std = float(_atr_inv.std())
                if _atr_std > 0:
                    _atr_z = (_atr_inv - _atr_inv.mean()) / _atr_std
                    df["score"] = df["score"].values + 0.3 * _atr_z.values

            percentile_map = {}
            for feat in REASON_FEATURES:
                if feat not in feature_df.columns:
                    continue
                vals = pd.to_numeric(feature_df[feat], errors="coerce")
                percentile_map[feat] = vals.rank(pct=True)

            df = _filter_overheated(df, feature_df, config)
            _df_pre_topn = df.copy()  # 供突破候補池使用（topN 前的完整候選集）
            df = risk.pick_topn(df, effective_topn)

            # ── 突破確認狀態標記（F+ 策略整合）──
            _df_backup: pd.DataFrame = pd.DataFrame()
            bt_status: Dict[str, Dict] = {}
            if getattr(config, "enable_breakthrough_entry", True):
                # 候補股：topN 以外的前 5 名（rank N+1 ~ N+5）
                _ext_n = effective_topn + 5
                _df_ext = risk.pick_topn(_df_pre_topn, _ext_n)
                if len(_df_ext) > effective_topn:
                    _df_backup = _df_ext.iloc[effective_topn:].copy()
                # feature_df.loc[df.index] 對應 pick_topn 後各股的特徵（含 foreign_buy_consecutive_days）
                _feat_for_bt = feature_df.loc[df.index].copy()
                _feat_for_bt.insert(0, "stock_id", df["stock_id"].values)
                bt_status = _bt.check_today(
                    price_df=price_df,
                    feature_df=_feat_for_bt,
                    stock_ids=df["stock_id"].astype(str).tolist(),
                    target_date=chosen_date,
                )
                df["breakthrough_ready"] = df["stock_id"].astype(str).map(
                    lambda sid: bt_status.get(sid, {}).get("ready", False)
                )
                df["breakthrough_type"] = df["stock_id"].astype(str).map(
                    lambda sid: bt_status.get(sid, {}).get("type")
                )
                df["days_waiting"] = 0
                # 輸出突破三清單
                _all_sids = (
                    df["stock_id"].astype(str).tolist()
                    + _df_backup["stock_id"].astype(str).tolist()
                )
                _stock_names = _load_stock_names(db_session, list(set(_all_sids)))
                _print_breakthrough_lists(df, _df_backup, bt_status, _stock_names, chosen_date)

            # ── 倉位配置 ──
            ps_method = str(getattr(config, "position_sizing_method", "vol_inverse"))
            scores_map = {str(row["stock_id"]): float(row["score"]) for _, row in df.iterrows()}
            try:
                weight_map = _pos_sizing.compute_weights(
                    prices=price_df.pivot_table(index="trading_date", columns="stock_id", values="close")
                    if not price_df.empty else pd.DataFrame(),
                    scores=scores_map,
                    method=ps_method,
                )
            except Exception:
                n = max(len(scores_map), 1)
                weight_map = {sid: 1.0 / n for sid in scores_map}

            for _, row in df.iterrows():
                reasons = {}
                for feat in REASON_FEATURES:
                    if feat not in feature_df.columns:
                        continue
                    value = float(feature_df.loc[row.name, feat])
                    pct = float(percentile_map[feat].loc[row.name]) if feat in percentile_map else None
                    reasons[feat] = {"value": value, "percentile": pct}
                reasons["_selection_meta"] = dict(selection_meta, dq_ctx=dq_ctx, selection_mode="model")
                sid = str(row["stock_id"])
                reasons["weight"] = round(weight_map.get(sid, 0.0), 6)

                records.append(
                    {
                        "pick_date": chosen_date,
                        "stock_id": row["stock_id"],
                        "score": float(row["score"]),
                        "model_id": model_version.model_id if model_version else "n/a",
                        "reason_json": reasons,
                    }
                )

        if selection_mode == "multi_agent":
            for _, row in df.iterrows():
                records.append(
                    {
                        "pick_date": chosen_date,
                        "stock_id": row["stock_id"],
                        "score": float(row["score"]),
                        "model_id": model_version.model_id if model_version else "multi_agent",
                        "reason_json": row["reason_json"],
                    }
                )

        if records:
            # 先清除同一天的舊 picks，避免殘留不同 stock_id 的過期資料
            db_session.query(Pick).filter(Pick.pick_date == chosen_date).delete()

            stmt = insert(Pick).values(records)
            stmt = stmt.on_duplicate_key_update(
                score=stmt.inserted.score,
                model_id=stmt.inserted.model_id,
                reason_json=stmt.inserted.reason_json,
            )
            db_session.execute(stmt)

        logs = {
            "rows": len(records),
            "pick_date": chosen_date.isoformat(),
            "model_id": model_version.model_id if model_version else "multi_agent",
            **fallback_logs,
            **impute_stats,
            **coverage_stats,
            "min_avg_turnover": config.min_avg_turnover,
            "min_amt_20": getattr(config, "min_amt_20", None),
            "selection_mode": selection_mode,
        }
        _write_run_manifest(
            job_id=job_id,
            chosen_date=chosen_date,
            rows_df=df,
            logs=logs,
            selection_mode=selection_mode,
            score_mode=str(coverage_stats.get("score_mode", "model")),
            config=config,
            selection_meta=selection_meta,
            dq_ctx=dq_ctx,
            weights_used=weights_used_manifest,
            agent_dump=agent_dump,
        )
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs={
            "error": str(exc),
            **coverage_stats,
        })
        raise
