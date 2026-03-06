"""FastAPI 應用：提供 REST API 端點，包含選股結果（/picks）、模型版本（/models）、
工作記錄（/jobs）、股票詳情（/stock/{stock_id}）、策略回測（/strategy_runs）、
WebSocket 即時效能監控（/ws/metrics）等。
"""
from __future__ import annotations

import asyncio
from datetime import date
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional

import psutil
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import func, text as sql_text

from app.config import load_config
from app.db import get_session
from app.job_utils import cleanup_stale_running_jobs
from app.models import (
    Feature,
    Job,
    ModelVersion,
    Pick,
    RawInstitutional,
    RawPrice,
    StrategyConfig,
    StrategyPosition,
    StrategyRun,
    StrategyTrade,
)
from app.strategy_doc import get_selection_logic
from app.schemas import (
    FeatureOut,
    InstitutionalOut,
    JobOut,
    ModelVersionOut,
    PickOut,
    PriceOut,
    StockDetailOut,
    StrategyConfigIn,
    StrategyConfigOut,
    StrategyRunIn,
    StrategyRunOut,
    StrategyTradeOut,
)
from skills.strategy_factory.data import compute_indicators, detect_regime, load_price_df, resolve_weights
from skills.strategy_factory.engine import BacktestConfig, BacktestEngine, StrategyAllocation
from skills.strategy_factory.registry import get as get_strategy, register_defaults

app = FastAPI(title="Stock Bot ML API", version="0.1.0")


def _to_float(value):
    if value is None:
        return None
    return float(value)


def _price_out(row: RawPrice) -> PriceOut:
    return PriceOut(
        stock_id=row.stock_id,
        trading_date=row.trading_date,
        open=_to_float(row.open),
        high=_to_float(row.high),
        low=_to_float(row.low),
        close=_to_float(row.close),
        volume=row.volume,
    )


def _inst_out(row: RawInstitutional) -> InstitutionalOut:
    return InstitutionalOut(
        stock_id=row.stock_id,
        trading_date=row.trading_date,
        foreign_buy=row.foreign_buy,
        foreign_sell=row.foreign_sell,
        foreign_net=row.foreign_net,
        trust_buy=row.trust_buy,
        trust_sell=row.trust_sell,
        trust_net=row.trust_net,
        dealer_buy=row.dealer_buy,
        dealer_sell=row.dealer_sell,
        dealer_net=row.dealer_net,
    )


def _feature_out(row: Feature) -> FeatureOut:
    return FeatureOut(
        stock_id=row.stock_id,
        trading_date=row.trading_date,
        features_json=row.features_json,
    )


def _pick_out(row: Pick) -> PickOut:
    return PickOut(
        pick_date=row.pick_date,
        stock_id=row.stock_id,
        score=_to_float(row.score) or 0.0,
        model_id=row.model_id,
        reason_json=row.reason_json,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/strategy")
def strategy_doc():
    """回傳選股邏輯說明（Markdown）"""
    config = load_config()
    return {"markdown": get_selection_logic(config)}


@app.get("/picks", response_model=List[PickOut])
def get_picks(date: Optional[date] = Query(default=None)):
    with get_session() as session:
        pick_date = date
        if pick_date is None:
            pick_date = session.query(func.max(Pick.pick_date)).scalar()
        if pick_date is None:
            return []
        rows = (
            session.query(Pick)
            .filter(Pick.pick_date == pick_date)
            .order_by(Pick.score.desc())
            .all()
        )
        return [_pick_out(row) for row in rows]


@app.get("/stock/{stock_id}", response_model=StockDetailOut)
def get_stock(stock_id: str, date: Optional[date] = Query(default=None)):
    with get_session() as session:
        target_date = date
        if target_date is None:
            target_date = (
                session.query(func.max(RawPrice.trading_date))
                .filter(RawPrice.stock_id == stock_id)
                .scalar()
            )
        if target_date is None:
            raise HTTPException(status_code=404, detail="stock not found")

        row = (
            session.query(RawPrice, RawInstitutional, Feature, Pick)
            .outerjoin(
                RawInstitutional,
                (RawPrice.stock_id == RawInstitutional.stock_id)
                & (RawPrice.trading_date == RawInstitutional.trading_date),
            )
            .outerjoin(
                Feature,
                (RawPrice.stock_id == Feature.stock_id)
                & (RawPrice.trading_date == Feature.trading_date),
            )
            .outerjoin(
                Pick,
                (RawPrice.stock_id == Pick.stock_id)
                & (RawPrice.trading_date == Pick.pick_date),
            )
            .filter(RawPrice.stock_id == stock_id, RawPrice.trading_date == target_date)
            .one_or_none()
        )

        if row is None:
            return StockDetailOut(stock_id=stock_id, trading_date=target_date)

        price, inst, feat, pick = row
        return StockDetailOut(
            stock_id=stock_id,
            trading_date=target_date,
            price=_price_out(price) if price else None,
            institutional=_inst_out(inst) if inst else None,
            features=_feature_out(feat) if feat else None,
            pick=_pick_out(pick) if pick else None,
        )


@app.get("/models", response_model=List[ModelVersionOut])
def get_models():
    with get_session() as session:
        rows = (
            session.query(ModelVersion)
            .order_by(ModelVersion.created_at.desc())
            .limit(20)
            .all()
        )
        return [ModelVersionOut.model_validate(row) for row in rows]


@app.get("/jobs", response_model=List[JobOut])
def get_jobs(limit: int = Query(default=50, ge=1, le=200)):
    with get_session() as session:
        cleanup_stale_running_jobs(session, stale_minutes=120, commit=False)
        rows = session.query(Job).order_by(Job.started_at.desc()).limit(limit).all()
        return [JobOut.model_validate(row) for row in rows]


@app.get("/strategy_configs", response_model=List[StrategyConfigOut])
def list_strategy_configs():
    with get_session() as session:
        rows = session.query(StrategyConfig).order_by(StrategyConfig.created_at.desc()).all()
        return [StrategyConfigOut.model_validate(r) for r in rows]


@app.post("/strategy_configs", response_model=StrategyConfigOut)
def create_strategy_config(payload: StrategyConfigIn):
    with get_session() as session:
        config_id = uuid.uuid4().hex
        row = StrategyConfig(
            config_id=config_id,
            name=payload.name,
            config_json=payload.config_json,
        )
        session.add(row)
        session.commit()
        return StrategyConfigOut.model_validate(row)


@app.post("/strategy_runs", response_model=StrategyRunOut)
def run_strategy_backtest(payload: StrategyRunIn):
    register_defaults()
    with get_session() as session:
        config_row = (
            session.query(StrategyConfig)
            .filter(StrategyConfig.config_id == payload.config_id)
            .one_or_none()
        )
        if config_row is None:
            raise HTTPException(status_code=404, detail="strategy config not found")

        config = load_config()
        start_date = payload.start_date
        end_date = payload.end_date
        raw = load_price_df(start_date, end_date)
        df = compute_indicators(raw)
        regime = detect_regime(df, config)
        weights = resolve_weights(regime, config, config_row.config_json or {})
        strategies = config_row.config_json.get("strategies") if config_row.config_json else None

        allocations = []
        for name, weight in weights.items():
            if strategies and name not in strategies:
                continue
            allocations.append(StrategyAllocation(strategy=get_strategy(name), weight=weight))
        if not allocations:
            raise HTTPException(status_code=400, detail="no strategies to run")

        bt_cfg = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=payload.initial_capital if payload.initial_capital is not None else 1_000_000.0,
            transaction_cost_pct=payload.transaction_cost_pct if payload.transaction_cost_pct is not None else 0.001425,
            slippage_pct=payload.slippage_pct if payload.slippage_pct is not None else 0.001,
            risk_per_trade=payload.risk_per_trade if payload.risk_per_trade is not None else 0.01,
            position_size_multiplier=(
                payload.position_size_multiplier if payload.position_size_multiplier is not None else 1.0
            ),
            target_exposure_pct=payload.target_exposure_pct if payload.target_exposure_pct is not None else 1.0,
            max_positions=payload.max_positions if payload.max_positions is not None else 6,
            rebalance_freq=(payload.rebalance_freq or "D").upper(),
            min_notional_per_trade=(
                payload.min_notional_per_trade if payload.min_notional_per_trade is not None else 1_000.0
            ),
            max_pyramiding_level=payload.max_pyramiding_level if payload.max_pyramiding_level is not None else 1,
        )
        engine = BacktestEngine(bt_cfg)
        result = engine.run(df, allocations)

        equity_curve = result["equity_curve"]
        final_equity = equity_curve[-1]["equity"] if equity_curve else bt_cfg.initial_capital
        total_return = final_equity / bt_cfg.initial_capital - 1
        metrics = {
            "regime": regime,
            "final_equity": final_equity,
            "total_return": total_return,
            "equity_curve": equity_curve,
            "trade_count": len(result["trades"]),
        }
        metrics = json.loads(json.dumps(metrics, default=str))

        run_id = uuid.uuid4().hex
        run_row = StrategyRun(
            run_id=run_id,
            config_id=config_row.config_id,
            start_date=start_date,
            end_date=end_date,
            initial_capital=bt_cfg.initial_capital,
            transaction_cost_pct=bt_cfg.transaction_cost_pct,
            slippage_pct=bt_cfg.slippage_pct,
            metrics_json=metrics,
        )
        session.add(run_row)

        for t in result["trades"]:
            trade_row = StrategyTrade(
                run_id=run_id,
                trade_id=uuid.uuid4().hex,
                trading_date=t["trading_date"],
                stock_id=t["stock_id"],
                strategy_name=t.get("strategy_name"),
                action=t["action"],
                qty=t["qty"],
                price=t["price"],
                fee=t["fee"],
                reason_json={
                    "reason": t.get("reason"),
                    "realized_pnl": t.get("realized_pnl"),
                    "avg_cost": t.get("avg_cost"),
                },
            )
            session.add(trade_row)

        for p in result["positions"]:
            pos_row = StrategyPosition(
                run_id=run_id,
                trading_date=p["trading_date"],
                stock_id=p["stock_id"],
                strategy_name=p.get("strategy_name"),
                qty=p["qty"],
                avg_cost=p["avg_cost"],
                market_value=p["market_value"],
                unrealized_pnl=p["unrealized_pnl"],
            )
            session.add(pos_row)

        session.commit()
        return StrategyRunOut.model_validate(run_row)


@app.get("/strategy_runs/{run_id}", response_model=StrategyRunOut)
def get_strategy_run(run_id: str):
    with get_session() as session:
        row = session.query(StrategyRun).filter(StrategyRun.run_id == run_id).one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="strategy run not found")
        return StrategyRunOut.model_validate(row)


@app.get("/strategy_runs/{run_id}/trades", response_model=List[StrategyTradeOut])
def get_strategy_trades(run_id: str):
    with get_session() as session:
        rows = (
            session.query(StrategyTrade)
            .filter(StrategyTrade.run_id == run_id)
            .order_by(StrategyTrade.trading_date.asc())
            .all()
        )
        return [StrategyTradeOut.model_validate(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# 效能監控 Endpoints
# ══════════════════════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SLOW_QUERIES_PATH = _PROJECT_ROOT / "artifacts" / "slow_queries.jsonl"
_FEATURE_PERF_PATH = _PROJECT_ROOT / "artifacts" / "feature_perf.csv"


def _system_snapshot() -> dict:
    """回傳 CPU / RAM 即時快照（psutil）。"""
    cpu = psutil.cpu_percent(interval=None)  # non-blocking; first call may be 0
    mem = psutil.virtual_memory()
    return {
        "ts": time.time(),
        "cpu_pct": cpu,
        "ram_used_gb": round(mem.used / 1e9, 2),
        "ram_total_gb": round(mem.total / 1e9, 2),
        "ram_pct": mem.percent,
    }


@app.get("/metrics/system")
def metrics_system():
    """回傳目前 CPU / RAM 使用狀況。"""
    return _system_snapshot()


@app.get("/metrics/latest")
def metrics_latest():
    """回傳系統資源快照 + 最近一筆 job 狀態。"""
    sys_info = _system_snapshot()
    with get_session() as session:
        latest_job = (
            session.query(Job)
            .order_by(Job.started_at.desc())
            .first()
        )
        job_info = None
        if latest_job:
            job_info = {
                "job_name": latest_job.job_name,
                "status": latest_job.status,
                "started_at": str(latest_job.started_at),
                "ended_at": str(latest_job.ended_at) if latest_job.ended_at else None,
            }
    return {"system": sys_info, "latest_job": job_info}


@app.get("/metrics/history")
def metrics_history(limit: int = Query(default=100, ge=1, le=1000)):
    """回傳最近 N 筆已完成 job 的耗時紀錄。"""
    with get_session() as session:
        rows = session.execute(
            sql_text("""
                SELECT job_name, status, started_at, ended_at,
                       TIMESTAMPDIFF(SECOND, started_at, ended_at) AS duration_s
                FROM jobs
                WHERE ended_at IS NOT NULL
                ORDER BY started_at DESC
                LIMIT :limit
            """),
            {"limit": limit},
        ).fetchall()
    return [
        {
            "job_name": r[0],
            "status": r[1],
            "started_at": str(r[2]),
            "ended_at": str(r[3]),
            "duration_s": r[4],
        }
        for r in rows
    ]


@app.get("/metrics/features")
def metrics_features():
    """回傳 artifacts/feature_perf.csv 的特徵耗時排名。"""
    if not _FEATURE_PERF_PATH.exists():
        return []
    import pandas as pd  # noqa: PLC0415
    df = pd.read_csv(_FEATURE_PERF_PATH)
    return df.to_dict(orient="records")


@app.get("/metrics/slow-queries")
def metrics_slow_queries(limit: int = Query(default=50, ge=1, le=500)):
    """回傳最近 N 筆 slow query 記錄（來自 artifacts/slow_queries.jsonl）。"""
    if not _SLOW_QUERIES_PATH.exists():
        return []
    records: list[dict] = []
    try:
        with open(_SLOW_QUERIES_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception:
        return []
    return records[-limit:]


@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket):
    """WebSocket：每 2 秒推送一次系統資源快照（CPU / RAM）。

    連線方式：ws://127.0.0.1:8000/ws/metrics
    回傳 JSON: {"ts": float, "cpu_pct": float, "ram_used_gb": float,
                "ram_total_gb": float, "ram_pct": float}
    """
    await websocket.accept()
    # warm-up psutil cpu_percent (first call always returns 0)
    psutil.cpu_percent(interval=None)
    try:
        while True:
            await asyncio.sleep(2)
            data = _system_snapshot()
            await websocket.send_json(data)
    except WebSocketDisconnect:
        pass
