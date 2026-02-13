from __future__ import annotations

from datetime import date
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import func

from app.config import load_config
from app.db import get_session
from app.job_utils import cleanup_stale_running_jobs
from app.models import Feature, Job, ModelVersion, Pick, RawInstitutional, RawPrice
from app.strategy_doc import get_selection_logic
from app.schemas import (
    FeatureOut,
    InstitutionalOut,
    JobOut,
    ModelVersionOut,
    PickOut,
    PriceOut,
    StockDetailOut,
)

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
def get_strategy():
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

        price = (
            session.query(RawPrice)
            .filter(RawPrice.stock_id == stock_id, RawPrice.trading_date == target_date)
            .one_or_none()
        )
        inst = (
            session.query(RawInstitutional)
            .filter(RawInstitutional.stock_id == stock_id, RawInstitutional.trading_date == target_date)
            .one_or_none()
        )
        feat = (
            session.query(Feature)
            .filter(Feature.stock_id == stock_id, Feature.trading_date == target_date)
            .one_or_none()
        )
        pick = (
            session.query(Pick)
            .filter(Pick.stock_id == stock_id, Pick.pick_date == target_date)
            .one_or_none()
        )

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
