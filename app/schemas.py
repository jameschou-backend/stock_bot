from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class PriceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    stock_id: str
    trading_date: date
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None


class InstitutionalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    stock_id: str
    trading_date: date
    foreign_buy: Optional[int] = None
    foreign_sell: Optional[int] = None
    foreign_net: Optional[int] = None
    trust_buy: Optional[int] = None
    trust_sell: Optional[int] = None
    trust_net: Optional[int] = None
    dealer_buy: Optional[int] = None
    dealer_sell: Optional[int] = None
    dealer_net: Optional[int] = None


class FeatureOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    stock_id: str
    trading_date: date
    features_json: Dict[str, Any]


class PickOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    pick_date: date
    stock_id: str
    score: float
    model_id: str
    reason_json: Dict[str, Any]


class ModelVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: str
    train_start: Optional[date] = None
    train_end: Optional[date] = None
    feature_set_hash: Optional[str] = None
    params_json: Optional[Dict[str, Any]] = None
    metrics_json: Optional[Dict[str, Any]] = None
    artifact_path: Optional[str] = None
    created_at: Optional[datetime] = None


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: str
    job_name: Optional[str] = None
    status: str
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    error_text: Optional[str] = None
    logs_json: Optional[Dict[str, Any]] = None


class StockDetailOut(BaseModel):
    stock_id: str
    trading_date: date
    price: Optional[PriceOut] = None
    institutional: Optional[InstitutionalOut] = None
    features: Optional[FeatureOut] = None
    pick: Optional[PickOut] = None
