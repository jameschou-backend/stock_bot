from __future__ import annotations

from sqlalchemy import BigInteger, Column, Date, DateTime, Enum, Index, String, Text
from sqlalchemy.dialects.mysql import DECIMAL, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class RawPrice(Base):
    __tablename__ = "raw_prices"

    stock_id = Column(String(16), primary_key=True)
    trading_date = Column(Date, primary_key=True)
    open = Column(DECIMAL(18, 6))
    high = Column(DECIMAL(18, 6))
    low = Column(DECIMAL(18, 6))
    close = Column(DECIMAL(18, 6))
    volume = Column(BigInteger)

    __table_args__ = (Index("idx_raw_prices_trading_date", "trading_date"),)


class RawInstitutional(Base):
    __tablename__ = "raw_institutional"

    stock_id = Column(String(16), primary_key=True)
    trading_date = Column(Date, primary_key=True)
    foreign_buy = Column(BigInteger)
    foreign_sell = Column(BigInteger)
    foreign_net = Column(BigInteger)
    trust_buy = Column(BigInteger)
    trust_sell = Column(BigInteger)
    trust_net = Column(BigInteger)
    dealer_buy = Column(BigInteger)
    dealer_sell = Column(BigInteger)
    dealer_net = Column(BigInteger)

    __table_args__ = (Index("idx_raw_inst_trading_date", "trading_date"),)


class Feature(Base):
    __tablename__ = "features"

    stock_id = Column(String(16), primary_key=True)
    trading_date = Column(Date, primary_key=True)
    features_json = Column(JSON, nullable=False)

    __table_args__ = (Index("idx_features_trading_date", "trading_date"),)


class Label(Base):
    __tablename__ = "labels"

    stock_id = Column(String(16), primary_key=True)
    trading_date = Column(Date, primary_key=True)
    future_ret_h = Column(DECIMAL(18, 8), nullable=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    model_id = Column(String(64), primary_key=True)
    train_start = Column(Date)
    train_end = Column(Date)
    feature_set_hash = Column(String(64))
    params_json = Column(JSON)
    metrics_json = Column(JSON)
    artifact_path = Column(String(255))
    created_at = Column(DateTime, server_default=func.now())


class Pick(Base):
    __tablename__ = "picks"

    pick_date = Column(Date, primary_key=True)
    stock_id = Column(String(16), primary_key=True)
    score = Column(DECIMAL(18, 8), nullable=False)
    model_id = Column(String(64), nullable=False)
    reason_json = Column(JSON, nullable=False)

    __table_args__ = (
        Index("idx_picks_pick_date", "pick_date"),
        Index("idx_picks_score", "score"),
    )


class Job(Base):
    __tablename__ = "jobs"

    job_id = Column(String(64), primary_key=True)
    job_name = Column(String(64))
    status = Column(Enum("running", "success", "failed"), nullable=False)
    started_at = Column(DateTime)
    ended_at = Column(DateTime)
    error_text = Column(Text)
    logs_json = Column(JSON)
