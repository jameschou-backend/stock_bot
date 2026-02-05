from __future__ import annotations

from sqlalchemy import BigInteger, Boolean, Column, Date, DateTime, Enum, Index, String, Text
from sqlalchemy.dialects.mysql import DECIMAL, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Stock(Base):
    """股票主檔表"""
    __tablename__ = "stocks"

    stock_id = Column(String(16), primary_key=True)
    name = Column(String(64))
    market = Column(String(16))  # TWSE/TPEX/...
    is_listed = Column(Boolean, default=True)
    listed_date = Column(Date)
    delisted_date = Column(Date)
    industry_category = Column(String(64))
    security_type = Column(String(16))  # stock/etf/warrant/...
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_stocks_market", "market"),
        Index("idx_stocks_security_type", "security_type"),
        Index("idx_stocks_is_listed", "is_listed"),
    )


class StockStatusHistory(Base):
    """股票狀態變更歷史表"""
    __tablename__ = "stock_status_history"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(String(16), nullable=False)
    effective_date = Column(Date, nullable=False)
    status_type = Column(String(32))  # listed/delisted/rename/...
    payload_json = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (Index("idx_ssh_stock_date", "stock_id", "effective_date"),)


class RawMarginShort(Base):
    """融資融券表"""
    __tablename__ = "raw_margin_short"

    stock_id = Column(String(16), primary_key=True)
    trading_date = Column(Date, primary_key=True)
    # 融資
    margin_purchase_buy = Column(BigInteger)
    margin_purchase_sell = Column(BigInteger)
    margin_purchase_cash_repay = Column(BigInteger)
    margin_purchase_limit = Column(BigInteger)
    margin_purchase_balance = Column(BigInteger)
    # 融券
    short_sale_buy = Column(BigInteger)
    short_sale_sell = Column(BigInteger)
    short_sale_cash_repay = Column(BigInteger)
    short_sale_limit = Column(BigInteger)
    short_sale_balance = Column(BigInteger)
    # 資券互抵
    offset_loan_and_short = Column(BigInteger)
    note = Column(String(255))

    __table_args__ = (Index("idx_raw_margin_trading_date", "trading_date"),)


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
