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


class RawFundamental(Base):
    """基本面原始資料表（月營收）"""
    __tablename__ = "raw_fundamentals"

    stock_id = Column(String(16), primary_key=True)
    trading_date = Column(Date, primary_key=True)  # 使用月份對應日期（通常為月初）
    revenue_current_month = Column(BigInteger)
    revenue_last_month = Column(BigInteger)
    revenue_last_year = Column(BigInteger)
    revenue_mom = Column(DECIMAL(18, 8))
    revenue_yoy = Column(DECIMAL(18, 8))

    __table_args__ = (Index("idx_raw_fundamentals_trading_date", "trading_date"),)


class RawThemeFlow(Base):
    """題材/金流聚合表（以產業為主題）"""
    __tablename__ = "raw_theme_flow"

    theme_id = Column(String(64), primary_key=True)  # 例如 industry_category
    trading_date = Column(Date, primary_key=True)
    turnover_amount = Column(DECIMAL(20, 2))  # 主題總成交值（元）
    turnover_ratio = Column(DECIMAL(18, 8))  # 主題成交值占全市場比例
    theme_return_5 = Column(DECIMAL(18, 8))
    theme_return_20 = Column(DECIMAL(18, 8))
    hot_score = Column(DECIMAL(18, 8))  # 綜合熱度分數

    __table_args__ = (Index("idx_raw_theme_flow_trading_date", "trading_date"),)


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


class StrategyConfig(Base):
    __tablename__ = "strategy_configs"

    config_id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    config_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class StrategyRun(Base):
    __tablename__ = "strategy_runs"

    run_id = Column(String(64), primary_key=True)
    config_id = Column(String(64), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    initial_capital = Column(DECIMAL(18, 6), nullable=False)
    transaction_cost_pct = Column(DECIMAL(10, 6), nullable=False)
    slippage_pct = Column(DECIMAL(10, 6), nullable=False)
    metrics_json = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_strategy_runs_config", "config_id"),
        Index("idx_strategy_runs_dates", "start_date", "end_date"),
    )


class StrategyTrade(Base):
    __tablename__ = "strategy_trades"

    run_id = Column(String(64), primary_key=True)
    trade_id = Column(String(64), primary_key=True)
    trading_date = Column(Date, nullable=False)
    stock_id = Column(String(16), nullable=False)
    action = Column(String(8), nullable=False)
    qty = Column(DECIMAL(18, 6), nullable=False)
    price = Column(DECIMAL(18, 6), nullable=False)
    fee = Column(DECIMAL(18, 6), nullable=False)
    reason_json = Column(JSON)

    __table_args__ = (
        Index("idx_strategy_trades_date", "trading_date"),
        Index("idx_strategy_trades_stock", "stock_id"),
    )


class StrategyPosition(Base):
    __tablename__ = "strategy_positions"

    run_id = Column(String(64), primary_key=True)
    trading_date = Column(Date, primary_key=True)
    stock_id = Column(String(16), primary_key=True)
    qty = Column(DECIMAL(18, 6), nullable=False)
    avg_cost = Column(DECIMAL(18, 6), nullable=False)
    market_value = Column(DECIMAL(18, 6), nullable=False)
    unrealized_pnl = Column(DECIMAL(18, 6), nullable=False)

    __table_args__ = (Index("idx_strategy_positions_date", "trading_date"),)
