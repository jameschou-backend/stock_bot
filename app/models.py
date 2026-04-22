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


class CorporateAction(Base):
    """公司行為事件（除權息/分割/合併/增資等）"""
    __tablename__ = "corporate_actions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(String(16), nullable=False)
    action_date = Column(Date, nullable=False)
    action_type = Column(String(32), nullable=False)
    adj_factor = Column(DECIMAL(18, 8))
    payload_json = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_ca_stock_date", "stock_id", "action_date"),
        Index("idx_ca_action_date", "action_date"),
        Index("idx_ca_action_type", "action_type"),
    )


class PriceAdjustFactor(Base):
    """每日累積還原因子（adj_close = close * adj_factor）"""
    __tablename__ = "price_adjust_factors"

    stock_id = Column(String(16), primary_key=True)
    trading_date = Column(Date, primary_key=True)
    adj_factor = Column(DECIMAL(18, 8), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (Index("idx_paf_trading_date", "trading_date"),)


class TradingCalendar(Base):
    """交易日曆（FULL/HALF/CLOSED）"""
    __tablename__ = "trading_calendar"

    trading_date = Column(Date, primary_key=True)
    is_open = Column(Boolean, nullable=False, default=False)
    session_type = Column(String(16), nullable=False, default="CLOSED")
    note = Column(String(255))
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_trading_calendar_is_open", "is_open"),
        Index("idx_trading_calendar_session_type", "session_type"),
    )


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

    __table_args__ = (
        Index("ix_labels_trading_date", "trading_date"),
    )


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
    strategy_name = Column(String(64))
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
    strategy_name = Column(String(64))
    qty = Column(DECIMAL(18, 6), nullable=False)
    avg_cost = Column(DECIMAL(18, 6), nullable=False)
    market_value = Column(DECIMAL(18, 6), nullable=False)
    unrealized_pnl = Column(DECIMAL(18, 6), nullable=False)

    __table_args__ = (Index("idx_strategy_positions_date", "trading_date"),)


# ─────────────────────────────────────────────────────────────
# Priority 1：分點券商追蹤（TaiwanStockTradingDailyReport）
# ─────────────────────────────────────────────────────────────
class RawBrokerTrade(Base):
    """分點券商每日買賣超聚合（Sponsor 專屬）

    不存原始逐筆資料（每日數百萬行），改存計算好的聚合指標：
    - top5_net：前5大淨買超分點的淨買超合計（張）
    - top5_concentration：前5大分點淨買超 / 全部分點 |淨買超| 之比（0-1）
    - buy_broker_count：今日淨買超分點數
    - sell_broker_count：今日淨賣超分點數
    - total_net：全部分點合計淨買超（應≈三大法人外資）
    """
    __tablename__ = "raw_broker_trades"

    stock_id       = Column(String(16), primary_key=True)
    trading_date   = Column(Date,       primary_key=True)
    top5_net       = Column(BigInteger)        # Top-5 分點淨買超（張）
    top5_concentration = Column(DECIMAL(10, 6))  # Top-5 集中度（0-1）
    buy_broker_count   = Column(BigInteger)    # 淨買超分點數
    sell_broker_count  = Column(BigInteger)    # 淨賣超分點數
    total_net      = Column(BigInteger)        # 全部分點合計淨買超（張）

    __table_args__ = (Index("idx_raw_broker_trading_date", "trading_date"),)


# ─────────────────────────────────────────────────────────────
# Priority 2：持股分級（TaiwanStockHoldingSharesPer）
# ─────────────────────────────────────────────────────────────
class RawHoldingDist(Base):
    """持股分級週報（每週五公布，反映散戶/大戶結構）

    - large_holder_pct：大戶（≥1000張）持股百分比（%）
    - small_holder_pct：散戶（<1000張）持股百分比（%）
    - top_level_pct：最大持股級別（>10萬張）百分比（%）
    - holder_count：總股東人數
    """
    __tablename__ = "raw_holding_dist"

    stock_id         = Column(String(16), primary_key=True)
    trading_date     = Column(Date,       primary_key=True)  # 公布日（通常週五）
    large_holder_pct = Column(DECIMAL(10, 4))  # 大戶持股 % (>=1000張)
    small_holder_pct = Column(DECIMAL(10, 4))  # 散戶持股 % (<1000張)
    top_level_pct    = Column(DECIMAL(10, 4))  # 頂級大戶持股 % (>10萬張或最高級)
    holder_count     = Column(BigInteger)       # 總股東人數

    __table_args__ = (Index("idx_raw_holding_dist_date", "trading_date"),)


# ─────────────────────────────────────────────────────────────
# Priority 3：分鐘 K 線日內聚合特徵（TaiwanStockKBar）
# ─────────────────────────────────────────────────────────────
class RawKBarDaily(Base):
    """分鐘K線日內特徵聚合（每日 1 筆，避免存海量分鐘資料）

    - morning_ret：開盤後 30 分鐘累積報酬（09:01-09:30）
    - close_vol_ratio：尾盤 30 分鐘成交量 / 日總成交量
    - intraday_high_pos：收盤相對日內高低點位置（0=最低, 1=最高）
    - vwap_dev：收盤偏離 VWAP 程度（(close-vwap)/vwap）
    """
    __tablename__ = "raw_kbar_daily"

    stock_id          = Column(String(16), primary_key=True)
    trading_date      = Column(Date,       primary_key=True)
    morning_ret       = Column(DECIMAL(10, 6))  # 早盤 30 分鐘報酬
    close_vol_ratio   = Column(DECIMAL(10, 6))  # 尾盤量佔比
    intraday_high_pos = Column(DECIMAL(10, 6))  # 收盤位置（0-1）
    vwap_dev          = Column(DECIMAL(10, 6))  # 收盤偏離 VWAP

    __table_args__ = (Index("idx_raw_kbar_daily_date", "trading_date"),)


# ─────────────────────────────────────────────────────────────
# Priority 4：官股銀行買賣超（TaiwanstockGovernmentBankBuySell）
# ─────────────────────────────────────────────────────────────
class RawGovBank(Base):
    """8大官股銀行每日合計淨買超（護盤/賣壓訊號）

    - gov_net：8 大行庫合計淨買超（張），正=護盤買入，負=倒貨
    - bank_count_buy：今日淨買超行庫數
    - bank_count_sell：今日淨賣超行庫數
    """
    __tablename__ = "raw_gov_bank"

    stock_id        = Column(String(16), primary_key=True)
    trading_date    = Column(Date,       primary_key=True)
    gov_net         = Column(BigInteger)   # 8行庫合計淨買超（張）
    bank_count_buy  = Column(BigInteger)   # 買超行庫數
    bank_count_sell = Column(BigInteger)   # 賣超行庫數

    __table_args__ = (Index("idx_raw_gov_bank_date", "trading_date"),)


# ─────────────────────────────────────────────────────────────
# Priority 5：CNN 恐懼貪婪指數（CnnFearGreedIndex）
# ─────────────────────────────────────────────────────────────
class RawFearGreed(Base):
    """CNN 恐懼貪婪指數（全球情緒指標，市場層級，非個股）

    - score：恐懼貪婪分數 0-100（0=極度恐懼，100=極度貪婪）
    - rating：文字評級（Extreme Fear/Fear/Neutral/Greed/Extreme Greed）
    """
    __tablename__ = "raw_fear_greed"

    date   = Column(Date, primary_key=True)
    score  = Column(BigInteger)       # 0-100
    rating = Column(String(32))       # Extreme Fear / Fear / Neutral / Greed / Extreme Greed

    __table_args__ = ()


class StrategyCTrade(Base):
    """Strategy C 每日進出場稽核 log（append-only）"""
    __tablename__ = "strategy_c_trades"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_date = Column(Date, nullable=False)          # 執行日期（今天）
    stock_id = Column(String(16), nullable=False)
    action = Column(String(16), nullable=False)       # buy / sell / hold / skip
    entry_date = Column(Date)                         # 實際進場日（buy 時設定）
    entry_score = Column(DECIMAL(18, 6))              # 進場時模型分數
    days_held = Column(BigInteger)                    # 已持有天數（sell/hold 時）
    exit_reason = Column(String(64))                  # 出場原因（sell 時）
    amount = Column(DECIMAL(18, 2))                   # 交易金額（正=買入，負=賣出）
    score_today = Column(DECIMAL(18, 6))              # 今日模型分數
    pct_to_breakthrough = Column(DECIMAL(18, 4))      # 距突破點距離（% ，正=尚未突破）
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_sc_trades_run_date", "run_date"),
        Index("ix_sc_trades_stock", "stock_id"),
    )
