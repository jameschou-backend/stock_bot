-- Migration 011: Sponsor 專屬資料表
-- TaiwanStockTradingDailyReport / TaiwanStockHoldingSharesPer /
-- TaiwanStockKBar / TaiwanstockGovernmentBankBuySell / CnnFearGreedIndex

-- ── Priority 1：分點券商聚合 ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw_broker_trades (
    stock_id            VARCHAR(16)  NOT NULL,
    trading_date        DATE         NOT NULL,
    top5_net            BIGINT,
    top5_concentration  DECIMAL(10, 6),
    buy_broker_count    BIGINT,
    sell_broker_count   BIGINT,
    total_net           BIGINT,
    PRIMARY KEY (stock_id, trading_date),
    INDEX idx_raw_broker_trading_date (trading_date)
);

-- ── Priority 2：持股分級週報 ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw_holding_dist (
    stock_id          VARCHAR(16)  NOT NULL,
    trading_date      DATE         NOT NULL,
    large_holder_pct  DECIMAL(10, 4),
    small_holder_pct  DECIMAL(10, 4),
    top_level_pct     DECIMAL(10, 4),
    holder_count      BIGINT,
    PRIMARY KEY (stock_id, trading_date),
    INDEX idx_raw_holding_dist_date (trading_date)
);

-- ── Priority 3：分鐘K線日內特徵 ─────────────────────────────────
CREATE TABLE IF NOT EXISTS raw_kbar_daily (
    stock_id          VARCHAR(16)  NOT NULL,
    trading_date      DATE         NOT NULL,
    morning_ret       DECIMAL(10, 6),
    close_vol_ratio   DECIMAL(10, 6),
    intraday_high_pos DECIMAL(10, 6),
    vwap_dev          DECIMAL(10, 6),
    PRIMARY KEY (stock_id, trading_date),
    INDEX idx_raw_kbar_daily_date (trading_date)
);

-- ── Priority 4：官股銀行 ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw_gov_bank (
    stock_id        VARCHAR(16)  NOT NULL,
    trading_date    DATE         NOT NULL,
    gov_net         BIGINT,
    bank_count_buy  BIGINT,
    bank_count_sell BIGINT,
    PRIMARY KEY (stock_id, trading_date),
    INDEX idx_raw_gov_bank_date (trading_date)
);

-- ── Priority 5：CNN 恐懼貪婪 ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw_fear_greed (
    date    DATE         NOT NULL,
    score   BIGINT,
    rating  VARCHAR(32),
    PRIMARY KEY (date)
);
