-- Additive repair: the legacy raw_securities_lending table is preserved.
CREATE TABLE IF NOT EXISTS securities_lending_days (
    trading_date DATE PRIMARY KEY,
    observed_at DATETIME NOT NULL,
    source_rows BIGINT NOT NULL,
    stock_rows BIGINT NOT NULL,
    payload_sha256 VARCHAR(64) NOT NULL,
    payload_json JSON NOT NULL
);
CREATE TABLE IF NOT EXISTS securities_lending_daily (
    stock_id VARCHAR(16) NOT NULL,
    trading_date DATE NOT NULL,
    transaction_volume BIGINT NOT NULL,
    transaction_count BIGINT NOT NULL,
    fee_rate DECIMAL(14,8) NOT NULL,
    methods_json JSON NOT NULL,
    PRIMARY KEY (stock_id, trading_date),
    INDEX idx_lending_daily_date (trading_date)
);
