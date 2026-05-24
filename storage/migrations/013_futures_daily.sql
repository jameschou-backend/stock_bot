-- Stage 11.0：台指期貨日資料 + 三大法人期貨持倉
-- 用途：作為「市場層級訊號」加入個股 model（不操作期貨）
-- 期貨領先指標：機構期貨倉位 → 5-10 個交易日後現貨方向

-- ── 1. 期貨日資料（OHLC + OI + 結算價）────────────────────────
-- contract_id 主要追蹤 "TX" 大台指近月合約
CREATE TABLE IF NOT EXISTS raw_futures_daily (
    contract_id        VARCHAR(16)  NOT NULL,
    trading_date       DATE         NOT NULL,
    contract_month     VARCHAR(8),                  -- 近月合約識別 例如 "202606"
    open               DECIMAL(12, 2),
    high               DECIMAL(12, 2),
    low                DECIMAL(12, 2),
    close              DECIMAL(12, 2),
    volume             BIGINT,
    open_interest      BIGINT,
    settlement_price   DECIMAL(12, 2),
    PRIMARY KEY (contract_id, trading_date),
    INDEX idx_raw_futures_daily_date (trading_date)
);

-- ── 2. 三大法人期貨持倉（多空淨額）────────────────────────────
-- foreign/trust/dealer 各自 long_oi / short_oi / net_oi
-- net_oi = long_oi - short_oi（正 = 看多，負 = 看空）
CREATE TABLE IF NOT EXISTS raw_futures_inst (
    contract_id        VARCHAR(16)  NOT NULL,
    trading_date       DATE         NOT NULL,
    foreign_long_oi    BIGINT,
    foreign_short_oi   BIGINT,
    foreign_net_oi     BIGINT,
    trust_long_oi      BIGINT,
    trust_short_oi     BIGINT,
    trust_net_oi       BIGINT,
    dealer_long_oi     BIGINT,
    dealer_short_oi    BIGINT,
    dealer_net_oi      BIGINT,
    PRIMARY KEY (contract_id, trading_date),
    INDEX idx_raw_futures_inst_date (trading_date)
);
