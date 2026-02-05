-- 003_margin_short.sql
-- 融資融券表：儲存個股融資融券日頻資料

CREATE TABLE IF NOT EXISTS raw_margin_short (
  stock_id VARCHAR(16) NOT NULL,
  trading_date DATE NOT NULL,
  -- 融資（Margin Purchase）
  margin_purchase_buy BIGINT,           -- 融資買進
  margin_purchase_sell BIGINT,          -- 融資賣出
  margin_purchase_cash_repay BIGINT,    -- 融資現金償還
  margin_purchase_limit BIGINT,         -- 融資限額
  margin_purchase_balance BIGINT,       -- 融資餘額（張）
  -- 融券（Short Sale）
  short_sale_buy BIGINT,                -- 融券買進（回補）
  short_sale_sell BIGINT,               -- 融券賣出
  short_sale_cash_repay BIGINT,         -- 融券現券償還
  short_sale_limit BIGINT,              -- 融券限額
  short_sale_balance BIGINT,            -- 融券餘額（張）
  -- 資券互抵
  offset_loan_and_short BIGINT,         -- 資券互抵
  -- 備註
  note VARCHAR(255),
  PRIMARY KEY (stock_id, trading_date),
  INDEX idx_raw_margin_trading_date (trading_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
