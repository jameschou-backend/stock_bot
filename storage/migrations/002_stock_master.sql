-- 002_stock_master.sql
-- Stock Master 表：管理股票基本資料與狀態歷史

-- stocks 主表：儲存股票基本資料
CREATE TABLE IF NOT EXISTS stocks (
  stock_id VARCHAR(16) NOT NULL PRIMARY KEY,
  name VARCHAR(64),
  market VARCHAR(16),               -- TWSE/TPEX/等
  is_listed TINYINT(1) DEFAULT 1,   -- 是否仍上市/櫃
  listed_date DATE,                 -- 上市日期
  delisted_date DATE,               -- 下市日期（若有）
  industry_category VARCHAR(64),    -- 產業類別
  security_type VARCHAR(16),        -- stock/etf/warrant/其他
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_stocks_market (market),
  INDEX idx_stocks_security_type (security_type),
  INDEX idx_stocks_is_listed (is_listed)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- stock_status_history：股票狀態變更歷史（下市/更名/市場異動等）
CREATE TABLE IF NOT EXISTS stock_status_history (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  stock_id VARCHAR(16) NOT NULL,
  effective_date DATE NOT NULL,
  status_type VARCHAR(32),          -- listed/delisted/rename/market_change/...
  payload_json JSON,                -- 額外資訊（如更名前後名稱等）
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_ssh_stock_date (stock_id, effective_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
