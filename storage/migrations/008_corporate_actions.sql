-- 008_corporate_actions.sql
-- 公司行為與價格還原因子（供特徵與回測使用）

CREATE TABLE IF NOT EXISTS corporate_actions (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  stock_id VARCHAR(16) NOT NULL,
  action_date DATE NOT NULL,
  action_type VARCHAR(32) NOT NULL,  -- DIVIDEND|SPLIT|MERGE|RIGHTS|OTHER
  adj_factor DECIMAL(18,8) NULL,     -- 若可推導，提供單筆事件調整係數
  payload_json JSON NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_ca_stock_date (stock_id, action_date),
  INDEX idx_ca_action_date (action_date),
  INDEX idx_ca_action_type (action_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 每日累積還原因子（adj_close = close * adj_factor）
CREATE TABLE IF NOT EXISTS price_adjust_factors (
  stock_id VARCHAR(16) NOT NULL,
  trading_date DATE NOT NULL,
  adj_factor DECIMAL(18,8) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (stock_id, trading_date),
  INDEX idx_paf_trading_date (trading_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
