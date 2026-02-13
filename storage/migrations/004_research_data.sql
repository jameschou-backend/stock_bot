-- 004_research_data.sql
-- 研究用資料表：基本面（月營收）與題材/金流（產業聚合）

CREATE TABLE IF NOT EXISTS raw_fundamentals (
  stock_id VARCHAR(16) NOT NULL,
  trading_date DATE NOT NULL,
  revenue_current_month BIGINT NULL,
  revenue_last_month BIGINT NULL,
  revenue_last_year BIGINT NULL,
  revenue_mom DECIMAL(18,8) NULL,
  revenue_yoy DECIMAL(18,8) NULL,
  PRIMARY KEY (stock_id, trading_date),
  INDEX idx_raw_fundamentals_trading_date (trading_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS raw_theme_flow (
  theme_id VARCHAR(64) NOT NULL,
  trading_date DATE NOT NULL,
  turnover_amount DECIMAL(20,2) NULL,
  turnover_ratio DECIMAL(18,8) NULL,
  theme_return_5 DECIMAL(18,8) NULL,
  theme_return_20 DECIMAL(18,8) NULL,
  hot_score DECIMAL(18,8) NULL,
  PRIMARY KEY (theme_id, trading_date),
  INDEX idx_raw_theme_flow_trading_date (trading_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
