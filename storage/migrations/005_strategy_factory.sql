-- Strategy Factory tables
CREATE TABLE IF NOT EXISTS strategy_configs (
  config_id VARCHAR(64) PRIMARY KEY,
  name VARCHAR(128) NOT NULL,
  config_json JSON NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS strategy_runs (
  run_id VARCHAR(64) PRIMARY KEY,
  config_id VARCHAR(64) NOT NULL,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  initial_capital DECIMAL(18,6) NOT NULL,
  transaction_cost_pct DECIMAL(10,6) NOT NULL,
  slippage_pct DECIMAL(10,6) NOT NULL,
  metrics_json JSON,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_strategy_runs_config (config_id),
  INDEX idx_strategy_runs_dates (start_date, end_date)
);

CREATE TABLE IF NOT EXISTS strategy_trades (
  run_id VARCHAR(64) NOT NULL,
  trade_id VARCHAR(64) NOT NULL,
  trading_date DATE NOT NULL,
  stock_id VARCHAR(16) NOT NULL,
  action ENUM('BUY','SELL','ADD') NOT NULL,
  qty DECIMAL(18,6) NOT NULL,
  price DECIMAL(18,6) NOT NULL,
  fee DECIMAL(18,6) NOT NULL,
  reason_json JSON,
  PRIMARY KEY (run_id, trade_id),
  INDEX idx_strategy_trades_date (trading_date),
  INDEX idx_strategy_trades_stock (stock_id)
);

CREATE TABLE IF NOT EXISTS strategy_positions (
  run_id VARCHAR(64) NOT NULL,
  trading_date DATE NOT NULL,
  stock_id VARCHAR(16) NOT NULL,
  qty DECIMAL(18,6) NOT NULL,
  avg_cost DECIMAL(18,6) NOT NULL,
  market_value DECIMAL(18,6) NOT NULL,
  unrealized_pnl DECIMAL(18,6) NOT NULL,
  PRIMARY KEY (run_id, trading_date, stock_id),
  INDEX idx_strategy_positions_date (trading_date)
);
