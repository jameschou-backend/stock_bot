CREATE TABLE IF NOT EXISTS raw_prices (
  stock_id VARCHAR(16) NOT NULL,
  trading_date DATE NOT NULL,
  open DECIMAL(18,6),
  high DECIMAL(18,6),
  low DECIMAL(18,6),
  close DECIMAL(18,6),
  volume BIGINT,
  PRIMARY KEY (stock_id, trading_date),
  INDEX idx_raw_prices_trading_date (trading_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS raw_institutional (
  stock_id VARCHAR(16) NOT NULL,
  trading_date DATE NOT NULL,
  foreign_buy BIGINT,
  foreign_sell BIGINT,
  foreign_net BIGINT,
  trust_buy BIGINT,
  trust_sell BIGINT,
  trust_net BIGINT,
  dealer_buy BIGINT,
  dealer_sell BIGINT,
  dealer_net BIGINT,
  PRIMARY KEY (stock_id, trading_date),
  INDEX idx_raw_inst_trading_date (trading_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS features (
  stock_id VARCHAR(16) NOT NULL,
  trading_date DATE NOT NULL,
  features_json JSON NOT NULL,
  PRIMARY KEY (stock_id, trading_date),
  INDEX idx_features_trading_date (trading_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS labels (
  stock_id VARCHAR(16) NOT NULL,
  trading_date DATE NOT NULL,
  future_ret_h DECIMAL(18,8) NULL,
  PRIMARY KEY (stock_id, trading_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS model_versions (
  model_id VARCHAR(64) NOT NULL,
  train_start DATE,
  train_end DATE,
  feature_set_hash VARCHAR(64),
  params_json JSON,
  metrics_json JSON,
  artifact_path VARCHAR(255),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (model_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS picks (
  pick_date DATE NOT NULL,
  stock_id VARCHAR(16) NOT NULL,
  score DECIMAL(18,8) NOT NULL,
  model_id VARCHAR(64) NOT NULL,
  reason_json JSON NOT NULL,
  PRIMARY KEY (pick_date, stock_id),
  INDEX idx_picks_pick_date (pick_date),
  INDEX idx_picks_score (score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS jobs (
  job_id VARCHAR(64) NOT NULL,
  job_name VARCHAR(64),
  status ENUM('running','success','failed') NOT NULL,
  started_at DATETIME,
  ended_at DATETIME,
  error_text TEXT NULL,
  logs_json JSON NULL,
  PRIMARY KEY (job_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
