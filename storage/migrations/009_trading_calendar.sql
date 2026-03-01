-- 009_trading_calendar.sql
-- 交易日曆（支援 FULL/HALF/CLOSED session）

CREATE TABLE IF NOT EXISTS trading_calendar (
  trading_date DATE NOT NULL,
  is_open TINYINT(1) NOT NULL DEFAULT 0,
  session_type VARCHAR(16) NOT NULL DEFAULT 'CLOSED',  -- FULL|HALF|CLOSED
  note VARCHAR(255) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (trading_date),
  INDEX idx_trading_calendar_is_open (is_open),
  INDEX idx_trading_calendar_session_type (session_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
