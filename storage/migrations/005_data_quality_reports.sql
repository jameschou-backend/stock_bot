-- 005_data_quality_reports.sql
-- Data quality 每日報表落地，供 dashboard 與追蹤使用

CREATE TABLE IF NOT EXISTS data_quality_reports (
  report_date DATE NOT NULL,
  table_name VARCHAR(64) NOT NULL,
  expected_rows BIGINT NULL,
  actual_rows BIGINT NOT NULL,
  missing_ratio DOUBLE NULL,
  max_trading_date DATE NULL,
  notes TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (report_date, table_name),
  INDEX idx_dq_reports_table_date (table_name, report_date),
  INDEX idx_dq_reports_report_date (report_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
