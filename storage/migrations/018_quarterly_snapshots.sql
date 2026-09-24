CREATE TABLE IF NOT EXISTS quarterly_fundamental_snapshots (
 stock_id VARCHAR(16) NOT NULL,
 report_date DATE NOT NULL,
 observed_at DATETIME(6) NOT NULL,
 available_date DATE NOT NULL,
 source_sha256 VARCHAR(64) NOT NULL,
 source_manifest VARCHAR(255) NOT NULL,
 definition_version VARCHAR(32) NOT NULL,
 timing_basis VARCHAR(48) NOT NULL,
 missing_metrics VARCHAR(255) NOT NULL,
 roe_ttm DECIMAL(18,6),
 roa_ttm DECIMAL(18,6),
 debt_ratio DECIMAL(18,6),
 operating_margin DECIMAL(18,6),
 net_margin DECIMAL(18,6),
 fcf_ttm DECIMAL(24,6),
 fcf_per_share DECIMAL(18,6),
 PRIMARY KEY(stock_id, report_date, observed_at),
 INDEX idx_quarterly_snapshot_available(available_date)
);
CREATE TABLE IF NOT EXISTS quarterly_ingest_state (
 stock_id VARCHAR(16) NOT NULL PRIMARY KEY,
 checked_at DATETIME(6) NOT NULL
);
