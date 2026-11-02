-- Runs ONLY in the dedicated US MySQL container's stock_bot_us database.
USE stock_bot_us;
CREATE TABLE IF NOT EXISTS ingest_snapshots (
  snapshot_id CHAR(64) CHARACTER SET ascii PRIMARY KEY,
  dataset VARCHAR(32) CHARACTER SET ascii NOT NULL,
  ticker VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  first_date DATE NOT NULL,
  last_date DATE NOT NULL,
  row_count BIGINT UNSIGNED NOT NULL,
  manifest_path VARCHAR(255) NOT NULL,
  manifest_sha256 CHAR(64) CHARACTER SET ascii NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX ix_lookup (dataset,ticker,first_date,last_date)
) ENGINE=InnoDB;
