CREATE TABLE IF NOT EXISTS validated_holding_dist (
 stock_id VARCHAR(16) NOT NULL,
 trading_date DATE NOT NULL,
 available_date DATE NOT NULL,
 large_holder_pct DECIMAL(10,4),
 small_holder_pct DECIMAL(10,4),
 top_level_pct DECIMAL(10,4),
 holder_count BIGINT,
 PRIMARY KEY(stock_id, trading_date),
 INDEX idx_validated_holding_available(available_date)
);
