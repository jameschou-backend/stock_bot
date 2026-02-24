-- Add strategy_name for trade/position attribution (idempotent)
SET @stmt := (
  SELECT IF(
    EXISTS(
      SELECT 1
      FROM information_schema.columns
      WHERE table_schema = DATABASE()
        AND table_name = 'strategy_trades'
        AND column_name = 'strategy_name'
    ),
    'SELECT 1',
    'ALTER TABLE strategy_trades ADD COLUMN strategy_name VARCHAR(64) NULL AFTER stock_id'
  )
);
PREPARE s FROM @stmt;
EXECUTE s;
DEALLOCATE PREPARE s;

SET @stmt := (
  SELECT IF(
    EXISTS(
      SELECT 1
      FROM information_schema.columns
      WHERE table_schema = DATABASE()
        AND table_name = 'strategy_positions'
        AND column_name = 'strategy_name'
    ),
    'SELECT 1',
    'ALTER TABLE strategy_positions ADD COLUMN strategy_name VARCHAR(64) NULL AFTER stock_id'
  )
);
PREPARE s FROM @stmt;
EXECUTE s;
DEALLOCATE PREPARE s;
