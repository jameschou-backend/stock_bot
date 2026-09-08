CREATE TABLE IF NOT EXISTS workbench_accounts (
 account_id VARCHAR(32) PRIMARY KEY,
 initial_cash DECIMAL(18,2) NOT NULL,
 created_at DATETIME NOT NULL
);
CREATE TABLE IF NOT EXISTS workbench_plans (
 plan_id VARCHAR(32) PRIMARY KEY,
 account_id VARCHAR(32) NOT NULL,
 stock_id VARCHAR(4) NOT NULL,
 entry_price DECIMAL(18,4) NOT NULL,
 stop_price DECIMAL(18,4) NOT NULL,
 qty INT NOT NULL,
 filled_qty INT NOT NULL DEFAULT 0,
 status VARCHAR(16) NOT NULL DEFAULT 'open',
 reason VARCHAR(500) NOT NULL DEFAULT '',
 created_at DATETIME NOT NULL,
 INDEX ix_workbench_plans_account_id (account_id),
 FOREIGN KEY (account_id) REFERENCES workbench_accounts(account_id)
);
CREATE TABLE IF NOT EXISTS workbench_fills (
 fill_id VARCHAR(32) PRIMARY KEY,
 sequence_no INT NOT NULL,
 account_id VARCHAR(32) NOT NULL,
 plan_id VARCHAR(32),
 stock_id VARCHAR(4) NOT NULL,
 side VARCHAR(4) NOT NULL,
 qty INT NOT NULL,
 price DECIMAL(18,4) NOT NULL,
 fee DECIMAL(18,2) NOT NULL,
 tax DECIMAL(18,2) NOT NULL,
 executed_at DATETIME(6) NOT NULL,
 created_at DATETIME(6) NOT NULL,
 UNIQUE (account_id, sequence_no),
 INDEX ix_workbench_fills_account_id (account_id),
 FOREIGN KEY (account_id) REFERENCES workbench_accounts(account_id),
 FOREIGN KEY (plan_id) REFERENCES workbench_plans(plan_id)
);
