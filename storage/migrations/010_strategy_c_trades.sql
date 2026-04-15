-- Strategy C 每日進出場稽核 log（append-only）
CREATE TABLE IF NOT EXISTS strategy_c_trades (
    id             BIGINT        NOT NULL AUTO_INCREMENT,
    run_date       DATE          NOT NULL COMMENT '執行日期（今天）',
    stock_id       VARCHAR(16)   NOT NULL,
    action         VARCHAR(16)   NOT NULL COMMENT 'buy / sell / hold / skip',
    entry_date     DATE          COMMENT '實際進場日（buy 時設定）',
    entry_score    DECIMAL(18,6) COMMENT '進場時模型分數',
    days_held      BIGINT        COMMENT '已持有天數（sell/hold 時）',
    exit_reason    VARCHAR(64)   COMMENT '出場原因（sell 時）',
    amount         DECIMAL(18,2) COMMENT '交易金額（正=買入，負=賣出）',
    score_today    DECIMAL(18,6) COMMENT '今日模型分數',
    pct_to_breakthrough DECIMAL(18,4) COMMENT '距突破點距離（正=尚未突破）',
    created_at     DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX ix_sc_trades_run_date (run_date),
    INDEX ix_sc_trades_stock (stock_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
