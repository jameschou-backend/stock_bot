-- Stage 11.1: 個股新聞表（FinMind TaiwanStockNews）
-- attention proxy + 後續可加 LLM sentiment

CREATE TABLE IF NOT EXISTS raw_stock_news (
    id              BIGINT PRIMARY KEY AUTO_INCREMENT,
    stock_id        VARCHAR(16)  NOT NULL,
    news_datetime   DATETIME     NOT NULL,
    source          VARCHAR(64),
    title           VARCHAR(500),
    link            VARCHAR(500),
    title_hash      VARCHAR(32),                  -- MD5(title) for dedup
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_raw_news_stock_dt (stock_id, news_datetime),
    INDEX idx_raw_news_dt (news_datetime),
    INDEX idx_raw_news_dedup (stock_id, news_datetime, title_hash)
);
