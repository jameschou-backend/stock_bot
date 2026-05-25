-- Stage 11.2: News sentiment 情感分數（LLM 分析結果）
-- Phase 2 設計：3.4M news → Claude Haiku batch → sentiment_score (-1/0/+1)

CREATE TABLE IF NOT EXISTS news_sentiment (
    news_id          BIGINT       NOT NULL,          -- FK to raw_stock_news.id
    sentiment_score  TINYINT      NOT NULL,          -- -1 利空 / 0 中性 / +1 利多
    confidence       DECIMAL(4,3),                   -- 0.0~1.0 LLM 信心
    llm_model        VARCHAR(32),                    -- 'claude-haiku-4.5' / 'gemini-flash'
    analyzed_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (news_id),
    INDEX idx_news_sent_score (sentiment_score)
);
