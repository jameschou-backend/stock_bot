-- Migration 012: 價值因子 + 借券 + 季報資料表
-- TaiwanStockPER / TaiwanStockSecuritiesLending /
-- TaiwanStockBalanceSheet + TaiwanStockFinancialStatements + TaiwanStockCashFlowsStatement

-- ── 1. 本益比/殖利率/本淨比（TaiwanStockPER）────────────────────
-- 每日資料，每股一筆，支援日期區間查詢（Sponsor）
CREATE TABLE IF NOT EXISTS raw_per (
    stock_id        VARCHAR(16)  NOT NULL,
    trading_date    DATE         NOT NULL,
    per             DECIMAL(12, 4),   -- 本益比（Price-to-Earnings Ratio）
    pbr             DECIMAL(12, 4),   -- 本淨比（Price-to-Book Ratio）
    dividend_yield  DECIMAL(10, 6),   -- 現金殖利率（%）
    PRIMARY KEY (stock_id, trading_date),
    INDEX idx_raw_per_trading_date (trading_date)
);

-- ── 2. 借券餘額聚合（TaiwanStockSecuritiesLending）────────────────
-- 原始逐筆借券資料彙整：每日每股借出餘額、平均費率
-- 資料意涵：借券餘額增加 = 機構投資人放空意願上升
CREATE TABLE IF NOT EXISTS raw_securities_lending (
    stock_id            VARCHAR(16)  NOT NULL,
    trading_date        DATE         NOT NULL,
    lending_balance     BIGINT,             -- 借券餘額（張）
    lending_fee_rate    DECIMAL(10, 6),     -- 加權平均借券費率（%年率）
    lending_transaction_count  BIGINT,     -- 當日借出筆數
    PRIMARY KEY (stock_id, trading_date),
    INDEX idx_raw_securities_lending_date (trading_date)
);

-- ── 3. 季報財務摘要（聚合自多個 FinMind dataset）─────────────────
-- 來源：TaiwanStockBalanceSheet + TaiwanStockFinancialStatements + TaiwanStockCashFlowsStatement
-- 約有 60 天公告延遲，使用 available_date = report_date + 60d 做 merge_asof
CREATE TABLE IF NOT EXISTS raw_quarterly_fundamental (
    stock_id            VARCHAR(16)  NOT NULL,
    report_date         DATE         NOT NULL,   -- 財報截止日（季底：0331/0630/0930/1231）
    roe                 DECIMAL(10, 6),           -- 股東權益報酬率（TTM，%）
    roa                 DECIMAL(10, 6),           -- 資產報酬率（%）
    debt_ratio          DECIMAL(10, 6),           -- 負債比率（負債/資產，%）
    operating_margin    DECIMAL(10, 6),           -- 營業利益率（%）
    net_margin          DECIMAL(10, 6),           -- 稅後淨利率（%）
    fcf_per_share       DECIMAL(12, 6),           -- 自由現金流量/股（元）
    PRIMARY KEY (stock_id, report_date),
    INDEX idx_raw_qfund_report_date (report_date)
);
