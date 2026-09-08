---
name: stock-data-audit
description: 檢查 stock_bot 台股資料的新鮮度、上市上櫃覆蓋、FinMind 用量與增量抓取問題。
---

先讀 stock-bot MCP `get_data_status`；工具未連線時在專案執行 `make api`，或讀 `app/workbench_service.py` 的共用服務。區分資料正確、策略通過驗證與 API 可連線三件事。

- 對每個市場與日期檢查，不把 MAX(date)、HTTP 200 或 jobs success 當作資料齊全。
- FinMind Sponsor 是 6,000 requests/hour，本專案共用帳本預設使用 5,400。所有 FinMind 查詢走 `app.finmind.fetch_dataset` 或 `query_finmind`。不要另外啟動繞过帳本的 FinMind MCP、curl 循環或自訂 client。
- `FinMindQuotaError` 表示暫停。保留已完成頁，等 retry_after 到期；不要反覆重跑或刪帳本。
- 已確認的全市場日／月查詢優先；分鐘 K 全市場匯出是 SponsorPro 能力。月營收 period date、create_time、正式公告時間不能混用。
- 不讀出 `.env` 內容、token、DB 密碼；不得修改 `.env`。設定範例及用量策略見 `docs/finmind-performance.md`。
- 還原因子全為 1 或最新日期前進不代表對帳成功。修復後重跑受影響特徵與標籤，再驗證當日候選名單。

回覆應指出哪個資料集／市場缺漏、應補哪個期間、預估請求數與實際驗證證據。大量回補前先估算，不盲目開啟所有 Sponsor ingest。
