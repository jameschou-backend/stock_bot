# FinMind Sponsor 請求與效能

所有 `app.finmind.fetch_dataset` 呼叫（pipeline、回補、畫面與專案 MCP）共用本機 SQLite 滑動視窗帳本。Sponsor 上限固定為 6,000 次／小時，預設使用 5,400 次、保留 10%；更低的 `FINMIND_REQUESTS_PER_HOUR` 仍生效。同一視窗內嚴格設定不會被另一工作放寬。重新啟動程式不會清空用量。

帳本預設 `~/.cache/stock-bot/finmind.sqlite3`，跨本機 checkout 共用。測試用 `FINMIND_STATE_PATH` 指定隔離檔案；正式工作應使用相同路徑。不要刪除帳本來清額度。其他機器、官方網頁或其他 MCP 的用量不在本機帳本內；收到供應商 402/429 時，全體本機工作依 Retry-After（未提供則一小時）暫停。

- 每次 HTTP 嘗試（包含重試與失敗）先扣一個許可；cache hit 不扣。
- 額度不足立即拋出 `FinMindQuotaError`，附可重試秒數；不在背景盲等一小時。
- 同一查詢以檔案鎖合併同時請求，重用 HTTP Session，成功且非空資料快取五分鐘。
- `DataFrame.attrs` 保留 `source`、`retrieved_at`、`cache_hit`。`force_refresh=True` 可明確重抓；過期、空值與錯誤不會當作可用快取。
- 成功頁可經 `batch_write_callback` 逐批保存；任何查詢失敗整個抓取會回報不完整。
- 不支援以 `rate_limit=False` 繞過額度。

官方文件只保證個股區間與特定日期全市場查詢，不能把逗號分隔 data_id 當成批次 API。已確認日價、還原價、PER 的全市場日期查詢，依「日期數」與「股票數」選擇較少請求的路徑。月營收按月初 period date 查全市場，每月一個請求；不是公告日，不能當作 point-in-time 可用時間。其他資料集預設單股區間，不暗中更換來源。

來源：[FinMind 技術面](https://finmind.github.io/tutor/TaiwanMarket/Technical/)、[基本面](https://finmind.github.io/tutor/TaiwanMarket/Fundamental/)。分鐘 K 的全市場匯出需要 SponsorPro，不把它當成 Sponsor 能力。

驗證：多程序競爭固定額度、重啟保留、短 timeout、402/429 冷卻、每次重試計費、同時查詢去重、強制刷新、不快取空資料。`tests/test_finmind_budget.py` 不連真實 API。
