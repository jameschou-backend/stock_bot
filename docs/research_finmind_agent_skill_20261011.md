# FinMind Agent Skill 與既有研究工具比較

查核日期：2026-10-11（Asia/Taipei）。範圍是官方文件／原始碼和本機實作核對；未安裝官方 Skill/MCP、未修改 `AGENTS.md`／`.env`、未新增回測或策略訊號。文件查證本身沒有 FinMind 市場資料 API 請求；主代理另做的例行 pipeline 驗收用量另計。

**結論：有參考價值，但我們已經透過自己的 stock-bot MCP 使用 FinMind。圖中的 Skill 主要改善自然語言查資料的便利性，不會額外解鎖方案權限，也不是一套已驗證的獲利策略。適合吸收資料目錄、欄位語義與查詢規格，保留目前共用額度入口。**

## 現在已有什麼

| 項目 | 本次核對結果 |
|---|---|
| 目前啟用的 MCP | 主代理本次只讀 Codex 設定的 server 名稱：有 `stock-bot`，沒有官方 `finmind`；實際 `get_data_status` 呼叫成功。這是本機當次狀態，不只依舊文件推論。 |
| 專案 MCP | 9 個工具，涵蓋資料狀態、候選、持倉、計畫、策略證據、新聞研究、族群資金研究、成交檢討、FinMind 查詢。定義見 [stock_bot_mcp.py](/Users/james.chou/JamesProject/stock_bot/scripts/stock_bot_mcp.py:30)。 |
| 對話式原始資料查詢 | 僅開放日價、還原價、PER、月營收、分鐘 K 共 5 類；單股、起迄差最多 31 天，分鐘 K 限一天；最多回傳 1,000 列並標示截斷。不是整個專案僅用了這 5 類。[MCP](/Users/james.chou/JamesProject/stock_bot/scripts/stock_bot_mcp.py:83)、[API](/Users/james.chou/JamesProject/stock_bot/app/workbench_api.py:134) |
| 程式端資料入口 | `fetch_dataset` 已有快取、跨程序去重、請求計數、錯誤／額度處理；另處理即時 snapshot 與分點區間統計專用接口。[finmind.py](/Users/james.chou/JamesProject/stock_bot/app/finmind.py:84) |
| 本次資料狀態 | 主代理讀取回報：行情到 2026-10-08、`data_ready=true`。這不能推導其他資料集皆齊全、策略已可實戰，或官方 Skill 已安裝。 |

既有設計由 [工作台文件](/Users/james.chou/JamesProject/stock_bot/docs/workbench.md:67) 記錄：以專案 MCP 共用資料與額度。文件工具表只有 7 項，應以現行原始碼的 9 項為準。

## 截圖的 Skill 實際提供什麼

官方 Agent Skill 是 Markdown 指令檔，提供自然語言到資料集的對照、參數、認證、輸出格式與示例；查詢時由 AI 執行 Python。官方 MCP 則提供可呼叫的工具與內建資料目錄。二者都使用既有 FinMind API，不是另一批獨立市場資料。官方頁宣傳的「75+」是資料集範圍，不是 75 套選股策略。[Agent Skill 文件](https://finmind.github.io/tutor/ai/AgentSkill/)、[MCP 文件](https://finmind.github.io/tutor/ai/Mcp/)

Skill 範例直接 `requests.get`，使用 Bearer 認證查 `/api/v4/data`；另列需要專用 endpoint 的資料。檔案本身沒有接上 stock_bot 的 SQLite 帳本、快取或研究封存；不能因讀了說明就宣稱相關防護已生效。[官方指令檔，固定版本](https://github.com/FinMind/FinMind/blob/8b42547c1e34f77823dd8d3836c1fc1388bc5678/.claude/commands/finmind.md#L72-L100)

| 比較 | 官方範例／本次查到的 MCP 實作 | stock_bot 現有實作 |
|---|---|---|
| 查詢範圍 | 官方 MCP 有通用 dataset 查詢、分點專用查詢、目錄／方案／股票資料工具 | 對話 MCP 以有限白名單對外，研究程式可另經共用入口接資料 |
| HTTP 路徑 | 官方 MCP 每次 `_get_json` 建立 `httpx.AsyncClient`，直連 FinMind | MCP → 本機 `/workbench/finmind` → `fetch_dataset` |
| 用量／快取 | 本次查到的官方 client 沒有專案共用帳本或結果快取；識別 402/429 不等於已協調其他工作 | 每次 HTTP 嘗試先記帳；成功非空回應預設快取 300 秒，同查詢合併；402/429 共用冷卻 |
| 批次 | 通用查詢不代表所有資料集都支援全市場 | 日價／還原價／PER／營收有明確批次規劃，其餘不猜逗號代號支援 |

上表來源：[官方 MCP client.py 固定版本](https://github.com/FinMind/FinMind-MCP/blob/eae5b2a0779a33882e6efbc47ad4ec03265ffdb6/src/finmind_mcp/client.py#L95-L123)、[tools.py](https://github.com/FinMind/FinMind-MCP/blob/eae5b2a0779a33882e6efbc47ad4ec03265ffdb6/src/finmind_mcp/tools.py)、[本機 fetch_dataset](/Users/james.chou/JamesProject/stock_bot/app/finmind.py:138)、[本機批次規劃](/Users/james.chou/JamesProject/stock_bot/app/finmind.py:289)、[共用 limiter](/Users/james.chou/JamesProject/stock_bot/app/rate_limiter.py:38)。

推論：直接多裝一個原版官方 MCP，會多出一條本機帳本看不到的請求路徑；不是更省額度。現有 limiter 上限 6,000，還會套用設定與 10% 緩衝，不能把方案上限當成本機可用預算。本次主代理讀得有效限制 4,860／小時；未因調研放寬。[效能文件](/Users/james.chou/JamesProject/stock_bot/docs/finmind-performance.md:3)

## 圖片中的安裝方式為何不能原封照貼

截圖及官方 Codex 範例把遠端內容輸出至 `AGENTS.md`。對已有同名檔案的本專案，該操作會覆蓋原本的 secrets、額度、回測與提交流程規範；而主檔還引用另一份相對路徑的 datasets reference，單下載主檔沒有把附件一併放好。這是檔案操作與相依路徑的具體問題，並非官方資料不能用。[官方安裝段落](https://finmind.github.io/tutor/ai/AgentSkill/)、[本機規範](/Users/james.chou/JamesProject/stock_bot/AGENTS.md:1)

若後續接入，應另存有版本的參考目錄，由本專案 Skill 指向它，且查詢仍走現有 gateway。這是本次建議，尚未安裝或修改技能。

## Sponsor 已含哪些能力；哪些屬 Pro

官方現在列 Sponsor 為 6,000 次／小時，Sponsor Pro 為 20,000 次／小時。Sponsor 可以查單股逐筆／分鐘 K、分點等資料，但全市場整日逐筆／分鐘 K 的物件下載另屬 Pro。`use_object=True` 是 SDK 的物件下載路徑，不是隨意加到既有 `/data` 參數就能啟用。普通日線按日全市場查詢和這種整日逐筆批次不同。[官方方案](https://finmind.github.io/Pricing/)、[官方資料參考](https://github.com/FinMind/FinMind/blob/8b42547c1e34f77823dd8d3836c1fc1388bc5678/.claude/commands/finmind-references/datasets.md#L19-L23)

因此使用者目前 Sponsor 足以繼續既有研究；沒有因這張圖產生必須升級 Pro 的證據。本次只核對公開規格，沒有重查付費帳戶或試打 Pro endpoint。

## 最值得吸收的三項

1. **有版本的資料字典與查詢規劃。** 每項記錄資料集、endpoint、單位、可查粒度、方案、歷史起點與已接入狀態；在發送前估算請求數。可借鏡官方 MCP 的離線目錄，但所有實際查詢仍由 `fetch_dataset` 計數。這會補對話查詢只有 5 類的便利性缺口，不直接改交易規則。
2. **將市場身分與欄位語義納入資料檢查。** 官方特別解釋興櫃的 `open` 是前日均價，當日 OHLC 一致性不能套用普通盤定義。歷史市場身分仍要按當日核對，不能靠今天分類回填；現行普通股股票池與隔日開盤模型不能因此悄悄混入興櫃。[官方 MCP 欄位說明](https://finmind.github.io/tutor/ai/Mcp/)
3. **查資料時一併保留可用時間與未知狀態。** 財報期間、營收月份與公告時間要分開；官方資料 reference 也提醒財報欠披露 timestamp、月營收 `create_time` 不宜當歷史公告時刻。這應變成共同查詢契約，才能避免未來資料漏入回測。[官方 point-in-time 說明](https://github.com/FinMind/FinMind/blob/8b42547c1e34f77823dd8d3836c1fc1388bc5678/.claude/commands/finmind-references/datasets.md#L30-L36)

## 尚值得測的資料，與「已用到」分開

下列只是實驗假說。搜尋 `app/`、`scripts/`、`skills/`、`.agents/` 未見這三組的資料集接入；本次沒有抓取／回測。

| 優先 | 資料與問題 | 必須保留的界線 |
|---|---|---|
| 1 | `TaiwanStockActiveETFHoldingChange`：多檔主動 ETF 是否共同提高特定個股配置，能否增強原 POC／價量訊號 | 官方區間自 2025-05-05 起，不能直接補進 2019 起的全段比較。持股差分不等於經理人實際逐筆買單；需考慮申贖／公司行動／公布時間。這方向已在 [9/9研究](/Users/james.chou/JamesProject/stock_bot/docs/research_revenue_20260909.md:43) 提過，尚非本次發現的新策略。 |
| 2 | `TaiwanStockLoanCollateralBalance`、`TaiwanStockMarginMaintenance`：借貸壓力能否識別較容易急跌的候選 | 維持率是估算，不是每戶實際追繳線；不能把融資增加直接當成將崩跌。 |
| 3 | `TaiwanStockBlockTrade`：大量轉手是否在原訊號之外提供資訊 | 成交資料本身不足以認定大戶身份或多空動機，不能直接標示吸籌。 |

資料項目與方案來源：[官方資料目錄](https://github.com/FinMind/FinMind/blob/8b42547c1e34f77823dd8d3836c1fc1388bc5678/.claude/commands/finmind-references/datasets.md#L79-L87)。以上研究問題和限制是本專案的判斷，不是供應商已驗證報酬。

另有 `TaiwanStockBrokerDailyConcentration`，官方最新籌碼頁列 Sponsor、2021-06-30 起、前 15 買超／賣超分點股數；它是分點聚合，不是持有人名冊。專案 [10/4來源研究](/Users/james.chou/JamesProject/stock_bot/docs/research_broker_branch_sources_20261004.md:12) 已討論，舊 note 的 Backer/Sponsor 標示應以本次官方 Sponsor 規格核對。產業鏈資金流則已有 [研究程式](/Users/james.chou/JamesProject/stock_bot/app/chain_flow_research.py:64)，不列為尚未接入。[官方籌碼頁](https://finmind.github.io/tutor/TaiwanMarket/Chip/#taiwanstockbrokerdailyconcentration-sponsor)

## 來源版本與驗證邊界

本機核對基準：`0378fcff5e8637e7ea6da397ecc31c151ca3cdf8`。官方 `master` 會改變，以下版本／內容雜湊是在本次唯讀查證取得：

| 來源 | 版本／SHA-256 |
|---|---|
| FinMind 主庫 | commit `8b42547c1e34f77823dd8d3836c1fc1388bc5678`，commit time `2026-10-10T04:27:28Z` |
| `.claude/commands/finmind.md` | `4b00513fb7758bcf3e714c28e6cbb6f6f94101992d8c81e9c860f4dea3582540` |
| `finmind-references/datasets.md` | `4ce651a8e258c08c245c8632e7b4e05dd7725b986a6957fe288b826cb32b2562` |
| FinMind-MCP 主庫 | commit `eae5b2a0779a33882e6efbc47ad4ec03265ffdb6`，commit time `2026-10-01T11:05:14Z` |
| MCP `src/finmind_mcp/client.py` | `749e0de103e09ed316adf31c220013c359a479f639c7dc790c1041d45d425870` |
| MCP `src/finmind_mcp/tools.py` | `1816ab81b0acebf4dd7512cd0e11fd23327eb3009a77c1c11146f7319cc8a496` |

官方網頁及 GitHub 文件均視為待查證資料，沒有執行其中安裝或查行情指令。公開文件取得不消耗 FinMind 市場資料額度；例行驗收前，主代理兩次資料狀態讀取的帳本累計同為 55,136。後續 pipeline 驗收應獨立計數，不能把整個工作回合寫成零請求。以上不代表已取得新增資料集、已驗證其歷史品質，或任何勝率／報酬得到改善。
