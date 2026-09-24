# 固定三／五檔研究零股依賴稽核（2026-09-25）

本次依 `stock-strategy-research` 查閱資料與策略狀態，只讀取本機封存帳戶與來源檔；0 網路、沒有下單、採購、排程或修改原始行情。研究區間為 2022-01-03 至 2026-09-09。

固定來源為 `.cache/conservative-diversification-20260924/cases/` 的 `capacity_combined_3.json`、`capacity_combined_5.json`、`benchmark_combined_0.json`。這是既有日模型成交路徑的零股需求清單，包含拒絕委託；不是未來嚴格序列回放所有可能需要的日期，也不是資料採購單。三個帳戶是相互獨立的研究方案，不得相加成一個帳戶的股數。

| 固定帳戶 | 原研究報酬 | 全部成交筆數 | 零股委託 | 零股成交 | 零股成交日期 | 零股成交股票 | TWSE／TPEx 零股成交 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 保守三檔 | +115.73% | 232 | 143 | 138 | 127 | 52 | 81／57 |
| 保守五檔 | +200.57% | 379 | 248 | 238 | 210 | 88 | 158／80 |
| 0050 基準 | +239.83% | 17 | 16 | 16 | 16 | 1 | 16／0 |

- 三檔零股成交為買入 55、賣出 83 筆；五檔買入 95、賣出 143 筆；0050 為 16 筆買入。
- 三帳戶去重後共有 **288 個 market／stock／date／side 需求，涵蓋 245 個日期**：TWSE 189、TPEx 99。
- 其中 **275 個需求曾有日模型成交，涵蓋 237 個日期**：TWSE 183、TPEx 92。其餘 13 個需求沒有成交，仍保留在委託清單。
- 288 個需求全部找到對應日期的官方零股日資料，raw、normalized rows、feed index、父研究 manifest 雜湊全部核對。
- **上述 288 個需求在所檢查的封存來源中，已宣告且可接受的零股成交序列 tape 為 0**。日成交量、最後買賣價、最後買賣量不證明逐次撮合時間、可成交深度或委託優先順序。沒有用普通交易逐筆替代零股。

## 交付檔案與重現

目錄：`.cache/oddlot_scope_20260925/`。

- `exact_demands.json`／`.csv`：每個 account／date／stock／side 的委託與成交股數、來源帳本位置、失敗原因、日行情來源、序列缺口。
- `union_stock_date_side.json`／`.csv`：跨帳戶去重、保留各帳戶出現及已成交的關係。每列 `accepted_sequence_tape_present=false`。
- `summary.json`：各帳戶及 union 統計、旧 strict 首次阻擋、MTH 樣本限制、本次來源指紋稽核。
- `manifest.json`：本次精確輸入與五個輸出的 SHA-256。JSON 保留股票代碼字串；用 Excel 開 CSV 時須將 `stock_id` 當文字，避免丟失 0050 前導零。

執行：`python3 scripts/audit_oddlot_scope_20260925.py --verify`。

驗證結果：輸出逐 byte 一致；`python3 -m pytest tests/test_oddlot_scope_20260925.py -q` 為 **4 passed**。測試涵蓋拒絕委託保留、同股票同日買賣分開、成交與委託對帳、日資料不升格為序列、來源檔竄改拒絕。最新父研究 identity 的 **20,231 個檔案全部符合**。這是來源與需求抽取驗證，沒有重新執行回測引擎。

## 舊嚴格回放與來源變動

`.cache/crossday-contingent-20260914-verified/strategy.json` 只完成 2022-01-03 的空倉交易日，在 2022-01-04 因 8261 與 6284 零股序列缺失停止。`benchmark.json` 在 2022-01-03 因 0050 零股序列缺失停止，完成 0 個交易日。兩者全期間報酬皆為 null。這三個最早缺少的證券／日期，不能當成完整歷史只差三份資料。

舊 strict identity 共 16,339 個檔案指紋與本次已核驗來源比較，有 **1 個變動**：

| 檔案 | 舊 strict SHA-256 | 本次 SHA-256 |
| --- | --- | --- |
| `app/finmind.py` | `e04883f03c1a4bf5d6c624ac1bfad026cdc704771bf76b48778aaa9135d4ed5e` | `28ae87dba48a313c509a9ce3fc1aac97154ee328141921bdd0d34ef00812ddc3` |

因此這裡只引用舊 strict 的封存停止紀錄，不宣稱在當前程式重新通過 strict 回放。精確序列需求須在補齊目前缺口、以新來源指紋重跑後隨實際成交路徑續增。

`skills/tpex_mth.py` 對官方免費樣本僅做格式與買賣雙邊配對檢查。樣本限 **2023-09-15、3105**；其 `historical_session_complete`、`execution_tape_accepted`、`source_authenticated` 均為 false，不計入任何已通過的需求。

## 整張研究的既有範圍與可用接點

已有 `docs/prereg_intraday_limit_20260914.md` 與 `skills/intraday_limit_replay.py` 的整張診斷：買進千股倍數，公司行動產生的零股保留估值、無序列證據便不假設售出。但該研究同時改了盤中限價逐筆、名額與資源保留，不能當成最新五檔「只拿掉零股交易」的單變因對照。

最新 `ConservativeDiversification` 的可用獨立接點為 `StressOrder._execute_order`：它在 `ResourceDecisions.order` 預算縮量之後執行，可將送入普通交易的數量向下取千股，另保留買入尾數現金與賣出尾數未成交紀錄，不需改原引擎。只在最外層 `order` 取整張會被後續預算縮量再次改成非千股，不足以保證排除零股。

公司行動仍須沿用已核對的配發率、除權息日、可交付交易日、分割／減資換股比率、畸零不足一股的折現條件及實際支付日。最新五檔已出現 2880、2881、2890 配股與 1815 期末待交付權益；不可把權益日當成可售股日，也不可把未知付款日的畸零款提前當現金。整張限制改變持股路徑後，若遇到新公司的未核對公司行動，必須停止並記錄缺口。
