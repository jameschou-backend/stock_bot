# 0050 2025/4/10 漲跌停價衝突修正

這是新回測執行入口的單日資料修正，不改選股、出場、成本或舊封存來源。僅修正 `0050 / 2025-04-10`；其他股票及日期保留來源原值，後續若再出現衝突須另行核對。

原始 FinMind 查詢為 `TaiwanStockPriceLimit / 0050 / 2018-01-01–2026-09-09`。當日競價參考價 146.20，來源上下限為 132.00／160.50。

已核對封存 TWSE MI_INDEX 原始日表及 HTTP/query receipt：4/9 收盤為 146.20；4/10 開高低收均為 160.80，漲跌價差 +14.60，與前收直接銜接。封存公司事件在 4/1–4/15 沒有 0050 事件，股息來源列出的 2025 除息日為 1/17 與 7/21。未發現當日除權息或分割造成另一參考價的證據。

ETF 在 50 元以上升降單位為 0.05 元，普通國內股票 ETF 漲跌幅 10%。依參考價向區間內取整：

- 上限：146.20 × 1.10 = 160.82，向下取 0.05 的合法價位為 **160.80**。
- 下限：146.20 × 0.90 = 131.58，向上取 0.05 的合法價位為 **131.60**。

原 160.50／132.00 恰好符合普通股此價位採 0.50 間距的結果。這支持價格間距誤用的診斷；無須以事後最高價任意放寬上限。

規則證據有兩層：既有[富邦投信 ETF 常見問題](https://websys.fsit.com.tw/FubonETF/Service/Question.aspx) 原始 HTML 與 receipt 已封存、綁 SHA256；另於 2026/10/3 查核 [TWSE ETF 交易規則](https://wwwc.twse.com.tw/zh/products/securities/etf/overview/rules.html)、[2016 官方簡報](https://www.twse.com.tw/staticFiles/news/event/event_download_201606220915_01.pdf)及[交易制度](https://wwwc.twse.com.tw/zh/products/system/trading.html)。後三個 URL 是本次官方規則頁與歷史簡報查核，**沒有封存原始 bytes，也沒有其檔案 hash 證據**。

機器可讀證據為 `benchmark_0050_limit_overlay_20261003.json`，綁定原始日表、查詢 receipt、FinMind raw/metadata、公司事件與既有規則 HTML 的 hash。新的 overlay 驗證來源後回傳 limits 副本；只有來源當日上下限仍精確等於 132.00／160.50 才套用。執行端與獨立 audit 必須使用同一個 feeds overlay，保留原值、衍生值、日期、公式與來源 hash。

這仍是**依已核對規則衍生的上下限**，不是取得交易所該日原始上下限欄位；`observed_official_daily_limits=false`、`strict_data_ready=false`、`live_qualified=false`。基準在修正前失敗的執行紀錄保留。修正後也不可推定在漲停價一定買得到；原本 HL2、量、限價及未成交檢查繼續生效。
