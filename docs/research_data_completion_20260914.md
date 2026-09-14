# 歷史行情、市場身分與零股資料修補結果

2026-09-14，研究與修復範圍：2022-01-03 至 2026-09-09 歷史驗證缺件。這次完成 22 檔已知行情衝突的修復；完整歷史市場身分與零股實際成交仍有明確來源缺件，沒有把三項全部標成完成。

## 1. 22 檔行情：64 筆已修復

將官方終止上市日、先前凍結的供應方每日行情、舊交易日 OHLCV，以及修復當下 MySQL 原始值逐筆交叉核對：

- 63 筆完整 OHLCV 與多年以前的行情完全相同；1 筆 9188 是全零占位資料。
- 全部位於終止上市之後，且未出現在對應日期的重新下載行情。
- 涉及 2026-06-09、06-11、06-17；只修這 64 個明確主鍵，沒有批量刪掉所有下市股歷史。
- 在同一個交易內移出 64 筆原始行情及 61 筆受污染標籤。標籤檢查沿著股票完整交易日序列比對未來終點，涵蓋原本下市前受影響的標籤，並非只刪六月的列。
- 備份保存完整原始行與標籤、登錄檔 SHA256；再執行核對得到 64 筆均不再存在、61 筆無效標籤沒有重新出現，不做第二次刪改。
- 正常 ingest、bootstrap、歷史回補、個股更新及封存回補入口均會拒絕寫入已隔離主鍵；即使改成其他價格也要求重新核對，不能自行解除。

登錄：[price_quarantine_20260914.json](price_quarantine_20260914.json)。修復及備份：`.cache/readiness-completion-20260914/price-repair/`。

重算 2026-06-09 至 09-11 的 125,940 筆特徵，同步重建行情、特徵、標籤快取，成功執行約 166 秒，0 次網路查詢。首次由 stdin 啟動的重算遇到 macOS 多程序無法匯入主模組，已停止並改用具名腳本完成；失敗與成功日誌皆保留。

這只結清 22 檔的已知衝突。原始三日稽核共 158 筆衝突，其餘 94 筆不在本次正式刪改範圍；它們先前的研究隔離證據仍保留。本次也沒有更動凍結回測行情、宣告重算歷史報酬或替換紙上策略。

可重跑命令：

```sh
python scripts/resolve_price_quarantine.py --output .cache/readiness-completion-20260914/price-repair
python scripts/resolve_price_quarantine.py --output .cache/readiness-completion-20260914/price-repair --rebuild-derived
python scripts/resolve_price_quarantine.py --output .cache/readiness-completion-20260914/price-repair --verify-storage
```

第二個命令在完成紀錄存在時回傳已重建，不重跑全量計算。第三個命令已於 pipeline 後驗證：MySQL 及行情、標籤、特徵快取、年度 FeatureStore 中均沒有受污染主鍵。備份不得删除；若需復原，應先核對備份與登錄 SHA，再逐筆復原到原主鍵並重算衍生資料，不能直接關閉隔離或回填舊快取。

## 2. 市場身分：目前名冊完成核對，完整歷史仍未齊

取得並保存 2026-09-14 [TWSE ISIN](https://isin.twse.com.tw/isin/C_public.jsp?strMode=2) 及 [TPEx ISIN](https://isin.twse.com.tw/isin/C_public.jsp?strMode=4) 官方快照，共 1,987 個四碼證券，其中股票類 1,945 檔；ETF、TDR、創新板保留獨立分類。此範圍不是所有長度證券代碼。

結合既有 60 筆終止紀錄與官方上市買賣日，建立開始含當日、結束不含當日的市場區間。6423 在 2026-01-22 的轉板邊界已通過測試；未知起日、重疊市場、快照之後的日期均不自動認定。

原先凍結的 458 筆候選，在各自訊號日期都能找到市場及掛牌區間；這不代表完整市場選股已消除存活者偏誤。

尚缺 41 筆歷史掛牌起日（TWSE 6、TPEx 35）、部分歷史證券類別與當時公告時間封存。詳細清單在 `identity-v2/report.json` 的 `missing_start_rows`。不以首筆行情日代替 IPO，也不把公開發行日當成上市日。例：目前 ISIN 查詢 2809 回傳的是公開發行日期，不能用來填原上市日期。

這次另取得 2022-01-03 TWSE 全市場收盤表，可作該日市場存在證據，尚未將單日存在外推成連續歷史區間。TPEx 同日歷史端點的 GET 與依網頁格式的 POST 均遇到傳輸中斷；TPEx 基本資料端點也無法完成 TLS 回應。沒有繞過 TLS 或以空資料當成功。已下櫃公司的官方個股頁也不一定保留原掛牌日，因此仍須官方歷史名冊或公告補證。

輸出：`.cache/readiness-completion-20260914/identity-v2/report.json`。重現需指定新目錄：

```sh
python scripts/audit_market_identity.py --output .cache/readiness-completion-20260914/identity-reproduced
```

此名冊目前用於研究核對，未直接覆寫股票主檔或啟用新的全市場候選池。

## 3. 歷史零股：解析完成，成交資料仍需供應

重新查核 [FinMind 技術資料目錄](https://finmind.github.io/tutor/TaiwanMarket/Technical/)，一般逐筆與分 K 資料不能直接視為零股成交證明；目前沒有取得能完成本案歷史零股回放的資料。

已下載 [TWSE H4 商品頁](https://eshop.twse.com.tw/zh/product/detail/0000000080da7fa70182334eb932009d) 的免費範例與兩份官方格式，加入 190-byte 舊版、2026-04-01 起 201-byte 新版解析及日期邊界檢查。格式具有試算／成交旗標；免費範例只含 2020-10-26 的 10 筆試算，實際成交 0 筆，不能補本案缺口。

格式文件的成交數量欄寫「成交張數」，買賣五檔欄則寫「股數」，價格縮放也需確認。解析器保留原始價格與數量，不猜單位、不宣稱完整盤別，不輸出可供成交模擬的 tape。

```sh
python scripts/audit_h4_odd_lot.py \
  --input .cache/readiness-completion-20260914/h4-sample.txt \
  --version legacy190 \
  --output .cache/readiness-completion-20260914/h4-sample-audit-reproduced.json
```

取得原始資料、單位與完整交易日的供應方證明後，才可接到既有 `normalized_auction_v1` 匯入與跨日回放。最早仍缺 0050／2022-01-03、8261／2022-01-04、6284／2022-01-04 的盤中零股實際成交。

[補件與詢價文件](readiness_data_request_20260914.md) 已列具體樣本、涵蓋日期、盤別、單位、授權及交付需求，未寄出、未採購。TWSE H4 每月全股票 1,500 元，2022/1 至 2026/7 共 55 月、82,500 元；最新兩個月不在一般商品可購範圍。TPEx MTH 頁面起始日為 2022/11，尚不能證明包含本案 2022/1 零股。應先確認 TPEx 早期資料與兩市場最新月份能否供應，再決定採購。

## 驗證與目前限制

- `make test`：2,569 個測試通過（29 個警告）；後續文件或測試更新以提交前最後日誌為準。
- 修復的再執行核對通過；紙上三份帳本原封存程式驗證通過。
- `make pipeline` 實際跑過：資料、特徵、標籤、模型訓練完成；最後選股因 2026-09-14 TAIEX 報酬指數尚未更新而明確停止，**整條 pipeline 未通過**。這次沒有放寬日期檢查或改用舊指數。
- 14:32 資料狀態：TWSE 已到 9/14，TPEx 仍在 9/11，`data_ready=false`。當日來源尚未同步，不能使用部分市場作正式新名單。
- API 重啟後 health 正常，picks 20 筆、models 9 筆、jobs 10 筆均可讀；既有 picks 可讀不代表今天已產出合格訊號。
- 本次重算使用既有還原因子；21.65% 的計算窗口列被既有特徵引擎標記缺因子。先前官方因子草稿仍未正式套用，不能把行情修復解讀為還原價全數正確。
- FinMind 本小時共用帳本由 12 增至 15，剩 5,385／5,400，未繞過 Sponsor 用量限制。

UI 在「歷史資料修補進度」顯示修復與來源缺件；摘要對應檔案以 SHA256 核對，不把缺失證據標成完成。全程未下單、未買資料、未修改 `.env` 或封存策略。
