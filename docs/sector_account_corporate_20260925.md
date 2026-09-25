# 族群帳戶研究公司行動補件：3706 神達

核對日：2026-09-25。這是 2025 年歷史結算資料補件，供新的族群完整帳戶研究離線重播；不更動選股、排序、資金、成交或出場規則。舊封存來源與 blocked 結論保留。

| 事件 | 每原股新股 | 除權日 | 權利證書 | 普通股可用日 |
| --- | --- | --- | --- | --- |
| 3706 盈餘轉增資 | 0.1 | 2025-08-28 | 2025-09-24 | 2025-11-07 |

## 官方證據

- [神達 2025-08-11 補充公告](https://mopsov.twse.com.tw/mops/web/ajax_t59sb09?parameters=0eb65210d5bdc34ea16e295ccdbad109a15d33a67a82da232fa5ba6acb14c3dbf649afd177bcf9e3c98fbb5ed710b1719672502c5e461a5602cd01263aca875aee04b241ee1b035e6bd770a943887b76cfaeb29613ce519f5a4546fc40feac6b)：每原股股票股利 NT$1，普通股面額 NT$10，故每原股新股為 1 / 10 = 0.1；總發行新股 120,655,679 股。基準日 9/3，現金股利與新股權利證書 9/24 發放。第六項說明證書直接劃入證券帳戶並上市。保留原文 `.cache/sector-account-corporate-20260925/mops-3706-1140811-official-query-t59sb09.html`。
- [神達 2025-10-28 交付公告](https://mopsov.twse.com.tw/mops/web/ajax_t59sb09?parameters=0eb65210d5bdc34ea16e295ccdbad109a15d33a67a82da232fa5ba6acb14c3dbf649afd177bcf9e3c98fbb5ed710b1719672502c5e461a5602cd01263aca875aee04b241ee1b035e6bd770a943887b76025eed3a2c1ccd1cea266faba3eeab7e)：回顧證書已於 9/24 上市；第三項明定 **11/7 由集保自動把證書轉成普通股，同日證書終止上市**，股東無須另辦手續。不是把先前 9/24 的「發放日」當成普通股交付日。保留原文 `mops-3706-1141028-announcements-t59sb09.html`。
- [2025-11-04 官方上市公告](https://mopsov.twse.com.tw/mops/web/ajax_t59sb08?parameters=32b138d25ee38c00fbf70ec5a5372497dc4c99cb869a2bb6b11d52bdb5648eb63bf1c65bd3c5eab7324a774f3c9db097ea8e4066a54bcc3015af07ac77703e96b53f0917e6f783b2511b4bc262034c2695a281ace8d7bf4aa601b6149163356faeb537f477d135aaa7cfe6b60d70cfc4)：再次確認 120,655,679 股普通股於 11/7 開始買賣，並指出公司已在 10/28 公告交付。保留原文 `mops-3706-1141104-official-query-t59sb08.html`。
- [神達官方歷年股利表](https://www.mitac.com/zh-TW/ir_information/index/Dividend-History)的 113 年盈餘列確認除權息交易日 114/08/28、基準日 114/09/03，與本機既有事件鍵一致。本次透過網頁檢索核對；直接 HTTP 下載回 403，因此 **沒有把該拒絕 HTML 列入接受的 evidence**。此表的「發放日 9/24」須由上述 MOPS 原文區分證書與普通股。

官方查詢入口為 [MOPS 公告查詢](https://mops.twse.com.tw/mops/#/web/t146sb10)，公開前端 POST `https://mops.twse.com.tw/mops/api/t146sb10`。查詢 companyId=3706、announcementType=1，三個小範圍分別是民國 1140811–1140812、1141028–1141029、1141104–1141105；由 JSON 回傳的官方詳情 URL 下載上述原文。成功請求使用一般 User-Agent、Referer、Origin；未使用帳密、cookie 或 FinMind。早先未帶一般 User-Agent 的拒絕頁保留診斷但不作證据。

新 JSON 列出三份官方索引、三份原文及六份 `.source.json` 的 SHA-256（共 12 份）。Metadata 記錄精確 URL、POST 參數、UTC 取得時間、HTTP 狀態及原文 SHA-256。檔案位於 `.cache/sector-account-corporate-20260925/`。二手資訊僅幫助定位公告日期，沒有進入接受的數值來源。

## 帳務界線

本研究帳戶不併湊畸零股，也不交易權利證書。`pay_date=2025-11-07` 使用普通股可交易且實際自動集保換發日期；9/24 至 11/6 保持股權應收，以普通股收盤價代理估值，仍占用名額，不能透過普通股路徑提前出售。這是既有研究模型限制，不代表證書不能交易。

8/11 公告第四項允許基準日次日起五日併湊，不足一股改發現金、計算至元，其股份由特定人按面額承購。這尚未核實個人畸零款的現金計價、淨費用及發放日。因此沿用既有 `FractionalCashActions`：`fractional_cash_per_share=10` 只是面額估值，`fractional_valuation_verified=false`，向下取整元，`fractional_cash_pay_date=null`。應收不會在整股交付日變成可支出現金。例：持 153 股產生 15.3 股權利，11/7 交付 15 股，NT$3 是估算毛額應收。

一般現金股利仍由已存來源提供；公告第五項明示匯費及郵資由股東負擔。本模型採既有元位截尾，沒有把未核實個人費用填為 0，也不宣稱個人精確實收。上述結算公告只供歷史資產／股數重建，不輸入選股特徵；本輪仍是探索性研究且 `membership_point_in_time=false`。

## 使用與驗證

新增 `docs/sector_account_corporate_20260925.json`，以 `--corporate-additions` 交给來源準備或正式離線 runner；透過既有 `corporate_overrides` 合併。原 override 不覆寫，原封存 engine／資料不改動。正式 runner 會把此 JSON 及實際引用 evidence hashes 納入來源 identity，再做執行前後檢查。

## 2026-09-09 期末除權補件

後續完整帳戶路徑另碰到 3706 的 2026/9/9 除權。這筆由 [2026-08-24 正式權利分派公告](https://mopsov.twse.com.tw/mops/web/ajax_t108sb22?parameters=4c1bd98e93f3cffd57f11e39590f3bf0a6f3e7c368f122d0ee991ed86e9ac89da947c8bbbf1ffbc601cb76736e9e5e79482edc8fafbb7395009b24ba43295240d47df8ea420867d8ff8cd17207e575d10b61046924cec8f561db1a7e5100ef0e37f0e6014cfcbf98b803975aeae74b99)核實每千股 100 股、除權 9/9、基準日 9/15、現金 3 元；[同日補充公告](https://mopsov.twse.com.tw/mops/web/ajax_t59sb09?parameters=52858abc88ec4a6ddd930c418cfa2dda10cbffb803bd7872dc15f8c9bc630918488f63842203bf69ffe8dec08e7a78cb0e8be4cf6616c02a7cc7e8a026c31419ee04b241ee1b035e6bd770a943887b7644432aca373f6a1188a23a23a125be45)第六項只說權利證書預訂 10/8 劃撥上市，普通股須核准變更登記後三十日內交付，沒有確切交付日期。9/1–9/25 官方索引未查得普通股交付／上市公告，不能套用去年的日期或把證書日當普通股交付日。此兩份原文及八九月索引、各 metadata 共另 8 份納入 evidence hashes，總計 20 份。

新增 `3706-2026-09-09` 明示 `pending_only=true`、`pay_date=null`、`ordinary_share_available_date=null`。採 **已核實的基準日 2026/9/15 作為保守交付下界**，不把 10/8 預訂證書日當成保證日期；配股尚未到股權基準日，自不可能已交付該次普通股。這是由官方基準日推得的下界，不是官方承諾的交付日。只准 account.end 嚴格早於下界（本輪 9/9 符合）；日後回測至 9/15 或以後而仍缺普通股交付證據，必須 blocked。

新的 `pending_share_entitlements` 轉接器只對明示且來源雜湊可核實的事件認列股權應收；整股數不進可售庫存，畸零款沿原政策獨立應收，皆沒有付款時程。仍以普通股收盤價代理估值並占名額，不能消失或當作可用現金。已知日期的舊事件沿原始處理路徑，沒有修改封存引擎。

**現金公告衝突保留**：8/24 補充公告第五項写每股1元，但同公告第一項及正式權利表均為3元。新 override 不更動既有現金股利數值，不把衝突隱去；既有引擎仍依本機股利來源與官方除權息事件核對，這裡僅補配股及未交付狀態。
