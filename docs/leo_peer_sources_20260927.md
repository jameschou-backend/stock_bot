# 低軌衛星固定同業組：原始揭露核對（2026-09-27）

依 2025-04-15 已固定的題材名單，核對聯發科、穩懋、正崴、康舒、台光電。沒有依之後股價選公司；本工作不讀報酬。原始資料與可程式讀取的分類見 [JSON](leo_peer_sources_20260927.json)。

## 結論

取得 5 家公司的 11 份官方／MOPS 歷史原文。這些文件無法單獨證明 LEO 營收占比至少 10%、已實現 LEO 成長至少 20%，或明確的 LEO 訂單／量產。因此相關欄位保留 `null`；不將缺乏拆分披露誤判成沒有受惠，也不將 AI、電信或能源收入挪用為 LEO 收入。

這是未通過經濟證據條件的「未知組」，不能宣稱為 LEO 未受惠／負面對照組。

## 五家公司保留結果

| 公司 | 原始文件覆蓋 | 判讀 |
|---|---|---|
| 2454 聯發科 | 3 份 | Q1/Q2/Q3 官方逐字稿已取；未證明單獨 LEO 經濟門檻。 |
| 3105 穩懋 | 3 份 | Q1/Q2/Q3 官方簡報已取。Q2 提低軌衛星，仍與航太、AI 混合在 Infra。 |
| 2392 正崴 | 1 份 | 取得 6/11 MOPS 原文；公司舊活動表格損坏，未找到 Q3 獨立法說原文。 |
| 6282 康舒 | 1 份 | 公司官網 403；取得 11/13 MOPS 原文及官方活動日期。未找到 Q2 獨立法說原文。 |
| 2383 台光電 | 3 份 | Q1/Q2/Q3 官方簡報已取；AI 基礎設施成長不視作 LEO 成長。 |

## 事件證據

每筆 `signal_eligible=true` 僅代表有日期與已讀原文，供探索性歷史事件對齊。所有 `first_publication_verified=false`；無歷史首次發布存檔，禁止宣稱 point-in-time 完整。日期只到日，後續程式仍須延後至公告後可交易時點。

### 2025-05-02 台光電 2025 年 5 月法說簡報

來源：[台光電 2025 年 5 月法說簡報](https://www.emctw.com/upload/media/New_Investors/Financial_Information/Presentation_Materials/CH/114/Investor%20Conference%20Presentation_202505-Chinese.pdf)，PDF 頁 1, 4, 6, 9。日期依據：官方法說簡報列表日期；檔名及封面僅作交叉核對，不以較早檔名日期視作公開日。

- Q1 集團營收年增 68%、基礎設施類別年增 99%；未提供 LEO 收入細分。
- 高階 CCL 市場與公司成長展望不等於 LEO 已實現業績。

LEO 已實現成長、重大曝險、訂單／量產、負面證據皆為 `null`。企業整體逆風另記 `corporate_negative=none`。

### 2025-07-31 台光電 2025 年 7 月法說簡報

來源：[台光電 2025 年 7 月法說簡報](https://www.emctw.com/upload/media/New_Investors/Financial_Information/Presentation_Materials/CH/114/Investor%20Conference%20Presentation_20250730%20Final%20Chinese.pdf)，PDF 頁 1, 4, 6, 10, 12。日期依據：官方法說簡報列表日期；檔名及封面僅作交叉核對，不以較早檔名日期視作公開日。

- Q2 集團營收年增 45.7%、基礎設施類別年增約 70%；未提供 LEO 收入細分。
- 成長優勢頁明列 AI 基礎設施設計；CCL 擴產未識別為 LEO 專屬訂單／量產。

LEO 已實現成長、重大曝險、訂單／量產、負面證據皆為 `null`。企業整體逆風另記 `corporate_negative=none`。

### 2025-10-31 台光電 2025 年 10 月法說簡報

來源：[台光電 2025 年 10 月法說簡報](https://www.emctw.com/upload/media/New_Investors/Financial_Information/Presentation_Materials/CH/114/Investor%20Conference%20Presentation_20251030%20final%20CH.pdf)，PDF 頁 1, 4, 6, 10, 12。日期依據：官方法說簡報列表日期；檔名及封面僅作交叉核對，不以較早檔名日期視作公開日。

- Q3 集團營收年增 44%、基礎設施類別年增約 68%；未提供 LEO 收入細分。
- AI 基礎設施與高階基材的優勢不能轉寫成 LEO 營收成長。

LEO 已實現成長、重大曝險、訂單／量產、負面證據皆為 `null`。企業整體逆風另記 `corporate_negative=none`。

### 2025-06-11 正崴 2025-06-11 法人說明會

來源：[正崴 2025-06-11 法人說明會](https://mopsov.twse.com.tw/nas/STR/239220250611M001.pdf)，PDF 頁 1, 4, 6, 13。日期依據：MOPS 原始 PDF 封面明列 2025 年 6 月 11 日。

- 1–5 月能源事業營收年增 56%；這是能源事業，不能列作 LEO 已實現成長。
- 產品展望包括 AI 線材、電力模組、AI 視訊、風電與 AI 運算中心，未提供可獨立驗證的 LEO 收入／量產指標。
- Q1 歸母淨利年減 67.1%；不得推論為 LEO 逆風。

LEO 已實現成長、重大曝險、訂單／量產、負面證據皆為 `null`。企業整體逆風另記 `corporate_negative=true`。

### 2025-04-30 聯發科 2025 Q1 法說逐字稿

來源：[聯發科 2025 Q1 法說逐字稿](https://www.mediatek.com/hubfs/MediaTek%20Assets/Pdfs/Quarterly%20Earnings%20Release/2025/Quarterly%20Earnings%20Release-2025Q1/Transcript.pdf)，PDF 頁 1, 2, 3, 4。日期依據：PDF 逐字稿首頁列明會議日期。

- Smart Edge Platforms 年增 32%、占總營收 39%，但涵蓋多種連網與運算產品，未拆出 LEO。
- 公司說明 AI、Wi-Fi 7、手機與車用產品；Q1 淨利年減 6.7%，不是 LEO 業務逆風證據。

LEO 已實現成長、重大曝險、訂單／量產、負面證據皆為 `null`。企業整體逆風另記 `corporate_negative=true`。

### 2025-07-30 聯發科 2025 Q2 法說逐字稿

來源：[聯發科 2025 Q2 法說逐字稿](https://www.mediatek.com/hubfs/MediaTek%20Assets/Pdfs/Quarterly%20Earnings%20Release/2025/Quarterly%20Earnings%20Release-2025Q2/Transcript.pdf)，PDF 頁 1, 3, 4。日期依據：PDF 逐字稿首頁列明會議日期。

- Smart Edge Platforms 年增 26%、占總營收 43%，不等於 LEO 收入。
- AI 平板、GB10、車用及 ASIC 的成長或量產展望不能改標成 LEO 訂單／量產。

LEO 已實現成長、重大曝險、訂單／量產、負面證據皆為 `null`。企業整體逆風另記 `corporate_negative=none`。

### 2025-10-31 聯發科 2025 Q3 法說逐字稿

來源：[聯發科 2025 Q3 法說逐字稿](https://www.mediatek.com/hubfs/MediaTek%20Assets/Pdfs/Quarterly%20Earnings%20Release/2025/Quarterly%20Earnings%20Release-2025Q3/Transcript.pdf)，PDF 頁 1, 2, 3, 4。日期依據：PDF 逐字稿首頁列明會議日期。

- Smart Edge Platforms 年增 14%、占總營收 42%；未拆出 LEO 收入。
- GB10 開始量產屬 AI 產品；雲端 ASIC 2026 年收入是預期，不是 LEO 已實現收入。
- 集團營業利益年減 7%、淨利年減 0.5%，未歸因 LEO。

LEO 已實現成長、重大曝險、訂單／量產、負面證據皆為 `null`。企業整體逆風另記 `corporate_negative=true`。

### 2025-04-30 穩懋 2025 Q1 法說簡報

來源：[穩懋 2025 Q1 法說簡報](https://www.winfoundry.com/zh-TW/Base/DownLoadFile/505?filename=1Q%202025%20%E6%B3%95%E8%AA%AA%E6%9C%83%20%E2%80%93%20%E7%B0%A1%E5%A0%B1.pdf&TargetTable=QuarterlyAttachment)，PDF 頁 1, 5, 7, 8。日期依據：PDF 封面日期與官方法說行事曆交叉核對。

- Q1 Infra 占比 30–35%；未拆出低軌衛星在 Infra 中的收入。
- 集團營收年減 20%，不等於 LEO 營收年減。

LEO 已實現成長、重大曝險、訂單／量產、負面證據皆為 `null`。企業整體逆風另記 `corporate_negative=true`。

### 2025-07-31 穩懋 2025 Q2 法說簡報

來源：[穩懋 2025 Q2 法說簡報](https://www.winfoundry.com/zh-TW/Base/DownLoadFile/511?filename=2Q%202025%20%E6%B3%95%E8%AA%AA%E6%9C%83%20%E2%80%93%20%E7%B0%A1%E5%A0%B1.pdf&TargetTable=QuarterlyAttachment)，PDF 頁 1, 5, 7。日期依據：PDF 封面日期與官方法說行事曆交叉核對。

- Infra 營收增加，應用包括低軌衛星、航太與 AI 資料中心；2025 上半年 Infra 收入占比與 Cellular 相當。
- 資料未提供獨立 LEO 收入占比、數值成長率或明確 LEO 訂單／量產數量。
- 同期[官方新聞稿](https://www.winfoundry.com/zh-TW/Base/DownLoadFile/508?filename=2Q%202025%20%E6%B3%95%E8%AA%AA%E6%9C%83%20%E2%80%93%20%E6%96%B0%E8%81%9E%E7%A8%BF.pdf&TargetTable=QuarterlyAttachment)：集團 Q2 營收年減 24%、歸母淨損 4.21 億元；不視作 LEO 逆風。

LEO 已實現成長、重大曝險、訂單／量產、負面證據皆為 `null`。企業整體逆風另記 `corporate_negative=true`。

### 2025-10-30 穩懋 2025 Q3 法說簡報

來源：[穩懋 2025 Q3 法說簡報](https://www.winfoundry.com/zh-TW/Base/DownLoadFile/517?filename=3Q%202025%20%E6%B3%95%E8%AA%AA%E6%9C%83%20%E2%80%93%20%E7%B0%A1%E5%A0%B1.pdf&TargetTable=QuarterlyAttachment)，PDF 頁 1, 5, 7, 11。日期依據：PDF 封面日期與官方法說行事曆交叉核對。

- Q3 Infra 占比 30–35%，未拆出 LEO。
- Q3 集團營收年增 3%、季增 19%，公司主要解釋手機旺季與 Wi-Fi 7；不能全部歸因 LEO。
- 前三季集團營收年減 14%、歸母淨利年減 41%；不是 LEO 專屬逆風證據。

LEO 已實現成長、重大曝險、訂單／量產、負面證據皆為 `null`。企業整體逆風另記 `corporate_negative=true`。

### 2025-11-13 康舒 2025 年 11 月法說簡報

來源：[康舒 2025 年 11 月法說簡報](https://mopsov.twse.com.tw/nas/STR/628220251113M001.pdf)，PDF 頁 1, 8, 9, 10, 11。日期依據：MOPS PDF 封面為 November 2025；臺灣指數官方法說行事曆列 2025/11/13 14:00。

- Q3 電信電源營收年增 55%、占營收 37%，但產品說明為基地台與電信機房，不可將全部電信收入視作低軌衛星。
- Q3 集團營收年減 4%、前三季 EPS -0.38；未說明為 LEO 業務惡化。
- 未量化 LEO 收入、獨立成長率或 LEO 訂單／量產。

LEO 已實現成長、重大曝險、訂單／量產、負面證據皆為 `null`。企業整體逆風另記 `corporate_negative=true`。

## 範圍限制

- 聯發科、穩懋、台光電以 Q1/Q2/Q3 檔為主要觀察窗；正崴與康舒只找到上述日期法說，沒有用不存在的季度資料補值。
- 正崴公司舊活動表格顯示損壞；康舒官網回應 403。兩者改讀在 MOPS 公開的公司原始 PDF，而非採用第三方摘要當成原文。
- 康舒 11/13 日期另由 [臺灣指數官方法說行事曆](https://irengage.taiwanindex.com.tw/ConferenceList?Month=11&Year=2025) 核對；穩懋日期由 [官方活動表](https://www.winfoundry.com/zh-CN/Invest/invest_activity?isOpen=True&page=1&year=2025) 核對；台光電由 [公司簡報列表](https://www.emctw.com/zh-TW/presentation_materials/index) 核對。
- 文件 URL、SHA-256、頁數與日期依據記入 JSON。SHA-256 是本次取得版本的指紋，不能追溯證明過去網站沒有改版。
- 沒有執行 FinMind 請求、修改環境設定或檢視價格結果。
