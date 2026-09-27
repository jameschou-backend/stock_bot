# 低軌衛星營運揭露：3491、6285、2313

2026-09-27 建立。本次未閱讀任何股價或報酬，僅依指定公司、時間範圍與公司官方資料建立營運事件表。可機讀資料為 [JSON](leo_catalyst_sources_20260927.json)：18 個已讀法說文件，另保留 2 個未納入訊號的已讀資源。

這份表支援「文件日期代理」研究，**沒有任何一列已證明歷史首次上網時間或當時版本**。`signal_eligible=true` 只代表沒有已知版本／日期問題而允許代理研究，不能視為嚴格 point-in-time 資料；實際取得日為 2026-09-27。需要真實首次取得時間的模式不得倒填。

## 判讀規則

- `realized_growth`：明確 LEO 部門或具名 LEO 客戶已實現營收／出貨成長至少 20%；客戶範圍另記，不能宣稱等於整個部門。
- `material_exposure`：明確低軌營收占比至少 10%；航太、網通等更大分類不能逕自代換。
- `order_or_production`：明確 LEO 在手訂單、已實現出貨或量產；只有產品型錄或未來預測不能通過。
- `negative`：必須明確綁定 LEO 的不利營運事實。公司整體或其他業務的衰退另存 `corporate_negative`。
- 未披露或不能確認為 `null`，不等於沒有該業務或沒有成長。正面與逆風資料一併保留。

## 昇達科（3491）：增補案例，與原七家公司分開

[官方法說清單](https://www.umt-tw.com/tw/institutional_investors.php)及[第 2 頁](https://www.umt-tw.com/tw/institutional_investors.php?p=2)提供會議日期與 PDF 連結。本次涵蓋官網所列 2025 年 1/8、3/20 兩份、5/20、8/12、8/22、9/9、11/11、12/10、12/12，以及 2026/1/22。第三方列表出現的 11/18 未獲這份官方清單確認，不補造文件。

| 文件日期 | 可核對的當期訊息 | 必須一起看的限制 |
|---|---|---|
| 2025/1/8 | 2024 前 11 月 LEO 占比 42.7%；個別 LEO 客戶出貨年增 250%。 | 客戶出貨成長的精確比較期間未寫明；不能當成整體 LEO 成長率。 |
| 2025/3/20 | 2024 LEO 占比 42.8%；2025 前 2 月 54.7%；兩個客戶 2024 出貨增長已實現。 | 同日兩份文件屬同一資訊事件；未來訂單增長預測未當成實現值。 |
| 2025/5/20 | Q1 LEO 占比約 57%；公司 Q1 營收年增 43%、母公司淨利年增 84.3%。 | 全公司成長不代替 LEO 成長；6 月工程業務退出合併會改變分母。 |
| 2025/8/12、8/22 | H1 LEO 營收年增 40%、占比 52.7%；在手訂單超過 4 億元。 | 公司 Q2 營收年減 20.7%；有匯率、合併範圍、重點客戶出貨遞延。未指明該延遲是哪業務，不強行標成 LEO 逆風。 |
| 2025/9/9 | 重申 H1 LEO 成長 40%、占比 52.7%；行動回傳業務衰退。 | 重複舊報告期，並非新一季成長證據。 |
| 2025/11/11 | 檔案帶有 `updated`。 | 保留內容但 `signal_eligible=false`；修訂時間與原版不明。 |
| 2025/12/10、12/12 | 1–10 月 LEO 年增 26%、10 月占比 69.7%；未交訂單超過去年 LEO 出貨金額。 | Q1–Q3 LEO 年增 19% 本身未達 20%；公司 Q3 營收與母公司淨利仍年減。兩份為重複揭露。 |
| 2026/1/22 | 2025 LEO 年增 45%、占比 59%；在手未交訂單超過 18 億。 | Q4 為自結未查核；這些數字不能用在 2025 年訊號。 |

依序來源：[1/8](https://www.umt-tw.com/upload/Investors/%283491%29_20250108.pdf)、[3/20 GS](https://www.umt-tw.com/upload/Investors/GS20250320.pdf)、[3/20 MEGA](https://www.umt-tw.com/upload/Investors/MEGA20250320.pdf)、[5/20](https://www.umt-tw.com/upload/Investors/349120250520M001.pdf)、[8/12](https://www.umt-tw.com/upload/Investors/3491_20250812.pdf)、[8/22](https://www.umt-tw.com/upload/Investors/3491_20250822.pdf)、[9/9](https://www.umt-tw.com/upload/Investors/UMT%283491%29_0909.pdf)、[11/11 修訂檔](https://www.umt-tw.com/upload/Investors/%283491%29_20251111%28updated%29.pdf)、[12/10](https://www.umt-tw.com/upload/Investors/%283491%29_20251210.pdf)、[12/12](https://www.umt-tw.com/upload/Investors/%283491%29_20251212.pdf)、[2026/1/22](https://www.umt-tw.com/upload/Investors/349120260122M001.pdf)。頁碼和各數字的範圍在 JSON。

8/12 第 8 頁的 4 億元訂單，搜尋工具文字索引漏掉，已以同一官方 URL 直接取得 PDF，再用 PyMuPDF 逐頁核對。該頁將 -17.2% 標為 YoY，但財務表是 QoQ；本表保留財務表的 YoY -20.7%，不混用。

## 啟碁（6285）：產品關聯已知，分項營運未證明

[官網會議清單](https://www.wnc.com.tw/en/investors/financial-information/investor-meetings)與三份官方 PDF 均已查閱。使用清單的 5/7、8/26、11/6 作保守代理日期；Q1、Q3 檔名分別含 5/5、11/4，但不能拿檔名當首次公布時間。

- [Q1 簡報](https://www.wnc.com.tw/uploads/files/shares/ir-presentation/IR_Presentation_tc_20250505.pdf)第 8–9 頁列有 LEO 寬頻終端／衛星模組；公司營收年增 10.9%、稅後淨利年增 22.4%。没有單列低軌營收占比、增幅或新量產訂單。
- [H1 簡報](https://www.wnc.com.tw/uploads/files/shares/ir-presentation/IR_Presentation_tc_20250826.pdf)第 3–4 頁是財務數字；H1 營收年減 0.8%、淨利年減 26.3%。沒有 LEO 分項，保留為未知揭露，不跳過弱季度。
- [Q3 簡報](https://www.wnc.com.tw/uploads/files/shares/ir-presentation/IR_Presentation_en_20251104.pdf)第 5、8–9 頁：前三季營收年增 1.2%、淨利年減 14.7%，LEO 終端仍列產品组合；仍不能把公司成長或產品存在當成已實現低軌成長。

因此三個正面旗標皆為 `null`，不是判定公司不受惠。直接 HTTP 存取官方清單收到 403 後未繞過，文件內容使用 web 工具可讀的官方 PDF。

## 華通（2313）：航太比重不能直接等同低軌比重

[公司法說清單](https://www.compeq.com.tw/accomplishments01_4_2.php)有 2025/5/23、8/29、11/24、11/25 的文件。

- [5/23 簡報](https://www.compeq.com.tw/doc/conference/c311d50a-322c-11f0-a9ba-005056a9c86d/20250523_135928_cn.pdf)第 7 頁：Aerospace 占比由 2024 年 18% 到 2025 Q1 23%。
- [8/29 簡報](https://www.compeq.com.tw/doc/conference/a71c5d79-8489-11f0-a762-005056a9c86d/20250829_134229_cn.pdf)第 7 頁：Q2 Aerospace 占比 22%。
- [11/24 清單連結](https://www.compeq.com.tw/doc/conference/8ac589b8-c689-11f0-a762-005056a9c86d/20251121_155159_cn.pdf)和[11/25 連結](https://www.compeq.com.tw/doc/conference/235724ee-c6af-11f0-a762-005056a9c86d/20251121_155406_cn.pdf)實際位元組完全相同，第 9 頁 Q3 Aerospace 占比 17%。PDF 封面沒日期，metadata 標題含 20251125，故保守統一 11/25 並標重複，不把檔名 11/21 当上網日。

航太類別可能還有其他業務，未取得可按當時日期驗證的 LEO 拆分；比重也不是營收成長率。因此不把 23%／22%／17% 自動標記成 LEO 曝險，不因比例回落便推論低軌訂單衰退。

另查閱[2024 年報](https://www.compeq.com.tw/doc/res_shareholder/79ad2ceb-03a0-11ef-a9ba-005056a9c86d/20250508_160621_cn_4.pdf)：封面刊印日是 **2025/3/20**，不是網址的 5/8；只有開發衛星、AI 等業務的計畫，位於本輪固定日期窗之前，未作催化訊號。[2024 永續報告](https://www.compeq.com.tw/doc/ent_csr/54373379-849a-11f0-a762-005056a9c86d/20251126_141856_cn.pdf)亦保留為已讀但未編碼，首次發布日期未確認。

## 研究含義

這批文件能支持明確分開「有概念」、「實際營收比重」、「實現增長」、「訂單」四件事。營運揭露品質不平均，若要求全部公司都公開相同比重，篩選結果也可能只是挑到披露較多的公司。下游應同時公布未知覆蓋、保留無訊號期，並把 3491 增補案例與七家公司原始群組分開。此表本身不證明策略勝率、報酬或可實戰。
