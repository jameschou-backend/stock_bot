# 2881：2025 年資本公積轉增資補件結果

正式新增 `overrides` 為空，配股交付資料缺口保留。這次只補來源查核與檔案 SHA256，沒有變更配股、成交、現金或策略規則，也沒有採用次級來源日期執行另一個情境。

`capacity_control_board_only.json` 的原停止原因為 `Stock dividend data missing or invalid: 2881 2025-09-25; shares_per_share, pay_date`。這個事件實際是**資本公積轉增資**，不能寫成盈餘配股。本地舊 override 只有 2024 年；2025 年 FinMind 股利快取中的 `2025-10-01` 是配股基準日，不能當作交付日。`2025-07-31` 則是現金股利發放日，也不能代替新股交付日。

| 欄位 | 已取得的證據 | 本次接納情況 |
| --- | --- | --- |
| 配股比例 | [富邦金控 2025/6/13 新聞](https://www.fubon.com/financialholdings/news/news_1250613_799862.htm) 說明每股配股金額 0.25 元、面額 10 元；[官方 2024 年報](https://www.fubon.com/financialholdings/governance/shareholders/2024_2881_AnnualReport_CN.pdf) 第 129 頁所列 2025 年股東會提案為每千股 25 股。 | 可支持研究核對 `shares_per_share=0.025`；網路搜尋索引文字並非本地完整 primary raw 檔，未單獨加入 override。 |
| 新股交付／上市日 | [MoneyDJ 轉載的 MOPS 公告](https://www.moneydj.com/kmdj/news/newsviewer.aspx?a=503f76a2-fdbd-413e-8184-de0c5a8814ea) 稱 2025/11/14 開始發放並上市，同日撥入指定集保帳戶。已保存完整 HTML 及 SHA256。 | 次級來源追查線索；未取得直接 primary 公告，不輸入回測。 |
| 官方執行狀態 | [2025 年股東會決議執行情形](https://www.fubon.com/financialholdings/governance/shareholders/114_C_EGMResolutionsAndImplementationStatus.pdf) 表示已完成新股發放。 | 未提供確切日期，無法填補 `pay_date`。 |
| 不足一股的處理 | 官方 2024 年報第 129 頁的 2025 提案說明，停止過戶起 5 日內得併湊，仍不足一股按面額折現至整元。 | 可供條款核對；尚缺完整 final primary 套件及實際淨額、扣款與支付日，不假定現金已可用。 |

截至事件前 2025/9/24，停止帳本留有 2881 的 150 股。若套用上述比例與折現條款，純算術為 `150 × 0.025 = 3.75` 股，也就是 3 股與 `floor(0.75 × 10) = 7` 元的毛額。這不是已確認收到的股票或現金；本次未入帳。交付日與零碎股款時點仍須取得可接納的 primary 證據。

直接 MOPS legacy 查詢的 3 次回覆分別是「查無所需資料」、「日期輸入錯誤」與「查無所需資料」。這只表示本次查詢未取得指定公告，不能推論公告不存在。完整回覆與查詢參數已保留。

請求已收尾：官方明示讀取嘗試 **10 次**，其中完整 tool read 8 次、DNS／未完整 PDF 傳輸失敗 2 次；網路搜尋 **8 次、16 個 queries**；次級來源 GET **1 次**；本子任務 FinMind **0 次**。官方計次包含失敗與無資料回覆；搜尋服務內部的 HTTP、轉址或重試次數不可觀測，沒有宣稱精確的遠端 HTTP 總量。主任務的 FinMind 限價補件另計。

可機器讀取的新增檔為 [board_only_corporate_additions_20260925.json](board_only_corporate_additions_20260925.json)，包含空 `overrides` 與證據 SHA256。完整狀態、逐次請求表及原文位於 `.cache/board-only-supplement-20260925/` 的 `2881-2025-source-status.json`、`corporate_request_ledger.json`、`corporate_evidence_manifest.json` 與相鄰 HTML。原封存來源和策略程式均未修改。

驗證：JSON 可解析、12 個 evidence SHA256 全部相符、正式新增 override 數為 0。這次只有資料與報告檔，未執行 pipeline、DB 寫入、下單、購買資料或排程。
