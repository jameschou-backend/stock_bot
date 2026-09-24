# 市場資料完整性與運算前檢查修正

2026-09-24，先處理使用者要求的第一優先項，沒有啟用或修改排程。

## 發現與修正

當日重新查詢時，全表股價及特徵最大日期為9/23，但TWSE只到9/22、TPEX到9/23；9/17、9/21、9/23存在市場覆蓋缺漏。9/23的data_quality與daily_pick仍曾記為success，說明全表新鮮度及最近多日平均門檻不足以保護同日橫斷面。

- 共用守門逐一檢查TWSE／TPEX，使用最近21日各市場最大筆數的90%門檻，要求最近7日曆天內交易日都齊全。另納入日曆日期，捕捉兩市場整日都沒有資料的情況。這是市場覆蓋檢查，不是每檔股票／所有歷史資料認證。
- 序列與DAG pipeline的data_quality入口，以及直接呼叫daily_pick，都必須通過；research／dev模式不能降級放行。
- 市場過濾器啟用時，同步檢查TAIEX報酬指數日期，移到特徵、標籤、訓練之前。未通過不發布新候選。
- 近期TAIEX查詢快取從24小時改為5分鐘，避免未完整發布的回應卡住一天；舊歷史仍保留24小時快取。維持共用FinMind限額，未刪限流帳本或繞過冷卻。
- 守門失敗在尚未寫候選／品質報表的階段，先保存failed job再拋錯，避免外層rollback把錯誤紀錄消掉。市場筆數查詢改用同一session connection，維持交易一致性。

## 實際資料與驗證

第一次手動pipeline已由FinMind增量補抓缺漏；截至本輪快照，TWSE與TPEX都到2026-09-24，分別1079／870檔有效普通股價格，最近缺漏清单清空。特徵從9/17重建12704列，標籤从8/25重建5657列。這不表示9/24所有來源已完整發布。

已修復的64筆行情、61筆受污染標籤再次查核：MySQL與四個價格／特徵／標籤快取都沒有重新出現。只驗證已審核主鍵，沒有宣稱剩餘94筆時序衝突、41個歷史起始日或全部還原因子已解決。

本輪共享FinMind帳本由3474增至3495，增加21次計費請求；當時有效上限5400/h，未等待限額。這是執行窗口的共用帳本變化，不把它說成獨立程序計量。

- `make test`：2581 passed、29 warnings，63.22秒。
- API重新啟動，curl驗證health、picks、models、jobs四端點均成功且JSON可解析。舊候選可讀不代表產生了9/24候選。
- `make pipeline`：**未完整成功**。最新股價為9/24，但FinMind TAIEX報酬指數未到9/24，現於data_quality的前置檢查阻擋；失敗紀錄已保存。修正前同一缺件直到建完特徵才發現，本輪最初那次特徵計算約63秒。後續不再為同一缺件重建特徵。
- 因完整pipeline驗收仍有外部來源阻擋，本輪程式先本機commit，未push。不可關掉市場過濾器或冒用9/23指數來宣稱驗收通過。

證據位於 `.cache/optimization-20260924/`：data-status.json、jobs-evidence.json、price-verification/storage-verification.json、pipeline-verified.log、preflight-blocked.log、make-test-verified.log。完整性檢查不會自動恢復排程，也不會下單。

## 同日後續驗收（17:52後）

TAIEX來源補至9/24後，原有守門未放寬，兩次手動`make pipeline`均正常完成；第二次features與labels新增列數均為0，daily_pick使用9/24指數／特徵且fallback_days=0，產出20個研究候選。重啟`make api`後四個curl端點成功，最新`make test`為2591 passed、29 warnings。驗收使用`INGEST_PRICES_SOURCE=finmind SPONSOR_INGEST=off AI_ASSIST_ENABLED=0`，避免擴大Sponsor抓取範圍；未修改`.env`、排程或正式策略。

資料品質仍報告5筆近期單日價格大幅變動警示，沒有據此認證全部行情、還原價或歷史母體。新驗收紀錄位於`.cache/broker-persistence-20260924/`的`pipeline.log`、`pipeline-repeat.log`、`api-*.json`與`make-test-final.log`。上述外部來源阻擋已解除；先前未push是較早驗收當下的狀態。
