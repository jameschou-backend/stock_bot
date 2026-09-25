# 歷史身分補件：精確上市日、日期衝突與停止交易（2026-09-25）

新增獨立 overlay，**原封存資料、原策略及歷史報酬均未改寫**。這次把剩餘兩筆上市起點補成有文件支持的精確日期，另釐清五筆 ISIN 日期差異，並加入已證實的停止交易排除區間。仍不是完整的歷史可投資母體。

| 檢查項目 | 前次 | 本次 |
|---|---:|---:|
| 既有名冊的精確上市起點未知 | 2 | 0 |
| 證券類別未確認 | 20 | 18 |
| 尚未逐件釐清的當前日期差異 | 19 | 14 |
| 新增有來源的 TPEx 產業異動 | — | 24 |

上述分母限於既有名冊，不代表已找齊所有歷史公司、上市櫃轉換、停止／恢復交易或分類異動。

## 兩筆上市日起點

- **1507 永大：1989-11-09**。TWSE 60 周年特刊《上市公司異動》PDF 第 3 頁（印刷頁 439）明列民國 78 年 11 月 9 日永大機電上市。與公司 2020 年報 PDF 第 65 頁的 `Common stock / Listed stock` 核對普通股類別；交易所終止上市新聞稿另外核對公司代號與名稱。[交易所歷史年表](https://www.twse.com.tw/staticFiles/product/publication/twse60/P17.pdf)、[公司年報](https://en.hitachi-yungtay.com.tw/wp-content/uploads/2022/01/109%E5%B9%B4%E5%A0%B1-%E8%8B%B1%E6%96%87lock.pdf.pdf)。
- **2358 廷鑫：1996-12-18**。已存的 2022 年報 PDF 第 98 頁（印刷頁 94）直接記錄完整年月日及美格科技更名廷鑫；第 1 頁綁定股票代號，第 194 頁核對全數普通股。先前只核對第 8 頁公司簡介，其日期僅精確到月份；本次讀取財報附註找到日。**1991-12 的公開發行不是上市日**。該份年報包含重編後財務報表，僅擷取歷史身分事實，沒有把重編數字或董事會通過日當作過去已可得特徵。

這些是官方／發行人的回溯文件；兩者的原初上市公告及當時可得時間仍未取得，`announcement_available_at=null`。

## ISIN 日期差異的原因

| 股票 | 更正後原上櫃日 | 原 ISIN 日期 | 已核對事件 |
|---|---|---|---|
| 3313 | 2006-05-29 | 2026-06-01 | 其他 → 建材營造 |
| 4905 | 2001-12-07 | 2026-06-01 | 通信網路業 → 生技醫療 |
| 5381 | 1999-03-19 | 2026-06-01 | 電子零組件業 → 電機機械 |
| 3521 | 2007-08-07 | 2025-06-02 | 電腦及週邊設備業 → 建材營造 |
| 5348 | 1998-04-29 | 2025-06-02 | 通信網路業 → 運動休閒 |

原上櫃日重用 2026-09-24 官方公司基本資料的 `上櫃日期` 欄；事件由 TPEx 原公告確證，兩公告均明示證券代號維持不變。因此不能把產業調整日當成首次上櫃日。[2025 公告](https://www.tpex.org.tw/storage/eb_data/11405/11402010541.html)、[2026 公告](https://www.tpex.org.tw/storage/eb_data/11505/11502011171.html)。

兩份公告完整抽出 11＋13 筆異動，包含官方 2025 年原文一處缺右括號及一處重複引號。解析器核對完整列數與代碼唯一性，缺列或重複即失敗。這是交易所單一產業分類，與 FinMind 多重供應鏈分類沒有直接對應證據；不回填原模型成分。公告只有日期，正式時分秒、前後完整區間仍留空。

## 上市不等於可以交易

1507 因股份轉換於 **2022-04-14 停止買賣、2022-04-21 終止上市**。若只用終止上市日當排除日，4/14–4/20 會被錯認為仍可下單。因此新 `resolve_followup()` 在這個半開區間回傳 `official_trading_suspension` 與 `tradable=false`，4/21 起為 `outside_verified_intervals`。[TWSE 2022-03-09 公告新聞稿](https://www.twse.com.tw/staticFiles/news/news/tsecnews/ff8080817d22b9cb017f6d9b33a80729.pdf)。

這只修補一個已核實區間；其他日期回傳 `identified` 也不代表已認證所有停止／恢復交易事件。報告永遠保留 `continuous_eligibility_proven=false`、`complete_historical_universe=false` 與 `live_qualified=false`。

## 重播與使用

```bash
python scripts/audit_historical_universe_followup.py
python scripts/audit_historical_universe_followup.py --verify
python -m pytest tests/test_historical_universe_followup.py -q
```

報告 `.cache/historical-universe-followup-20260925/report-final.json` 與同名 `.sha256`，schema 為 `historical_universe_followup_v1`。`scripts.audit_historical_universe_followup.verify_report(path)` 驗報告、舊來源閉包、新原始文件／回應 metadata／已檢視 PNG 及程式指紋，並重新抽取實際頁面與資料列。PDF 驗證只讀所需頁面，不每次解析整份數百頁年報。

```python
from scripts.audit_historical_universe_followup import verify_report
from skills.historical_universe_followup import resolve_followup
report = verify_report()
identity = resolve_followup(report, '1507', '2022-04-14')
assert identity['status'] == 'official_trading_suspension'
```

19 項針對上市前後、停止交易前後、終止日、日期更正前後、來源 hash 與官方原文異常的測試通過。報告另附 13 筆實際來源 `before_after_examples`；可離線查詢，沒有 DB 回寫、策略回測、排程或下單。本次新增 **3 次官方 HTTP 擷取、0 次 FinMind**，其餘重用已保存來源；驗證重播 0 網路。

剩餘 18 類別與 14 日期差異清單完整保存在報告中。包括換股、改名、轉板、重新上市等可能性，沒有官方逐項文件支持前不擅自判定。
