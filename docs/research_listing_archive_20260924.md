# 原始上櫃日期補件

原先41檔缺起日，上輪1258已取得原公告；本輪官方年鑑再核實23檔，合計24檔，剩17檔。原始文件保存在本機快取，沒有把搜尋摘要、第一筆行情或公司成立日當成掛牌日。

| 代號 | 原始上櫃日 | 一手來源位置 |
|---|---|---|
| 1258 | 2011-12-12 | [factbook100](https://www.tpex.org.tw/storage/publish/factbook/100/100.pdf) PDF第28頁 |
| 1752 | 2009-08-10 | [factbook98](https://www.tpex.org.tw/storage/publish/factbook/98/98.pdf) PDF第28頁 |
| 3089 | 2003-10-14 | [factbook92](https://www.tpex.org.tw/storage/publish/factbook/92/92.pdf) PDF第41頁 |
| 3202 | 2005-07-26 | [factbook94](https://www.tpex.org.tw/storage/publish/factbook/94/94.pdf) PDF第38頁 |
| 3642 | 2011-09-29 | [factbook100](https://www.tpex.org.tw/storage/publish/factbook/100/100.pdf) PDF第27頁 |
| 3652 | 2009-08-27 | [factbook98](https://www.tpex.org.tw/storage/publish/factbook/98/98.pdf) PDF第28頁 |
| 4130 | 2012-01-12 | [new101](https://www.tpex.org.tw/storage/publish/factbook/101/03_1.doc) 新上櫃公司表 |
| 4429 | 2010-09-15 | [factbook99](https://www.tpex.org.tw/storage/publish/factbook/99/99.pdf) PDF第25頁 |
| 4736 | 2010-12-01 | [factbook99](https://www.tpex.org.tw/storage/publish/factbook/99/99.pdf) PDF第25頁 |
| 4944 | 2011-05-31 | [factbook100](https://www.tpex.org.tw/storage/publish/factbook/100/100.pdf) PDF第27頁 |
| 4945 | 2020-11-20 | [new109](https://www.tpex.org.tw/storage/publish/factbook/109/03_1_一般類股新上櫃公司彙總表New Listing on TPEx in 2020.doc) 新上櫃公司表 |
| 4987 | 2012-03-21 | [new101](https://www.tpex.org.tw/storage/publish/factbook/101/03_1.doc) 新上櫃公司表 |
| 6247 | 2003-07-04 | [factbook92](https://www.tpex.org.tw/storage/publish/factbook/92/92.pdf) PDF第40頁 |
| 6287 | 2003-10-02 | [factbook92](https://www.tpex.org.tw/storage/publish/factbook/92/92.pdf) PDF第40頁 |
| 6404 | 2014-07-23 | [new103](https://www.tpex.org.tw/storage/publish/factbook/103/03_1.doc) 新上櫃公司表 |
| 6457 | 2015-06-12 | [new104](https://www.tpex.org.tw/storage/publish/factbook/104/03_1.doc) 新上櫃公司表 |
| 6472 | 2017-04-19 | [new106](https://www.tpex.org.tw/storage/publish/factbook/106/03_1_106年上櫃股票新掛牌公司彙總表.doc) 新上櫃公司表 |
| 6514 | 2015-12-02 | [new104](https://www.tpex.org.tw/storage/publish/factbook/104/03_1.doc) 新上櫃公司表 |
| 6589 | 2019-06-28 | [new108](https://www.tpex.org.tw/storage/publish/factbook/108/03_1_一般類股新上櫃公司彙總表New Listing on TPEx in 2019.doc) 新上櫃公司表 |
| 6594 | 2017-04-28 | [new106](https://www.tpex.org.tw/storage/publish/factbook/106/03_1_106年上櫃股票新掛牌公司彙總表.doc) 新上櫃公司表 |
| 8406 | 2012-04-27 | [new101](https://www.tpex.org.tw/storage/publish/factbook/101/03_1.doc) 新上櫃公司表 |
| 8418 | 2011-12-06 | [factbook100](https://www.tpex.org.tw/storage/publish/factbook/100/100.pdf) PDF第28頁 |
| 8420 | 2014-11-10 | [new103](https://www.tpex.org.tw/storage/publish/factbook/103/03_1.doc) 新上櫃公司表 |
| 8476 | 2017-03-28 | [new106](https://www.tpex.org.tw/storage/publish/factbook/106/03_1_106年上櫃股票新掛牌公司彙總表.doc) 新上櫃公司表 |

PDF頁數包含封面。92年PDF中文字編碼部分損壞，股票代號、掛牌日及英文欄頭可直接核對；僅解析事先審閱的新上櫃頁面，排除後面興櫃登錄表。1258與前輪2011-12-12原公告一致。每份原檔、下載紀錄及轉錄文字均記SHA256；稽核時由原檔重新轉文字比對，再解析，不單獨信任可編輯文字檔。

年鑑證實歷史日期，但不是原始公告逐版檔案。announcement_available_at仍空白，不回填為當時可得訊號；起日與終止日之間的停牌、恢復、改名與特殊交易方式仍須逐日核對，continuous_eligibility_proven=false。本輪沒有修改DB或既有封存回測。

未完成：5820、5102、5306、5281、4712、5383、6446、6747、5371、5236、3426、2809、2358、1701、2841、1507、9188。

部分舊年鑑需不同章節格式；105與110年目錄目前取回失敗，96年完整PDF傳输失敗，均保留失敗紀錄、不當空資料。TWSE六個舊公司PDF本機未成功取得；2809搜尋快取雖有日期，但目前官網原件不存在，未納入核實數。

歷史零股的逐次真實成交（不是試撮、整股逐筆或日統計）及完整公告修訂仍缺。這輪不購買資料，原始檔取得後才能完成嚴格成交重播。已另備資料商詢問清單於readiness_data_request_20260914.md，尚未寄送。

重現：`python scripts/audit_listing_archive.py --output /tmp/listing-audit.json`。需本機已保存來源、Poppler的pdftotext及macOS textutil。報告`.cache/listing-sources-20260924/report.json`包含全部來源SHA256與未解項目；0 DB寫入，live_qualified=false。
