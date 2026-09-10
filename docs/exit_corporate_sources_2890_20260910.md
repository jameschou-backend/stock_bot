# 永豐金2890：2026-07-23權息帳本來源查證

查證日：2026-09-10。只核公司行動來源，沒有修改override、交易引擎、資料庫或收益報告，沒有呼叫FinMind。

## 可供review的欄位

| 欄位 | 查證值 | 依據 |
| --- | --- | --- |
| stock_id / ex_date | 2890 / 2026-07-23 | 發行人官網、2026-07-07分派公告 |
| 每千股無償配發 | 20.00000000股 | 7/7公告「五、（五）權利分派內容」直接列示 |
| shares_per_share | 0.02 | 上列明示配發率除以1000，不從權息參考價或總發行股數反推 |
| 新股交付及開始買賣日 | 2026-08-24 | 6/26公告的新股發放日，加上8/14上市公告確認開始買賣日 |
| fractional_cash_per_share | 0，限本研究集保劃撥帳戶且未另外併湊 | 7/7公告七（2）：不足一股原按面額計至元，集保獲配者的畸零股款充抵劃撥費。因此本帳戶淨入帳為0，不能當成按市價出售或再加交易佣金 |
| 現金股利每股 | NT$1.10000000 | 發行人官網及7/7公告 |
| 現金股利付款日 | 2026-08-24 | 發行人官網、6/26及7/7公告 |
| 現金股利金額小數 | floor_ntd | 7/7公告七（1）：每位股東應領股利計算至整元，元以下捨去；不是把每股1.1元先取整 |

一般未透過集保且未併湊的畸零股，公告另訂面額折現及整元處理；不能把本研究的淨額0推广至所有股東。公告也列匯費、郵資由股東負擔；本輪既定帳本未計郵匯費，仍須保留此限制。

## 一手與轉載來源分開標示

1. **Direct issuer：永豐金官網股東會訊息**。頁面列114年度股利、7/23除權息、現金每股1.1元、股票每股0.2元、現金8/24發放。頁面沒有明示此次更新時間，不能用本次抓取時間冒充歷史公告時間。
   https://www.sinopac.com/investors/20211210153456867633/20211210153457700590.html

2. **MOPS內容轉載，不是本次直接從MOPS取得：nStock，2026-07-07 12:02:08**。標題「永豐金融控股股份有限公司115年盈餘轉增資發行新股暨分派現金股利公告」，公告序號1。完整內文藏在原HTML的`window.__NUXT__`之`article.content`；搜尋引擎與純body文字只顯示標題。已只解析字串、未執行JavaScript。原文標示由永豐金輸入，頁面資料鍵為MOPS解析公告查詢。
   https://www.nstock.tw/news/article_m?id=506514
   本次核心來源：精確20.00000000股/千股、現金1.10000000元、集保畸零款抵费、現金整元捨去。不是沿用2025年的相似條款。

3. **交易所上市公告轉載，不是本次直接從TWSE取得：nStock，2026-08-14 18:02:45**。標題「永豐金融控股股份有限公司（公司代號：2890）115年除權配股股票上市掛牌日期」。列本次普通股289,846,446股、8/24開始買賣、8/5變更登記完成，以及公司已於8/7將交付前公告輸入MOPS。此項補上實際核准上市日，沒有只依6/26預訂日期認列可交易股數。
   https://www.nstock.tw/news/article_m?id=534593

4. **MOPS重大訊息轉載：玉山證券／MoneyDJ，2026-06-26 14:35:01**。股利基準日公告第12項列現金8/24發放，第13項列增資新股同日發放；僅作與後續上市公告交叉核對。
   https://m.esunsec.com.tw/news/instant-detail.aspx?id=%7B24390692-F30C-4B49-A2AC-03D066CC175D%7D

## 本地資料交叉核對

- `.cache/exit-research-inputs/dividends/2890.parquet` 的114年度列：AnnouncementDate=2026-07-07、AnnouncementTime=11:03:34；CashExDividendTradingDate及StockExDividendTradingDate均為2026-07-23；StockEarningsDistribution=0.2、CashEarningsDistribution=1.1、CashDividendPaymentDate=2026-08-24。其公告時間與nStock轉載時間分開保留。
- `.cache/million-replay-inputs/events.parquet` 同股同日TWSE ex_rights記錄：event_type=權息、prev_close=40.4、ref_price=38.52、opening_ref=38.5；payload為合併權息值1.870589。這筆參考價不是精確股數或現金分項的證據。

## 取得證據的雜湊

以下是本次HTTP 200原HTML及解碼後公告內文的雜湊，方便再次抓取比對。沒有另外保存完整網頁副本；HTML含動態推薦內容，重抓的整頁雜湊可能改變，公告內文雜湊可單獨核對。

```json
[
  {
    "url": "https://www.nstock.tw/news/article_m?id=506514",
    "published": "2026-07-07 12:02:08",
    "retrieved_at": "2026-09-10T08:50:33.904203+00:00",
    "html_sha256": "41ae68a77e6489286ee0e9277c33de5261896eb83b53b3109bb344a9d97959eb",
    "decoded_content_sha256": "489ec69bef43219fe98f0b5960ec25601c804c4aa29dfc31d2ebccda43c56184"
  },
  {
    "url": "https://www.nstock.tw/news/article_m?id=534593",
    "published": "2026-08-14 18:02:45",
    "retrieved_at": "2026-09-10T08:50:34.545286+00:00",
    "html_sha256": "1ed9caf8f7742ac0240c76367e5948c50638dfb2fd8f8a8450c7d3c49a0c0056",
    "decoded_content_sha256": "1bbc58891e767285df619342e0115d21998e3cd55bf03f03c9d3d68a73f55669"
  }
]
```
