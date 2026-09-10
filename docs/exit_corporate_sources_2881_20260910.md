# 富邦金2881：2024-09-09配股帳本來源查證

查證日：2026-09-10。只核公司行動來源，沒有修改override、引擎、封存輸入或收益報告，沒有呼叫FinMind。配股率與實際新股交付日已核對；**集保畸零股款淨額及其付款日尚未取得足夠證據**，不可標成全部事實已驗證。

## 可供review的欄位

| 欄位 | 查證值 | 依據與範圍 |
| --- | --- | --- |
| stock_id / ex_date | 2881 / 2024-09-09 | 8/23分派公告與本地TWSE事件一致，為單獨除權 |
| 每千股無償配發 | 50.00000000股 | 8/23分派公告五（五）明示資本公積配發率 |
| shares_per_share | 0.05 | 明示配發率除以1000；不從總發行股數或除權參考價反推 |
| 新股交付日 / pay_date | 2024-10-15 | 10/9轉載的正式發放公告參、一明示當日直接撥入指定集保帳戶；不是只採8月的暫訂日期 |
| 新股開始買賣日 | 2024-10-15 | 正式發放公告與10/8交易所上市公告轉載一致 |
| 本次普通股現金股利 | 0 | 9/9是除權，8/23公告的除息、現金股利及現金付款日欄均空白，本地資料亦為0 |
| 不併湊的不足1股權利毛額 | floor(fraction × 10)元，其中fraction為不足1股的部分 | 2024年發行人股東會通知第2頁第六聯第四項；這是面額折算至元的毛額規則，**尚不等於集保實際淨入帳** |
| fractional_cash_per_share，若此欄代表集保淨額 | **未核實，不建議填0或10並標成verified** | 已閱當年通知兩頁及後續發放公告，未找到畸零款充抵集保劃撥費、免收費或精確淨額的條款 |
| 畸零股款現金付款日 | **未核實** | 新股10/15交付公告只明示股票撥入集保，沒有明示不足1股現金於同日入帳 |
| 同年度另一次現金股利 | 每股NT$2.5；2024-07-19除息，2024-08-21付款 | 本地股利另有獨立列，發行人決議執行情形亦確認8/21現金發放完成；不能在9/9再加一次 |
| 同年度普通股現金取整規則 | floor_ntd，按每位股東應領總額處理 | 當年通知第2頁第六聯第二項（五）列元以下捨去；不是把每股2.5元先取整，也不是9/9另有現金股利 |

2024年通知允許在停止過戶日起5日內自行併湊成整股；本研究沒有主動併湊操作。原始取整短句位於PDF第2頁最下方「第六聯」第四項，為「按面額折發現金至元為止」。不足1股現金的面額折算，與可在交易所買賣的1至999股零股，是不同事項。面額折算不得改用當日市價，也不能套一般股票賣出佣金。

## 一手與轉載來源分開標示

1. **Direct issuer：富邦金控2024年股東常會通知。** [發行人歷年股東會頁](https://www.fubon.com/financialholdings/governance/shareholders.html)的113年股東常會項目連到[當年開會通知PDF](https://www.fubon.com/financialholdings/governance/shareholders/MEETING-NOTICE.pdf)。PDF為2頁，印製日期2024/5/3，會議日期2024/6/14；本次已將兩頁渲染並逐頁檢視，確認文件年份與頁面小字。第2頁第六聯第四項載明每千股50股、畸零股併湊期限及面額折現至元；第二項（五）載明普通股現金總額元以下捨去。第1頁的集保徵詢說明載明股票發放日撥入指定帳戶，**沒有畸零股款抵費條款**。當頁的20/100是融資擔保品的無償配股門檻，不是NT$20或NT$100費用。

2. **MOPS內容轉載，非本次直接從MOPS取得：nStock，2024-08-23 10:02:29。** [分派112年度資本公積轉增資發行新股基準日](https://www.nstock.tw/news/article_m?id=217751)。五（五）列每千股50.00000000股；每股面額10.0000元、本次發行普通股650,748,662股、參與權利分派普通股13,014,973,243股。五（七）列9/9除權、9/15基準日。股利所屬112年是2023會計年度，不是2023發放。此時新股交付及上市只暫訂10/15。公告內文從原HTML的`article.content`字串解碼取得，沒有執行JavaScript。

3. **MOPS發放公告轉載，非本次直接從MOPS取得：玉山證券／MoneyDJ，2024-10-09 13:36:38。** [資本公積轉增資新股10/15起發放並上市買賣](https://m.esunsec.com.tw/news/instant-detail.aspx?id=%7BF61EF278-230A-4488-8C78-AC48385DF1CB%7D)。公告列2024/10/7變更登記核准，並在參、一明示10/15新股發放、同日上市以及當日直接撥入集保帳戶。這是交付日期的主要後續證據。沒有列畸零股款付款日或集保抵費細則。

4. **交易所上市公告轉載，非本次直接從TWSE取得：nStock，2024-10-08 20:01:09。** [113年資本公積轉增資股票上市掛牌日期](https://www.nstock.tw/news/article_m?id=226118)。明示650,748,662股、113年10月15日開始買賣、113年10月7日登記核准。原文伍、二寫成「112年10月08日」將交付前公告輸入MOPS，與本次113年事件年份不一致；保留為來源文字異常，不擅自改成113年，也不使用這一段推定公司的精確申報日。新股10/15交付另有上列完整公告支援。

5. **Direct issuer：富邦金控2024年股東會決議執行情形。** [發行人PDF](https://www.fubon.com/financialholdings/governance/shareholders/113_C_EGMResolutionsAndImplementationStatus.pdf)。1頁已渲染檢視；內文標題是2024年決議執行情形，搜尋或PDFmetadata可能顯示不相干的舊新聞稿標題，不能採metadata判斷年份。第2項列普通股現金每股2.5元，且8/21發放完成；第3項確認資本公積增資及新股發行已完成，但本表本身未列股票發放日期。

## 未核實欄位的處理界線

- 已查到的當年原始條款僅支持畸零權利按面額NT$10折現至元。不能因為未看到費用文字就宣稱集保完全免扣費；也不能把其他公司的抵費條款移植到富邦金。搜尋結果中富邦媒、其他委任富邦證券股務代理的公司或其他年度文件均不作本次證據。
- 若帳本欄位`fractional_cash_per_share`代表扣除集保費後的入帳金額，現有證據不足以將它核實為0或10。`pay_date=2024-10-15`已核實的範圍是整數新股，不能以同一日期補畸零現金的實際入帳時間。
- 已知畸零毛額可獨立記為付款日未核實的現金應收，不轉成可動用現金；整數新股仍依已核實日期交付。這樣的帳務分離不構成費用或付款日已驗證，須繼續顯示未核實費用及入帳時間的限制。
- 若後續採按面額、忽略另列費用、假定同日付款的研究假設，須與已驗證公司事實分開記錄，不能標成fully verified。若規格要求精確淨額與付款日，就仍需當年股務配股通知、費用說明或實際股利入帳證據。這份查證沒有以新增假設解除來源檢查。

## 本地資料交叉核對

- `.cache/exit-research-inputs/dividends/2881.parquet` 的配股列：year=`112年`、date=`2024-09-15`、AnnouncementDate=`2024-08-23`、AnnouncementTime=`09:58:04`；StockStatutorySurplus=`0.5`、StockEarningsDistribution=`0`、StockExDividendTradingDate=`2024-09-09`；現金分項均0，現金除息日及付款日空白。資料源公告時間與nStock轉載時間分開保留。
- 同檔另一次現金列：date=`2024-07-27`、AnnouncementDate=`2024-07-04`、AnnouncementTime=`16:55:59`、CashEarningsDistribution=`2.5`、CashExDividendTradingDate=`2024-07-19`、CashDividendPaymentDate=`2024-08-21`；配股分項均0。
- `.cache/million-replay-inputs/events.parquet` 同股9/9為TWSE `ex_rights`、event_type=`權`、prev_close=`92.5`、ref_price=`88.09`、opening_ref=`88.1`、payload為`{"value_amount":4.404762,"kind":"權"}`。同股7/19另為`息`、payload的value_amount=`2.5`。除權價差不是精確股數、畸零現金或付款時間的證據。
- 查證時股利檔SHA256：`1ff40f5d24d21899b3b170df171201b29cfdfd8a19241b7a825efc0e6972ea11`；官方事件檔SHA256：`e132c22c58f1e5814fcfffe44dd33dd87e5996c49e5923dd1874466426e0c709`。

## 取得證據的雜湊

本次公開網頁及PDF均HTTP 200。以下雜湊可核對此次來源版本，未將完整來源頁保存進repo。PDF通用檔名可能被發行人覆寫，故同時保留文件年份、頁數與雜湊；HTML動態區塊可能使重抓整頁雜湊改變，nStock另列解碼後公告內文雜湊。

```json
[
  {
    "url": "https://www.fubon.com/financialholdings/governance/shareholders/MEETING-NOTICE.pdf",
    "retrieved_at": "2026-09-10T09:48:37.165962+00:00",
    "sha256": "5e2ace93f6ee5173a8404acac313e583188c3812850b28f1e462a16798ba80ae"
  },
  {
    "url": "https://www.fubon.com/financialholdings/governance/shareholders/113_C_EGMResolutionsAndImplementationStatus.pdf",
    "retrieved_at": "2026-09-10T09:48:24.896589+00:00",
    "sha256": "a30301185fee41928ffd899126924081179bd0a092fd7b1fb21a98b2b1756707"
  },
  {
    "url": "https://www.nstock.tw/news/article_m?id=217751",
    "published": "2024-08-23 10:02:29",
    "retrieved_at": "2026-09-10T09:48:24.488883+00:00",
    "html_sha256": "11ee98e338f3136ff9b1de5c4674e6b2058f279766ca25d5f02826b06fd57367",
    "decoded_content_sha256": "9c1813b9c307647dea22a2685267d13c9a46d19de29c6d85f3f71a72c70f7297"
  },
  {
    "url": "https://www.nstock.tw/news/article_m?id=226118",
    "published": "2024-10-08 20:01:09",
    "retrieved_at": "2026-09-10T09:48:24.487542+00:00",
    "html_sha256": "eb40afb74b9a041ec77fe44aec52e30acc96763c725188d26d53ceb427f6f135",
    "decoded_content_sha256": "36925714e84948d93e84d0e99d0f8f0cc64de0dd711048cf52715b6a87f88531"
  },
  {
    "url": "https://m.esunsec.com.tw/news/instant-detail.aspx?id=%7BF61EF278-230A-4488-8C78-AC48385DF1CB%7D",
    "published": "2024-10-09 13:36:38",
    "retrieved_at": "2026-09-10T09:54:53.264458+00:00",
    "html_sha256": "684874bba3eb0a0e5b88595d010be959abaa8ff1b70a262b503515f46c2c4a23"
  }
]
```
