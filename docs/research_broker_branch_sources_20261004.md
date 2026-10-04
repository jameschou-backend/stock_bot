# 分點連買與權證買盤：公開一手資料核對

查核日：2026-10-04。此子任務只讀公開文件與既有研究文件；FinMind 資料 API 請求為 0，未使用 token、未下載分點交易資料、未改策略或回測。以下是資料可行性與研究建議，不是已驗證的超額報酬。

## 可拿什麼資料

| FinMind 資料 | 文件歷史起點／更新 | Sponsor 與欄位要點 |
|---|---|---|
| `TaiwanStockTradingDailyReport` | 2021-06-30；平日21:00 | 可用；每次1日，依股票或分點查。`price`成交價、`buy/sell`股數，另有股票／分點代碼與日期。全市場整日檔限 Sponsor Pro。 |
| `TaiwanStockWarrantTradingDailyReport` | 2023-06-21；**同頁分別寫01:00與23:00** | 可用；每次1日、價格層買賣數。整日檔限 Sponsor Pro。資料商品是權證，數量不可當母股股數。 |
| `TaiwanStockTradingDailyReportSecIdAgg` | 2021-06-30；平日21:00 | 可用；官方示例股票＋分點＋起迄日，回每日 `buy_volume/sell_volume`股數、`buy_price/sell_price`均價。 |
| `TaiwanStockBrokerDailyConcentration` | 2021-06-30；盤後 | Backer/Sponsor；`top_k=15`、前15買超與賣超總股數，沒有個別分點身份。 |

以上依 [FinMind 籌碼文件](https://finmind.github.io/tutor/TaiwanMarket/Chip/)。分點明細不含鉅額；已揭露買賣合計可能不平衡，不可自行補平；文件列缺漏2022-10-31～11-03、2023-01-11～01-17。更新時刻只是文件時間，實際可用性仍需收件證據。

| 其他必要資料 | 文件與用途 |
|---|---|
| `TaiwanStockDayTrading` | 2014-01-01起；標的／`BuyAfterSale`盤前，量值21:30。`Volume`股數、`BuyAmount/SellAmount`金額（元）；個股區間免費，全市場單日限Backer/Sponsor。 |
| `TaiwanStockInfoWithWarrantSummary` | Sponsor，每日01:30；上市／上櫃權證母股映射。只明示上櫃歷史回溯2011-01-03，未確認上市完整起點。欄含上市／末交易日、母股、認購售別、行使比例、履約價。 |
| `TaiwanStockInfoWithWarrant` | 每日01:30，名稱／代碼／市場／產業／更新日；此節未明示完整歷史起點或額外方案門檻，不能視為歷史身分全認證。 |
| `TaiwanSecuritiesTraderInfo` | 免費分點目錄：代碼、名称、開業日、地址、電話。文件未承諾完整更名／合併沿革或更新時刻。 |

依 [FinMind 技術文件](https://finmind.github.io/tutor/TaiwanMarket/Technical/)及 [FinMind 官方 MCP 資料目錄](https://github.com/FinMind/FinMind-MCP/blob/master/knowledge/datasets.md)。權證代號會重用；母股關聯必須以交易日落在 `date`～`end_date` 區間連接，不能套今日對照到全部歷史。

## 官方資料的意思與時序限制

- 證交所說明券商對個股買賣資訊包含自營與經紀受託客戶的合計。**分點不是受益人帳戶**。由此推論，連買可作為該通路持續淨流入特徵，不能直接稱「同一大戶加碼」「尚未出貨」「真實持股成本」。轉倉、不同客戶互抵與既有庫存也無法靠這張流量表辨認。[TWSE 買賣日報說明](https://eshop.twse.com.tw/zh/category/sub/29)
- TPEx 一般交易買賣日報頁公告當日資料約16:00，與供應商21:00不是同一發布管道；網頁有驗證碼，未嘗試繞過。[TPEx 券商買賣日報](https://www.tpex.org.tw/zh-tw/mainboard/trading/info/brokerBS.html)
- 官方當沖量值是**股／元**；TPEx 明示 T、T+1 可修訂，最終以 T+2 更新為準。FinMind 現在取得的歷史最終版不能無版本證據當作 T 當晚已知。若無歷史收件快照，研究可明示修訂風險，或採保守延後可用日，且將延後與即時版分開。[TPEx 當沖統計說明](https://www.tpex.org.tw/storage/zh-tw/web/stock/trading/intraday_stat/intraday_trading_statD.htm)
- FinMind 標示2014-01-01起是供應商區間；TWSE頁明示2014-01-06起提供，不能假設2014-01-01有實際交易列。[TWSE 每日當沖標的](https://www.twse.com.tw/zh/trading/day-trading.html)
- 權證發行商依模型動態避險，發行權證不必然代表方向看法。買進認購權證、買進認售權證、發行人買回權證、發行人買進母股避險是不同事件，不能相加成同一种看多資金。[TWSE 權證教育](https://www.twse.com.tw/zh/products/securities/warrant/educate/qa.html)、[TWSE 避險說明](https://wwwc.twse.com.tw/market_insights/zh/detail/ff8080818cc7b692018cf62c2b26010e)
- 權證數量需考慮行使比例；若估算避險母股等值，還需 Delta、流通在外／發行人持有、認購認售互抵與其他避險工具。**成交量不是未平倉部位**，現有分點＋母股對照不足以還原真實避險庫存。[TWSE 權證教育](https://www.twse.com.tw/zh/products/securities/warrant/educate/qa.html)、[證券商公會 Delta 避險說明](https://www.twsa.org.tw/B01/doc/104Q3.pdf)

## 截圖分點的代碼對應

| 可辨識名稱 | 官方可對應代碼 | 證據／注意 |
|---|---|---|
| 永豐金內湖 | `9A9g` | [永豐金官方據點](https://www.sinotrade.com.tw/Friendly_service/Location)。小寫 g；`9A9G` 是天母，不能統一轉大寫。 |
| 元大南屯 | `9853` | [元大官方分公司文件](https://static.yuanta.com.tw/staticFile/eyuanta/resourcesFile/f67e23fc-5ad1-483e-af02-3557bef30b68.pdf)、[元大合併對照](https://www.yuanta.com.tw/file-repository/content/2013sitax/cb_2012y/pl2.htm)。 |
| 富邦公益 | `961F` | [TPEx 2024-04-23公告](https://www.tpex.org.tw/storage/eb_data/11304/1130059533.html)：2024-04-29起由中港更名公益。歷史顯示名稱應按日期。 |
| 兆豐嘉義 | `7001` | [兆豐官方據點](https://www.emega.com.tw/emegaTran/saleSpot.do)。 |
| 元大館前 | `984K` | [元大官方分公司文件](https://static.yuanta.com.tw/staticFile/eyuanta/resourcesFile/f67e23fc-5ad1-483e-af02-3557bef30b68.pdf)、[元大2014遷址公告](https://www.yuanta.com.tw/file-repository/content/20140414/notice103_0318/notice103_0318/notice103_0318.html)。 |

只有「竹科／嘉義／台中」而無可辨識券商全名的列保持未知，不補猜。上表核實名稱與代碼，不代表已取得全期間有效性名冊，更不證明這些分點具有穩定選股優勢。

## 建議的有限研究範圍（研究設計，尚未執行）

1. 先研究既有候選股的分點連買，不再用事後飆股名單選樣。固定原始買進／出場规则，比較有無特徵，保留所有虧損與未知案例。
2. 對指定分點使用每日淨買、連買天數、買賣不平衡率 `(buy-sell)/(buy+sell)`；分點淨流入集中度與 TDCC 持股集中度分開命名。先檢驗分點不平衡／缺日，不把缺資料當零。
3. 同時控制價格動能、成交值、市場、產業、當沖與法人自營避險，回答「是否提供原先價量之外的新增訊息」。個股整體當沖比率不能直接扣某個分點來推其隔夜持股。
4. 把券商通路買股、權證買盤、母股自營避險分三組假說；後兩組以2023-06-21後且歷史映射完整的共同樣本為主。權證價款可按成交價格×單位數彙總，不能將不同權證單位直接當同一母股曝險相加。
5. 不預先認定截图分點是「主力名單」。先固定列表，按年度／股票分群驗證；同股重複訊號及同日市場共振需分群統計，避免把密集訊號當獨立樣本。保留未知涵蓋率與漏掉大漲股的比例。
6. 效能優先重用 `SecIdAgg` 區間快取。此接口已在[2026-09-24既有研究](/Users/james.chou/JamesProject/stock_bot/docs/research_broker_persistence_20260924.md)實測2,212次請求（含2次探測），成功取得的訊號日加總亦與封存價位層相符；不能寫成接口尚未實測。該研究458候選僅251個具完整5日、125個具完整20日，缺列仍未知，不補零。純連買不必重抓價格層；未有證據的全歷史單次查詢上限另列未知。共用使用者每小時6000上限與持久快取，不與POC抓取競速。本輪不新增任何API額度。

未解事項：權證分點01:00/23:00矛盾、當沖歷史版本、完整券商沿革、權證實際庫存與Delta、分點披露完整性與各市場成交範圍。這些缺口不能靠模糊名稱、淨買累加或空表補零解決。
