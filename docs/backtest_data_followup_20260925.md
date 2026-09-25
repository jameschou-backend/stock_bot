# 回測資料續補：獨立日表、上市身分與公告版本

本輪接續前20個日資料帳戶，補資料證據並找出衝突；沒有改選股條件、重算原報酬、啟動排程或送券商委託。舊封存結果保留，最新補件在工作台「策略驗證」最上方另外顯示。

## 普通盤檔案已有，對帳仍會揭露問題

前輪917個必要股日有逐筆檔，這輪開始與獨立官方日表按相同交易時段核對。已取得全部必要上櫃日期的官方「不含定價」日表，另用上市樣本驗證分項扣除方法。

FinMind逐筆檔含14:30盤後定價成交，不能把全檔總量當普通盤。上市日表又含普通盤、盤中／盤後零股、盤後定價及鉅額交易，必須取得所有分項後再扣除；有任何分項或組合鉅額明細缺漏，都不推定零成交。原逐筆模擬已限制09:01～13:25，並沒有把14:30用作盤中成交。

目前194股日的股數、金額與OHLC一致，42股日有差異，681股日仍缺相同口徑的獨立日表。42筆中5筆為股數缺量、37筆只有金額差異；後者可能涉及行情訊息聚合或價格語意，不能直接全稱為漏成交。原始差額與來源hash保留，不設容忍值把衝突消掉。

即使日總量一致，仍無法證明每個時間點、委託優先序或自己的成交。這輪日表核對不升級為完整逐筆／實盤資格。全部TWSE股日若沿目前分項端點補齊，最低另需約2,000次官方HTTP且仍可能缺組合明細；本輪只抓有界必要資料，沒有無限制爬取或消耗FinMind額度。精確結果、衝突分類及新查核入口見 [普通盤對帳紀錄](board_tape_reconciliation_20260925.md)。

## 上市身分已補上具體缺口

從官方歷史資料和公司年報核對：1507正式上市日為1989-11-09，2358為1996-12-18，原有未知起日2檔降為0檔。證券類別待查20降為18，日期差異19降為14；另外保留24筆官方產業異動。

1507在2022-04-14至04-20已停止交易，4/21才終止上市。新身分查核函式把停牌期間回傳不可交易，不能只依上市／下市日期認定仍可買進。這些是獨立overlay，未改原封存母體或歷史績效；其餘股票的完整停復牌史與歷史可投資母體仍不齊全。詳見 [歷史身分補件](historical_universe_followup_20260925.md)。

## 公告日期與版本分開保存

已擷取華新科官網索引中的8篇營收公告和來源收據。營收月份、可見刊登日期、CMS更新時間、實際取得原文的時間分開處理；較晚更正不覆蓋較早版本。同一份報告對歷史時點只回傳當時已被本系統觀察到的內容，今天抓到的舊公告不會自動回填到2022年。

例如2026年3月營收這篇官方新聞刊登於4/9，不能成為4/2的已知訊號。這批只是局部原文，不是完整MOPS版本鏈；MOPS目前入口遇到安全阻擋，沒有繞過或宣稱取得內容。詳見 [公告版本紀錄及手動保存方式](publication_versions_20260925.md)。

## 零股仍有外部來源限制

完整需求已整理為1,303個股日、204檔，含未成交委託；日表缺列不推斷零成交。官方免費資料、FinMind公開schema和Shioaji公開歷史介面目前沒有提供已核實可補完本案的獨立歷史零股序列。

已備妥市場／月份CSV及具體詢問稿，沒有寄送或採購。TWSE H4依先前核實的列價估算部分期間需逾8萬元，且仍不含全部近期與上櫃缺口；不能把買了某個產品當作全部完成。詳見 [零股來源查核與供應需求](odd_lot_source_followup_20260925.md)。

新版入口保留 `live_qualified=false`、`performance_recomputed=false`。本輪沒有新的策略報酬可報告；使用前仍需核對資料及成交假設。

## 驗收與重現

- `make test`：3,115項通過。新增獨立來源、日期／單位、衝突拒絕、公告版本時間、來源竄改、並行發布及畫面測試；帳本操作測試隔離本機研究快取，補件畫面另有專屬測試與真實資料驗收。
- `make pipeline`：連續兩次成功；`make api`後以curl核對`/health`、`/picks`、`/models`與`/jobs?limit=10`均正常。
- 4份補件來源重播／驗證通過；整合讀取約4.9秒，全程本機核對。真實Streamlit AppTest無例外、無錯誤、1個預期衝突警示、4個下載入口；瀏覽器已確認最新版數字可見。
- 對帳與原文擷取有界進行，來源快取保留在本機`.cache`，不把未取得或缺失來源當作已完成。`artifacts/forward_simulation/backtest_data_followup_20260925.json`綁定四份報告及原20案的SHA；缺少快取或指紋變更時，畫面拒絕顯示核對結果。

```bash
python scripts/audit_board_tape_reconciliation.py --verify
python scripts/audit_historical_universe_followup.py --verify --output .cache/historical-universe-followup-20260925/report-final.json
python scripts/prepare_odd_lot_evidence_request.py --verify --output .cache/odd-lot-provider-request-20260925-v3
python scripts/capture_publication_versions.py --verify .cache/publication-versions-20260925/walsin-v1/archive.json
```

上市身分核對使用`scripts.audit_historical_universe_followup.verify_report(Path('.cache/historical-universe-followup-20260925/report-final.json'))`；整合核對使用`app.backtest_data_followup_ui.load()`。新普通盤讀取入口拒絕衝突／缺件，尚未改寫原封存引擎或績效。
