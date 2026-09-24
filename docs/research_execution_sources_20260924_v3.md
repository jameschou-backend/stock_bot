# 歷史零股來源再查與 TPEx 實樣核對 — 2026-09-24

本輪取得並解析了 **TPEx 官方 MTH 免費實樣**，核對出零股買賣雙邊列與盤後成交必須分開。這比前輪只查商品頁多了一層實證，但仍沒有取得首個缺件日的撮合序列；跨日回放維持缺件停止，沒有改寫原有績效、DB 或成交來源。

## MTH 有零股，但不能直接把每列量相加

[TPEx MTH 格式](https://eshop.tpex.org.tw/uploadFile/upload/product/2c92e0139984eab70199892f7fcf0005.pdf) 明列交易種類 0／1／2 為普通／鉅額／零股，成交量為股數。它也說明成交資料已做錯帳調整及綜合帳戶重分配，未必能找到對應原委託，因此不是當時券商接收回報的原始事件紀錄。格式沒有額外的盤中／盤後旗標，也未清楚交代十二位時間的精度與歷史版本。

[官方免費樣本](https://eshop.tpex.org.tw/uploadFile/upload/product/2c92e0139984eab70199892fbfab0006.TXT) 是 **2023-09-15、3105 穩懋**，632,316 bytes、9,164 列，每列資料為 67 bytes，另有 CRLF。以交易日、股票、交易種類、成交序號作鍵，4,582 組均恰好一列 B 與一列 S，量價時間一致；直接加總兩側會把量加倍。

| 樣本零股範圍 | 原始列數 | 配對後筆數 | 單邊計一次股數 |
|---|---:|---:|---:|
| 全部 Trade Type 2 | 1,954 | 977 | 54,078 |
| 原始時鐘 09:10～13:30 | 1,914 | 957 | 53,340 |
| 原始時鐘 14:30 | 40 | 20 | 738 |

零股全部 B 側與 S 側各54,078股，直接相加為108,156股。依[官方盤中／盤後制度](https://www.tpex.org.tw/zh-tw/mainboard/trading/rules/odd-lot.html)，14:30 是盤後撮合時段。這份樣本的時間分布支持「MTH 零股混含盤中與盤後」的判讀；程式仍將時段歸類標為推論，不把鐘點判斷升格為所有歷史版本的盤別認證。

再讀取[2023-09-15 免費盤中零股日行情](https://www.tpex.org.tw/www/zh-tw/afterTrading/oddQuote?date=2023/09/15&response=json)，驗證回傳日期、表名及3105列：官方957筆、53,340股，與前述盤中時鐘區間完全吻合。

**金額有需要保留的差異：** 配對後逐筆價格乘股數合計7,771,349元，官方日表為7,771,271元，相差78元。把每筆不足1元捨去再合計恰為官方數字，但這只是診斷；尚未取得官方金額取整政策，未標記金額原值完全相同。量與筆數對得上，也不能單獨證明每個時間戳、試撮排除、委託優先順序或2022年覆蓋都已驗證。

## 已加入獨立唯讀檢查器

`skills/tpex_mth.py` 的 `inspect_mth_sample(raw, format_id="tpex_mth_67_v1")` 僅處理此已檢查的67-byte結構與四碼證券；本地格式名稱不是官方版本保證。它會拒絕不明格式、非ASCII／不符長度、未知分類、不合法日期或時間、零量、格式不明價格，以及缺側、重複側、兩側價量時間不一致。成交鍵包含日期及股票，避免跨股或跨日誤去重。

輸出保留十二位原始時間、各側原始股數與配對後股數／金額，**不生成 `normalized_auction_v1`、不接 execution feeds、不自動認證来源或全日完整性**。`historical_session_complete`、`execution_tape_accepted`、`source_authenticated`、`live_qualified` 都固定為false。未知時段保留為other，不當成零量或自動刪除。

19項合成測試已通過，涵蓋雙邊容量不能重算、缺側與不同量價時間拒絕、跨股跨日同序號、盤後時點及不明輸入。官方樣本另外實際解析，沒有把供應商原始檔放進測試或版本庫。

本機重現解析（零網路請求）：

```bash
python -m pytest -q tests/test_tpex_mth.py
python - <<'PY'
import json
from pathlib import Path
from skills.tpex_mth import inspect_mth_sample, FORMAT_ID
p = Path('.cache/execution-sources-20260924-v3/tpex-mth-sample.txt')
print(json.dumps(inspect_mth_sample(p.read_bytes(), format_id=FORMAT_ID), indent=2))
PY
```

## 首個缺件仍未解除

| 必要資料 | 本輪可确认的程度 | 尚缺什麼 |
|---|---|---|
| TWSE 0050，2022-01-03；8261，2022-01-04 | [H4商品](https://eshop.twse.com.tw/zh/product/detail/0000000080da7fa70182334eb932009d)日期範圍涵蓋，未採購 | 真實全日盤中零股揭示原檔、成交／試算及量價單位核實 |
| TPEx 6284，2022-01-04 | 免費日表確有570筆、16,236股；當日零股低72.40、高74.70元 | 73.10元事前限價出現後可成交多少股的時間序列 |
| TPEx MTH 路徑 | 格式與2023實樣確認含零股；可做雙邊與日彙總交叉核對 | [商品頁](https://eshop.tpex.org.tw/zh/product/detail/2c92e0139984eab70199892c78bf0004)只標2022-11-01起，未能證明2022年1月可得；內部使用報價與交付範圍仍須官方確認 |

6284的[日表](https://www.tpex.org.tw/www/zh-tw/afterTrading/oddQuote?date=2022/01/04&response=json)有落在限價以下的最低價，仍無法推算533股能否成交：最低價不包含各價位的可成交容量、先後順序和前面的排隊委託。這三個股票日仍是回放最早缺口，並非全期間只差三天。

TWSE新[H4格式](https://eshop.twse.com.tw/uploadFile/upload/8a82e9e69c3aecea019da8fa4128015b.docx)載明2026-04-01起190改201 bytes、價格欄加長；有成交及試算旗標，但成交量中文仍標「張數」，與零股語意需要供應方核實，不能自動套整股乘1000。旧免费样本未重新下载或当成这三天真实成交。

另查了TPEx [O60盤中零股價格資料檔](https://eshop.tpex.org.tw/zh/product/detail/703BD71401F177340A45B08188F82BAD)：雖起始2020-10-26，商品列示07:50產製，屬另一條盤前價格路徑，不能因名稱含零股就當成交序列。下載的格式ZIP也沒有O60明細。本輪未找到可自動補足上述2022撮合缺口的免費官方新來源。

## FinMind Sponsor及歷史公告版本

目前[技術面官方文件](https://finmind.github.io/tutor/TaiwanMarket/Technical/)仍說明 `TaiwanStockPriceTick` 為2019起、一般查詢一天一檔、上市／上櫃volume以張計；整日下載列為SponsorPro。其schema沒有能把上市上櫃tick可靠分成盤中零股的欄位。[92種資料集目錄](https://finmind.github.io/tutor/TaiwanMarket/DataList/)也未列独立歷史零股逐次撮合集。不能把現有Sponsor的整股tick改標零股；新增MCP或skill不會增加資料內容。

公告時點的限制已更明確：[FinMind基本面文件](https://finmind.github.io/tutor/TaiwanMarket/Fundamental/)指出月營收 `create_time` 自2026-04-21才有值，是入庫日期；更早歷史為空，啟用當日初始值也不代表公告日。它不是原公告時間，也没有每次更正版本與生效時間，無法據此重建2022年的事前可知資料。

[2026-09-06官方校正公告](https://finmind.github.io/WhatIsNew/)另披露2024Q2資產負債表補回316家公司，25家後續下市或終止公開發行公司的原來源歷史頁無法補回。這提供了具體缺漏及存續公司偏差的檢查方向，但公告沒有給這25家完整代碼；本輪未把它們任意對應到策略股票，也沒有以今天回補的值覆寫封存資料。

可繼續的路徑是逐事件取得MOPS／公司原公告與更正版本，對股票、報告期、發布時間、修訂鏈及原始檔雜湊；現行公開頁或抓取日不能替代當時版本。缺原件的訊號只能保留「未驗證」或另做排除敏感度研究，不能宣稱歷史公告缺口自動補完。

## 證據保存與請求成本

全部下載、失敗metadata、檢查結果在 `.cache/execution-sources-20260924-v3/`，原件附URL、UTC取得時間、HTTP結果與SHA256。共16次直接HTTP嘗試（15成功，1次沙盒DNS失敗後獲准讀公開來源），FinMind資料API **0次**；沒有採購、寄信、下單或重啟排程。

- `tpex-mth-sample.txt` SHA256：`2f3fbe0b93ba71b624c24e9bae273af804092ea49fd8c6e0509d455b7762470e`。
- `tpex-mth-format.pdf` SHA256：`11c62e2d8adec7131b00a67111a5b7833ac9f54c376583203f99979674c4916a`。
- `mth-sample-inspection.json`：可由上述命令重新產生的唯讀診斷。
- `mth-sample-daily-comparison.json`：保存957筆／53,340股核對與78元取整差異；沒有將差異隱藏。

本輪補的是來源契約及可重現檢查能力。原始缺件、歷史版本與券商成交回報仍保持未完成狀態。
