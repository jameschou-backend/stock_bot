# 前三名驗證操作與重現

在儀表板「回測實驗室」最上方的「前三名：成本壓力與樣本外資格」，切換原成本、佣金加倍、滑價加倍、兩者加倍。各列的「曲線與明細」可查看逐日資產及交易；POC 版本差異可在表格下方展開。

本區固定比較原31組排名的前三名與0050，買價為訊號隔日各通路的 `(H+L)/2`，賣價維持 `L+0.3*(H-L)`；舊31組結果保留其買70／賣30標示。不得把不同價格版本混排為全31組新名次。

樣本外資格與歷史壓力測試分開：全部2024～2026/10/2資料已看過，分年與252日窗口只是穩定性診斷。HL2全日高低事後才知道，未完成可執行成交認證前不能把它當紙上實際成交。0個未見交易日不是零報酬，也不是驗證通過。通過條件見 [固定規格](prereg_top3_validation_20261012.md)。

既有A線訊號價NAV及leader前瞻帳戶採不同策略／資金規則；本次沒有往那些帳本填寫前三名假想成交。

## 重播

每次使用新輸出目錄，保留cache裡的原始資料、帳本與來源。以下單一命令跑四個帳戶；把 `baseline` 依序換成 `fee2`、`slip2`、`combined`：

```bash
python scripts/research_top3_validation_v2.py --mode baseline --output .cache/top3-validation-20261012/new-baseline
```

來源不齊會明確停止，使用者已授權必要補件時可加 `--fetch`，仍使用共用FinMind限額及既有持久預算／來源限制。一次只跑一個財務程序，共用互斥鎖，不能覆寫已封存目錄。費用是在資金回放時扣除，會改變股數及後續交易。

`research_top3_validation.py` 保留開發失敗試次所綁定的來源，正式重播使用上述 `v2`。前版的獨立資金核帳仍固定原佣金，會拒絕佣金加倍帳本；`v2` 用來源指紋鎖定的私有核帳副本套用該組佣金，其餘現金、稅、滑價、股數與公司行動核對不變。失敗、被中止與完成的試次都保留紀錄，不把核帳失敗報成績效。

四個成本組都有嘗試結果後發布：

```bash
python scripts/summarize_top3_validation.py \
  --report .cache/top3-validation-20261012/new-baseline/report.json \
  --report .cache/top3-validation-20261012/new-fee2/report.json \
  --report .cache/top3-validation-20261012/new-slip2/report.json \
  --report .cache/top3-validation-20261012/new-combined/report.json \
  --output artifacts/forward_simulation/new_top3_validation.json
```

彙總工具核對來源閉包、666日資產算術、費稅、證書庫存及各成本組基準，未完成者不列完整績效。儀表板只讀明確登記的發布版本與SHA收據，不任意讀取cache最新檔。

佣金倍率和最低費是本研究假設，並非使用者券商的實際報價；股票／ETF賣出稅及手續費原則可查 [證交所交易制度說明](https://www.twse.com.tw/en/products/system/trading.html)。
