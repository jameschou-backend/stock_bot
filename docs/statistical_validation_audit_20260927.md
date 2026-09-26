# 統計驗證修正：防止錯把回測檢定當作通過

2026-09-27，範圍為 `skills/statistics.py` 與既有回測 CLI 統計入口。沒有更換選股規則，也不因修正工具提升策略資格。

## 已修正的可重現錯誤

1. **DSR 單次試驗誤判**：原公式在 N=1 代入常態分位數零，預期最大 Sharpe 變成負無限大，連 Sharpe=0 都可能得到分數1。現在單一零均值試驗的預期最大值為0，同樣輸入分數為0.5、不通過。大試驗數使用 survival quantile，避免 `1 - tiny` 浮點消去。
2. **PBO 兩候選永遠偏樂觀**：原來 rank/N 且只計 <0.5，兩候選中最差者仍為0.5，不算失敗。依論文改為 rank/(N+1)，非正 logit 計為未超越中位數。合成的六種互補分割，每次樣本內勝者都在測試段落後，修正後 PBO=1。平手不算超越中位數。
3. **尾端資料遺漏**：PBO 原來捨棄 T%S 個資料，CPCV 則讓尾端資料永遠不進測試。現在分割涵蓋全部資料，組大小最多差一筆。
4. **標籤重疊未清除**：CPCV 原來只有 embargo。新增明確的 `label_end_indices`，按每筆標籤涵蓋的時間區間排除 train/test 重疊；未提供時只適用點觀測。這是組合診斷，會使用測試期前後資料，不能稱為未見樣本前推驗證。
5. **缺基準被補成零**：CLI 原來把 missing benchmark return 視為0，並跳過策略缺值。現在拒絕計算這份統計區塊，保留明確錯誤，避免虛增相對績效。Bootstrap 也拒絕 NaN、infinity、錯誤維度及不合法頻率。
6. **無效動差被夾成極小分母**：DSR 原來將非正變異修正夾到1e-12，可能產生極端顯著值。現在明確拒絕非法輸入與非正／非有限變異修正。

## 統計結果的使用限制

保留既有 JSON 欄位 `p_value` 的相容性，但清楚標為 DSR normal CDF 分數，數字越大代表在該模型假設下越強；它不是未來賺錢的機率，一般單尾 p 值為1減去該分數。CLI 新增欄位揭露試驗覆蓋尚未驗證、試驗 Sharpe 離散程度仍為假設。

DSR 須使用與樣本數同頻率的未年化 Sharpe。全歷史 trial registry 仍包含估計的歷史基數80，不能將這個慣例當作已查核的完整獨立候選數。PBO 也不能省略失敗／未完成候選後再宣稱整個選擇流程沒有過擬合。這些修正不證明原415%或更早700%以上回測已取得樣本外優勢。

CPCV helper 目前沒有正式策略訓練呼叫者；新增正確接口與測試，並不表示所有模型訓練已使用這個介面。既有封存完整帳戶來源不包含本次修改的統計 helper／CLI，所以其現金、持股與收益計算沒有因本修正改寫。

## 第一手方法來源

- [Bailey & López de Prado, The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)：試驗次數、Sharpe 離散程度、樣本數與高階動差共同影響檢定。
- [Bailey et al., The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)：第2.2節相對排名與 logit 定義；測試需要完整候選試驗資訊。

## 驗收紀錄

見本文件後續的實際命令結果；不以新增測試數宣稱系統不存在任何漏洞。

- 最終 `AI_ASSIST_ENABLED=0 make test`：3,383項通過、30個既有警告、67.00秒。
- 本次統計與部位／門檻相關測試：110項通過，2.97秒。測試包含原有案例，不全是新增。
- `make pipeline` 在既有 TWSE HTTP 前阻擋保護下退出0；TWSE仍記 fetch failed，不能聲稱官方來源已恢復。
- `make api` 後 curl `/health`、`/picks`、`/models`、`/jobs?limit=10` 全為200、有效JSON；health=ok。
- 研究本身零請求；pipeline驗收另增加5次FinMind請求，累計7,064→7,069。觀察時小時窗16／保守上限5,400。排程未啟用，未下單。
