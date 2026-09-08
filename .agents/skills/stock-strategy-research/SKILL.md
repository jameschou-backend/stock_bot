---
name: stock-strategy-research
description: 為 stock_bot 設計與評估台股波段選股、持股及出場實驗，重用訓練以控制回測時間。
---

先讀 `get_strategy_evidence` 與 `get_data_status`，再看對應程式和 prereg 文件。使用者偏重提高報酬，可接受較大波動；金額與最大可承受回撤仍需以當次已知設定為準，不能把 UI 試算初值當成使用者偏好。

- 先明確基準、單一變因、可交易時序、稅費滑價與樣本切分。A 線回測預設延遲至少 1 交易日；rotation 必須顯式 `--signal-lag 1`。lag 0 只可作舊口徑診斷，不能宣稱可實現績效。
- `make workbench` 的策略驗證會背景執行，避免重複提交；已有工作先看狀態。完整 CLI 可用 `scripts/run_backtest.py` 與 `scripts/backtest_rotation.py`，先讀 help / 實作確認參數。
- `skills.training_cache` 依實際訓練陣列內容、標籤、權重、群組、參數及版本重用模型。不要為改 TopN 或出場條件刪訓練快取。`BACKTEST_TRAIN_CACHE=off` 僅用於重訓對照。
- 價格還原、公告可用時間、存活者偏誤、樣本外區間須獨立確認。用過來調參的區間不再是未見測試集。快速模式與正式模式不能混為一談。
- 報告實際回測期間、淨報酬、基準、最大回撤、換手與成本；列出未驗證的限制。不因某次回測高報酬就自動切換實盤策略。

使用既有模型與程式為起點。第一個新實驗可研究降低換手的出場確認，但停損不能被持有確認延後。不要為了得到好看的結果重複搜尋同一測試期。
