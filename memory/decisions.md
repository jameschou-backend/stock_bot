# 重要策略決策與實驗記錄

> 最後更新：2026-03-16

---

## 核心架構決策

### Label 定義
- **20 交易日 forward return**（`LABEL_HORIZON_DAYS=20`）
- 定義：`close[T+20] / close[T] - 1`
- `build_labels.py: group["future_ret_h"] = group["close"].shift(-horizon) / group["close"] - 1`

### 再平衡頻率
- **月頻（M）**，約等於 20 個交易日
- 與 label horizon 對齊，避免 4:1 mismatch（見「已知錯誤修正」）

### 特徵數量
- 目前：**56 個特徵**（FEATURE_COLUMNS）
- 最新新增（2026-03-11）：`foreign_buy_streak`、`volume_surge_ratio`、`foreign_buy_intensity`

### 停損設計
- **現行生產：無固定停損**（`stoploss_pct=0.0`，`--no-stoploss`）
- 月底換股即出場，不設中途停損
- 單筆最大虧損 clip `-50%`（退市股保護）

### 大盤過濾
- **漸進式大盤過濾**：`[(-0.05, 0.5), (-0.10, 0.25), (-0.15, 0.10)]`
- 最低持股數：2 檔（防止單押集中風險）

### 季節性降倉
- 3月、10月：topN × 0.5，floor=5
- `daily_pick.py` 與 `backtest.py` 行為一致（2026-03-11 修正）

---

## 重要回測結果記錄

### 現行生產基準：Exp D（2026-03-15）

| 指標 | 數值 |
|------|------|
| 期間 | 2016-05-03 ~ 2026-02-03（118 期）|
| 累積報酬 | **+2637.11%** |
| 年化報酬 | +39.92% |
| 大盤報酬 | +53.68% |
| 超額報酬 | +2583.43% |
| MDD | **-29.20%** |
| Sharpe | **1.042** |
| Calmar | **1.367** |
| 交易次數 | 2009 筆 |
| 停損觸發 | 0 次 |

**配置**：無停損 + 漸進大盤過濾（-5%:×0.5, -10%:×0.25, -15%:×0.10）+ 最少 2 檔 + seasonal filter

**生產 CLI**：
```bash
python scripts/run_backtest.py --months 120 --seasonal-filter --no-stoploss \
  --market-filter-tiers="-0.05:0.5,-0.10:0.25,-0.15:0.10" --market-filter-min-pos 2
```

---

## 已試過的方向與結論

### ✅ 已採用

| 實驗 | 改動 | 效果 | 狀態 |
|------|------|------|------|
| Clip -50% | `max(ret, -0.50)` 退市股保護 | +338% → +1216% | ✅ 生產 |
| enable_seasonal_filter | 3/10月 topN×0.5 與生產對齊 | +1216% → +9553% | ✅ 生產 |
| 無停損 baseline | 移除 stoploss_pct=-0.07 | +1792% 無過濾 | ✅ 生產基礎 |
| 漸進式大盤過濾 Exp D | (-5%:×0.5, -10%:×0.25, -15%:×0.10) + 最少2檔 | +2637%, Sharpe 1.042 | ✅ 現行生產 |
| 強勢訊號特徵（Exp E）| foreign_buy_streak、volume_surge_ratio、foreign_buy_intensity | +2252% → +10005%（⚠️ 有 label 洩漏）| ✅ 特徵已在生產 |
| EMERGING 過濾 | 排除興櫃股（2340→1965 股）| 修正 foreign_buy_* 永遠為 0 | ✅ 生產 |
| label_horizon_buffer=20 | 消除訓練標籤前向洩漏 | 去偏後真實基準 | ✅ 生產 |

### ❌ 已試過但放棄

| 實驗 | 改動 | 結果 | 原因 |
|------|------|------|------|
| 固定停損 -7% | stoploss_pct=-0.07 | Sharpe 0.268，累積 +69% | 過度截斷月內正常波動 |
| 固定停損 -10%/-12% | stoploss_pct=-0.10/-0.12 | 累積 +69%~+122% | 同上 |
| ATR 動態停損 | 低波動-15%/高波動-25% | MDD 惡化 -60% | MDD 反而惡化 |
| 週頻再平衡 | rebalance_freq="W" | 累積 -84.96%，Sharpe -0.36 | Label 4:1 mismatch（根本錯誤）|
| 原始大盤過濾 | >5%半倉, >10%全現金 | 累積 +1301% | 2018/2020 錯過反彈 |
| Exp G：RSI 45-70 進場過濾 | RSI 過濾 | Sharpe 0.84（-0.20） | 模型已隱式學習 RSI，人為截斷干擾 |
| Exp H/I：多條件過濾 | streak+RSI+bias+volume | Sharpe 0.77 | 條件越多越差 |
| 退市過濾 B1/B2 | 零成交量 + 月跌幅門檻 | 2021: -28pp，2023: -46pp | 誤殺強勢反彈股 |
| 時間加權訓練 | 近期樣本 weight=2.0 | 無改善 | equal-weight+月頻下無效 |
| Market Regime Switching | bull/bear/sideways 三態 | sideways 長期損耗 | `market_regime.py` 標記 NOT IN USE |
| 日頻 Strategy B | 每日進出場 | Sharpe 0.48，MDD -54% | 預測尺度（月）與持倉（8天）不匹配 |

### 🔬 有潛力但尚未完整驗證

| 方向 | 說明 |
|------|------|
| 突破確認進場 | F+ 實驗：Sharpe 0.86（vs 基準 0.49），但基準為含 label 洩漏版本，需在去偏版本驗證 |
| foreign_buy_streak<=3 過濾（Exp F）| 微幅改善 +66pp，差異太小不足以採用 |

---

## 已知錯誤與修正記錄

### 訓練標籤前向洩漏（2026-03-13 修正）
- **問題**：`label_horizon_buffer=0` 時，訓練截止前 20 個交易日的標籤使用到測試期收盤價
- **影響**：回測績效虛高（+10004% 去偏後只有 +205%）
- **修正**：`label_horizon_buffer=20`，`train_ranker.py LABEL_HORIZON_BUFFER_DAYS=20`

### 週頻 Label Mismatch（2026-03-10 發現）
- **問題**：`rebalance_freq="W"`（每~5天換股），label horizon 20 天 → 4:1 mismatch
- **影響**：10y 累積 -84.96%，Sharpe -0.36
- **修正**：改回 `rebalance_freq="M"`

### ATR Bug（2026-03-10 修正）
- `enable_slippage=True` 時 `atr_df=None` 靜默失效
- 修正：`if atr_stoploss_multiplier is not None or position_sizing == "vol_inverse" or enable_slippage:`

### fund revenue 前向洩漏（2026-03-04 修正）
- 月營收加 45 天公告延遲：`available_date = trading_date + 45 days`
- 改為 per-stock groupby `merge_asof`（全域 merge 需要全局單調，跨股資料不符合）
