# CLEANUP_REPORT.md

> 本次清理與強化作業執行日期：2026-03-04

---

## 一、專案健診發現

### 1.1 Bug 修復

| 檔案 | 問題 | 修復方式 |
|------|------|---------|
| `skills/train_ranker.py` line 157 | `train_end` 在賦值前被使用（`NameError` 風險）。原始寫法 `effective_train_end = train_end - timedelta(...)` 中 `train_end` 尚未定義 | 改為 `train_end = max_label_date - timedelta(days=LABEL_HORIZON_BUFFER_DAYS)`，直接從 `max_label_date` 計算 |
| `app/api.py` line 106 | 路由函式 `def get_strategy():` 命名與 line 43 的 `from skills.strategy_factory.registry import get as get_strategy` import 衝突，導致 line 243 的 `get_strategy(name)` 呼叫的是 FastAPI 路由函式而非 registry 函式（`TypeError`） | 路由函式重命名為 `strategy_doc()` |

### 1.2 未使用 Import / 死碼

經逐一檢查，核心模組（`app/`、`skills/`、`pipelines/`）中：
- 所有 import 均有被使用
- 私有函式（`_xxx`）均有被呼叫
- 無孤兒函式

`_LiquidityConfig` dataclass（`skills/risk.py` line 16）目前未被外部呼叫，僅作型別提示存在。標注如下（保留，非安全刪除項目）：

```python
# DEAD CODE? _LiquidityConfig 目前未被使用，原設計為 apply_liquidity_filter 的型別安全包裝
```

### 1.3 TODO / FIXME 位置（仍未修復的已知問題）

| 檔案 | 位置 | 內容 | CLAUDE.md 對應章節 |
|------|------|------|-------------------|
| `skills/ingest_trading_calendar.py` | line 25 | `"note": "TODO: replace with TWSE official calendar (holiday/half-day aware)"` | ✅ 已知，週末 heuristic 仍在使用 |
| `skills/ingest_corporate_actions.py` | line 35 | `# TODO: 接上正式來源（TWSE / MOPS / FinMind）` | ✅ 已知，`adj_factor=1.0` 保底 |

以上兩個 TODO 屬於資料來源整合問題，需接上外部 API，**非本次清理範圍**。

### 1.4 CLAUDE.md 已知問題狀態

| 問題 | 狀態 |
|------|------|
| `ingest_trading_calendar.py` 使用 weekday heuristic | ⚠️ 仍未修復（TODO 存在） |
| `ingest_corporate_actions.py` 外部來源未接妥 | ⚠️ 仍未修復（TODO 存在） |
| `pd.merge_asof(by="stock_id")` 全局單調限制 | ✅ 已正確使用 per-stock groupby loop |
| 回測交易成本口徑（單邊 vs 來回） | ✅ `backtest.py` 已標注「來回合計」 |

---

## 二、清理作業

### 2.1 Bug 修復
- `skills/train_ranker.py`: 修復 `train_end` 使用前未定義的 bug
- `app/api.py`: 修復 `get_strategy` 命名衝突

### 2.2 新增模組頂部說明 Docstring

以下檔案原本缺少模組級說明，已補充：

| 檔案 | 說明 |
|------|------|
| `skills/build_features.py` | 特徵工程模組用途說明 |
| `skills/daily_pick.py` | 每日選股模組用途說明 |
| `skills/risk.py` | 風控模組用途說明 |
| `skills/train_ranker.py` | 模型訓練模組用途說明 |
| `app/api.py` | FastAPI 應用端點說明 |

### 2.3 標注死碼（待人工確認）

| 位置 | 標注 | 說明 |
|------|------|------|
| `skills/risk.py` `_LiquidityConfig` | `# DEAD CODE?` | dataclass 目前無外部使用，可能是重構中間態 |

---

## 三、新增特徵（`skills/build_features.py`）

所有特徵均嚴禁資料洩漏（只使用當日可得資訊）。

### 3.1 外資連續買超天數 `foreign_buy_consecutive_days`

```python
is_buy = (group["foreign_net"] > 0).astype(int)
run_id = (is_buy != is_buy.shift()).cumsum()
group["foreign_buy_consecutive_days"] = is_buy * (is_buy.groupby(run_id).cumcount() + 1)
```

與既有 `foreign_buy_streak_5`（5 日窗格加總）不同，此特徵計算**真實連續天數**（例如連買 10 日為 10，中斷後重新計算）。無資料洩漏風險：只使用截至當日的歷史序列。

### 3.2 月營收 YoY 加速度 `fund_revenue_yoy_accel`

```python
fund_df["fund_revenue_yoy_accel"] = fund_df.groupby("stock_id")["fund_revenue_yoy"].diff(1)
```

計算位置：`_fetch_data()` 的 fund_df 處理階段，在 45 天公告延遲調整**之前**計算（`available_date = trading_date + 45 days`），透過 `merge_asof` 以 `available_date` 與 `trading_date` 對齊，確保不使用未公開資料。

正值代表 YoY 成長率加速（相較上月更強），負值代表放緩。

### 3.3 布林帶位置百分位 `boll_pct`

```python
boll_mid = close.rolling(20).mean()
boll_std = close.rolling(20).std()
boll_upper = boll_mid + 2 * boll_std
boll_lower = boll_mid - 2 * boll_std
boll_range = (boll_upper - boll_lower).replace(0, np.nan)
group["boll_pct"] = ((close - boll_lower) / boll_range).clip(0, 1)
```

0 = 價格在下軌，1 = 價格在上軌。使用 20 日標準差（2σ），並 clip 至 [0, 1] 處理突破情況。

### 3.4 近 60 日報酬偏態與峰態 `ret_60_skew` / `ret_60_kurt`

```python
group["ret_60_skew"] = daily_ret.rolling(60, min_periods=30).skew()
group["ret_60_kurt"] = daily_ret.rolling(60, min_periods=30).kurt()
```

使用 pandas rolling `.skew()` 和 `.kurt()`，`min_periods=30` 避免資料不足時的不可靠估計。正偏態代表右尾厚（偶爾大漲），正超額峰態（>0）代表尾部風險高。

### 3.5 價量背離信號 `price_volume_divergence`

```python
vol_ma_10 = volume.rolling(10).mean()
price_near_high = close >= close.rolling(10).max() * 0.98
price_near_low = close <= close.rolling(10).min() * 1.02
vol_shrink = volume < vol_ma_10 * 0.8
group["price_volume_divergence"] = (price_near_low & vol_shrink).astype(int) - (price_near_high & vol_shrink).astype(int)
```

- `+1`：正背離（股價接近 10 日低點但量縮，賣壓衰竭）
- `-1`：負背離（股價接近 10 日高點但量縮，上攻乏力）
- `0`：無明顯背離

---

## 四、Sharpe 提升措施

### 4.1 LOW_IC 特徵標注

以下特徵已在 `FEATURE_COLUMNS` 中加上 `# LOW_IC?` 註解，建議以實際 IC 數據確認後決定是否刪除或調整：

| 特徵 | 位置 | 標注理由 |
|------|------|---------|
| `vol_20` | `CORE_FEATURE_COLUMNS` | 波動率本身非方向性 alpha，IC 方向不確定 |
| `dealer_net_5`, `dealer_net_20` | `CORE_FEATURE_COLUMNS` | 自營商操作含避險性質，IC 不穩 |
| `drawdown_60` | `EXTENDED_FEATURE_COLUMNS` | 多為風險衡量指標，未必有預測力 |
| `fund_revenue_trend_3m` | `EXTENDED_FEATURE_COLUMNS` | 60 日滾動均值可能過度平滑 |
| `theme_hot_score` | `EXTENDED_FEATURE_COLUMNS` | 題材資料品質不穩（見 CLAUDE.md） |

**注意：這些特徵尚未被刪除，需實際計算 IC 後再決定。**

### 4.2 簡易滑價模型（`skills/backtest.py`）

在 `_simulate_period()` 加入 `enable_slippage: bool = True` 參數，計算邏輯：

```python
# 單邊滑價 = min(ATR/close * 0.1, 0.003)；來回（進 + 出）共 × 2
atr_pct = atr_value / entry_price
slippage_one_way = min(atr_pct * 0.1, 0.003)  # 上限 0.3%
slippage_total = slippage_one_way * 2  # 進出場各一次
```

報酬計算：`ret = exit_px / entry_px - 1 - transaction_cost_pct - slippage_pct`

大盤 benchmark 不套用滑價（`enable_slippage=False`）以保持指數特性。

### 4.3 Market Regime Filter 狀態確認

`AppConfig.market_filter_enabled: bool = True` 已為預設開啟。`load_config()` 中默認值為 `"true"`，**無需修改**。

---

## 五、仍需人工確認的項目

| 項目 | 類型 | 說明 |
|------|------|------|
| `skills/risk.py` `_LiquidityConfig` | DEAD CODE? | 目前未被使用，需確認是否可刪除 |
| `vol_20` | LOW_IC? | 計算實際 IC 後決定是否保留在 CORE（保留則無法訓練時被 dropna） |
| `dealer_net_5`, `dealer_net_20` | LOW_IC? | 考慮移至 EXTENDED（允許 NaN 填補），降低因缺資料被 dropna 的影響 |
| `drawdown_60` | LOW_IC? | 若 IC < 0.01 可考慮移除 |
| `fund_revenue_trend_3m` | LOW_IC? | 60 日窗口的平滑 YoY 均值，需驗證是否優於直接使用 `fund_revenue_yoy` |
| `theme_hot_score` | LOW_IC? | 題材資料品質不穩，建議在充足資料下計算 IC 後再決定 |
| `ingest_trading_calendar.py` TODO | 未修復 | 仍使用 weekday heuristic，假日不準確 |
| `ingest_corporate_actions.py` TODO | 未修復 | `adj_factor=1.0` 保底，除權息調整不準確 |
| `backtest.py` 無 `run()` 函式 | 設計確認 | 只有 `run_backtest()`，與其他 skill 的 `run()` 介面不一致。若未來需由 pipeline 呼叫，需補 wrapper |

---

## 六、測試結果

```
93 passed, 10 warnings in 1.29s
```

所有原有測試全部通過，無新增測試失敗。
