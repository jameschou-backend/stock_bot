# Dashboard v2 — 多頁系統

Streamlit 多頁 dashboard，深色高 contrast 設計，5 個視圖：

| Page | 用途 |
|------|------|
| `main.py` | Entry + summary cards |
| `1_📊_Today.py` | 投資者每日操作（picks / 持倉 / 大盤 regime） |
| `2_🧪_Backtest_Lab.py` | 回測結果視覺化（equity overlay / heatmap / KPI） |
| `3_🎯_Features.py` | SHAP 特徵歸因（importance / regime breakdown / redundant pairs） |
| `4_⚕️_System_Health.py` | data coverage / pipeline jobs / sponsor datasets |
| `5_📈_Experiments.py` | MLflow runs + Optuna trials |

## 啟動

```bash
# 跑 v2（port 8502，避免跟 v1 衝突）
make dashboard-v2

# 或直接跑
streamlit run app/dashboard_v2/main.py --server.port 8502

# 對照原版 v1（port 8501）
make dashboard
```

瀏覽器開 http://localhost:8502

## 設計

- **深色背景** `#0E1117` + **glassmorphism** panels
- **Plotly 互動圖表**（hover、zoom）
- **KPI cards** with flavor（success/warning/danger）
- **Regime tags** with color coding
- **Mobile responsive**（streamlit 內建）

## 資料需求

| Page | 需要 |
|------|------|
| Today | DB picks + raw_prices |
| Backtest Lab | `artifacts/optuna_10y/*.json`, `artifacts/stage10_*/*.json` |
| Features | `artifacts/shap_analysis/global_importance.csv` 等（先跑 `python scripts/shap_v2.py`）|
| System Health | DB jobs / raw_* tables |
| Experiments | `mlruns/` (MLflow), `optuna.db` (Optuna) |

## 注意

- v2 跟 v1 並存（v1 = `app/dashboard.py`），不互相影響
- 預設 port 8502，避免衝突
- 所有 query 都 cached（ttl 60-600s）降低 DB 負擔
- 跟 backfill 並存安全（純 read，DB pool 20 connections 充足）

## TODO（未來增強）

- [ ] Position tracking 從 portfolio.json 讀（目前只算建議部位）
- [ ] WebSocket 即時推送 picks 更新
- [ ] Mobile-optimized layout
- [ ] 個人化 alert 規則（MDD 警告、regime 切換）
- [ ] 多語系（中/英 toggle）
