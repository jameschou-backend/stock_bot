#!/bin/bash
# Stage 11.1 完整驗證：5y news backfill 完成後跑此 script
# 1. Rebuild features (force recompute 5 year) 讓 news features 進 features_json
# 2. 跑 10y baseline (no news) + news-enabled 對照
# 3. 比較 Sharpe / MDD / Calmar

set -e
cd "$(dirname "$0")/.."

echo "=== Stage 11.1 News 10y Validation ==="
TS=$(date +%Y%m%d_%H%M)
mkdir -p artifacts/stage11_news

# Step 1: Rebuild features 5y（強制重算讓 news features 入 features_json）
echo "[1/3] Rebuild features 5y (force recompute)..."
FORCE_RECOMPUTE_DAYS=1825 python -c "
from app.config import load_config
from app.db import get_session
from skills import build_features
cfg = load_config()
with get_session() as s:
    build_features.run(cfg, s)
" 2>&1 | tee "artifacts/stage11_news/rebuild_${TS}.log"

# Step 2: Baseline (no news features, current production)
echo "[2/3] Baseline 10y (production: topn=30, no news)..."
python scripts/run_backtest.py --months 120 --topn 30 --seasonal-filter --no-stoploss \
  --market-filter-tiers="-0.05:0.5,-0.10:0.25,-0.15:0.10" --market-filter-min-pos 2 \
  --liq-weighted --pruned-features \
  --output "artifacts/stage11_news/baseline_${TS}.json" \
  2>&1 | tee "artifacts/stage11_news/baseline_${TS}.log"

# Step 3: With news features
# 需要先把 news_count_5d / news_source_diversity_5d 移出 _PRUNE_SET
# 用 env var 或修 code? 簡化: 改 _PRUNE_SET 寫法
echo "[3/3] With news features 10y..."
# 暫時用 env var 控制 (需要修 build_features.py 支援 NEWS_FEATURES_ENABLED env)
NEWS_FEATURES_ENABLED=1 python scripts/run_backtest.py --months 120 --topn 30 \
  --seasonal-filter --no-stoploss \
  --market-filter-tiers="-0.05:0.5,-0.10:0.25,-0.15:0.10" --market-filter-min-pos 2 \
  --liq-weighted --pruned-features \
  --output "artifacts/stage11_news/with_news_${TS}.json" \
  2>&1 | tee "artifacts/stage11_news/with_news_${TS}.log"

# Step 4: 對照
echo "[4/4] 對照結果"
echo "Baseline:"
grep -E "(累積|MDD|Sharpe|Calmar)" "artifacts/stage11_news/baseline_${TS}.log" | head -5
echo
echo "With news:"
grep -E "(累積|MDD|Sharpe|Calmar)" "artifacts/stage11_news/with_news_${TS}.log" | head -5

echo
echo "=== DONE: 結果存於 artifacts/stage11_news/ ==="
