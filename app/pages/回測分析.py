"""回測分析 Dashboard – 即時進度、績效指標、權益曲線、Fold 明細"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "backtest"
PROGRESS_FILE = ARTIFACTS_DIR / "progress.json"
RESULTS_FILE = ARTIFACTS_DIR / "latest.json"

st.set_page_config(page_title="回測分析", page_icon="📊", layout="wide")
st.title("📊 回測分析")

# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt_pct(val, suffix="%"):
    if val is None:
        return "N/A"
    return f"{val}{suffix}"


# ──────────────────────────────────────────────
# Section 1: 回測控制面板
# ──────────────────────────────────────────────
st.subheader("🚀 啟動回測")

col_a, col_b, col_c = st.columns(3)
with col_a:
    folds = st.number_input("Folds 數量", min_value=2, max_value=20, value=6)
with col_b:
    val_months = st.number_input("驗證月數", min_value=1, max_value=12, value=3)
with col_c:
    topn = st.number_input("Top N 選股", min_value=5, max_value=50, value=20)

if st.button("▶ 啟動回測", type="primary"):
    st.info("正在背景啟動回測…請稍候。")
    subprocess.Popen(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_backtest.py"),
        ],
        env={
            **__import__("os").environ,
            "BACKTEST_FOLDS": str(folds),
            "BACKTEST_VAL_MONTHS": str(val_months),
            "TOPN": str(topn),
        },
    )
    st.rerun()

st.divider()

# ──────────────────────────────────────────────
# Section 2: 即時進度
# ──────────────────────────────────────────────
progress = _load_json(PROGRESS_FILE)

if progress:
    status = progress.get("status", "unknown")
    total = progress.get("total_folds", 0)
    done = progress.get("completed_folds", 0)
    elapsed = progress.get("elapsed_seconds", 0)

    if status == "running" or status == "preparing":
        st.subheader("⏳ 回測進行中")
        pct = done / total if total else 0
        st.progress(pct, text=f"已完成 {done}/{total} folds（{elapsed:.0f} 秒）")

        col1, col2, col3 = st.columns(3)
        col1.metric("狀態", "⚙️ 運行中")
        col2.metric("已完成", f"{done}/{total}")
        col3.metric("耗時", f"{elapsed:.0f}s")

        st.caption("💡 頁面會自動重新載入以顯示最新進度")
        import time
        time.sleep(3)
        st.rerun()
    elif status == "completed":
        st.success(f"✅ 回測完成（{total} folds，耗時 {elapsed:.0f} 秒）")
    elif status == "failed":
        st.error(f"❌ 回測失敗：{progress.get('error', '未知錯誤')}")
else:
    st.info("尚未執行回測。按上方按鈕啟動。")

st.divider()

# ──────────────────────────────────────────────
# Section 3: 回測結果
# ──────────────────────────────────────────────
results = _load_json(RESULTS_FILE)

if results and results.get("status") == "completed":
    agg = results.get("aggregate", {})

    st.subheader("📈 綜合績效")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("平均 IC", _fmt_pct(agg.get("avg_ic"), ""))
    m2.metric("平均 Sharpe", _fmt_pct(agg.get("avg_sharpe"), ""))
    m3.metric("平均報酬率", _fmt_pct(agg.get("avg_total_return_pct")))
    m4.metric("平均最大回撤", _fmt_pct(agg.get("avg_max_drawdown_pct")))
    m5.metric("平均勝率", _fmt_pct(agg.get("avg_win_rate_pct")))

    # 指標解讀卡片
    with st.expander("📖 指標說明（什麼是 IC / Sharpe / 回撤？）"):
        st.markdown("""
| 指標 | 意義 | 越好越… |
|------|------|---------|
| **IC (Information Coefficient)** | 預測分數與實際報酬的相關性。> 0.05 算合格，> 0.1 算不錯。 | **高** |
| **Sharpe Ratio** | 風險調整後的年化報酬。> 1.0 好，> 2.0 很好。 | **高** |
| **報酬率** | 期間內策略的總報酬。正數代表獲利。 | **高** |
| **最大回撤** | 策略從高點到低點的最大跌幅。-10% 以內算穩。 | **接近 0** |
| **勝率** | 每日持股組合獲利的天數比例。> 55% 表示勝多敗少。 | **高** |
        """)

    st.divider()

    # ── 權益曲線 ──
    equity = results.get("equity_curve", {})
    eq_dates = equity.get("dates", [])
    eq_values = equity.get("values", [])

    if eq_dates and eq_values:
        st.subheader("💰 策略權益曲線")
        st.caption("若從 1 元開始投資，到每個時間點的淨值變化")

        eq_df = pd.DataFrame({"日期": eq_dates, "淨值": eq_values})
        eq_df["日期"] = pd.to_datetime(eq_df["日期"])
        eq_df = eq_df.set_index("日期")
        st.line_chart(eq_df, use_container_width=True)

        final_val = eq_values[-1] if eq_values else 1
        delta_pct = round((final_val - 1) * 100, 2)
        st.caption(f"最終淨值：**{final_val:.4f}**（{'🟢' if delta_pct >= 0 else '🔴'} {delta_pct:+.2f}%）")

    st.divider()

    # ── Fold 明細表 ──
    folds_data = results.get("folds", [])
    if folds_data:
        st.subheader("📋 各 Fold 明細")
        st.caption("每個 Fold 代表一段時間的驗證區間，Walk-forward 確保不偷看未來")

        fold_df = pd.DataFrame(folds_data)
        fold_df = fold_df.rename(columns={
            "fold": "Fold",
            "period": "驗證期間",
            "ic": "IC",
            "total_return_pct": "報酬率%",
            "sharpe": "Sharpe",
            "max_drawdown_pct": "最大回撤%",
            "win_rate_pct": "勝率%",
            "val_days": "交易天數",
        })

        st.dataframe(
            fold_df.style.format({
                "IC": "{:.4f}",
                "報酬率%": "{:+.2f}%",
                "Sharpe": "{:.2f}",
                "最大回撤%": "{:.2f}%",
                "勝率%": "{:.1f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ── 回測設定 ──
    st.subheader("⚙️ 回測設定")
    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
    cfg_col1.metric("Top N", results.get("topn"))
    cfg_col2.metric("Embargo 天數", results.get("embargo_days"))
    cfg_col3.metric("最低日均量", f"{results.get('min_avg_volume'):,} 張")
    cfg_col4.metric("來回成本", f"{results.get('round_trip_cost', 0) * 100:.3f}%")

    st.caption(f"完成時間：{results.get('completed_at', 'N/A')}　|　耗時：{results.get('elapsed_seconds', 0):.0f} 秒")

elif results and results.get("status") == "failed":
    st.error(f"回測失敗：{results.get('error', '未知錯誤')}")
else:
    st.info("目前沒有回測結果。請先執行回測。")
