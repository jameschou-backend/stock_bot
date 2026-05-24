"""Page 5: Experiments — MLflow runs + Optuna trials 視覺化。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.dashboard_v2.helpers import (
    apply_style, render_top_banner, render_kpi,
    COLOR_ACCENT, COLOR_DANGER, COLOR_SUCCESS, COLOR_WARNING,
    COLOR_TEXT_PRIMARY, COLOR_GRID, COLOR_PANEL, PLOTLY_LAYOUT,
)


apply_style()
render_top_banner("📈 Experiments", "MLflow runs · Optuna trials")

# ── MLflow ──
st.markdown('<div class="section-header">🔬 MLflow Experiments</div>', unsafe_allow_html=True)

mlruns_dir = PROJECT_ROOT / "mlruns"
if not mlruns_dir.exists():
    st.warning("無 mlruns/ 目錄，請先跑：`python scripts/run_backtest.py --mlflow ...`")
else:
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:{mlruns_dir}")
        client = mlflow.tracking.MlflowClient()
        exps = client.search_experiments()
        if exps:
            exp_options = {e.name: e.experiment_id for e in exps if e.name != "Default"}
            if exp_options:
                selected_exp = st.selectbox("選 experiment", list(exp_options.keys()))
                exp_id = exp_options[selected_exp]
                runs = client.search_runs(experiment_ids=[exp_id], max_results=50)
                if runs:
                    rows = []
                    for r in runs:
                        m = r.data.metrics
                        p = r.data.params
                        rows.append({
                            "run_id": r.info.run_id[:8],
                            "run_name": r.info.run_name or "—",
                            "Sharpe": m.get("sharpe_ratio"),
                            "MDD": m.get("max_drawdown"),
                            "Calmar": m.get("calmar_ratio"),
                            "cum": m.get("cumulative_return"),
                            "topn": p.get("topn"),
                            "start_time": pd.to_datetime(r.info.start_time, unit="ms"),
                        })
                    runs_df = pd.DataFrame(rows)
                    st.dataframe(runs_df, use_container_width=True, hide_index=True, height=400)

                    # Sharpe vs MDD scatter
                    if "Sharpe" in runs_df.columns and not runs_df["Sharpe"].isna().all():
                        st.markdown('<div class="section-header">📊 Sharpe vs MDD 散布圖</div>',
                                    unsafe_allow_html=True)
                        valid = runs_df.dropna(subset=["Sharpe", "MDD"])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=valid["MDD"], y=valid["Sharpe"],
                            mode="markers+text",
                            text=valid["run_name"], textposition="top center",
                            textfont=dict(size=9, color=COLOR_TEXT_PRIMARY),
                            marker=dict(
                                size=10, color=valid["Sharpe"],
                                colorscale=[[0, COLOR_DANGER], [0.5, COLOR_WARNING], [1, COLOR_SUCCESS]],
                                showscale=True,
                                colorbar=dict(title="Sharpe"),
                                line=dict(width=1, color=COLOR_TEXT_PRIMARY),
                            ),
                            hovertemplate="<b>%{text}</b><br>MDD: %{x:.2%}<br>Sharpe: %{y:.2f}<extra></extra>",
                        ))
                        fig.update_layout(
                            **PLOTLY_LAYOUT,
                            height=500,
                            xaxis_title="MDD（越正越好）", yaxis_title="Sharpe（越高越好）",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"experiment `{selected_exp}` 無 runs")
            else:
                st.info("無 user experiment")
        else:
            st.info("無 experiments")
    except Exception as e:
        st.error(f"MLflow 讀取失敗: {e}")

# ── Optuna ──
st.markdown('<div class="section-header">🎯 Optuna Studies</div>', unsafe_allow_html=True)
optuna_db = PROJECT_ROOT / "optuna.db"
if not optuna_db.exists():
    st.info("無 optuna.db，請先跑：`python scripts/optuna_search.py --n-trials 30 ...`")
else:
    try:
        import optuna
        storage = f"sqlite:///{optuna_db}"
        summaries = optuna.get_all_study_summaries(storage=storage)
        if not summaries:
            st.info("無 study")
        else:
            study_names = [s.study_name for s in summaries]
            selected_study = st.selectbox("選 study", study_names)
            study = optuna.load_study(study_name=selected_study, storage=storage)
            trials = [t for t in study.trials if t.value is not None]
            if not trials:
                st.info(f"study `{selected_study}` 無完成 trial")
            else:
                rows = []
                for t in trials:
                    row = {"#": t.number, "value": t.value}
                    row.update(t.params)
                    row["MDD"] = t.user_attrs.get("max_drawdown")
                    row["Calmar"] = t.user_attrs.get("calmar_ratio")
                    rows.append(row)
                tdf = pd.DataFrame(rows).sort_values("value", ascending=False)
                st.dataframe(tdf, use_container_width=True, hide_index=True, height=400)

                # value 進步趨勢
                fig2 = go.Figure()
                trial_nums = [t.number for t in trials]
                trial_vals = [t.value for t in trials]
                best_so_far = []
                bsf = -1e9
                for v in trial_vals:
                    bsf = max(bsf, v)
                    best_so_far.append(bsf)
                fig2.add_trace(go.Scatter(
                    x=trial_nums, y=trial_vals,
                    mode="markers", name="trial Sharpe",
                    marker=dict(color=COLOR_ACCENT, size=8),
                ))
                fig2.add_trace(go.Scatter(
                    x=trial_nums, y=best_so_far,
                    mode="lines", name="Best so far",
                    line=dict(color=COLOR_SUCCESS, width=2),
                ))
                fig2.update_layout(
                    **PLOTLY_LAYOUT,
                    height=400,
                    title="Optuna trial 進步軌跡",
                    xaxis_title="Trial #", yaxis_title="Sharpe",
                    legend=dict(orientation="h", y=1.05, x=0),
                )
                st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Optuna 讀取失敗: {e}")
