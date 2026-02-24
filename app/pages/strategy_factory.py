from __future__ import annotations

import json
import re
import traceback
import uuid
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from sqlalchemy import text

from app.config import load_config
from app.db import get_engine, get_session
from skills.strategy_factory.data import (
    compute_indicators,
    detect_regime,
    load_price_df,
    resolve_weights,
)
from skills.strategy_factory.engine import BacktestConfig, BacktestEngine, StrategyAllocation
from skills.strategy_factory.registry import get as get_strategy, list_strategies, register_defaults


def _parse_json(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:
        return {}


def _format_reason(value) -> str:
    obj = _parse_json(value)
    reason = obj.get("reason") if isinstance(obj, dict) else None
    action_map = {
        "entry_rule": "進場規則",
        "exit_rule": "出場規則",
        "pyramiding": "加碼規則",
    }

    def _translate_rule_name(name: str) -> str:
        text = str(name or "").strip()
        if not text:
            return ""
        fixed = {
            "close_near_high_60": "接近 60 日高點",
            "close_near_low_20": "接近 20 日低點",
            "volume_20_gt_volume_60_mean": "20 日均量大於 60 日均量",
            "close_below_ma_20": "收盤跌破 20 日均線",
            "close_above_ma_20": "收盤站上 20 日均線",
            "close_above_ma_60": "收盤站上 60 日均線",
            "trailing_stop_0.1": "10% 移動停損",
            "trailing_stop_0.12": "12% 移動停損",
            "time_stop_10": "時間停損 10 日",
            "time_stop_15": "時間停損 15 日",
            "time_stop_20": "時間停損 20 日",
            "level_1": "第一層加碼",
            "level_2": "第二層加碼",
        }
        if text in fixed:
            return fixed[text]

        m = re.match(r"^([a-z0-9_]+)_gt_(-?\\d+(?:\\.\\d+)?)$", text)
        if m:
            return f"{m.group(1)} > {m.group(2)}"
        m = re.match(r"^([a-z0-9_]+)_lt_(-?\\d+(?:\\.\\d+)?)$", text)
        if m:
            return f"{m.group(1)} < {m.group(2)}"
        m = re.match(r"^([a-z0-9_]+)_gte_(-?\\d+(?:\\.\\d+)?)$", text)
        if m:
            return f"{m.group(1)} >= {m.group(2)}"
        m = re.match(r"^([a-z0-9_]+)_lte_(-?\\d+(?:\\.\\d+)?)$", text)
        if m:
            return f"{m.group(1)} <= {m.group(2)}"
        m = re.match(r"^stoploss_fixed_(-?\\d+(?:\\.\\d+)?)$", text)
        if m:
            return f"固定停損 {float(m.group(1)):.1%}"
        m = re.match(r"^takeprofit_fixed_(-?\\d+(?:\\.\\d+)?)$", text)
        if m:
            return f"固定停利 {float(m.group(1)):.1%}"
        m = re.match(r"^trailing_stop_(-?\\d+(?:\\.\\d+)?)$", text)
        if m:
            return f"移動停損 {float(m.group(1)):.1%}"
        return text

    if isinstance(reason, dict):
        code = reason.get("code", "")
        detail = reason.get("detail", [])
        if isinstance(detail, list):
            detail_text = ", ".join(_translate_rule_name(x) for x in detail if str(x).strip())
        else:
            detail_text = _translate_rule_name(detail)
        code_text = action_map.get(str(code), str(code))
        return f"{code_text}: {detail_text}" if detail_text else code_text
    if isinstance(reason, list):
        return ", ".join(_translate_rule_name(x) for x in reason if str(x).strip())
    if reason is None:
        return ""
    return str(reason)


def _trade_display_df(trades_df: pd.DataFrame) -> pd.DataFrame:
    df = trades_df.copy()
    action_map = {"BUY": "買進", "SELL": "賣出", "ADD": "加碼"}
    df["action"] = df["action"].map(action_map).fillna(df["action"])
    df["reason_text"] = df["reason_json"].apply(_format_reason)
    if "strategy_name" in df.columns:
        df["strategy_name"] = df["strategy_name"].fillna("").apply(lambda s: _strategy_label(s) if s else "")
    parsed = df["reason_json"].apply(_parse_json)
    df["sell_realized_pnl"] = parsed.apply(
        lambda x: ((x.get("realized_pnl") if isinstance(x, dict) else None) if isinstance(x.get("reason"), dict) else None)
        if isinstance(x, dict)
        else None
    )
    df["qty_shares"] = df["qty"].round(0).astype("Int64")
    cols = [
        "trading_date",
        "stock_id",
        "strategy_name",
        "action",
        "qty_shares",
        "price",
        "fee",
        "sell_realized_pnl",
        "reason_text",
    ]
    return df[cols]


def fetch_strategy_configs(engine) -> pd.DataFrame:
    try:
        return pd.read_sql("SELECT * FROM strategy_configs ORDER BY created_at DESC", engine)
    except Exception:
        return pd.DataFrame()


def fetch_strategy_runs(engine) -> pd.DataFrame:
    try:
        return pd.read_sql("SELECT * FROM strategy_runs ORDER BY created_at DESC", engine)
    except Exception:
        return pd.DataFrame()


def fetch_strategy_trades(engine, run_id: str) -> pd.DataFrame:
    try:
        return pd.read_sql(
            text(
                """
                SELECT trading_date, stock_id, strategy_name, action, qty, price, fee, reason_json
                FROM strategy_trades
                WHERE run_id = :run_id
                ORDER BY trading_date ASC
                """
            ),
            engine,
            params={"run_id": run_id},
        )
    except Exception:
        return pd.DataFrame()


def fetch_strategy_positions(engine, run_id: str) -> pd.DataFrame:
    try:
        return pd.read_sql(
            text(
                """
                SELECT trading_date, stock_id, strategy_name, qty, avg_cost, market_value, unrealized_pnl
                FROM strategy_positions
                WHERE run_id = :run_id
                ORDER BY trading_date ASC
                """
            ),
            engine,
            params={"run_id": run_id},
        )
    except Exception:
        return pd.DataFrame()


def _normalize_weights(weights: dict) -> dict:
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


STRATEGY_LABELS = {
    "MomentumTrend": "動能趨勢",
    "MeanReversion": "均值回歸",
    "DefensiveLowVol": "防禦低波動",
    "CourseVolumeMomentum": "課程版｜量大動能",
    "CourseBreakout": "課程版｜價量突破",
    "CoursePullback": "課程版｜回檔進場",
}


def _strategy_label(name: str) -> str:
    return STRATEGY_LABELS.get(name, name)


HELP_TEXT = {
    "name": "策略設定名稱，方便在回測與查詢時辨識。",
    "strategies": "選擇要啟用的策略組合。",
    "weight_bull": "多頭市場時各策略的資金配置權重。",
    "weight_bear": "空頭市場時各策略的資金配置權重。",
    "start_date": "回測起始日期。",
    "end_date": "回測結束日期。",
    "initial_capital": "回測初始資金。",
    "transaction_cost": "單邊手續費率（預設 0.1425%，每筆最低 20 元）。",
    "slippage": "成交滑價比例（買提高、賣降低）。",
    "risk_per_trade": "單筆最大風險占資金比例。",
    "max_positions": "同時持倉的最大股票數。",
    "rebalance_freq": "再平衡頻率：D=每日、W=每週、M=每月。",
    "min_notional_per_trade": "單筆最低成交金額，小於門檻則跳過，避免碎單成本過高。",
    "max_pyramiding_level": "同一檔股票最多允許加碼層數（建議 0~1）。",
}


def run_strategy_backtest(
    config_json: dict,
    start_date: date,
    end_date: date,
    initial_capital: float,
    transaction_cost_pct: float,
    slippage_pct: float,
    risk_per_trade: float,
    max_positions: int,
    rebalance_freq: str,
    min_notional_per_trade: float,
    max_pyramiding_level: int,
) -> tuple[str, dict]:
    register_defaults()
    config = load_config()
    raw = load_price_df(start_date, end_date)
    df = compute_indicators(raw)
    regime = detect_regime(df, config)
    weights = resolve_weights(regime, config, config_json or {})
    strategies = config_json.get("strategies") if config_json else None

    allocations = []
    for name, weight in weights.items():
        if strategies and name not in strategies:
            continue
        allocations.append(StrategyAllocation(strategy=get_strategy(name), weight=weight))
    if not allocations:
        raise ValueError("no strategies to run")

    bt_cfg = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        transaction_cost_pct=transaction_cost_pct,
        slippage_pct=slippage_pct,
        risk_per_trade=risk_per_trade,
        max_positions=max_positions,
        rebalance_freq=rebalance_freq.upper(),
        min_notional_per_trade=min_notional_per_trade,
        max_pyramiding_level=max_pyramiding_level,
    )
    engine = BacktestEngine(bt_cfg)
    result = engine.run(df, allocations)

    equity_curve = result["equity_curve"]
    final_equity = equity_curve[-1]["equity"] if equity_curve else bt_cfg.initial_capital
    total_return = final_equity / bt_cfg.initial_capital - 1
    metrics = {
        "regime": regime,
        "final_equity": final_equity,
        "total_return": total_return,
        "equity_curve": equity_curve,
        "trade_count": len(result["trades"]),
    }
    run_id = uuid.uuid4().hex

    with get_session() as session:
        session.execute(
            text(
                """
                INSERT INTO strategy_runs (
                    run_id, config_id, start_date, end_date, initial_capital,
                    transaction_cost_pct, slippage_pct, metrics_json, created_at
                ) VALUES (
                    :run_id, :config_id, :start_date, :end_date, :initial_capital,
                    :transaction_cost_pct, :slippage_pct, :metrics_json, NOW()
                )
                """
            ),
            {
                "run_id": run_id,
                "config_id": config_json.get("config_id"),
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": bt_cfg.initial_capital,
                "transaction_cost_pct": bt_cfg.transaction_cost_pct,
                "slippage_pct": bt_cfg.slippage_pct,
                "metrics_json": json.dumps(metrics, ensure_ascii=False, default=str),
            },
        )

        for t in result["trades"]:
            session.execute(
                text(
                    """
                    INSERT INTO strategy_trades (
                        run_id, trade_id, trading_date, stock_id, strategy_name, action, qty, price, fee, reason_json
                    ) VALUES (
                        :run_id, :trade_id, :trading_date, :stock_id, :strategy_name, :action, :qty, :price, :fee, :reason_json
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "trade_id": uuid.uuid4().hex,
                    "trading_date": t["trading_date"],
                    "stock_id": t["stock_id"],
                    "strategy_name": t.get("strategy_name") or "",
                    "action": t["action"],
                    "qty": t["qty"],
                    "price": t["price"],
                    "fee": t["fee"],
                    "reason_json": json.dumps(
                        {
                            "reason": t.get("reason"),
                            "realized_pnl": t.get("realized_pnl"),
                            "avg_cost": t.get("avg_cost"),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            )

        for p in result["positions"]:
            session.execute(
                text(
                    """
                    INSERT INTO strategy_positions (
                        run_id, trading_date, stock_id, strategy_name, qty, avg_cost, market_value, unrealized_pnl
                    ) VALUES (
                        :run_id, :trading_date, :stock_id, :strategy_name, :qty, :avg_cost, :market_value, :unrealized_pnl
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "trading_date": p["trading_date"],
                    "stock_id": p["stock_id"],
                    "strategy_name": p.get("strategy_name") or "",
                    "qty": p["qty"],
                    "avg_cost": p["avg_cost"],
                    "market_value": p["market_value"],
                    "unrealized_pnl": p["unrealized_pnl"],
                },
            )
        session.commit()

    return run_id, metrics


st.set_page_config(page_title="Strategy Factory", layout="wide")
st.title("Strategy Factory")

config = load_config()
engine = get_engine()
register_defaults()

tab_builder, tab_result, tab_holdings = st.tabs(
    ["Strategy Builder", "Backtest Result", "Holdings & Exposure"]
)

with tab_builder:
    st.subheader("建立策略設定")
    configs_df = fetch_strategy_configs(engine)
    if not configs_df.empty:
        st.dataframe(configs_df, use_container_width=True)

    name = st.text_input("設定名稱", value="default_strategy", help=HELP_TEXT["name"])
    all_strategies = list_strategies()
    selected = st.multiselect(
        "策略選擇",
        all_strategies,
        default=["MomentumTrend", "MeanReversion", "DefensiveLowVol"],
        format_func=_strategy_label,
        help=HELP_TEXT["strategies"],
    )

    st.caption("多頭權重（總和會自動正規化為 1）")
    bull_weights = {}
    for s in selected:
        bull_weights[s] = st.number_input(
            f"{_strategy_label(s)}（多頭）",
            min_value=0.0,
            max_value=1.0,
            value=0.6 if s == "MomentumTrend" else 0.2,
            step=0.05,
            key=f"bull_{s}",
            help=HELP_TEXT["weight_bull"],
        )
    bull_total = sum(bull_weights.values())
    st.caption(f"多頭權重合計：{bull_total:.2f}")

    st.caption("空頭權重（總和會自動正規化為 1）")
    bear_weights = {}
    for s in selected:
        bear_weights[s] = st.number_input(
            f"{_strategy_label(s)}（空頭）",
            min_value=0.0,
            max_value=1.0,
            value=0.5 if s == "DefensiveLowVol" else (0.3 if s == "MeanReversion" else 0.2),
            step=0.05,
            key=f"bear_{s}",
            help=HELP_TEXT["weight_bear"],
        )
    bear_total = sum(bear_weights.values())
    st.caption(f"空頭權重合計：{bear_total:.2f}")

    payload = {
        "strategies": selected,
        "weights_bull": _normalize_weights(bull_weights),
        "weights_bear": _normalize_weights(bear_weights),
    }

    with st.expander("進階設定 JSON（唯讀）", expanded=False):
        st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")

    if st.button("建立設定"):
        if not selected:
            st.error("請至少選擇一個策略")
        else:
            try:
                with get_session() as session:
                    session.execute(
                        text(
                            "INSERT INTO strategy_configs (config_id, name, config_json) VALUES (:id, :name, :json)"
                        ),
                        {
                            "id": uuid.uuid4().hex,
                            "name": name,
                            "json": json.dumps(payload, ensure_ascii=False),
                        },
                    )
                    session.commit()
                st.success("已建立策略設定")
            except Exception as exc:
                st.error(f"建立失敗: {exc}")

    st.divider()
    st.subheader("執行回測")

    config_options = configs_df["config_id"].tolist() if not configs_df.empty else []
    config_id = st.selectbox(
        "選擇策略設定",
        config_options,
        format_func=lambda cid: (
            f"{configs_df[configs_df['config_id'] == cid].iloc[0]['name']} ({cid[:8]})"
            if not configs_df.empty and cid in configs_df["config_id"].values
            else cid
        ),
    )
    start_date = st.date_input(
        "開始日期",
        value=date.today() - timedelta(days=365 * 5),
        help=HELP_TEXT["start_date"],
    )
    end_date = st.date_input("結束日期", value=date.today(), help=HELP_TEXT["end_date"])
    initial_capital = st.number_input(
        "初始資金",
        min_value=100_000.0,
        value=1_000_000.0,
        step=100_000.0,
        help=HELP_TEXT["initial_capital"],
    )
    transaction_cost_pct = st.number_input(
        "交易成本比例",
        min_value=0.0,
        value=0.001425,
        step=0.0001,
        help=HELP_TEXT["transaction_cost"],
    )
    slippage_pct = st.number_input(
        "滑價比例",
        min_value=0.0,
        value=0.001,
        step=0.0001,
        help=HELP_TEXT["slippage"],
    )
    if "sf_risk_per_trade" not in st.session_state:
        st.session_state["sf_risk_per_trade"] = 0.01
    risk_per_trade = st.number_input(
        "單筆風險比例",
        min_value=0.001,
        max_value=0.02,
        value=float(st.session_state["sf_risk_per_trade"]),
        step=0.001,
        format="%.4f",
        key="sf_risk_per_trade",
        help=HELP_TEXT["risk_per_trade"],
    )
    max_positions = st.number_input(
        "同時持倉上限",
        min_value=1,
        max_value=20,
        value=6,
        step=1,
        help=HELP_TEXT["max_positions"],
    )
    rebalance_freq = st.selectbox(
        "再平衡頻率",
        ["D", "W", "M"],
        help=HELP_TEXT["rebalance_freq"],
    )
    min_notional_per_trade = st.number_input(
        "單筆最低成交金額",
        min_value=0.0,
        value=1_000.0,
        step=500.0,
        help=HELP_TEXT["min_notional_per_trade"],
    )
    max_pyramiding_level = st.number_input(
        "最多加碼層數",
        min_value=0,
        max_value=3,
        value=1,
        step=1,
        help=HELP_TEXT["max_pyramiding_level"],
    )

    if st.button("執行回測"):
        if not config_id:
            st.error("請先選擇策略設定")
        else:
            try:
                row = configs_df[configs_df["config_id"] == config_id].iloc[0]
                config_json = _parse_json(row.get("config_json") or {})
                config_json["config_id"] = config_id
                run_id, metrics = run_strategy_backtest(
                    config_json=config_json,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    transaction_cost_pct=transaction_cost_pct,
                    slippage_pct=slippage_pct,
                    risk_per_trade=risk_per_trade,
                    max_positions=max_positions,
                    rebalance_freq=rebalance_freq,
                    min_notional_per_trade=min_notional_per_trade,
                    max_pyramiding_level=int(max_pyramiding_level),
                )
                st.success(f"回測完成 run_id={run_id}")
                st.json(metrics)
            except Exception as exc:
                st.error(f"回測失敗: {exc}")
                st.code(traceback.format_exc(), language="text")

with tab_result:
    runs_df = fetch_strategy_runs(engine)
    if runs_df.empty:
        st.info("尚無策略回測紀錄")
    else:
        run_id = st.selectbox("選擇 run_id", runs_df["run_id"].tolist())
        row = runs_df[runs_df["run_id"] == run_id].iloc[0]
        metrics = _parse_json(row.get("metrics_json"))
        st.json(metrics)
        equity_curve = metrics.get("equity_curve", [])
        if equity_curve:
            curve_df = pd.DataFrame(equity_curve)
            curve_df["trading_date"] = pd.to_datetime(curve_df["trading_date"])
            st.line_chart(curve_df.set_index("trading_date")["equity"])
        trades_df = fetch_strategy_trades(engine, run_id)
        if not trades_df.empty:
            st.dataframe(_trade_display_df(trades_df), use_container_width=True)

with tab_holdings:
    runs_df = fetch_strategy_runs(engine)
    if runs_df.empty:
        st.info("尚無持倉資料")
    else:
        run_id = st.selectbox("選擇 run_id (持倉)", runs_df["run_id"].tolist(), key="run_positions")
        pos_df = fetch_strategy_positions(engine, run_id)
        if pos_df.empty:
            st.info("此 run 無持倉快照")
        else:
            st.dataframe(pos_df, use_container_width=True)
