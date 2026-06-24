#!/usr/bin/env python
"""示範：multi_agent 模式的『有依據進場』輸出。

凍結資料下，對最新一日 universe 跑 5-agent 選股，印出每檔的
各面向訊號 + 中文理由 + supervisor 三視角裁決（= /why 的內容）。
純示範，不寫 DB。
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["DATA_STORE_FREEZE"] = "1"

import pandas as pd
from datetime import date

from app.config import load_config
from app.db import get_session
from sqlalchemy import text
import skills.data_store as ds
from skills.multi_agent_selector import run_multi_agent_selection, explain_stock, AGENT_NAMES, _zscore

SIG = {-2: "強烈看空", -1: "看空", 0: "中立", 1: "看多", 2: "強烈看多"}
AG_LABEL = {"tech": "技術線型", "flow": "法人籌碼", "margin": "融資券", "fund": "基本面", "theme": "題材/資金輪動"}


def main():
    import dataclasses
    cfg = load_config()
    try:
        cfg = dataclasses.replace(cfg, selection_mode="multi_agent",
                                  multi_agent_weights={a: 0.2 for a in AGENT_NAMES})
    except Exception:
        # 非 dataclass 或欄位不可 replace 時退回 SimpleNamespace
        from types import SimpleNamespace
        d = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)} if dataclasses.is_dataclass(cfg) else dict(vars(cfg))
        d.update(selection_mode="multi_agent", multi_agent_weights={a: 0.2 for a in AGENT_NAMES})
        cfg = SimpleNamespace(**d)

    with get_session() as s:
        feat_all = ds.get_features(s, date(2000, 1, 1), date(2026, 12, 31))
        last_d = feat_all["trading_date"].max()
        day = feat_all[feat_all["trading_date"] == last_d].copy()
        # universe：四碼普通股 + 流動性門檻（聚焦可交易，剔除微型雜訊）
        day = day[day["stock_id"].astype(str).str.fullmatch(r"\d{4}")]
        if "amt_20" in day.columns:
            day = day[pd.to_numeric(day["amt_20"], errors="coerce") >= 1e8]  # 日均量 >= 1 億
        names = {str(r[0]): r[1] for r in s.execute(text("SELECT stock_id, name FROM stocks")).fetchall()}

    print(f"資料日：{last_d}　可交易 universe（四碼、日均量≥1億）：{len(day)} 檔　5 面向等權")
    dq_ctx = {"degraded_mode": False, "degraded_datasets": []}
    picks, _meta = run_multi_agent_selection(
        feature_df=day.reset_index(drop=True),
        stock_ids=day["stock_id"].astype(str).reset_index(drop=True),
        pick_date=last_d if isinstance(last_d, date) else pd.Timestamp(last_d).date(),
        topn=10, config=cfg, dq_ctx=dq_ctx, selection_meta={},
    )

    # 為 explain_stock 準備 z_map / ratio_median（與 selector 內部一致）
    z_map = {
        "ret_20": _zscore(day.get("ret_20", pd.Series(0.0, index=day.index))),
        "foreign_net_20": _zscore(day.get("foreign_net_20", pd.Series(0.0, index=day.index))),
    }
    ratio_median = float(pd.to_numeric(day.get("margin_short_ratio"), errors="coerce").median(skipna=True) or 0.2)
    day_idx = day.reset_index(drop=True).set_index(day.reset_index(drop=True)["stock_id"].astype(str))

    print("\n" + "=" * 78)
    print("今日多面向選股（依綜合分數排序，每檔附完整進場依據）")
    print("=" * 78)
    for rank, (_, p) in enumerate(picks.head(8).iterrows(), 1):
        sid = str(p["stock_id"])
        row = day_idx.loc[sid] if sid in day_idx.index else None
        if row is None:
            continue
        rep = explain_stock(row, dq_ctx, z_map, ratio_median)
        sup = rep["supervisor"]
        nm = names.get(sid, "")
        print(f"\n#{rank}  {sid} {nm}　綜合分 {p['score']:+.3f}　【{sup['verdict']}】 {sup['explanation']}")
        for a in AGENT_NAMES:
            ao = rep["agents"].get(a, {})
            if ao.get("unavailable"):
                print(f"     · {AG_LABEL[a]:10s}：(資料不足)")
                continue
            sig = SIG.get(int(ao.get("signal", 0)), "?")
            reasons = "，".join(str(x) for x in ao.get("reasons", [])[:4])
            print(f"     · {AG_LABEL[a]:10s}：{sig}　{reasons}")


if __name__ == "__main__":
    main()
