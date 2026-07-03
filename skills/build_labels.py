from __future__ import annotations

from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import Label, PriceAdjustFactor, RawPrice


def _compute_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values(["stock_id", "trading_date"])

    def apply_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("trading_date").copy()
        group["future_ret_h"] = group["close"].shift(-horizon) / group["close"] - 1
        return group

    return df.groupby("stock_id", group_keys=False).apply(apply_group)


# ──────────────────────────────────────────────
# Triple-Barrier labels (López de Prado, "Advances in Financial Machine Learning" Ch 3)
#
# 每個 entry 同時設三個 barrier，誰先觸到就決定 label：
#   profit-take (upper):  close >= entry × (1 + upper_pt)   → tb_label=+1
#   stop-loss   (lower):  close <= entry × (1 + lower_sl)   → tb_label=-1
#   time        (max_h):  既沒 pt 也沒 sl 撐到第 max_horizon 天 → tb_label= 0
#
# 比固定 horizon 好的地方：
#   - 自然編碼風險回報比（pt/sl ratio）
#   - 避免「20 天後剛好回檔」造成 false negative
#   - 對 outlier 友善（不會被 +200% 或 -100% 主導訓練）
#
# 為什麼 opt-in（不寫進 labels 表）：
#   - DB schema 不動，與既有 fixed-horizon label 並存
#   - 走 parquet 路徑（artifacts/labels/triple_barrier.parquet），由
#     scripts/build_triple_barrier_labels.py 產出
# ──────────────────────────────────────────────


def triple_barrier_labels(
    prices: pd.DataFrame,
    upper_pt: float = 0.15,
    lower_sl: float = -0.07,
    max_horizon: int = 20,
) -> pd.DataFrame:
    """Triple-Barrier Method 純函式（每股向量化，不打 DB / 不打網路）。

    Args:
        prices: DataFrame 需含 stock_id / trading_date / close 三欄
        upper_pt: profit-take 上界（如 +0.15 = +15%）
        lower_sl: stop-loss 下界（如 -0.07 = -7%；應為負數）
        max_horizon: 時間 barrier（交易日數，預設 20）

    Returns:
        DataFrame 含欄位：
          stock_id, trading_date,
          tb_label (-1/0/+1),
          tb_return (float)：觸 barrier 當下的實際 return（pt 通常略 >= upper_pt，
                            sl 通常略 <= lower_sl，time barrier 為當期 max_horizon return）
          tb_exit_type ('pt'/'sl'/'time')
          tb_exit_day_offset (1~max_horizon)
        後 max_horizon 列因 forward window 不足無法 label，**會被 drop**。

    Raises:
        ValueError: upper_pt <= 0 / lower_sl >= 0 / max_horizon < 1 / 缺欄位
    """
    if upper_pt <= 0:
        raise ValueError(f"upper_pt 必為正數（+15% = 0.15），got {upper_pt}")
    if lower_sl >= 0:
        raise ValueError(f"lower_sl 必為負數（-7% = -0.07），got {lower_sl}")
    if max_horizon < 1:
        raise ValueError(f"max_horizon >= 1，got {max_horizon}")
    required = {"stock_id", "trading_date", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices 缺欄位: {sorted(missing)}")

    df = prices.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["stock_id", "trading_date", "close"])
    df = df.sort_values(["stock_id", "trading_date"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=[
            "stock_id", "trading_date", "tb_label",
            "tb_return", "tb_exit_type", "tb_exit_day_offset",
        ])

    out_chunks = []
    for stock_id, g in df.groupby("stock_id", sort=False):
        g = g.reset_index(drop=True)
        close = g["close"].to_numpy()
        n = len(close)
        if n <= 1:
            continue

        labels = np.zeros(n, dtype=np.int8)
        returns = np.full(n, np.nan)
        exit_types = np.empty(n, dtype=object)
        exit_offsets = np.zeros(n, dtype=np.int32)
        valid = np.zeros(n, dtype=bool)

        # 對每個 entry index i，往後看 max_horizon 天
        for i in range(n):
            entry_px = close[i]
            if not np.isfinite(entry_px) or entry_px <= 0:
                continue
            # forward window 末端（不含當日）
            end = min(i + max_horizon, n - 1)
            if end <= i:
                # 後面沒任何資料，無法 label
                continue
            upper = entry_px * (1 + upper_pt)
            lower = entry_px * (1 + lower_sl)
            hit_pt = -1
            hit_sl = -1
            for offset in range(1, end - i + 1):
                p = close[i + offset]
                if not np.isfinite(p):
                    continue
                if p >= upper:
                    hit_pt = offset
                    break
                if p <= lower:
                    hit_sl = offset
                    break
            if hit_pt > 0:
                labels[i] = 1
                returns[i] = close[i + hit_pt] / entry_px - 1
                exit_types[i] = "pt"
                exit_offsets[i] = hit_pt
                valid[i] = True
            elif hit_sl > 0:
                labels[i] = -1
                returns[i] = close[i + hit_sl] / entry_px - 1
                exit_types[i] = "sl"
                exit_offsets[i] = hit_sl
                valid[i] = True
            else:
                # 時間 barrier：撐到 end 都沒觸發
                if end - i >= max_horizon:
                    labels[i] = 0
                    returns[i] = close[i + max_horizon] / entry_px - 1
                    exit_types[i] = "time"
                    exit_offsets[i] = max_horizon
                    valid[i] = True
                # 若 forward window 不足 max_horizon 天，整列丟棄
        if not valid.any():
            continue
        chunk = pd.DataFrame({
            "stock_id": g["stock_id"].iloc[valid],
            "trading_date": g["trading_date"].iloc[valid],
            "tb_label": labels[valid],
            "tb_return": returns[valid],
            "tb_exit_type": exit_types[valid],
            "tb_exit_day_offset": exit_offsets[valid],
        })
        out_chunks.append(chunk)

    if not out_chunks:
        return pd.DataFrame(columns=[
            "stock_id", "trading_date", "tb_label",
            "tb_return", "tb_exit_type", "tb_exit_day_offset",
        ])
    return pd.concat(out_chunks, ignore_index=True)


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "build_labels")
    try:
        max_price_date = db_session.query(func.max(RawPrice.trading_date)).scalar()
        if max_price_date is None:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        horizon = config.label_horizon_days
        last_label_date = db_session.query(func.max(Label.trading_date)).scalar()
        if last_label_date is None:
            target_start = db_session.query(func.min(RawPrice.trading_date)).scalar()
        else:
            target_start = last_label_date + timedelta(days=1)

        if target_start is None:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        stmt = (
            select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close)
            .where(RawPrice.trading_date.between(target_start, max_price_date))
            .order_by(RawPrice.stock_id, RawPrice.trading_date)
        )
        df = pd.read_sql(stmt, db_session.get_bind())
        if df.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        # ── 還原 close（與 build_features 一致：委派 apply_adj_factors）──
        # label = 還原後 forward return，避免配息股因除權息跌價被誤標「未來差」。
        # factor 缺日 per-stock ffill/bfill（直接 fillna(1.0) 會在 factor<1 區段
        # 中間製造單日假跳動，污染 T 或 T+20 落在缺日的所有 label）；
        # 整檔無 factor（無 adj 的下市股）才回退 1.0 = 未還原。
        if bool(getattr(config, "use_adjusted_price", True)):
            try:
                f_stmt = (
                    select(PriceAdjustFactor.stock_id, PriceAdjustFactor.trading_date,
                           PriceAdjustFactor.adj_factor)
                    .where(PriceAdjustFactor.trading_date.between(target_start, max_price_date))
                )
                fdf = pd.read_sql(f_stmt, db_session.get_bind())
            except Exception:
                fdf = pd.DataFrame()
            if not fdf.empty:
                from skills.build_features import apply_adj_factors

                df["stock_id"] = df["stock_id"].astype(str)
                df = apply_adj_factors(df, fdf)
                df["close"] = df["adj_close"]
                df = df.drop(columns=["adj_factor", "factor_missing", "adj_close"])

        df = _compute_labels(df, horizon)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["future_ret_h"])
        if df.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        records: List[Dict] = df[["stock_id", "trading_date", "future_ret_h"]].to_dict("records")
        # 分批寫入，避免單次 INSERT 語句過大
        BATCH_SIZE = 5000
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            insert_stmt = insert(Label).values(batch)
            insert_stmt = insert_stmt.on_duplicate_key_update(future_ret_h=insert_stmt.inserted.future_ret_h)
            db_session.execute(insert_stmt)
            db_session.commit()

        end_date = df["trading_date"].max()
        logs = {
            "rows": len(records),
            "start_date": target_start.isoformat(),
            "end_date": end_date.isoformat() if end_date is not None else None,
        }
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        db_session.rollback()
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs={"error": str(exc)})
        raise
