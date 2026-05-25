#!/usr/bin/env python
"""Stage 11.2: News sentiment via Claude Haiku batch.

對 raw_stock_news 跑 LLM 情感分析（-1 利空 / 0 中性 / +1 利多）+ confidence。
結果寫入 news_sentiment 表。

設計：
  - 批次 50 篇/call（單 prompt 含 50 個標題）→ 降 API call 次數
  - Claude Haiku 4.5：~$0.80/1M input + $4/1M output
  - 3.4M news × 50/batch = 68K calls
  - 平均每 batch 3500 input + 200 output tokens
  - cost 估: 68K × 3700 tokens × ($0.8+$4)/2 / 1M ≈ **~$80**
  - 時間: 68K calls × 1s/call ≈ 19 hours（需要分批 + parallel）

用法：
  python scripts/news_sentiment_llm.py --dry-run --limit 5
    → 試跑 5 batch（不寫 DB，看 LLM output）
  python scripts/news_sentiment_llm.py --limit 100
    → 跑 100 batch (~$0.12，~2min)
  python scripts/news_sentiment_llm.py
    → 跑全部待分析 news (~$80, ~10h with parallelism)

PREREQUISITE:
  pip install anthropic
  export ANTHROPIC_API_KEY=sk-ant-xxx
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 50  # 篇 news / call
MODEL = "claude-haiku-4-5-20251001"
PROMPT_TEMPLATE = """你是台股金融新聞分析師。以下是 {n} 則台股個股新聞標題，請對每則判斷對該股**短期 (1-2 週)** 股價影響：
  +1 = 利多（業績/合約/外資買超等）
  0 = 中性（一般報導、不明影響）
  -1 = 利空（業績下修/裁員/訴訟等）

也評估你的信心 (0.0-1.0)。

新聞清單：
{news_list}

請以 JSON 陣列回覆，順序對應上述清單（不解釋，純 JSON）：
[
  {{"id": 1, "sentiment": +1, "confidence": 0.85}},
  ...
]"""


def fetch_pending_news(limit=None):
    """從 raw_stock_news 取尚未分析的 news（news_id not in news_sentiment）。"""
    from sqlalchemy import text
    from app.db import get_session
    with get_session() as s:
        q = """
            SELECT n.id, n.stock_id, n.title, n.source, n.news_datetime
            FROM raw_stock_news n
            LEFT JOIN news_sentiment ns ON ns.news_id = n.id
            WHERE ns.news_id IS NULL
            ORDER BY n.id
        """
        if limit:
            q += f" LIMIT {limit * BATCH_SIZE}"
        rows = s.execute(text(q)).fetchall()
    return rows


def call_llm(client, batch_news, dry_run=False):
    """送一個 batch 到 Claude Haiku，回 sentiment list。"""
    news_list = "\n".join(f"{i+1}. [{n.stock_id}] {n.title}" for i, n in enumerate(batch_news))
    prompt = PROMPT_TEMPLATE.format(n=len(batch_news), news_list=news_list)
    if dry_run:
        logger.info(f"[DRY-RUN] Prompt preview (first 500 chars):\n{prompt[:500]}")
        return [{"id": i+1, "sentiment": 0, "confidence": 0.5} for i in range(len(batch_news))]

    resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    # 去除可能的 markdown code block
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"  parse fail: {e}; raw: {text[:200]}")
        return []


def write_sentiment(records):
    """寫入 news_sentiment 表（INSERT IGNORE）。"""
    if not records:
        return 0
    from sqlalchemy import text
    from sqlalchemy.dialects.mysql import insert
    from app.db import get_session
    from app.models import Base
    # 用 raw SQL 因為 news_sentiment 沒在 models.py（migration only）
    with get_session() as s:
        sql = """
            INSERT IGNORE INTO news_sentiment (news_id, sentiment_score, confidence, llm_model)
            VALUES (:news_id, :sentiment, :confidence, :model)
        """
        for r in records:
            s.execute(text(sql), r)
        s.commit()
    return len(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="只跑 N 個 batch（dev / cost control）")
    p.add_argument("--dry-run", action="store_true",
                   help="只 print prompt，不 call API")
    args = p.parse_args()

    if not args.dry_run:
        try:
            import anthropic
        except ImportError:
            logger.error("pip install anthropic 後再跑")
            sys.exit(1)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("export ANTHROPIC_API_KEY=sk-ant-... 後再跑")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
    else:
        client = None

    logger.info(f"fetching pending news (batch_size={BATCH_SIZE}, limit={args.limit})...")
    rows = fetch_pending_news(limit=args.limit)
    n_news = len(rows)
    n_batches = (n_news + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"  {n_news:,} pending news → {n_batches} batches")

    if n_news == 0:
        logger.info("nothing to do")
        return

    if not args.dry_run:
        est_cost_usd = n_batches * 3700 / 1_000_000 * 2.4  # blended price
        logger.warning(f"  💰 預估成本: ${est_cost_usd:.2f} USD")
        logger.warning(f"  預估時長: {n_batches} × 1s = {n_batches/60:.0f} min")
        logger.warning(f"  10 秒後開始... (Ctrl-C 中止)")
        time.sleep(10)

    total_written = 0
    t0 = time.time()
    for batch_idx in range(n_batches):
        batch_news = rows[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]
        try:
            sentiments = call_llm(client, batch_news, dry_run=args.dry_run)
            records = []
            for i, s in enumerate(sentiments):
                if i >= len(batch_news):
                    break
                news = batch_news[i]
                records.append({
                    "news_id": news.id,
                    "sentiment": int(s.get("sentiment", 0)),
                    "confidence": float(s.get("confidence", 0.5)),
                    "model": MODEL,
                })
            if not args.dry_run:
                n_written = write_sentiment(records)
                total_written += n_written
        except Exception as exc:
            logger.warning(f"  batch {batch_idx} failed: {exc}")
            continue

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            pct = (batch_idx + 1) / n_batches * 100
            eta_min = elapsed * (n_batches - batch_idx - 1) / (batch_idx + 1) / 60
            logger.info(f"  {batch_idx+1}/{n_batches} batches ({pct:.1f}%), "
                        f"{total_written:,} written, ETA {eta_min:.0f} min")

    elapsed = time.time() - t0
    logger.info(f"\n=== DONE ===")
    logger.info(f"  batches: {n_batches}, news written: {total_written:,}")
    logger.info(f"  elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
