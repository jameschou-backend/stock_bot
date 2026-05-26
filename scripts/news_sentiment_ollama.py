#!/usr/bin/env python
"""Stage 11.2 Ollama 版 sentiment：本地免費跑 qwen2.5:7b。

prereq:
  brew install ollama
  ollama serve &  (background daemon at localhost:11434)
  ollama pull qwen2.5:7b  (~5GB)

設計：
  - 跟 Claude 版同個 prompt + 同個 batch_size = 50
  - 用 ollama /api/generate REST API
  - JSON mode (format=json) 強制結構化輸出
  - retry on parse fail
  - 寫入 news_sentiment 表（llm_model 標記 qwen2.5:7b）

效能 (M-series Mac):
  - qwen2.5:7b ~5-10 articles/sec
  - 3.4M news / 7.5 = 453,000 sec = 5.2 days
  - batch_size=50 加速：68K batches × 5s/batch = 95 hours = 4 days

用法:
  python scripts/news_sentiment_ollama.py --dry-run --limit 2
  python scripts/news_sentiment_ollama.py --limit 100
  python scripts/news_sentiment_ollama.py  (full backfill, ~5 days bg)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 50
MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE = """你是台股金融新聞分析師。以下 {n} 則台股個股新聞標題，請對每則判斷對該股**短期 (1-2 週)** 股價影響：
  +1 = 利多（業績/合約/外資買超等）
  0 = 中性（一般報導、不明影響）
  -1 = 利空（業績下修/裁員/訴訟等）
也評估你的信心 (0.0-1.0)。

新聞清單：
{news_list}

請以 JSON 物件回覆，key 為 results 陣列：
{{
  "results": [
    {{"id": 1, "sentiment": 1, "confidence": 0.85}},
    {{"id": 2, "sentiment": 0, "confidence": 0.5}}
  ]
}}

只輸出 JSON，不要其他文字。"""


def fetch_pending_news(limit=None):
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


def call_ollama(batch_news, dry_run=False, max_retries=2):
    news_list = "\n".join(f"{i+1}. [{n.stock_id}] {n.title}" for i, n in enumerate(batch_news))
    prompt = PROMPT_TEMPLATE.format(n=len(batch_news), news_list=news_list)

    if dry_run:
        logger.info(f"[DRY-RUN] Prompt preview (first 500 chars):\n{prompt[:500]}")
        return [{"id": i+1, "sentiment": 0, "confidence": 0.5} for i in range(len(batch_news))]

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",  # 強制 JSON 結構化輸出
                    "options": {"temperature": 0.1, "num_predict": 2000},
                },
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            parsed = json.loads(text)
            results = parsed.get("results", [])
            if results:
                return results
            logger.warning(f"  empty results: {text[:200]}")
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            logger.warning(f"  ollama call failed: {exc}")
    return []


def write_sentiment(records):
    if not records:
        return 0
    from sqlalchemy import text
    from app.db import get_session
    with get_session() as s:
        sql = """
            INSERT IGNORE INTO news_sentiment (news_id, sentiment_score, confidence, llm_model)
            VALUES (:news_id, :sentiment, :confidence, :model)
        """
        for r in records:
            s.execute(text(sql), r)
        s.commit()
    return len(records)


def health_check():
    """confirm Ollama daemon + model available."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if MODEL not in models and not any(m.startswith(MODEL.split(":")[0]) for m in models):
            logger.error(f"  Ollama 缺少 {MODEL}！請跑：ollama pull {MODEL}")
            return False
        logger.info(f"  ✓ Ollama OK, models: {models[:5]}")
        return True
    except Exception as exc:
        logger.error(f"  Ollama daemon 不通：{exc}")
        logger.error(f"  請先：ollama serve &")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.dry_run:
        if not health_check():
            sys.exit(1)

    logger.info(f"fetching pending news (batch_size={BATCH_SIZE}, limit={args.limit})...")
    rows = fetch_pending_news(limit=args.limit)
    n_news = len(rows)
    n_batches = (n_news + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"  {n_news:,} pending news → {n_batches} batches")

    if n_news == 0:
        logger.info("nothing to do")
        return

    if not args.dry_run:
        logger.warning(f"  預估時長: {n_batches} × ~5s = {n_batches*5/60:.0f} min ({n_batches*5/3600:.1f}h)")
        logger.warning(f"  5 秒後開始... (Ctrl-C 中止)")
        time.sleep(5)

    total = 0
    t0 = time.time()
    for batch_idx in range(n_batches):
        batch = rows[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]
        try:
            sentiments = call_ollama(batch, dry_run=args.dry_run)
            records = []
            for i, s in enumerate(sentiments):
                if i >= len(batch):
                    break
                news = batch[i]
                try:
                    sent = int(s.get("sentiment", 0))
                    sent = max(-1, min(1, sent))  # clamp
                    conf = float(s.get("confidence", 0.5))
                    conf = max(0.0, min(1.0, conf))
                except (TypeError, ValueError):
                    continue
                records.append({
                    "news_id": news.id,
                    "sentiment": sent,
                    "confidence": conf,
                    "model": MODEL,
                })
            if not args.dry_run:
                total += write_sentiment(records)
        except Exception as exc:
            logger.warning(f"  batch {batch_idx} failed: {exc}")

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            pct = (batch_idx + 1) / n_batches * 100
            eta_min = elapsed * (n_batches - batch_idx - 1) / (batch_idx + 1) / 60
            logger.info(f"  {batch_idx+1}/{n_batches} ({pct:.1f}%), {total:,} written, "
                        f"ETA {eta_min:.0f} min ({eta_min/60:.1f}h)")

    elapsed = time.time() - t0
    logger.info(f"\n=== DONE ===  batches: {n_batches}, written: {total:,}, "
                f"elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
