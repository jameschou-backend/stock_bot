"""Prospective evidence capture. Explicit clicks/CLI only; never sends orders."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from app import forward_journal as journal

DEFAULT_SIGNALS = None


def freeze_today(path=journal.DEFAULT_PATH, signals_path=DEFAULT_SIGNALS):
    from app.workbench_service import data_status
    status = data_status()
    source = Path(signals_path) if signals_path is not None else (
        journal.ROOT/'.cache/forward-validation/signals'/str(datetime.now(ZoneInfo('Asia/Taipei')).date())/'signals.json')
    manifest = source.with_name('manifest.json')
    if source.exists():
        if not manifest.exists(): raise ValueError('Signal source manifest missing')
        meta = json.loads(manifest.read_text())
        for name, expected in meta['sha256'].items():
            if hashlib.sha256((source.parent/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Signal inputs changed; cannot freeze this source')
        for name, expected in meta['code_sha256'].items():
            if hashlib.sha256((journal.ROOT/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Signal generator changed; prepare a separately versioned run')
    raw = source.read_bytes() if source.exists() else b'{}'
    signals = json.loads(raw)
    if source.exists() and signals.get('source_kind') == 'original_rule_forward_extension':
        import pandas as pd
        prices = pd.read_parquet(source.parent/'raw-close.parquet').set_index('date')
        prices.index = pd.to_datetime(prices.index)
        for entry in signals.get('entries', []):
            entry['planning_reference_close'] = float(prices.loc[pd.Timestamp(entry['signal_date']), entry['members'][0]])
    with journal.connection(path) as con:
        return journal.freeze(con, status, signals, hashlib.sha256(raw).hexdigest())


def capture_quotes(stock_ids=('0050',), path=journal.DEFAULT_PATH):
    from app.config import load_config
    from app.finmind import fetch_dataset
    config = load_config()
    # One all-market snapshot instead of N per-stock calls, including shared retries/quota.
    frame = fetch_dataset('TaiwanStockTickSnapshot', datetime.now(ZoneInfo('Asia/Taipei')).date(),
        token=config.finmind_token, requests_per_hour=config.finmind_requests_per_hour,
        max_retries=0, timeout=20, cache_ttl=10)
    if frame.empty: raise ValueError('FinMind snapshot is empty; no fill can be inferred')
    subset = frame[frame.stock_id.astype(str).isin(stock_ids)].copy()
    if set(subset.stock_id.astype(str)) != set(stock_ids):
        raise ValueError('Requested stock missing from snapshot')
    subset.attrs = frame.attrs.copy()
    with journal.connection(path) as con:
        return journal.snapshot(con, subset)


def plan_frozen_candidates(path=journal.DEFAULT_PATH):
    from decimal import Decimal, ROUND_FLOOR
    with journal.connection(path) as con:
        rows = journal.read_events(con)
        today = str(datetime.now(ZoneInfo('Asia/Taipei')).date())
        signal = next((r for r in reversed(rows) if r['kind']=='signal' and r['body']['signal_date']==today), None)
        if signal is None: raise ValueError('今日沒有通過資料檢查的事前封存訊號')
        planned = []
        for candidate in sorted(signal['body']['candidates'], key=lambda e:(-e['priority'],e['members'][0]))[:3]:
            price = journal.number(candidate['planning_reference_close'])
            liquidity = candidate.get('liquidity_before_entry', {})
            if not liquidity.get('complete_20_sessions') or liquidity.get('mean_turnover20_twd', 0)<50000000:
                raise ValueError('候選缺少完整20日流動性證據')
            # Initial evidence account only; no claims of a rolling NAV simulation.
            budget = Decimal(journal.RULES['initial_cash'])/3
            qty = int((budget-40)/(price*Decimal('1.001425')))
            board = min(qty//1000*1000, int(liquidity['adv20_shares']*.01)//1000*1000)
            odd = qty%1000
            for channel, size in [('board',board),('odd',odd)]:
                if size:
                    planned.append(journal.order(con,signal['hash'],candidate['members'][0],'buy',channel,
                        size,price,candidate['entry_date']))
        return planned
