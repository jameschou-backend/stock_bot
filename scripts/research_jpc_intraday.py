#!/usr/bin/env python3
"""Fixed 6197 case study; cache-only analysis and chronological exit replay."""
import hashlib
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from scripts import research_midpoint as study
from skills.intraday_limit_replay import normalize_ticks
from skills.intraday_peak_exit import replay

OUT = ROOT/'artifacts/forward_simulation/jpc_intraday_20260929'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str)+'\n')


def run(output=OUT):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve published case study; choose a new version for changes')
    publication = study.old.load_selector()
    with study.old.offline_only():
        data, _, _, _, quote_refs = study.strict.repaired_data(publication)
    quotes = data.quotes[data.quotes.stock_id.eq('6197')].copy()
    quotes['date'] = pd.to_datetime(quotes.date)
    quotes = quotes.sort_values('date')
    chip_root = ROOT/'.cache/leader-chip-inputs-20260928'
    manifest = json.loads((chip_root/'manifest.json').read_text())
    for rel, expected in manifest['sources'].items():
        if sha(ROOT/rel) != expected:
            raise ValueError('Chip source drift: '+rel)
    paths = [chip_root/'flows.parquet', chip_root/'weekly.parquet',
        ROOT/'.cache/chip-inputs/margin_verified.parquet',
        ROOT/'.cache/chip-inputs/dividends/6197.parquet',
        ROOT/'.cache/jpc-intraday-20260929/6197-2026-07-13.parquet',
        ROOT/'.cache/jpc-intraday-20260929/6197-2026-07-13.json',
        ROOT/'.cache/leader-chip-prepared-20260928/inputs/execution-feeds/limits-6197.rows.json',
        ROOT/'.cache/leader-chip-prepared-20260928/inputs/execution-feeds/limits-6197.raw.json']
    frames = [pd.read_parquet(p) for p in paths[:4]]
    flow, weekly, margin, dividends = [f.loc[f.stock_id.eq('6197')].copy() for f in frames]
    for f in (flow, weekly, margin):
        f['date'] = pd.to_datetime(f.date)
    flows = {}
    for start, end in [('2026-07-08','2026-07-30'), ('2026-07-24','2026-07-30'),
                       ('2026-07-07','2026-07-09')]:
        rows = flow[flow.date.between(start, end)]
        if not rows.schema_supported.all():
            raise ValueError('Unsupported institutional schema')
        flows[start+'..'+end] = rows[['foreign','trust','dealer','total']].sum().to_dict()
    # This single-case seed is known before 7/13: post-dividend 7/7 high exceeds
    # even entry-session high. No unknown intraday entry timestamp is needed.
    peak_rows = quotes[quotes.date.between('2026-07-02','2026-07-09')]
    peak = float(peak_rows.high.max())
    if peak != 405 or not quotes.loc[quotes.date.eq('2026-07-07'),'high'].eq(405).all():
        raise ValueError('Reviewed pre-session peak changed')
    if (peak_rows.low <= 344.25).any():
        # 7/2 precedes the 7/7 new peak; do not evaluate it with a future high.
        prior = quotes[quotes.date.between('2026-07-07','2026-07-09')]
        if (prior.low <= 344.25).any():
            raise ValueError('Earlier trigger requires investigation')
    day = '2026-07-13'
    meta = json.loads(paths[5].read_text())
    if meta['sha256'] != sha(paths[4]):
        raise ValueError('Tick source digest mismatch')
    tape = normalize_ticks(pd.read_parquet(paths[4]), '6197', day, 'TWSE')
    tape = tape[tape.time.ge(pd.Timedelta('09:00:00')) & tape.time.lt(pd.Timedelta('13:25:00'))]
    limits_file = json.loads(paths[6].read_text())
    if limits_file['raw_sha256'] != sha(paths[7]):
        raise ValueError('Limit source digest mismatch')
    limits = limits_file['rows'][day]
    prior = quotes[quotes.date.lt(day)].tail(20)
    if len(prior) != 20:
        raise ValueError('Incomplete prior ADV')
    basis = '6197_raw_after_2026-07-02_ex_dividend'
    payload = dict(position=dict(stock_id='6197', venue='board', qty=2000, peak=peak,
        asof='2026-07-09T13:30:00+08:00', basis=basis,
        source='Existing midpoint cohort bought 2000 board shares 2026-06-10; known post-dividend peak 405'),
        events=[dict(kind='session', date=day, known_at=day+'T08:59:00+08:00',
            lower=limits['lower'], upper=limits['upper'], prior_adv_shares=float(prior.volume.mean()),
            basis=basis, source=str(paths[6].relative_to(ROOT)))])
    for time, group in tape.groupby('time', sort=True):
        at = (pd.Timestamp(day, tz='Asia/Taipei')+time).isoformat()
        payload['events'].append(dict(kind='prints', at=at, stock_id='6197', venue='board',
            basis=basis, prints=[dict(price=float(r.price), shares=int(r.shares)) for r in group.itertuples()]))
    result = replay(payload)
    if result != replay(json.loads(json.dumps(payload))):
        raise ValueError('Independent replay differs')
    window = quotes[quotes.date.between('2026-07-07','2026-08-10')]
    prices = window.set_index(window.date.dt.strftime('%Y-%m-%d'))
    weekly = weekly[weekly.date.between('2026-06-26','2026-08-07')]
    weekly['assumed_available_lag8'] = weekly.date+pd.Timedelta(days=8)
    weekly['assumed_available_lag15'] = weekly.date+pd.Timedelta(days=15)
    report = dict(stock_id='6197', window='2026-07-07..2026-08-10',
        peak_intraday=405, trough_intraday=float(prices.loc['2026-07-30','low']),
        high_to_low_drawdown=float(prices.loc['2026-07-30','low']/405-1),
        peak_close=372.5, close_drawdown=float(prices.loc['2026-07-30','close']/372.5-1),
        rebound_next_close=float(prices.loc['2026-07-31','close']/prices.loc['2026-07-30','close']-1),
        rebound_to_aug6_close=float(prices.loc['2026-08-06','close']/prices.loc['2026-07-30','close']-1),
        institutional_net_shares=flows, weekly_holders=weekly.to_dict('records'),
        margin=margin[margin.date.between('2026-07-07','2026-07-31')].to_dict('records'),
        price_rows=window.to_dict('records'), intraday_board_exit=result,
        limitation=['Only this known 2000-share board position is replayed; no portfolio return claim',
            'Odd-lot timestamp tape absent: remaining odd shares not assumed filled',
            '8/15-day holder publication lags are assumptions; historical first releases unverified',
            'Net institutional trades and account-size tiers do not identify retail panic or hidden ownership',
            'Tape participation is a fill estimate, not queue or brokerage execution evidence'],
        live_qualified=False, unseen_validation=False, actual_fill_verified=False,
        research_tick_acquisition_requests=1, replay_network_requests=0,
        # Parent manifests retain their own complete hash chains; avoid copying
        # 33,000 unrelated files into a single-stock report.
        source_sha256=quote_refs |
            {str(p.relative_to(ROOT)):sha(p) for p in paths} |
            {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/'skills/intraday_peak_exit.py',
             chip_root/'manifest.json', ROOT/'artifacts/forward_simulation/historical_selector_replay_20260925.json']})
    output.mkdir(parents=True)
    write(output/'input.json', payload)
    write(output/'report.json', report)
    (output/'report.sha256').write_text(sha(output/'report.json')+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('high_to_low_drawdown','close_drawdown','rebound_next_close','intraday_board_exit')},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args().output)
