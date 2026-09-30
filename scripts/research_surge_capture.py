#!/usr/bin/env python3
"""Offline census of frozen-universe surges versus the current selector/accounts."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import socket
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.surge_capture import (DEFINITIONS, labels, nonoverlap_starts, causal_gates,
                                 window_gate_counts, label_statistics)


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def deny_network(*args, **kwargs):
    raise RuntimeError('This diagnosis must use frozen local inputs only')


def finite(v):
    return float(v) if np.isfinite(v) else None


class AccountIndex:
    def __init__(self, case, close):
        self.account = a = case['account']
        self.close = close
        self.orders, self.trades, self.holdings = defaultdict(list), defaultdict(list), defaultdict(list)
        self.cohorts = {c['event_id']: c for c in a['cohorts']}
        self.nav = {r['date']: r['nav'] for r in a['daily']}
        for r in a['orders']:
            self.orders[r['event_id']].append(r)
        for r in a['trades']:
            self.trades[r['stock_id']].append(r)
        for r in a['holdings']:
            self.holdings[r['stock_id']].append(r)
        self.profit = defaultdict(float)
        for r in a['cash_ledger']:
            if r.get('stock_id'):
                self.profit[r['stock_id']] += r['cash_change']
        for r in case['summary']['final_holdings']:
            self.profit[r['stock_id']] += r['market_value']
        for r in case['summary']['final_receivables']:
            if r['kind'] != 'cash':
                raise ValueError('Add explicit valuation for non-cash final receivables')
            self.profit[r['stock_id']] += r['amount']
        if abs(sum(self.profit.values())-case['summary']['profit']) > .001:
            raise ValueError('Stock attribution fails to reconcile to account net profit')
        self.summary = case['summary']

    def window(self, sid, start, end, events):
        hs = [r for r in self.holdings[sid] if start <= r['date'] <= end and r['qty'] > 0]
        # Starting-close holdings are pre-existing exposure, not a new T+1 purchase.
        buys = [r for r in self.trades[sid] if start < r['date'] <= end and r['side'] == 'buy']
        failures = Counter()
        filled_events = set()
        for e in events:
            orders = [o for o in self.orders[e['event_id']] if o['side'] == 'buy']
            if not orders:
                raise ValueError('Sealed signal has no account order: '+e['event_id'])
            failures.update(set(o['failure'] for o in orders if o.get('failure')))
            if any(o['filled_qty'] > 0 for o in orders):
                filled_events.add(e['event_id'])
        exits = [c for c in self.cohorts.values() if c['stock_id'] == sid
                 and c.get('exit_date') and start < c['exit_date'] < end]
        last_exit = max((c['exit_date'] for c in exits), default=None)
        held_end = any(r['date'] == end for r in hs)
        after = None
        if last_exit and not held_end:
            after = finite(self.close.at[pd.Timestamp(end), sid]/self.close.at[pd.Timestamp(last_exit), sid]-1)
        weight = max((r['market_value']/self.nav[r['date']] for r in hs), default=0.)
        return dict(held_any=bool(hs), held_at_start=any(r['date'] == start for r in hs),
                    held_at_end=held_end, material_held=weight >= .01, max_nav_weight=weight,
                    days_held=len(hs), bought_in_window=bool(buys),
                    first_buy_date=min((r['date'] for r in buys), default=None),
                    filled_signal_events=len(filled_events), order_failures=dict(failures),
                    last_full_exit=last_exit, after_last_exit_to_end_return=after)


def cohort_rows(index, label_sets, days):
    out = []
    all_trades = index.account['trades']
    for c in index.account['cohorts']:
        eid, sid = c['event_id'], c['stock_id']
        tr = [t for t in all_trades if t['event_id'] == eid]
        orders = [o for o in index.orders[eid] if o['side'] == 'sell']
        entry, end = pd.Timestamp(c['entry_date']), c.get('exit_date')
        row = dict(stock_id=sid, name=c['name'], event_id=eid, entry_date=c['entry_date'],
                   signal_date=c['signal_date'], exit_date=end,
                   first_exit_date=min((o['date'] for o in orders), default=None),
                   exit_reason=orders[0]['reason'] if orders else None,
                   trade_cash_difference=sum(t['cash_change'] for t in tr),
                   trade_cost=sum(t['total_cost'] for t in tr), closed=end is not None)
        for h, data in label_sets.items():
            row[f'entry_forward_{h}_status'] = int(data['status'].at[entry, sid])
            row[f'entry_forward_{h}_return'] = finite(data['price_return'].at[entry, sid])
        if end:
            last = pd.Timestamp(end)
            row['holding_sessions'] = int(days.get_loc(last)-days.get_loc(entry))
            row['entry_to_exit_adjusted_return'] = finite(index.close.at[last, sid]/index.close.at[entry, sid]-1)
            for h in (20, 60):
                row[f'post_exit_{h}_status'] = int(label_sets[h]['status'].at[last, sid])
                row[f'post_exit_{h}_return'] = finite(label_sets[h]['price_return'].at[last, sid])
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to(ROOT)
    if out.exists():
        raise ValueError('Choose a new output; preserve prior research')
    out.mkdir(parents=True)
    socket.socket.connect = deny_network
    socket.create_connection = deny_network
    base = ROOT/'.cache/partial-risk-2019-20260929/inputs-final'
    manifest = read(base/'manifest.json')
    verified, direct = {}, {}

    def bind(p, expected=None):
        p = Path(p)
        relative = str(p.relative_to(ROOT))
        digest = verified.get(relative)
        if digest is None:
            digest = sha(p)
            verified[relative] = digest
        if expected and digest != expected:
            raise ValueError('Frozen input changed: '+relative)
        return digest

    def root_ref(p):
        direct[str(p.relative_to(ROOT))] = bind(p)

    root_ref(base/'manifest.json')
    for n, digest in manifest['files_sha256'].items():
        bind(base/n, digest)
    for p, digest in manifest['source_sha256'].items():
        bind(ROOT/p, digest)
    signal_path = ROOT/'.cache/stock-universe-2019-20260929/signals-v2.json'
    root_ref(signal_path)
    sealed = read(signal_path)
    for p, digest in sealed['source_sha256'].items():
        bind(ROOT/p, digest)
    cases = {}
    for key, folder in [('three', 'stock-universe-2019-20260929'), ('five', 'stock-universe-five-20260929')]:
        p = ROOT/'.cache'/folder/'final-a/report.json'
        root_ref(p)
        report = read(p)
        if not report['all_completed'] or not report['validated'] or report['preparation']:
            raise ValueError('Incomplete source account')
        for name, digest in report['source_sha256'].items():
            bind(ROOT/name, digest)
        p = p.parent/'liquid_universe.json'
        bind(p, report['cases']['liquid_universe']['sha256'])
        root_ref(p)
        cases[key] = read(p)
    for p in (Path(__file__).resolve(), ROOT/'skills/surge_capture.py',
              ROOT/'docs/prereg_surge_capture_20260930.md'):
        root_ref(p)
    print('verified source files', len(verified), flush=True)
    frames = {n: pd.read_parquet(base/(n+'.parquet')).set_index('date') for n in
              ('close-official', 'close-quality', 'raw-close', 'raw-volume', 'eligibility')}
    c, other, eligibility = [frames[n] for n in ('close-official', 'close-quality', 'eligibility')]
    days = c.index
    companies = pd.read_parquet(base/'companies.parquet')
    names = dict(zip(companies.stock_id, companies.name))
    ids = [s for s in c.columns if s != '0050']
    if set(ids) != set(names) or any(len(s) != 4 or not s.isdigit() or s.startswith('0') for s in ids):
        raise ValueError('Individual-stock cohort does not match price columns')
    start = days.get_loc(pd.Timestamp('2019-01-02'))
    feature = causal_gates(frames, companies)
    generated = feature['gates']['signal'].loc['2019-01-02':'2026-09-08']
    ii, jj = np.where(generated.to_numpy())
    observed = {(str(generated.index[i].date()), generated.columns[j]) for i, j in zip(ii, jj)}
    entries = sealed['entries']['liquid_universe']
    expected = {(e['signal_date'], e['members'][0]) for e in entries}
    if len(expected) != len(entries) or expected != observed:
        raise ValueError(f'Reconstructed selector differs: expected={len(expected)} actual={len(observed)}')
    by_stock = defaultdict(list)
    for e in entries:
        if days[days.get_loc(pd.Timestamp(e['signal_date']))+1].date().isoformat() != e['entry_date']:
            raise ValueError('Signal is not T-close / next-session execution')
        by_stock[e['members'][0]].append(e)
    accounts = {k: AccountIndex(v, c) for k, v in cases.items()}
    label_sets, stats, episodes, signal_rows, stock_rows, coverage = {}, [], [], [], [], []
    coords = [(days.get_loc(pd.Timestamp(e['signal_date'])), c.columns.get_loc(e['members'][0])) for e in entries]
    for horizon, threshold in DEFINITIONS:
        data = labels(c, other, eligibility, horizon, threshold)
        label_sets[horizon] = data
        ss, rr = data['status'], data['price_return']
        # Retain every daily observation, including non-surges and unknowns, locally.
        pd.DataFrame(ss.loc['2019-01-02':, ids]).to_parquet(out/f'daily-labels-{horizon}.parquet')
        for year in sorted(set(days[start:].year)):
            mask = days.year == year
            x = label_statistics(ss.loc[mask, ids].to_numpy(), rr.loc[mask, ids].to_numpy())
            stats.append(dict(scope='all_anchors', year=int(year), horizon=horizon, **x))
            q = feature['gates']['quality'].loc[mask, ids].to_numpy()
            stats.append(dict(scope='quality_anchors', year=int(year), horizon=horizon,
                              **label_statistics(ss.loc[mask, ids].to_numpy()[q], rr.loc[mask, ids].to_numpy()[q])))
        signals = []
        for e, (i, j) in zip(entries, coords):
            sid, day = e['members'][0], days[i]
            row = dict(event_id=e['event_id'], stock_id=sid, name=names[sid], signal_date=e['signal_date'],
                       entry_date=e['entry_date'], year=day.year, horizon=horizon,
                       status=int(ss.iat[i, j]), forward_return=finite(rr.iat[i, j]),
                       excess_return=finite(data['excess_return'].iat[i, j]),
                       relative20=finite(feature['relative20'].iat[i, j]),
                       volume_ratio=finite(feature['volume_ratio'].iat[i, j]),
                       prior_adv20=finite(feature['adv20'].iat[i, j]))
            for key, idx in accounts.items():
                orders = [o for o in idx.orders[e['event_id']] if o['side'] == 'buy']
                if not orders:
                    raise ValueError('Missing order for sealed signal')
                row[key+'_filled'] = any(o['filled_qty'] > 0 for o in orders)
                row[key+'_failures'] = '|'.join(sorted(set(o['failure'] for o in orders if o.get('failure'))))
            signals.append(row)
        signal_rows.extend(signals)
        for year in sorted({r['year'] for r in signals}):
            part = [r for r in signals if r['year'] == year]
            stats.append(dict(scope='formal_signals', year=year, horizon=horizon,
                              **label_statistics([r['status'] for r in part], [r['forward_return'] for r in part])))
        h_episodes = []
        for sid in ids:
            j = c.columns.get_loc(sid)
            status = ss[sid].to_numpy()
            windows = nonoverlap_starts(np.flatnonzero((status == 5) & (np.arange(len(days)) >= start)), horizon)
            stock_rows.append(dict(stock_id=sid, name=names[sid], horizon=horizon,
                                   nonoverlap_windows=len(windows),
                                   **label_statistics(status[start:], rr[sid].to_numpy()[start:])))
            for i in windows:
                end = i+horizon
                s, t = str(days[i].date()), str(days[end].date())
                ev = [e for e in by_stock[sid] if s <= e['signal_date'] < t]
                counts, blocker = window_gate_counts(feature['gates'], j, i, end)
                if counts['signal'] != len(ev):
                    raise ValueError('Window signals differ from reconstructed gates')
                first = min((e['signal_date'] for e in ev), default=None)
                at = days.get_loc(pd.Timestamp(first)) if first else None
                row = dict(window_id=f'{horizon}-{sid}-{s}', horizon=horizon, stock_id=sid,
                           name=names[sid], year=days[i].year, start=s, end=t,
                           price_return=float(rr.iat[i, j]), excess_return=float(data['excess_return'].iat[i, j]),
                           first_signal=first, signal_count=len(ev),
                           first_signal_delay=at-i if first else None,
                           early_signal=bool(first and at-i < horizon/3),
                           gain_before_signal=float(c.iat[at, j]/c.iat[i, j]-1) if first else None,
                           gain_after_signal=float(c.iat[end, j]/c.iat[at, j]-1) if first else None,
                           first_empty_gate=blocker, gate_counts=counts,
                           low_liquidity_days=int(feature['adv20'].iloc[i:end, j].lt(50_000_000).sum()))
                for key, idx in accounts.items():
                    row[key] = idx.window(sid, s, t, ev)
                h_episodes.append(row)
        episodes.extend(h_episodes)
        for year in sorted({r['year'] for r in h_episodes}):
            subset = [r for r in h_episodes if r['year'] == year]
            coverage.append(dict(horizon=horizon, year=year, windows=len(subset), stocks=len({r['stock_id'] for r in subset}),
                with_signal=sum(bool(r['signal_count']) for r in subset), early_signal=sum(r['early_signal'] for r in subset),
                **{key+'_'+field: sum(r[key][field] for r in subset) for key in accounts
                   for field in ('held_any', 'material_held', 'bought_in_window', 'held_at_start')}))
        print('horizon', horizon, 'nonoverlap windows', len(h_episodes), flush=True)
    tables = dict(coverage=coverage, statistics=stats, stocks=stock_rows, signals=signal_rows)
    flat_episodes = []
    for r in episodes:
        flat = {k: v for k, v in r.items() if k not in ('three', 'five', 'gate_counts')}
        flat.update({'gate_'+k: v for k, v in r['gate_counts'].items()})
        for key in accounts:
            flat.update({key+'_'+k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, dict) else v
                         for k, v in r[key].items()})
        flat_episodes.append(flat)
    tables['episodes'] = flat_episodes
    for key, idx in accounts.items():
        tables[key+'-cohorts'] = cohort_rows(idx, label_sets, days)
        tables[key+'-stock-profit'] = [dict(stock_id=s, name=names[s], net_profit=v) for s, v in sorted(idx.profit.items())]
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out/(name+'.csv'), index=False, encoding='utf-8-sig', float_format='%.10g')
    # Verify source stability again after calculation. Do not compare only mtimes.
    for name, digest in verified.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Source changed during diagnosis: '+name)
    write(out/'report.json', dict(schema='surge_capture_20260930', start='2019-01-02', end='2026-09-09',
        stocks=len(ids), sealed_signals=len(entries), signal_reconstruction_exact=True,
        source_files_verified=len(verified), source_roots_sha256=direct,
        exports_sha256={p.name: sha(p) for p in sorted(out.iterdir())},
        coverage=coverage, statistics=stats,
        no_signal_first_empty_gate={str(h): dict(Counter(r['first_empty_gate'] for r in episodes
                                         if r['horizon'] == h and not r['signal_count'])) for h, _ in DEFINITIONS},
        account_metrics={key: {k: idx.summary[k] for k in ('total_return', 'max_drawdown', 'final_nav', 'profit')}
                         for key, idx in accounts.items()},
        complete_historical_universe=False, live_qualified=False, unseen_validation=False,
        new_backtest=False, network_requests=0, db_writes=0,
        scope='Fixed-endpoint nonoverlap retrospective windows; not all peaks or a new tradable return'))
    print('complete', out, flush=True)


if __name__ == '__main__':
    main()
