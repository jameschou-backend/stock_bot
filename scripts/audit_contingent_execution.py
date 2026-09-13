#!/usr/bin/env python3
"""Offline evidence inventory, not a historical fill simulator or a new return."""
from pathlib import Path
from collections import Counter
import argparse
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read, write, sha, encoded
from scripts.research_reservation_bridge import inspect_ledger, OUTPUT as BRIDGE
from scripts.research_intraday_limit import TickCache, OUTPUT as TICKS

TARGET=ROOT/'artifacts/forward_simulation/contingent_audit_20260914.json'
CODE=['skills/contingent_execution.py','scripts/audit_contingent_execution.py',
      'docs/prereg_contingent_execution_20260914.md']


def build():
    # Full parent evidence verification is CLI-only; the UI reads a small result.
    manifest=read(BRIDGE/'manifest.json')
    for name,digest in manifest.items():
        if sha(BRIDGE/name)!=digest:
            raise ValueError('Sealed bridge evidence changed: '+name)
    for name,digest in read(BRIDGE/'identity.json').items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Sealed bridge identity changed: '+name)
    summary=read(BRIDGE/'summary.json')
    offline=read(BRIDGE/'offline.json')
    if not offline['identical'] or offline['manifest_sha256']!=sha(BRIDGE/'manifest.json'):
        raise ValueError('Missing current offline proof')
    inputs={str((BRIDGE/'manifest.json').relative_to(ROOT)):sha(BRIDGE/'manifest.json')}
    ticks=TickCache(TICKS/'ticks',online=False)
    cases={}
    for key,old in summary['legacy_ledger_checks'].items():
        path=BRIDGE/'cases'/f'{key}.json';account=read(path)['account']
        inputs[str(path.relative_to(ROOT))]=sha(path)
        markets={r['stock_id']:r['market'] for r in old['source_requests']}
        if encoded(inspect_ledger(account,markets))!=encoded(old):
            raise ValueError('Dependency inventory mismatch')
        rows=[]
        for dep in old['dependencies']:
            day=dep['date']
            trades=[t for t in account['trades'] if t['date']==day]
            orders=[o for o in account['orders'] if o['date']==day and o['requested_qty']]
            cash=[c for c in account['cash_ledger'] if c['date']==day and c['kind'] not in ('buy','sell')]
            needs=sorted({(o['stock_id'],o['channel']) for o in orders})
            evidence=[]
            for sid,channel in needs:
                item=dict(stock_id=sid,channel=channel,market=markets[sid],verified_ticks=False)
                if channel=='board':
                    tick_path=TICKS/'ticks'/f'{sid}-{day}.parquet'
                    meta_path=tick_path.with_suffix('.json')
                    if tick_path.exists() and meta_path.exists():
                        frame,digest=ticks.get(sid,day,markets[sid])
                        item.update(verified_ticks=True,rows=len(frame),sha256=digest)
                        for p in (tick_path,meta_path):inputs[str(p.relative_to(ROOT))]=sha(p)
                    else:item['missing']='board_tick_cache_missing'
                else:item['missing']='historical_odd_auction_evidence'
                evidence.append(item)
            closing=[t for t in trades if t['side']=='sell' and t['remaining_shares']==0]
            board_proceeds=sum(t['cash_change'] for t in trades if t['side']=='sell' and t['channel']=='board')
            missing=['original_precommitted_order_plan','broker_available_funds_timestamps']
            missing+=sorted({e['missing'] for e in evidence if 'missing' in e})
            rows.append(dict(**dep, trades=trades, orders=orders, nontrade_cash=cash,
                closing_sales=[dict(stock_id=t['stock_id'],channel=t['channel'],qty=t['qty']) for t in closing],
                board_sale_proceeds=round(board_proceeds,2),
                cash_gap_after_all_board_sales=round(max(0,dep['buy_outflow']-dep['previous_cash']-board_proceeds),2),
                # This is a need inventory; no assumption of a timestamp for an EOD reference.
                evidence=evidence, status='unverified', missing=missing))
        counts=Counter(x for r in rows for x in r['missing'])
        cases[key]=dict(dependency_days=len(rows),missing_days=dict(counts),rows=rows,
            odd_final_sale_days=sum(any(s['channel']=='odd' for s in r['closing_sales']) for r in rows),
            slot_release_with_odd_final_sale_days=sum(r['needs_same_day_slot_release'] and
                any(s['channel']=='odd' for s in r['closing_sales']) for r in rows),
            board_tick_requests=sum(e['channel']=='board' for r in rows for e in r['evidence']),
            board_tick_cached=sum(e['channel']=='board' and e['verified_ticks'] for r in rows for e in r['evidence']))
    if ticks.calls:raise ValueError('Offline audit made network requests')
    return dict(scope='legacy_path_dependency_inventory',audit_completed=True,
        historical_replay_completed=False,total_return=None,live_qualified=False,
        network_calls=0,cases=cases,code_sha256={p:sha(ROOT/p) for p in CODE},inputs_sha256=inputs)


def run(target=TARGET,verify=False):
    started=time.monotonic();result=build()
    if verify:
        if encoded(read(target))!=encoded(result):raise ValueError('Offline audit differs')
        if target.with_suffix('.sha256').read_text().strip()!=sha(target):raise ValueError('Publication hash changed')
    else:
        if target.exists() and read(target)!=result:raise ValueError('Published inventory is immutable')
        write(target,result);target.with_suffix('.sha256').write_text(sha(target)+'\n')
    print(dict(verified=verify,seconds=round(time.monotonic()-started,3),network_calls=0,
        cases={k:{x:v[x] for x in ('dependency_days','odd_final_sale_days','slot_release_with_odd_final_sale_days',
                                  'board_tick_requests','board_tick_cached','missing_days')} for k,v in result['cases'].items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();run(verify=args.verify)
