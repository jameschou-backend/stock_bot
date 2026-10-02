#!/usr/bin/env python3
"""Acquire only the five dated provider gaps found by the offline market audit."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_three_black_market_inputs import BASE, RUN, SIGNALS, collect_sources, encode, read, sha
from skills.market_input_validation import compare_quote, numeric, require

DIRECTORY = ROOT/'.cache/market-input-validation-20261002/provider-gaps'
PLAN = [('3426','2026-05-22'), ('4130','2026-05-22'), ('5236','2026-05-22'),
        ('5371','2026-05-22'), ('5371','2026-07-30')]


def normalized_repair(raw, sid, stamp, primary):
    require(raw['query'] == dict(dataset='TaiwanStockPrice',data_id=sid,start_date=stamp,end_date=stamp)
            and len(raw['data']) == 1, 'Wrong dated provider repair scope')
    r = raw['data'][0]
    require(r['date'] == stamp and r['stock_id'] == sid
            and primary['date'] == stamp and primary['stock_id'] == sid,
            'Provider or official returned another stock/date')
    quote = dict(date=stamp,stock_id=sid,open=numeric(r['open']),high=numeric(r['max']),
        low=numeric(r['min']),close=numeric(r['close']),volume=numeric(r['Trading_Volume'],integral=True))
    checks = compare_quote(quote,primary)
    require(all(checks[f] == 'matched' for f in ('open','high','low','close')), 'Repair price conflicts with official source')
    require(primary['volume_scope'] == 'ordinary_session' and quote['volume'] >= primary['volume'],
            'Daily total is smaller than ordinary volume or source scope differs')
    return dict(quote=quote,official=primary,field_checks=checks)


def repair_exposure(account, entries, ids, cutoff):
    """Unfilled orders and not-yet-bought candidates can change too."""
    counts = {f'affected_{key}_after_repair':sum(r['stock_id'] in ids and r['date'] >= cutoff
               for r in account[key]) for key in ('holdings','trades','orders')}
    counts['affected_candidates_after_repair'] = sum(e['members'][0] in ids and e['entry_date'] >= cutoff
                                                    for e in entries)
    return counts


def aligned_quality(series, existing, stamp):
    """Fill from the independent series only after surrounding scale agreement."""
    import numpy as np
    import pandas as pd
    day = pd.Timestamp(stamp)
    require(series.index.is_unique and day in series.index and float(series.at[day]) > 0,
            'Independent adjusted price missing or duplicated')
    common = series.reindex(existing.index)
    good = existing.gt(0) & common.gt(0)
    before = existing.index[good & (existing.index < day)]
    after = existing.index[good & (existing.index > day)]
    require(len(before) and len(after), 'Cannot anchor independent adjusted price on both sides')
    anchors = [before[-1],after[0]]
    ratios = existing.loc[anchors]/common.loc[anchors]
    scale = float(ratios.mean())
    require(np.isfinite(scale) and scale > 0 and (abs(ratios/scale-1) < 1e-6).all(),
            'Independent adjustment basis differs across repair')
    return dict(value=float(series.at[day])*scale,scale=scale,anchor_dates=[str(d.date()) for d in anchors])


def fetch():
    from app.config import load_config
    from app.file_lock import file_lock
    from app.finmind import fetch_dataset
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with file_lock(DIRECTORY/'.lock', timeout=0):
        path = DIRECTORY/'ledger.json'
        ledger = json.loads(path.read_text()) if path.exists() else dict(maximum=5,attempts={})
        require(ledger['maximum'] == 5, 'Unexpected request budget')
        config = load_config()
        for sid, stamp in PLAN:
            key = sid+'-'+stamp
            query = dict(dataset='TaiwanStockPrice',data_id=sid,start_date=stamp,end_date=stamp)
            if key in ledger['attempts']:
                item = ledger['attempts'][key]
                require(item['query'] == query and item['status'] == 'received', 'Unfinished request; do not retry automatically')
                require(sha(ROOT/item['path']) == item['sha256'], 'Acquired provider source changed')
                continue
            require(len(ledger['attempts']) < 5, 'Five-request repair budget exhausted')
            ledger['attempts'][key] = dict(query=query,status='started')
            path.write_text(encode(ledger))
            frame = fetch_dataset('TaiwanStockPrice',date.fromisoformat(stamp),date.fromisoformat(stamp),
                token=config.finmind_token,data_id=sid,requests_per_hour=5400,max_retries=0,timeout=30)
            raw = DIRECTORY/(key+'.json')
            with raw.open('x') as out:
                out.write(encode(dict(query=query,source='finmind',retrieved_at=frame.attrs.get('retrieved_at'),
                    cache_hit=frame.attrs.get('cache_hit'),data=frame.to_dict('records'))))
            ledger['attempts'][key].update(status='received',path=str(raw.relative_to(ROOT)),sha256=sha(raw),rows=len(frame))
            path.write_text(encode(ledger))
            print(encode(dict(stock_id=sid,date=stamp,rows=len(frame))),flush=True)


def prepare(output):
    """Create a separate repair bundle and recompute the fixed candidate rules."""
    import pandas as pd
    from scripts.prepare_million_signals import official_adjusted
    from skills.stock_universe_2019 import generate
    from skills.liquidity_candidates import filter_candidates
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), 'Choose a new repository output directory')
    seal = read(RUN/'report.json')
    refs = {}
    def source(path):
        name = str(path.relative_to(ROOT))
        require(sha(path) == seal['source_sha256'][name], 'Frozen source changed: '+name)
        refs[name] = sha(path)
        return path
    official, descriptors, _ = collect_sources(seal['source_sha256'], refs)
    for name in ('scripts/prepare_million_signals.py','skills/official_adj_factors.py',
                 'skills/stock_universe_2019.py','skills/candidate_quality.py','skills/diffusion_signals.py',
                 'skills/regime_state.py','skills/liquidity_candidates.py','skills/liquidity_diagnostics.py'):
        source(ROOT/name)
    refs[str((RUN/'report.json').relative_to(ROOT))] = sha(RUN/'report.json')
    require(sha(RUN/'three_black.json') == seal['cases']['three_black']['sha256'], 'Original account changed')
    refs[str((RUN/'three_black.json').relative_to(ROOT))] = sha(RUN/'three_black.json')
    ledger = read(DIRECTORY/'ledger.json')
    refs[str((DIRECTORY/'ledger.json').relative_to(ROOT))] = sha(DIRECTORY/'ledger.json')
    require(set(ledger['attempts']) == {s+'-'+d for s,d in PLAN}, 'Incomplete/unexpected repair request set')
    repairs = []
    for sid, stamp in PLAN:
        item = ledger['attempts'][sid+'-'+stamp]
        path = ROOT/item['path']
        require(item['status'] == 'received' and sha(path) == item['sha256'], 'Repair provider bytes changed')
        refs[item['path']] = item['sha256']
        repair = normalized_repair(read(path),sid,stamp,official[('TPEX',stamp,sid)])
        repairs.append(dict(repair,provider_path=item['path']))
    quotes = pd.read_parquet(source(BASE/'quotes-unmasked.parquet'))
    quotes.date = pd.to_datetime(quotes.date)
    additions = pd.DataFrame([r['quote'] for r in repairs])
    additions.date = pd.to_datetime(additions.date)
    require(not pd.MultiIndex.from_frame(additions[['date','stock_id']]).isin(
        pd.MultiIndex.from_frame(quotes[['date','stock_id']])).any(), 'Repair would overwrite an existing quote')
    frames = {name:pd.read_parquet(source(BASE/(name+'.parquet'))).set_index('date')
              for name in ('raw-close','raw-volume','close-official','close-quality','eligibility')}
    for f in frames.values():
        f.index = pd.to_datetime(f.index)
    companies = pd.read_parquet(source(BASE/'companies.parquet'))
    prior = read(source(ROOT/'.cache/stock-universe-2019-20260929/signals-v2.json'))
    groups = read(source(BASE/'signals.json'))
    liquid = read(source(SIGNALS))
    before = generate(frames,companies,groups,'2026-09-08')['liquid_universe']
    require(before == prior['entries']['liquid_universe'], 'Original candidate reproduction differs')
    cutoff = min(d for s,d in PLAN)
    for row in additions.itertuples(index=False):
        require(bool(frames['eligibility'].at[row.date,row.stock_id]), 'Repair falls outside known listing interval')
        frames['raw-close'].at[row.date,row.stock_id] = row.close
        frames['raw-volume'].at[row.date,row.stock_id] = row.volume
    changed_ids = sorted({s for s,d in PLAN})
    adjusted = official_adjusted(frames['raw-close'][changed_ids],pd.read_parquet(source(BASE/'events.parquet')))
    for row in additions.itertuples(index=False):
        frames['close-official'].at[row.date,row.stock_id] = adjusted.at[row.date,row.stock_id]
    quality_repairs = []
    quality_manifest = read(source(ROOT/'.cache/historical-selector-quality-20260925/manifest.json'))
    for sid in changed_ids:
        if sid == '5236':
            source_path = ROOT/'.cache/growth-flow-research/quotes.parquet'
            f = pd.read_parquet(source(source_path))
            f = f.loc[f.stock_id.eq(sid)].rename(columns={'trading_date':'date','adj_close':'close'})
        else:
            source_path = ROOT/'.cache/historical-selector-quality-20260925'/(sid+'.parquet')
            require(sha(source_path) == quality_manifest['files_sha256'][sid+'.parquet'], 'Independent quality source changed')
            f = pd.read_parquet(source(source_path))
            require(set(f.stock_id) == {sid}, 'Independent adjusted-price identity differs')
        f['date'] = pd.to_datetime(f.date)
        series = f.set_index('date')['close']
        for stock,stamp in PLAN:
            if stock != sid:
                continue
            detail = aligned_quality(series,frames['close-quality'][sid],stamp)
            frames['close-quality'].at[pd.Timestamp(stamp),sid] = detail['value']
            quality_repairs.append(dict(stock_id=sid,date=stamp,source=str(source_path.relative_to(ROOT)),**detail))
    after = generate(frames,companies,groups,'2026-09-08')['liquid_universe']
    # Compare original filters against their original frame, not the repaired one.
    original_raw = pd.read_parquet(BASE/'raw-close.parquet').set_index('date')
    original_volume = pd.read_parquet(BASE/'raw-volume.parquet').set_index('date')
    baseline,_ = filter_candidates(original_raw,original_volume,before,'2026-09-08')
    require(baseline['median50m'] == liquid['entries']['median50m'], 'Original liquidity reproduction differs')
    filtered,_ = filter_candidates(frames['raw-close'],frames['raw-volume'],after,'2026-09-08')
    old = {e['event_id']:e for e in baseline['median50m']}
    new = {e['event_id']:e for e in filtered['median50m']}
    require([e for e in after if e['signal_date'] < cutoff] == [e for e in before if e['signal_date'] < cutoff],
            'Later quote repair changed earlier candidates')
    added = [new[k] for k in sorted(new.keys()-old.keys())]
    removed = [old[k] for k in sorted(old.keys()-new.keys())]
    changed = [k for k in sorted(new.keys() & old.keys()) if new[k] != old[k]]
    account = read(RUN/'three_black.json')['account']
    exposure = repair_exposure(account,filtered['median50m'],set(changed_ids),cutoff)
    output.mkdir(parents=True)
    additions.to_parquet(output/'quote-supplement.parquet',index=False)
    pd.concat([quotes,additions],ignore_index=True).sort_values(['date','stock_id']).to_parquet(
        output/'quotes-unmasked.parquet',index=False)
    companies.to_parquet(output/'companies.parquet',index=False)
    for name in ('identity.json','events.parquet'):
        (output/name).write_bytes(source(BASE/name).read_bytes())
    for name,f in frames.items():
        f.rename_axis('date').reset_index().to_parquet(output/(name+'.parquet'),index=False)
    (output/'signals.json').write_text(encode(dict(entries=filtered,live_qualified=False,unseen_validation=False)))
    report = dict(schema='market_input_repairs_v1',repairs=repairs,quality_repairs=quality_repairs,provider_requests=5,
        missing_quotes_repaired=5,original_candidate_count=len(old),repaired_candidate_count=len(new),
        added_candidates=added,removed_candidates=removed,changed_candidates=changed,
        earlier_candidates_unchanged=True,**exposure,
        full_account_replay_required=bool(added or removed or changed or any(exposure.values())),
        live_qualified=False,source_sha256=refs,
        output_sha256={str(p.relative_to(ROOT)):sha(p) for p in output.iterdir() if p.is_file()})
    refs[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    refs['scripts/validate_three_black_market_inputs.py'] = sha(ROOT/'scripts/validate_three_black_market_inputs.py')
    refs['skills/market_input_validation.py'] = sha(ROOT/'skills/market_input_validation.py')
    (output/'report.json').write_text(encode(report))
    (output/'report.sha256').write_text(sha(output/'report.json')+'\n')
    print(encode({k:v for k,v in report.items() if k not in ('source_sha256','output_sha256','repairs')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.fetch and args.output:
        parser.error('Separate acquisition and offline preparation')
    if args.fetch:
        fetch()
    elif args.output:
        prepare(args.output)
    else:
        parser.error('Choose --fetch (at most five shared-quota requests) or offline --output')
