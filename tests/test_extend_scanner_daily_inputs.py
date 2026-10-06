from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from scripts.extend_scanner_daily_inputs import digest, extend, original_candidates, write
from skills.strategy_scanner.data import load_bundle


def seal(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    manifest['files_sha256'] = {p.name: digest(p) for p in path.iterdir()
                               if p.name not in ('manifest.json', 'manifest.sha256')}
    write(path / 'manifest.json', manifest)
    (path / 'manifest.sha256').write_text(digest(path / 'manifest.json') + '\n')


@pytest.fixture
def sources(tmp_path):
    base = tmp_path / 'base'; base.mkdir()
    end = '2026-10-05'; days = pd.bdate_range(end='2026-10-02', periods=260)
    ids = ['0050', '2330', '9999']; n = len(days)
    close = pd.DataFrame({'0050': 50. + np.arange(n) * .01,
                          '2330': 100. + np.arange(n) * .10,
                          '9999': 80. + np.arange(n) * .05}, index=days)
    volume = pd.DataFrame(1_000_000., index=days, columns=ids)
    volume.at[days[-1], '2330'] = 2_000_000.
    eligible = pd.DataFrame(True, index=days, columns=ids)
    frames = {'close-official': close.copy(), 'close-quality': close.copy(),
              'raw-close': close.copy(), 'raw-volume': volume, 'eligibility': eligible}
    for name, frame in frames.items():
        frame.reset_index(names='date').to_parquet(base / (name + '.parquet'), index=False)
    quotes = pd.DataFrame([dict(stock_id=sid, date=d, open=float(close.at[d, sid] - .2),
        high=float(close.at[d, sid] + .5), low=float(close.at[d, sid] - .5),
        close=float(close.at[d, sid]), volume=float(volume.at[d, sid])) for d in days for sid in ids])
    quotes.to_parquet(base / 'quotes-unmasked.parquet', index=False)
    companies = pd.DataFrame([dict(stock_id=sid, name=sid, market='TWSE', industry='01',
                                   listed_date=pd.Timestamp('2000-01-01')) for sid in ids[1:]])
    companies.to_parquet(base / 'companies.parquet', index=False)
    identity = dict(coverage_start=str(days[0].date()), coverage_end='2026-10-02', identity_repair={},
        episodes=[dict(stock_id=sid, start='2000-01-01', end=None, market='TWSE',
                       category='ETF' if sid == '0050' else '股票', snapshot_date='2026-10-02') for sid in ids],
        trading_exclusions=[], complete_historical_universe=False,
        extension_observation=dict(provisional=True, official_identity_verified_through='2026-09-09'))
    write(base / 'identity.json', identity)
    pending, _ = original_candidates(frames, companies, identity, '2026-10-02')
    assert [e['members'] for e in pending] == [['2330']]
    old = dict(deepcopy(pending[0]), event_id='liquid_universe-2026-10-01-2330',
               signal_date='2026-10-01', entry_date='2026-10-02')
    write(base / 'signals.json', dict(entries={'median50m': [old]}, pending_signal_file='pending-signals.json'))
    write(base / 'pending-signals.json', dict(schema='pending_last_close_signals_v1',
                                             signal_date='2026-10-02', entries=pending))
    pd.DataFrame(dict(stock_id=['2330'], event_date=pd.to_datetime(['2026-08-01']).date,
                     payload_json=['{"cash":10}'])).to_parquet(base / 'events.parquet', index=False)
    write(base / 'manifest.json', dict(schema='poc_latest_input_bundle_v1', start=str(days[0].date()),
        end='2026-10-02', historical_end='2026-09-09', executable_candidate_count=1,
        pending_candidate_count=1, events_extension_complete=True, files_sha256={}, source_sha256={}, limitations=[]))
    seal(base)
    raw = quotes.tail(3).copy(); raw['date'] = pd.Timestamp(end)
    for column in ('open', 'high', 'low', 'close'):
        raw[column] *= 1.04
    raw.loc[raw.stock_id == '2330', 'volume'] = 3_000_000.
    raw.to_parquet(tmp_path / 'quotes.parquet', index=False)
    aa = quotes.tail(3)[['date', 'stock_id', 'close']].copy(); aa.close /= 2
    aa.to_parquet(tmp_path / 'adj-anchor.parquet', index=False)
    ae = raw[['date', 'stock_id', 'close']].copy(); ae.close /= 2
    ae.to_parquet(tmp_path / 'adj-end.parquet', index=False)
    pd.DataFrame([dict(stock_id=sid, type='twse', stock_name=sid, industry_category='test',
                       date='2026-10-06') for sid in ids]).to_parquet(tmp_path / 'snapshot.parquet', index=False)
    return dict(base=base, quotes=tmp_path / 'quotes.parquet', adj_anchor=tmp_path / 'adj-anchor.parquet',
                adj_end=tmp_path / 'adj-end.parquet', market_snapshot=tmp_path / 'snapshot.parquet',
                end=end, output=tmp_path / 'new', root=tmp_path)


def test_extension_preserves_history_promotes_terminal_and_runs_scanner(sources):
    before = {p.name: digest(p) for p in sources['base'].iterdir()}
    result = extend(**sources)
    out, base = sources['output'], sources['base']
    for name in ('raw-close', 'raw-volume', 'close-official', 'close-quality', 'eligibility'):
        old = pd.read_parquet(base / (name + '.parquet'))
        new = pd.read_parquet(out / (name + '.parquet'))
        pd.testing.assert_frame_equal(old, new.iloc[:-1].reset_index(drop=True))
        if name == 'close-official':
            assert new.iloc[-1]['2330'] == pytest.approx(old.iloc[-1]['2330'] * 1.04)
    assert {p.name: digest(p) for p in base.iterdir()} == before
    assert result['parent_terminal_candidates_reproduced'] is True
    assert result['events_extension_complete'] is False
    assert result['network_requests'] == 0 and result['database_mutations'] is False
    assert result['parent_terminal_promotions'] == 1
    ledger = json.loads((out / 'signals.json').read_text())['entries']['median50m']
    assert ledger[-1]['entry_date'] == '2026-10-05'
    pending = json.loads((out / 'pending-signals.json').read_text())['entries']
    assert [e['members'] for e in pending] == [['2330']]
    assert all(e['entry_date'] is None and e['signal_date'] == '2026-10-05' for e in pending)
    bundle = load_bundle(out, '2026-10-05', '2026-10-05')
    assert bundle['provenance']['original_candidates_complete'] is True
    assert bundle['provenance']['officially_crosschecked_extension'] is False
    assert bundle['provenance']['source_end'] == '2026-10-05'
    identity = json.loads((out / 'identity.json').read_text())
    assert identity['extension_observation']['snapshot_record_date_max'] == '2026-10-06'
    assert identity['publication_time_archive_complete'] is False


@pytest.mark.parametrize('change,reason', [('missing', 'snapshot_stock_missing'),
                                         ('market', 'snapshot_market_changed'),
                                         ('date', 'snapshot_record_date_unknown')])
def test_snapshot_unknown_or_changed_never_blindly_extends_eligibility(sources, change, reason):
    path = sources['market_snapshot']; frame = pd.read_parquet(path)
    if change == 'missing':
        frame = frame.loc[frame.stock_id != '2330']
    elif change == 'market':
        frame.loc[frame.stock_id == '2330', 'type'] = 'tpex'
    else:
        frame.loc[frame.stock_id == '2330', 'date'] = 'None'
    frame.to_parquet(path, index=False)
    extend(**sources)
    mask = pd.read_parquet(sources['output'] / 'eligibility.parquet')
    assert bool(mask.iloc[-2]['2330']) and not bool(mask.iloc[-1]['2330'])
    coverage = json.loads((sources['output'] / 'extension-coverage.json').read_text())
    assert next(r for r in coverage['snapshot_identity_checks'] if r['stock_id'] == '2330')['reason'] == reason


def test_industry_only_duplicates_are_collapsed_but_conflicting_market_is_rejected(sources):
    path = sources['market_snapshot']; frame = pd.read_parquet(path)
    duplicate = frame.iloc[[1]].copy(); duplicate.industry_category = 'broad-industry'
    pd.concat([frame, duplicate], ignore_index=True).to_parquet(path, index=False)
    extend(**sources)
    identity = json.loads((sources['output'] / 'identity.json').read_text())
    assert identity['extension_observation']['duplicate_industry_rows_collapsed'] == 1
    duplicate['type'] = 'tpex'
    pd.concat([frame, duplicate], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(ValueError, match='Conflicting snapshot'):
        extend(**dict(sources, output=sources['root'] / 'conflict'))


def test_missing_individual_adjustment_is_nan_not_raw_fallback(sources):
    path = sources['adj_anchor']; frame = pd.read_parquet(path)
    frame.loc[frame.stock_id == '2330', 'close'] = np.nan
    frame.to_parquet(path, index=False)
    extend(**sources)
    close = pd.read_parquet(sources['output'] / 'close-official.parquet')
    assert pd.isna(close.iloc[-1]['2330'])
    report = json.loads((sources['output'] / 'extension-coverage.json').read_text())
    assert {r['stock_id'] for r in report['bridge_missing']} == {'2330'}


def test_missing_benchmark_blocks_original_candidate_completeness(sources):
    path = sources['adj_end']; frame = pd.read_parquet(path)
    frame = frame.loc[frame.stock_id != '0050']; frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match='0050 identity'):
        extend(**sources)
    assert not sources['output'].exists()


@pytest.mark.parametrize('change,reason', [('duplicate', 'Duplicate daily'), ('date', 'wrong date'),
                                         ('numeric_id', 'stock_id must'), ('future_adj', 'wrong date')])
def test_invalid_source_dates_duplicates_and_nonstrings_are_rejected(sources, change, reason):
    path = sources['adj_end'] if change == 'future_adj' else sources['quotes']
    frame = pd.read_parquet(path)
    if change == 'duplicate': frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif change in ('date', 'future_adj'): frame['date'] = pd.Timestamp('2026-10-06')
    else: frame.stock_id = frame.stock_id.astype(int)
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match=reason): extend(**sources)


def test_finmind_column_mapping_is_explicit(sources):
    path = sources['quotes']; frame = pd.read_parquet(path)
    frame = frame.rename(columns={'high': 'max', 'low': 'min', 'volume': 'Trading_Volume'})
    frame.to_parquet(path, index=False)
    assert extend(**sources)['pending_candidate_count'] == 1


def test_tampered_parent_and_modified_terminal_are_rejected(sources):
    path = sources['base'] / 'pending-signals.json'
    pending = json.loads(path.read_text()); pending['entries'][0]['priority'] += .01
    write(path, pending)
    with pytest.raises(ValueError, match='hash mismatch'): extend(**sources)
    seal(sources['base'])
    with pytest.raises(ValueError, match='terminal candidates changed'): extend(**sources)


def test_no_overwrite_and_no_skipped_weekday(sources):
    sources['output'].mkdir()
    with pytest.raises(ValueError, match='new repository'): extend(**sources)
    sources['output'].rmdir()
    with pytest.raises(ValueError, match='Multiple weekdays'):
        extend(**dict(sources, end='2026-10-06'))


def event_report(sources):
    path = sources['root'] / 'official'; path.mkdir()
    events = path / 'events.parquet'
    pd.DataFrame(dict(stock_id=['9999'], event_date=pd.to_datetime(['2026-10-05']).date,
                     payload_json=['{"cash":1}'])).to_parquet(events, index=False)
    payload = dict(schema='poc_latest_official_extension_v1', start='2026-10-05', end='2026-10-05',
        corporate_events_extension_complete=True, source_sha256={},
        events_path='official/events.parquet', output_sha256={'official/events.parquet': digest(events)},
        corporate_action_coverage=[dict(kind=m+'_'+k, start='2026-10-05', end='2026-10-05', complete=True)
            for m in ('twse', 'tpex') for k in ('ex_rights', 'capital_reduction', 'par_value_change')])
    report = path / 'report.json'; write(report, payload)
    report.with_suffix('.sha256').write_text(digest(report))
    return report


def test_complete_six_kind_report_extends_events_but_not_live_certification(sources):
    report = event_report(sources)
    result = extend(**sources, event_extension_report=report)
    assert result['events_extension_complete'] and result['corporate_events_extended']
    assert result['live_qualified'] is False
    old = pd.read_parquet(sources['base'] / 'events.parquet')
    new = pd.read_parquet(sources['output'] / 'events.parquet')
    pd.testing.assert_frame_equal(new.iloc[:len(old)].reset_index(drop=True), old)
    assert len(new) == len(old) + 1
    assert 'official/report.json' in result['source_sha256']


@pytest.mark.parametrize('mutation', ['incomplete', 'missing_kind', 'wrong_interval', 'unbound_events'])
def test_partial_or_unbound_event_report_cannot_claim_completion(sources, mutation):
    report = event_report(sources); payload = json.loads(report.read_text())
    if mutation == 'incomplete': payload['corporate_events_extension_complete'] = False
    elif mutation == 'missing_kind': payload['corporate_action_coverage'].pop()
    elif mutation == 'wrong_interval': payload['corporate_action_coverage'][0]['start'] = '2026-10-02'
    else: payload['output_sha256'] = {}
    write(report, payload); report.with_suffix('.sha256').write_text(digest(report))
    with pytest.raises(ValueError, match='Corporate'):
        extend(**sources, event_extension_report=report)
