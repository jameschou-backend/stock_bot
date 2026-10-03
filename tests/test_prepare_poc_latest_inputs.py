from copy import deepcopy
import json

import pandas as pd
import pytest

from scripts.prepare_poc_latest_inputs import (
    bind, digest, load_event_extension, matrix_prefix, merge_events,
    partition_signals, routing_audit, verify_bundle,
)


CALENDAR = pd.to_datetime(['2026-09-08', '2026-09-09', '2026-09-10', '2026-10-02'])


def event(day, entry):
    return dict(event_id='liquid_universe-' + day + '-2330', signal_date=day,
                entry_date=entry, members=['2330'], priority=0.5)


def signals():
    old = [event('2026-09-08', '2026-09-09')]
    new = dict(prefix_exact=True, entries=old + [
        event('2026-09-09', '2026-09-10'), event('2026-10-02', None)])
    return old, new


def test_pending_close_is_retained_but_not_sent_to_account():
    old, new = signals()
    executable, pending = partition_signals(old, new, CALENDAR)
    assert executable == new['entries'][:2]
    assert pending == [new['entries'][-1]]
    assert pending[0]['entry_date'] is None


@pytest.mark.parametrize('mutation,reason', [
    ('history', 'Historical candidate prefix'),
    ('same_day', 'observed T\\+1'),
    ('invented_future', 'observed T\\+1'),
    ('duplicate', 'Duplicate signal'),
    ('etf', 'individual stocks'),
])
def test_signal_adaptation_rejects_causal_or_identity_changes(mutation, reason):
    old, new = signals()
    new = deepcopy(new)
    if mutation == 'history':
        new['entries'][0]['priority'] = 0.9
    elif mutation == 'same_day':
        new['entries'][1]['entry_date'] = '2026-09-09'
    elif mutation == 'invented_future':
        new['entries'][-1]['entry_date'] = '2026-10-05'
    elif mutation == 'duplicate':
        new['entries'].append(new['entries'][-1])
    else:
        new['entries'][-1]['members'] = ['0050']
    with pytest.raises(ValueError, match=reason):
        partition_signals(old, new, CALENDAR)


def test_matrix_extension_cannot_rebase_historical_prices(tmp_path):
    base, extension = tmp_path / 'base', tmp_path / 'extension'
    base.mkdir(); extension.mkdir()
    before = pd.DataFrame({'date': CALENDAR[:2], '2330': [100., 101.]})
    after = pd.DataFrame({'date': CALENDAR, '2330': [100., 101., 110., 120.]})
    before.to_parquet(base / 'close.parquet', index=False)
    after.to_parquet(extension / 'close.parquet', index=False)
    assert matrix_prefix(base, extension, 'close').equals(CALENDAR)
    after.loc[0, '2330'] = 100.01
    after.to_parquet(extension / 'close.parquet', index=False)
    with pytest.raises(ValueError, match='Historical matrix prefix changed'):
        matrix_prefix(base, extension, 'close')


def identity():
    return dict(coverage_start='2018-01-01', coverage_end='2026-10-02',
        extension_observation=dict(provisional=True, official_identity_verified_through='2026-09-09'),
        trading_exclusions=[], episodes=[dict(stock_id=sid, market='TWSE',
            category='ETF' if sid == '0050' else '股票', start='2000-01-01', end=None,
            snapshot_date='2026-10-02') for sid in ('0050', '2330')])


def test_extended_routes_are_available_but_never_certified():
    old, new = signals()
    executable, pending = partition_signals(old, new, CALENDAR)
    rows = routing_audit(identity(), executable, pending, CALENDAR)
    assert {(r['date'], r['stock_id']) for r in rows} == {
        ('2026-09-10', '2330'), ('2026-10-02', '2330'),
        ('2026-09-10', '0050'), ('2026-10-02', '0050')}
    assert all('not_daily_certification' in r['identity_basis'] for r in rows)


def test_unknown_extension_route_fails_before_replay():
    old, new = signals()
    executable, pending = partition_signals(old, new, CALENDAR)
    value = identity()
    value['episodes'] = [e for e in value['episodes'] if e['stock_id'] != '2330']
    with pytest.raises(ValueError, match='Missing extended execution route'):
        routing_audit(value, executable, pending, CALENDAR)


def test_source_verification_rejects_changed_payload_and_path_escape(tmp_path):
    bundle = tmp_path / 'bundle'; bundle.mkdir()
    raw = bundle / 'data.json'; raw.write_text('{}')
    manifest = bundle / 'manifest.json'
    manifest.write_text(json.dumps(dict(files_sha256={'data.json': digest(raw)}, source_sha256={})))
    (bundle / 'manifest.sha256').write_text(digest(manifest))
    refs = {}
    verify_bundle(bundle, tmp_path, refs)
    assert refs['bundle/data.json'] == digest(raw)
    raw.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Sealed source changed'):
        verify_bundle(bundle, tmp_path, {})
    with pytest.raises(ValueError, match='escapes repository'):
        bind(tmp_path.parent / 'outside', tmp_path, {})


def test_corporate_extension_preserves_parent_and_rejects_revised_history():
    old = pd.DataFrame(dict(stock_id=['2330'], event_date=pd.to_datetime(['2026-08-01']).date,
                            payload_json=['{"cash":10}']))
    extension = pd.DataFrame(dict(stock_id=['2330'], event_date=pd.to_datetime(['2026-09-11']),
                                  payload_json=['{"cash":11}']))
    merged = merge_events(old, extension)
    assert merged.iloc[:1].reset_index(drop=True).equals(old)
    assert str(merged.iloc[1].event_date) == '2026-09-11'
    with pytest.raises(ValueError, match='outside-period'):
        merge_events(old, old)
    with pytest.raises(ValueError, match='Duplicate corporate'):
        merge_events(old, pd.concat([extension, extension], ignore_index=True))


def test_event_report_cannot_promote_partial_coverage_or_unbound_data(tmp_path):
    event_path = tmp_path / 'events.parquet'
    pd.DataFrame(dict(stock_id=['2330'], event_date=['2026-09-11'])).to_parquet(event_path, index=False)
    report_path = tmp_path / 'report.json'
    report = dict(corporate_events_extension_complete=False, start='2026-09-10', end='2026-10-02',
        events_path='events.parquet', output_sha256={'events.parquet': digest(event_path)}, source_sha256={})
    def seal():
        report_path.write_text(json.dumps(report))
        report_path.with_suffix('.sha256').write_text(digest(report_path))
    seal()
    with pytest.raises(ValueError, match='not complete'):
        load_event_extension(report_path, tmp_path, {})
    report['corporate_events_extension_complete'] = True
    seal()
    assert len(load_event_extension(report_path, tmp_path, {})) == 1
    report['output_sha256'] = {}
    seal()
    with pytest.raises(ValueError, match='not bound'):
        load_event_extension(report_path, tmp_path, {})
