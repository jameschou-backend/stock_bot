"""Source closure and historical-query guards for the latest POC adapters."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import skills.poc_latest_data as latest
from scripts.prepare_poc_latest_inputs import digest
from skills.replay_market_feeds import ReplayDataUnavailable


def seal(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    path.with_suffix('.sha256').write_text(digest(path))


def test_checked_json_never_replaces_an_already_bound_source(tmp_path):
    path = tmp_path / 'source.json'
    path.write_text('{"version":1}')
    refs = {}
    assert latest.checked_json(path, refs, tmp_path) == {'version': 1}
    path.write_text('{"version":2}')
    with pytest.raises(ValueError, match='Source changed'):
        latest.checked_json(path, refs, tmp_path)


def test_closure_rejects_different_hashes_for_same_source_and_output(tmp_path):
    path = tmp_path / 'source.json'; path.write_text('{}')
    report = dict(source_sha256={'source.json':'0' * 64}, output_sha256={'source.json':digest(path)})
    with pytest.raises(ValueError):
        latest.bind_closure(report, {}, tmp_path)


def test_closure_rejects_changed_files_and_paths_outside_root(tmp_path):
    path = tmp_path / 'source.json'; path.write_text('{}')
    with pytest.raises(ValueError):
        latest.bind_closure({'source_sha256': {'source.json':'0' * 64}}, {}, tmp_path)
    with pytest.raises(ValueError):
        latest.bind_closure({'source_sha256': {'../outside.json':'0' * 64}}, {}, tmp_path)


def test_all_new_adapter_sources_and_tests_are_bound_and_immutable(tmp_path):
    for name in latest.ADAPTER_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# source fixture\n')
    refs = {}
    latest.bind_adapter_sources(tmp_path, refs)
    assert set(refs) == set(latest.ADAPTER_SOURCES)
    changed = tmp_path / 'skills/poc_latest_execution.py'
    changed.write_text('# changed after study creation\n')
    with pytest.raises(ValueError, match='Conflicting source hashes'):
        latest.bind_adapter_sources(tmp_path, refs)


def test_missing_adapter_test_is_not_silently_omitted(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest.bind_adapter_sources(tmp_path, {})


def profile_fixture():
    result = dict(event_id='event-old', stock_id='2330', signal_date='2026-09-08',
                  available=True, poc_up=True, profile={'poc_before': 100., 'poc_after': 110.})
    value = latest.LatestAccountData.__new__(latest.LatestAccountData)
    value.query_ids = {'poc_red': {'event-old'}}
    value.profile_index = {'poc_red': {'event-old': result}}
    value.used = {'poc_red': {}}
    value.profiles = lambda event: dict(event_id=event['event_id'], available=False, reason='raw_tape_unavailable')
    return value, result


def test_historical_profiles_reuse_exact_frozen_arm_evidence_and_do_not_alias():
    value, original = profile_fixture()
    event = dict(event_id='event-old', members=['2330'], signal_date='2026-09-08', entry_date='2026-09-09')
    def prohibited(event):
        raise AssertionError('Old profile should never be recomputed')
    value.profiles = prohibited
    observed = value.profile('poc_red', event)
    assert observed == original
    observed['profile']['poc_after'] = -1.
    assert original['profile']['poc_after'] == 110.


def test_historical_profile_cannot_query_a_previously_unqueried_candidate():
    value, _ = profile_fixture()
    event = dict(event_id='other', members=['2330'], signal_date='2026-09-08', entry_date='2026-09-09')
    with pytest.raises(ValueError, match='Historical POC query path changed'):
        value.profile('poc_red', event)


def test_extended_profile_uses_new_adapter_even_on_anchor_close():
    value, _ = profile_fixture()
    event = dict(event_id='new', members=['2330'], signal_date='2026-09-09', entry_date='2026-09-10')
    result = value.profile('poc_red', event)
    assert result['reason'] == 'raw_tape_unavailable'
    assert value.used['poc_red']['new'] == result


def test_execution_refs_cannot_overwrite_a_frozen_source():
    value = latest.LatestAccountData.__new__(latest.LatestAccountData)
    value.refs = {'sealed.json': 'old'}
    value.execution = SimpleNamespace(refs={'sealed.json':'changed'}, finmind=lambda *args: pd.DataFrame())
    with pytest.raises(ValueError):
        value.finmind('2330', 'TaiwanStockPriceLimit')


def test_new_odd_refs_cannot_overwrite_a_frozen_source():
    value = latest.LatestAccountData.__new__(latest.LatestAccountData)
    value.refs = {'sealed.json': 'old'}
    value.odds = SimpleNamespace(refs={'sealed.json':'changed'}, get=lambda *args: {'volume': 10})
    with pytest.raises(ValueError):
        value.get_odd('2026-09-10', '2330', 'TWSE')


def profile_files(tmp_path, monkeypatch):
    """Small two-session official profile fixture; the expensive old loader is stubbed."""
    bundle, official = tmp_path / 'bundle', tmp_path / 'official'
    bundle.mkdir(); official.mkdir()
    pd.DataFrame({'date':pd.to_datetime(['2026-09-09', '2026-10-02'])}).to_parquet(bundle / 'close-official.parquet', index=False)
    pd.DataFrame({'stock_id':[], 'event_date':[]}).to_parquet(bundle / 'events.parquet', index=False)
    matrix_refs = {p.name:digest(p) for p in bundle.iterdir()}
    seal(bundle / 'manifest.json', dict(end='2026-10-02', files_sha256=matrix_refs,
         events_extension_complete=True, source_sha256={}))
    normalized = pd.DataFrame([dict(stock_id=sid, date='2026-10-02', market=market, source_id=market + '-new')
                               for sid, market in [('2330','TWSE'), ('6206','TPEX')]])
    normalized.to_parquet(official / 'official-normalized.parquet', index=False)
    sources = {'sources':{m + '-new':{} for m in ('TWSE','TPEX')}}
    (official / 'sources.json').write_text(json.dumps(sources))
    outputs = {str(p.relative_to(tmp_path)):digest(p) for p in official.iterdir()}
    report = dict(schema='poc_latest_official_extension_v1', start='2026-09-10', end='2026-10-02',
                  corporate_events_extension_complete=True, daily_tables_extension_complete=True,
                  required_market_days=2, accepted_market_days=2, missing_market_days=[],
                  normalized_path='official/official-normalized.parquet', sources_path='official/sources.json',
                  source_sha256={}, output_sha256=outputs)
    seal(official / 'report.json', report)
    prereg = tmp_path / 'prereg.md'; prereg.write_text('fixed rules')
    monkeypatch.setattr(latest, 'ROOT', tmp_path)
    monkeypatch.setattr(latest, 'PREREG', prereg)
    def initialize(self, *args, **kwargs):
        self.refs = {}
        self.official = pd.DataFrame([dict(stock_id='2330', date='2026-09-09', market='TWSE',
                                          source_id='old')]).set_index(['stock_id','date'])
        self.official_meta = {'sources': {'old':{}}}
    def mark(self, path, expected=None):
        path = Path(path).resolve()
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError('Profile input hash changed')
        self.refs[str(path)] = actual
        return actual
    monkeypatch.setattr(latest.AccountProfileData, '__init__', initialize)
    monkeypatch.setattr(latest.LatestProfiles, '_mark', mark)
    return bundle, official, report


def test_profiles_append_official_rows_without_replacing_historical_rows(tmp_path, monkeypatch):
    bundle, official, _ = profile_files(tmp_path, monkeypatch)
    profile = latest.LatestProfiles(bundle, official, online=False)
    assert len(profile.official) == 3
    assert profile.official.loc[('2330','2026-09-09'), 'source_id'] == 'old'


def test_profiles_reject_an_unbound_normalized_table(tmp_path, monkeypatch):
    bundle, official, report = profile_files(tmp_path, monkeypatch)
    del report['output_sha256']['official/official-normalized.parquet']
    seal(official / 'report.json', report)
    with pytest.raises(ValueError):
        latest.LatestProfiles(bundle, official, online=False)


def test_profiles_reject_changed_report_sidecar(tmp_path, monkeypatch):
    bundle, official, report = profile_files(tmp_path, monkeypatch)
    report['end'] = '2026-09-09'
    (official / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        latest.LatestProfiles(bundle, official, online=False)


def test_profiles_reject_a_missing_new_market_day(tmp_path, monkeypatch):
    bundle, official, report = profile_files(tmp_path, monkeypatch)
    path = official / 'official-normalized.parquet'
    rows = pd.read_parquet(path)
    rows.loc[rows.market.eq('TWSE')].to_parquet(path, index=False)
    report['output_sha256']['official/official-normalized.parquet'] = digest(path)
    seal(official / 'report.json', report)
    with pytest.raises(ValueError, match='market-day coverage'):
        latest.LatestProfiles(bundle, official, online=False)


def odd_fixture(market='twse', high='102.00'):
    from test_replay_market_feeds import odd_record
    from skills.replay_market_feeds import FIELDS
    record = odd_record(market, '2024-01-02')
    table = record['payload'] if market == 'twse' else record['payload']['tables'][0]
    if market == 'twse':
        table['title'] = '113年01月02日 盤中零股交易行情單'
    table['data'][0][table['fields'].index(FIELDS[market]['odd_high'])] = high
    return record


def seal_odd_raw(root, folder, record, *, changed_rows=False):
    from skills import replay_market_feeds
    directory = root / '.cache' / folder
    directory.mkdir(parents=True, exist_ok=True)
    market, day = record['provider'], record['day']
    stem = 'odd-' + market + '-' + day
    raw = directory / (stem + '.raw.json')
    normalized = directory / (stem + '.rows.json')
    raw.write_text(json.dumps(record))
    rows = latest.parse_odd(record, market, day)
    if changed_rows:
        next(iter(rows.values()))['odd_shares'] += 1
    normalized.write_text(json.dumps(dict(schema=1, raw_sha256=digest(raw), rows=rows)))
    index = directory / 'index.json'
    index.write_text(json.dumps(dict(schema=1, parser_sha256=digest(Path(replay_market_feeds.__file__)),
        files_sha256={raw.name:digest(raw), normalized.name:digest(normalized)},
        entries={'odd:' + market + ':' + day:dict(raw_file=raw.name, rows_file=normalized.name,
            row_count=len(rows), first_date=day, last_date=day, empty_verified_response=False)})))
    return {str(p.relative_to(root)):digest(p) for p in (index, raw, normalized)}


def seal_odd_simple(root, record):
    directory = root / latest.CACHE_ORDER[0]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('odd-' + record['provider'] + '-' + record['day'] + '.json')
    path.write_text(json.dumps(record))
    return {str(path.relative_to(root)):digest(path)}


@pytest.mark.parametrize('market,sid', [('twse','0050'), ('tpex','6488')])
def test_first_day_frozen_odd_raw_index_and_rows_are_reused(tmp_path, market, sid):
    refs = seal_odd_raw(tmp_path, 'old-execution/feeds', odd_fixture(market))
    source = latest.FrozenOddSources(tmp_path, refs)
    row = source.get('2024-01-02', sid, market)
    assert row['odd_shares'] == 1200 and row['odd_high'] == 102.
    assert source.refs == refs
    row['odd_shares'] = 0
    assert source.get('2024-01-02', sid, market)['odd_shares'] == 1200


def test_raw_sources_take_precedence_over_simple_wrapper(tmp_path):
    refs = seal_odd_raw(tmp_path, 'old-execution/feeds', odd_fixture())
    refs.update(seal_odd_simple(tmp_path, odd_fixture(high='103.00')))
    assert latest.FrozenOddSources(tmp_path, refs).get('2024-01-02','0050','twse')['odd_high'] == 102.


def test_simple_wrapper_used_only_when_no_bound_raw_source(tmp_path):
    refs = seal_odd_simple(tmp_path, odd_fixture())
    # A new unsealed cache cannot change old execution even when present on disk.
    seal_odd_raw(tmp_path, 'new-unsealed', odd_fixture(high='103.00'))
    source = latest.FrozenOddSources(tmp_path, refs)
    assert source.get('2024-01-02','0050','twse')['odd_high'] == 102.
    assert source.refs == refs


def test_disagreeing_frozen_raw_versions_fail_for_requested_stock(tmp_path):
    refs = seal_odd_raw(tmp_path, 'one', odd_fixture())
    refs.update(seal_odd_raw(tmp_path, 'two', odd_fixture(high='103.00')))
    with pytest.raises(ReplayDataUnavailable, match='Conflicting frozen odd rows'):
        latest.FrozenOddSources(tmp_path, refs).get('2024-01-02','0050','twse')


def test_frozen_raw_rows_mismatch_does_not_fall_back_to_wrapper(tmp_path):
    refs = seal_odd_raw(tmp_path, 'one', odd_fixture(), changed_rows=True)
    refs.update(seal_odd_simple(tmp_path, odd_fixture()))
    with pytest.raises(ReplayDataUnavailable, match='raw and normalized rows disagree'):
        latest.FrozenOddSources(tmp_path, refs).get('2024-01-02','0050','twse')


def test_frozen_raw_cache_requires_bound_normalized_rows_and_unchanged_bytes(tmp_path):
    refs = seal_odd_raw(tmp_path, 'one', odd_fixture())
    unbound = {k:v for k,v in refs.items() if not k.endswith('.rows.json')}
    with pytest.raises(ReplayDataUnavailable, match='outside the sealed closure'):
        latest.FrozenOddSources(tmp_path, unbound).get('2024-01-02','0050','twse')
    source = latest.FrozenOddSources(tmp_path, refs)
    raw_name = next(k for k in refs if k.endswith('.raw.json'))
    (tmp_path / raw_name).write_text('{}')
    with pytest.raises(ReplayDataUnavailable, match='source changed'):
        source.get('2024-01-02','0050','twse')
