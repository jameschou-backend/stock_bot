import json
from pathlib import Path

import pytest

from skills import replay_market_feeds
from skills.official_daily_acquisition import digest
from skills.poc_broker_odd import OddMarketDayMissing, SupplementaryOddSources
from skills.poc_latest_execution import CACHE_ORDER
from skills.replay_market_feeds import FIELDS, ReplayDataUnavailable, parse_odd
from test_replay_market_feeds import odd_record
from test_prepare_volume_profile_odd import context, completed_followup


def record(market='twse', day='2024-01-02', high='102.00'):
    result = odd_record(market, day)
    table = result['payload'] if market == 'twse' else result['payload']['tables'][0]
    if market == 'twse':
        from datetime import date
        d = date.fromisoformat(day)
        table['title'] = f'{d.year-1911}年{d.month:02d}月{d.day:02d}日 盤中零股交易行情單'
    table['data'][0][table['fields'].index(FIELDS[market]['odd_high'])] = high
    return result


def seal(root, folder, value, *, parser=None):
    directory = root / '.cache' / folder
    directory.mkdir(parents=True)
    market, day = value['provider'], value['day']
    stem = 'odd-' + market + '-' + day
    raw, norm = directory / (stem + '.raw.json'), directory / (stem + '.rows.json')
    raw.write_text(json.dumps(value))
    parsed = parse_odd(value, market, day)
    norm.write_text(json.dumps(dict(schema=1, raw_sha256=digest(raw), rows=parsed)))
    index = directory / 'index.json'
    index.write_text(json.dumps(dict(schema=1, parser_sha256=parser or digest(Path(replay_market_feeds.__file__)),
        entries={'odd:'+market+':'+day:dict(raw_file=raw.name, rows_file=norm.name, row_count=len(parsed))},
        files_sha256={raw.name:digest(raw), norm.name:digest(norm)})))
    return {str(p.relative_to(root)):digest(p) for p in (index, raw, norm)}


def simple(root, value):
    path = root / CACHE_ORDER[0] / ('odd-'+value['provider']+'-'+value['day']+'.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize('market,sid', [('twse', '0050'), ('tpex', '6488')])
def test_reuses_full_market_source_and_binds_only_consumed_evidence(tmp_path, market, sid):
    refs = seal(tmp_path, 'additional', record(market))
    data = SupplementaryOddSources(tmp_path)
    assert data.refs == {}
    row = data.get('2024-01-02', sid, market.upper())
    assert row['odd_shares'] == 1200
    assert data.refs == refs
    assert data.selected_sources['odd:'+market+':2024-01-02']['stock_ids'] == [sid]
    row['odd_shares'] = 0
    assert data.get('2024-01-02', sid, market)['odd_shares'] == 1200


def test_distinguishes_market_day_absence_from_missing_stock(tmp_path):
    seal(tmp_path, 'additional', record())
    data = SupplementaryOddSources(tmp_path)
    with pytest.raises(OddMarketDayMissing):
        data.get('2024-01-03', '0050', 'twse')
    with pytest.raises(ReplayDataUnavailable, match='stock absent') as caught:
        data.get('2024-01-02', '9999', 'twse')
    assert not isinstance(caught.value, OddMarketDayMissing)


def test_explicit_zero_is_preserved_not_manufactured(tmp_path):
    source = record()
    table = source['payload']
    for key in ('odd_shares', 'odd_last', 'odd_low', 'odd_high'):
        table['data'][0][table['fields'].index(FIELDS['twse'][key])] = '0' if key == 'odd_shares' else '--'
    seal(tmp_path, 'zero', source)
    assert SupplementaryOddSources(tmp_path).get('2024-01-02', '0050', 'twse')['odd_shares'] == 0


def test_different_raw_sources_conflict_for_requested_stock(tmp_path):
    seal(tmp_path, 'one', record())
    seal(tmp_path, 'two', record(high='103.00'))
    with pytest.raises(ReplayDataUnavailable, match='Conflicting supplementary odd rows'):
        SupplementaryOddSources(tmp_path).get('2024-01-02', '0050', 'twse')


def test_simple_source_cannot_override_conflicting_raw(tmp_path):
    seal(tmp_path, 'one', record())
    simple(tmp_path, record(high='103.00'))
    with pytest.raises(ReplayDataUnavailable, match='Conflicting supplementary odd rows'):
        SupplementaryOddSources(tmp_path).get('2024-01-02', '0050', 'twse')


def test_restricted_inventory_ignores_later_unsealed_addition(tmp_path):
    refs = seal(tmp_path, 'one', record())
    seal(tmp_path, 'two', record(high='103.00'))
    data = SupplementaryOddSources(tmp_path, refs)
    assert data.get('2024-01-02', '0050', 'twse')['odd_high'] == 102
    assert data.refs == refs


def test_discovery_is_frozen_before_later_files_appear(tmp_path):
    seal(tmp_path, 'one', record())
    data = SupplementaryOddSources(tmp_path)
    seal(tmp_path, 'two', record(high='103.00'))
    assert data.get('2024-01-02', '0050', 'twse')['odd_high'] == 102


def test_changed_cached_bytes_are_detected_after_first_read(tmp_path):
    refs = seal(tmp_path, 'one', record())
    data = SupplementaryOddSources(tmp_path)
    data.get('2024-01-02', '0050', 'twse')
    raw = next(k for k in refs if k.endswith('.raw.json'))
    (tmp_path / raw).write_text('{}')
    with pytest.raises(ReplayDataUnavailable, match='source changed'):
        data.get('2024-01-02', '0050', 'twse')


@pytest.mark.parametrize('bad', ['normalized', 'date', 'partial', 'row_count'])
def test_parser_and_provenance_are_not_relaxed(tmp_path, bad):
    refs = seal(tmp_path, 'one', record())
    paths = {Path(k).suffixes[-2] if len(Path(k).suffixes)>1 else 'index':tmp_path/k for k in refs}
    raw, norm, index = paths['.raw'], paths['.rows'], paths['index']
    r, n, i = (json.loads(p.read_text()) for p in (raw, norm, index))
    if bad == 'normalized':
        next(iter(n['rows'].values()))['odd_shares'] += 1
    elif bad == 'date':
        r['payload']['date'] = '20240103'
    elif bad == 'partial':
        r['payload']['total'] = len(r['payload']['data']) + 1
    else:
        next(iter(i['entries'].values()))['row_count'] += 1
    raw.write_text(json.dumps(r))
    n['raw_sha256'] = digest(raw)
    norm.write_text(json.dumps(n))
    i['files_sha256'] = {raw.name:digest(raw), norm.name:digest(norm)}
    index.write_text(json.dumps(i))
    with pytest.raises(ReplayDataUnavailable):
        SupplementaryOddSources(tmp_path).get('2024-01-02', '0050', 'twse')


def test_wrong_parser_not_adopted_and_unbound_rows_rejected(tmp_path):
    bad = seal(tmp_path, 'incompatible', record(), parser='0'*64)
    source = SupplementaryOddSources(tmp_path)
    assert len(source.excluded_indexes) == 1
    with pytest.raises(OddMarketDayMissing):
        source.get('2024-01-02', '0050', 'twse')
    refs = seal(tmp_path, 'one', record())
    refs = {k:v for k,v in refs.items() if not k.endswith('.rows.json')}
    with pytest.raises(ReplayDataUnavailable, match='outside sealed closure'):
        SupplementaryOddSources(tmp_path, refs)


def test_simple_legacy_source_and_bound_replay(tmp_path):
    path = simple(tmp_path, record())
    data = SupplementaryOddSources(tmp_path)
    assert data.get('2024-01-02', '0050', 'twse')['odd_shares'] == 1200
    replay = SupplementaryOddSources(tmp_path, data.source_refs)
    assert replay.get('2024-01-02', '0050', 'twse') == data.get('2024-01-02', '0050', 'twse')
    assert str(path.relative_to(tmp_path)) in replay.refs


def test_initial_inventory_contains_linked_ancestors_before_any_get(context):
    root, _, _, _ = context
    completed_followup(context)
    original = SupplementaryOddSources(root)
    assert original.refs == {} and original.queries == []
    saved = json.loads(json.dumps(original.source_refs))
    assert any('/authorization.json' in name for name in saved)
    replay = SupplementaryOddSources(root, saved)
    assert replay.get('2025-02-14', '6558', 'twse')['odd_shares'] == 10000
    assert original.get('2025-02-14', '6558', 'twse') == replay.get('2025-02-14', '6558', 'twse')
    assert original.source_refs == saved == replay.source_refs
    assert original.refs == replay.refs


def test_linked_wrapper_without_complete_receipt_cannot_be_downgraded(tmp_path):
    value = record()
    value.update(source_receipt='.cache/missing/receipt.json', source_receipt_sha256='0'*64,
                 source_raw='.cache/missing/raw.json', source_raw_sha256='0'*64)
    simple(tmp_path, value)
    with pytest.raises(ReplayDataUnavailable, match='receipt chain'):
        SupplementaryOddSources(tmp_path).get('2024-01-02', '0050', 'twse')


def test_stripped_receipt_link_is_not_an_ordinary_wrapper(tmp_path):
    value = record()
    value.update(source_raw='.cache/missing/raw.json', source_raw_sha256='0'*64)
    simple(tmp_path, value)
    with pytest.raises(ReplayDataUnavailable, match='receipt chain is incomplete'):
        SupplementaryOddSources(tmp_path).get('2024-01-02', '0050', 'twse')


def test_unrelated_in_progress_index_does_not_enter_odd_inventory(tmp_path):
    seal(tmp_path, 'one', record())
    unrelated = tmp_path / '.cache' / 'other' / 'index.json'
    unrelated.parent.mkdir()
    unrelated.write_text('{')
    assert SupplementaryOddSources(tmp_path).get('2024-01-02', '0050', 'twse')['odd_shares'] == 1200


def test_index_changed_during_discovery_rejected(tmp_path, monkeypatch):
    seal(tmp_path, 'one', record())
    original = SupplementaryOddSources._fingerprint
    reads = []
    def changed(path):
        value = original(path)
        if path.name == 'index.json':
            reads.append(True)
            return value[0], value[1], value[2] + len(reads)
        return value
    monkeypatch.setattr(SupplementaryOddSources, '_fingerprint', staticmethod(changed))
    with pytest.raises(ReplayDataUnavailable, match='changed during discovery'):
        SupplementaryOddSources(tmp_path)


def test_newer_dates_supported_offline_and_outside_scope_rejected(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('No HTTP permitted')
    monkeypatch.setattr('requests.sessions.Session.send', fail)
    seal(tmp_path, 'new-day', record(day='2026-10-02'))
    data = SupplementaryOddSources(tmp_path)
    assert data.get('2026-10-02', '0050', 'twse')['source_date'] == '2026-10-02'
    for args in [('2023-12-29', '0050', 'twse'), ('2026-10-05', '0050', 'twse'),
                 ('2026-10-02', '00631L', 'twse'), ('2026-10-02', '0050', 'unknown')]:
        with pytest.raises(ValueError):
            data.get(*args)


def test_path_escape_rejected(tmp_path):
    refs = seal(tmp_path, 'one', record())
    index = tmp_path / next(k for k in refs if k.endswith('index.json'))
    value = json.loads(index.read_text())
    next(iter(value['entries'].values()))['raw_file'] = '../../outside.json'
    index.write_text(json.dumps(value))
    with pytest.raises(ReplayDataUnavailable, match='unsafe filename'):
        SupplementaryOddSources(tmp_path)
