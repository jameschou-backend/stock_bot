from pathlib import Path
import pytest

from skills.all_entry_odd_cache import CachedSupplement
from skills.poc_broker_odd import SupplementaryOddSources
from skills.replay_market_feeds import ReplayDataUnavailable
from test_poc_broker_odd import seal, simple, record
from test_prepare_volume_profile_odd import context, completed_followup


@pytest.mark.parametrize('kind', ['raw', 'simple', 'both'])
def test_repeated_tables_keep_values_references_selection_and_errors(tmp_path, kind):
    if kind in ('raw', 'both'):seal(tmp_path, 'source', record())
    if kind in ('simple', 'both'):simple(tmp_path, record())
    scalar = SupplementaryOddSources(tmp_path)
    cached = CachedSupplement.from_verified_inventory(SupplementaryOddSources(tmp_path))
    for _ in range(3):
        assert cached.get('2024-01-02', '0050', 'twse') == scalar.get('2024-01-02', '0050', 'twse')
        assert cached.refs == scalar.refs
        assert cached.selected_sources == scalar.selected_sources
        assert cached.queries == scalar.queries
        for provider in (scalar, cached):
            with pytest.raises(ReplayDataUnavailable, match='stock absent'):
                provider.get('2024-01-02', '9999', 'twse')
    row = cached.get('2024-01-02', '0050', 'twse')
    row['odd_high'] = -1
    assert cached.get('2024-01-02', '0050', 'twse')['odd_high'] == 102


@pytest.mark.parametrize('change', ['raw', 'normalized', 'index', 'symlink'])
def test_cached_parse_still_rejects_changed_or_escaping_sources(tmp_path, change):
    refs = seal(tmp_path, 'source', record())
    cached = CachedSupplement.from_verified_inventory(SupplementaryOddSources(tmp_path))
    cached.get('2024-01-02', '0050', 'twse')
    suffix = {'raw':'.raw.json', 'normalized':'.rows.json', 'index':'index.json', 'symlink':'.raw.json'}[change]
    path = tmp_path / next(k for k in refs if k.endswith(suffix))
    if change == 'symlink':
        outside = tmp_path.parent / ('outside-' + tmp_path.name + '.json')
        outside.write_bytes(path.read_bytes());path.unlink();path.symlink_to(outside)
    else:path.write_text('{}')
    with pytest.raises(ReplayDataUnavailable):cached.get('2024-01-02', '0050', 'twse')


def test_receipt_ancestors_are_validated_after_cached_parse(context):
    root, _, _, _ = context
    completed_followup(context)
    scalar = SupplementaryOddSources(root)
    cached = CachedSupplement.from_verified_inventory(SupplementaryOddSources(root))
    for _ in range(2):
        assert cached.get('2025-02-14', '6558', 'twse') == scalar.get('2025-02-14', '6558', 'twse')
        assert cached.refs == scalar.refs
        assert cached.selected_sources == scalar.selected_sources
    authorization = next(k for k in cached.refs if k.endswith('/authorization.json'))
    (root / authorization).write_text('{}')
    with pytest.raises(ReplayDataUnavailable):cached.get('2025-02-14', '6558', 'twse')
