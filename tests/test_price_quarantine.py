from datetime import date
import json
import pytest
from app.price_quarantine import load_registry, reject_quarantined, same_observation
from scripts.resolve_price_quarantine import invalidated_labels, build_registry


def test_reviewed_evidence_reproduces_registry():
    from app.price_quarantine import REGISTRY
    # Official/cache evidence is optional on clean checkouts, not a silent runtime fallback.
    from scripts.resolve_price_quarantine import MEMBERSHIP
    if not MEMBERSHIP.exists(): pytest.skip('Local source evidence is not distributed in git')
    assert build_registry() == json.loads(REGISTRY.read_text())


def test_known_key_rejects_original_and_revised_values():
    registry = load_registry()
    key, evidence = next(iter(registry.items()))
    row = dict(stock_id=key[0], trading_date=date.fromisoformat(key[1]), **evidence['original'])
    with pytest.raises(ValueError, match='refusing write'): reject_quarantined([row])
    row['close'] = 12345
    with pytest.raises(ValueError, match='refusing write'): reject_quarantined([row])
    row['trading_date'] = date(2020, 1, 2)
    assert reject_quarantined([row]) == [row]


def test_registry_is_required_and_duplicate_keys_fail(tmp_path):
    with pytest.raises(FileNotFoundError): load_registry(tmp_path/'missing.json')
    row=next(iter(load_registry().values()))
    path=tmp_path/'duplicates.json'
    path.write_text(json.dumps(dict(schema='reviewed_price_quarantine_v1',rows=[row,row])))
    with pytest.raises(ValueError,match='Duplicate'): load_registry(path)


def test_fingerprint_has_no_float_tolerance_or_missing_value_fallback():
    row=dict(open='1.000001',high=2,low=1,close=2,volume=3)
    assert same_observation(row,row)
    assert not same_observation({**row,'open':'1.000002'},row)
    assert not same_observation({**row,'volume':None},row)


def test_deleted_tail_invalidates_labels_before_delisting_not_only_bad_days():
    days=['2022-01-01','2022-01-02','2022-01-03','2022-01-04','2026-06-09']
    assert invalidated_labels(days,{'2026-06-09'},2)==['2022-01-03','2026-06-09']
    assert invalidated_labels(days,set(),2)==[]


def test_middle_removal_changes_multiple_forward_endpoints():
    assert invalidated_labels(['01','02','03','04','05'],{'03'},2)==['01','02','03']
