from datetime import date
import json
import threading
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_volume_profile import write
from skills import volume_profile_data as data


def bare_provider(tmp_path, monkeypatch, *, maximum=1, online=True):
    # Exercise acquisition independently of the large sealed project bundle.
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    monkeypatch.setattr(data, 'PILOT', tmp_path/'pilot')
    monkeypatch.setattr(data, 'local', lambda p: p)
    import scripts.prepare_volume_profile as preparation
    monkeypatch.setattr(preparation, 'local', lambda p: Path(p) if Path(p).is_absolute() else tmp_path/p)
    p = object.__new__(data.AccountProfileData)
    p.directory = tmp_path/'new'; p.directory.mkdir()
    p.online, p.maximum = online, maximum
    p._lock, p._stop, p._config = threading.Lock(), threading.Event(), None
    p.refs, p.reuse_index, p.receipt_hashes = {}, {}, {}
    return p


@pytest.mark.parametrize('reason', sorted(data.QUALITY_REASONS))
def test_unknown_quality_is_explicit_not_negative_poc(reason):
    value = data.unknown(reason)
    assert value['poc_up'] is None and not value['available']
    assert value['recoverable'] is False


def test_missing_request_never_eligible_for_quality_fallback():
    value = data.unknown('raw_tape_unavailable')
    assert value['recoverable'] is True and value['poc_up'] is None


@pytest.mark.parametrize('members,signal,entry', [(['00631L'],'2024-01-02','2024-01-03'),(['0050'],'2024-01-02','2024-01-03'),
    (['2330','2303'],'2024-01-02','2024-01-03'),(['2330'],'2024-01-03','2024-01-03')])
def test_event_rejects_non_individual_or_future_signal(members,signal,entry):
    with pytest.raises(ValueError):
        data.validate_event(dict(members=members,signal_date=signal,entry_date=entry))


def test_request_reservation_no_automatic_retry_and_budget(tmp_path,monkeypatch):
    p = bare_provider(tmp_path,monkeypatch)
    import app.config, app.finmind
    from types import SimpleNamespace
    monkeypatch.setattr(app.config,'load_config',lambda:SimpleNamespace(finmind_token='test-only',finmind_requests_per_hour=6000))
    calls = []
    def fetch(*args,**kwargs):
        calls.append((args,kwargs))
        assert len(list((p.directory/'attempts').glob('*.json'))) == 1
        return pd.DataFrame({'stock_id':['2330'],'date':['2023-12-29'],'price':[10.],'volume':[1.]})
    monkeypatch.setattr(app.finmind,'fetch_dataset',fetch)
    first = p._raw(('2330','2023-12-29'))
    assert first['status'] == 'received'
    assert p._raw(('2330','2023-12-29')) == first
    assert p._raw(('2303','2023-12-29'))['status'] == 'request_budget_or_quota_paused'
    assert len(calls) == 1
    assert calls[0][0] == ('TaiwanStockPriceTick',date(2023,12,29))
    assert calls[0][1]['requests_per_hour'] == 5400 and calls[0][1]['max_retries'] == 0


def test_started_receipt_cannot_be_retried(tmp_path,monkeypatch):
    p = bare_provider(tmp_path,monkeypatch)
    item = dict(query=dict(dataset='TaiwanStockPriceTick',data_id='2330',start_date='2023-12-29'),status='started')
    write(p.directory/'receipts/2330-2023-12-29.json',item)
    assert p._raw(('2330','2023-12-29'))['status'] == 'started'
    assert not (p.directory/'attempts').exists()


def test_offline_missing_does_not_write_negative_evidence(tmp_path,monkeypatch):
    p = bare_provider(tmp_path,monkeypatch,online=False)
    assert p._raw(('2330','2023-12-29'))['status'] == 'not_requested'
    assert not (p.directory/'receipts').exists()


def test_query_tampering_rejected(tmp_path,monkeypatch):
    p = bare_provider(tmp_path,monkeypatch)
    write(p.directory/'receipts/2330-2023-12-29.json',dict(status='empty',query=dict(dataset='TaiwanStockPriceTick',data_id='2303',start_date='2023-12-29')))
    with pytest.raises(ValueError, match='query changed'):
        p._raw(('2330','2023-12-29'))


def test_orphaned_request_reservation_is_not_resent(tmp_path,monkeypatch):
    p = bare_provider(tmp_path,monkeypatch)
    item = dict(query=dict(dataset='TaiwanStockPriceTick',data_id='2330',start_date='2023-12-29'),status='started')
    path = p.directory/'attempts/2330-2023-12-29.json'
    write(path,item)
    before = path.read_bytes()
    assert p._raw(('2330','2023-12-29'))['status'] == 'orphaned_started_attempt'
    assert path.read_bytes() == before
    assert not (p.directory/'receipts').exists()


def test_publication_rechecks_every_used_source(tmp_path,monkeypatch):
    p = bare_provider(tmp_path,monkeypatch)
    source = tmp_path/'source.json';source.write_text('{}')
    p._mark(source)
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='hash changed'):
        p.snapshot(tmp_path/'publication')
