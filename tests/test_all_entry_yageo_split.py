from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from skills.all_entry_yageo_split import DOCUMENT, KEY, DAY, install_split, load_split
from skills.million_replay import Replay
from skills.replay_corporate_actions import CorporateActions

ROOT = Path(__file__).resolve().parents[1]


def terms():
    if not (ROOT / '.cache/all-entry-accounts-20261011/corporate-evidence/2327/taifex-2025-split.pdf').exists():
        pytest.skip('Archived primary split evidence unavailable')
    return load_split(ROOT, ROOT / DOCUMENT)


def test_official_event_and_notice_install_without_mutating_frozen_terms():
    action, halt, refs = terms()
    assert action['multiplier'] == 4 and halt['announcement_date'] < halt['start'] < halt['end']
    assert len(refs) == 5
    source = dict(verified_halts=[], source_sha256={})
    runner = SimpleNamespace(read=lambda _: source)
    provider = SimpleNamespace(corporate_overrides={})
    install_split(runner, provider, ROOT / DOCUMENT, ROOT)
    assert provider.corporate_overrides == {KEY: action}
    assert runner.read(ROOT / 'docs/stock_universe_2019_corporate_terms.json')['verified_halts'] == [halt]
    assert source == dict(verified_halts=[], source_sha256={})
    with pytest.raises(ValueError, match='conflicts'): install_split(runner, provider, ROOT / DOCUMENT, ROOT)


def test_real_corporate_parser_and_account_split_preserve_reference_value(tmp_path):
    action, _, _ = terms()
    events = pd.DataFrame([dict(stock_id='2327', event_date=DAY, source='par_value_change', payload_json='{}')])
    pd.DataFrame().to_parquet(tmp_path / '2327.parquet', index=False)
    corp = CorporateActions(events, tmp_path, offline=True, overrides={KEY: action})
    parsed = corp.on_date('2327', DAY)
    assert len(parsed) == 1 and parsed[0]['multiplier'] == 4
    obj = object.__new__(Replay)
    obj.corporate = corp; obj.holdings = {'2327': dict(qty=336, event_id='held')}
    obj.marks = {'2327': dict(price=546.)}; obj.actions = []
    obj.receivables = [dict(stock_id='2327', amount=123., kind='cash', pay_date=None)]
    before = deepcopy(obj.receivables)
    assert obj.corporate_day(pd.Timestamp(DAY)) == 0
    assert obj.holdings['2327']['qty'] == 1344 and obj.marks['2327']['price'] == 136.5
    assert 336 * 546 == 1344 * 136.5 and obj.receivables == before
    assert obj.actions[0]['entitled_qty'] == 336 and obj.actions[0]['qty_after'] == 1344
