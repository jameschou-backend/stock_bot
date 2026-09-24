import hashlib
import json
from pathlib import Path

import pytest

from scripts import prepare_board_only_supplement as subject


def test_supplement_stops_before_network_at_budget_limit(tmp_path, monkeypatch):
    source, output = tmp_path / 'old', tmp_path / 'new'
    source.mkdir()
    (source / 'manifest.json').write_text('{}')
    feeds = output / 'inputs' / 'execution-feeds'
    feeds.mkdir(parents=True)
    (feeds / 'index.json').write_text(json.dumps({'entries': {}, 'request_counters': {}}))
    (output / 'source-ledger.json').write_text(json.dumps(dict(
        parent_manifest_sha256=hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest(),
        preparation_code_sha256=hashlib.sha256(Path(subject.__file__).read_bytes()).hexdigest(),
        finmind_fetch_attempts=subject.MAX_FETCHES, requests=[])))
    monkeypatch.setattr(subject, 'SOURCE', source)
    monkeypatch.setattr(subject, 'OUTPUT', output)

    class NeverFetch:
        def __init__(self, *args, **kwargs):
            pass

        def get_limits(self, sid):
            pytest.fail('Exceeded the preregistered API budget')

    monkeypatch.setattr(subject, 'ReplayMarketFeeds', NeverFetch)
    with pytest.raises(ValueError, match='budget exhausted'):
        subject.prepare(['3687'])


@pytest.mark.parametrize('stock', ['TAIEX', '2330;echo', '２３３０', 'abc0'])
def test_supplement_rejects_non_stock_ids(stock):
    with pytest.raises(ValueError, match='four-digit'):
        subject.prepare([stock])
