from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pandas as pd
import pytest

from skills.frozen_dividend_copy import ensure_dividend_copy
from skills.replay_market_feeds import ReplayDataUnavailable


def test_offline_replay_never_creates_a_missing_source(tmp_path):
    path = tmp_path/'absent'/'6409.parquet'
    with pytest.raises(ReplayDataUnavailable, match='before offline replay'):
        ensure_dividend_copy(pd.DataFrame({'cash': [21.]}), path, prepare=False)
    assert not path.parent.exists()


def test_existing_dividend_copy_is_read_only_and_mismatch_is_blocked(tmp_path):
    path = tmp_path/'6409.parquet'
    frame = pd.DataFrame({'cash': [21.]})
    ensure_dividend_copy(frame, path, prepare=True)
    before = path.read_bytes(), path.stat().st_mtime_ns
    ensure_dividend_copy(frame, path, prepare=False)
    with pytest.raises(ValueError, match='copy differs'):
        ensure_dividend_copy(pd.DataFrame({'cash': [22.]}), path, prepare=True)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_competing_preparations_only_publish_complete_parquet(tmp_path, monkeypatch):
    path = tmp_path/'6409.parquet'
    frame = pd.DataFrame({'cash': [21.]})
    ready = Barrier(2)
    original = pd.DataFrame.to_parquet

    def finish_together(self, staged, **kwargs):
        result = original(self, staged, **kwargs)
        assert not path.exists()  # Neither staged file is visible as the source.
        ready.wait(timeout=10)
        return result

    monkeypatch.setattr(pd.DataFrame, 'to_parquet', finish_together)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(ensure_dividend_copy, frame, path, prepare=True) for _ in range(2)]
        for future in futures:
            future.result(timeout=15)
    assert pd.read_parquet(path).equals(frame)
    assert list(tmp_path.iterdir()) == [path]
