"""Prepare dividend copies atomically; official replays may only read them."""
import os
from pathlib import Path
import tempfile

import pandas as pd

from skills.replay_market_feeds import ReplayDataUnavailable


def ensure_dividend_copy(frame, path, *, prepare):
    path = Path(path)
    if not path.exists():
        if not prepare:
            raise ReplayDataUnavailable('Prepare dividend execution copy before offline replay: '+str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.parquet', delete=False) as temporary:
            staged = Path(temporary.name)
        try:
            frame.to_parquet(staged, index=False)
            try:
                # Publish only a closed, complete file; never replace a competing
                # writer's copy. Compare that copy below before accepting it.
                os.link(staged, path)
            except FileExistsError:
                pass
        finally:
            staged.unlink(missing_ok=True)
    if not frame.equals(pd.read_parquet(path)):
        raise ValueError('Dividend execution copy differs: '+str(path))
