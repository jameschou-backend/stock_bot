"""Bounded advisory locks for local workers (macOS/Linux)."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import time


@contextmanager
def file_lock(path: Path, timeout: float = 60):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        started = time.monotonic()
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                left = timeout - (time.monotonic() - started)
                if left <= 0:
                    raise TimeoutError("已有工作處理相同資料，請稍後查看工作狀態")
                time.sleep(min(0.05, left))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
