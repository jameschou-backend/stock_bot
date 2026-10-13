"""Serialize shared evidence reads/acquisition while account arithmetic is isolated."""
from contextlib import contextmanager
from pathlib import Path
import threading

from app.file_lock import file_lock
from skills.poc_followup_reuse import ReusedFollowupData

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT/'.cache/poc-followup-20261014/.shared-acquisition.lock'


class SharedEvidenceGuard:
    def __init__(self, path=LOCK):
        self.path = Path(path)
        self.thread_lock = threading.RLock()
        self.depth = 0

    @contextmanager
    def hold(self):
        with self.thread_lock:
            if self.depth:
                self.depth += 1
                try:
                    yield
                finally:
                    self.depth -= 1
            else:
                with file_lock(self.path, timeout=600):
                    self.depth = 1
                    try:
                        yield
                    finally:
                        self.depth = 0


class SynchronizedFollowupData(ReusedFollowupData):
    def __init__(self, *, online=False, donor_manifest):
        self.shared_evidence = SharedEvidenceGuard()
        with self.shared_evidence.hold():
            super().__init__(online=online, donor_manifest=donor_manifest)
            original = self.board_ticks.get
            def board(*args, **kwargs):
                with self.shared_evidence.hold():
                    return original(*args, **kwargs)
            self.board_ticks.get = board
            self._bind(Path(__file__))

    def profile(self, *args, **kwargs):
        with self.shared_evidence.hold():
            return super().profile(*args, **kwargs)

    def finmind(self, *args, **kwargs):
        with self.shared_evidence.hold():
            return super().finmind(*args, **kwargs)

    def get_odd(self, *args, **kwargs):
        with self.shared_evidence.hold():
            return super().get_odd(*args, **kwargs)

    def profile_snapshot(self, *args, **kwargs):
        with self.shared_evidence.hold():
            return super().profile_snapshot(*args, **kwargs)
