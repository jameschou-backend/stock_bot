from concurrent.futures import ThreadPoolExecutor
import threading
import time

from skills.poc_followup_shared_acquisition import SharedEvidenceGuard


def test_shared_evidence_guard_is_exclusive_across_instances_and_reentrant(tmp_path):
    path = tmp_path/'acquire.lock'
    active, overlap, completed = 0, [], []
    check = threading.Lock()
    def worker(index):
        nonlocal active
        guard = SharedEvidenceGuard(path)
        with guard.hold():
            with guard.hold():
                with check:
                    active += 1
                    if active != 1: overlap.append(index)
                time.sleep(.03)
                with check:
                    active -= 1
                    completed.append(index)
        assert guard.depth == 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(worker, range(3)))
    assert not overlap and sorted(completed) == [0,1,2]
