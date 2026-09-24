import hashlib
import json
import os

import pytest

from skills import backtest_case_cache as cache
from skills.backtest_case_cache import CaseStore, content_digest, file_identities


IDENTITY = {"source": {"prices.csv": "source-hash"}, "code": {"replay.py": "code-hash"}}
CONFIG = {"top_n": 5, "execution": "offline"}
RESULT = {"status": "blocked", "reason": "missing input", "research_only": True, "live_qualified": False}


def test_content_digest_is_canonical_and_sensitive():
    assert content_digest({"b": [2, 3], "a": "台股"}) == content_digest({"a": "台股", "b": [2, 3]})
    assert content_digest([2, 3]) != content_digest([3, 2])
    assert content_digest({"x": 1}) != content_digest({"x": True})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), {1: "one"}, (1, 2), {"nested": [float("nan")]}])
def test_nonportable_json_is_rejected(value, tmp_path):
    with pytest.raises(ValueError, match="portable JSON"):
        content_digest(value)
    store = CaseStore(tmp_path, IDENTITY)
    with pytest.raises(ValueError, match="portable JSON"):
        store.save("case", CONFIG, value)
    assert not (tmp_path / "cases").exists()


def test_file_identities_are_relative_and_hash_current_bytes(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    nested = root / "source"
    nested.mkdir()
    source = nested / "prices.csv"
    source.write_bytes(b"price=10")
    monkeypatch.chdir(tmp_path)
    before = source.stat()
    initial = file_identities(["source/prices.csv"], root)
    assert initial == {"source/prices.csv": hashlib.sha256(b"price=10").hexdigest()}
    assert initial == file_identities([source], root)
    source.write_bytes(b"price=99")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert source.stat().st_size == before.st_size
    changed = file_identities([source], root)
    assert changed != initial
    first_store = CaseStore(tmp_path / "cache", {"files": initial})
    first_store.save("case", CONFIG, RESULT)
    with pytest.raises(ValueError, match="identity mismatch"):
        CaseStore(tmp_path / "cache", {"files": changed})


@pytest.mark.parametrize("relative", ["../outside", "source/../../outside", "source\\prices.csv", "."])
def test_source_traversal_and_nonfiles_rejected(tmp_path, relative):
    with pytest.raises((ValueError, OSError)):
        file_identities([relative], tmp_path)


def test_outside_source_and_symlinks_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("data")
    with pytest.raises(ValueError, match="outside root"):
        file_identities([outside], root)
    (root / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        file_identities(["link"], root)
    (root / "nested").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        file_identities(["nested/outside"], root)


def test_blocked_case_roundtrip_resumes_and_result_is_detached(tmp_path):
    store = CaseStore(tmp_path, IDENTITY)
    assert store.load("case", CONFIG) is None
    receipt = store.save("case", CONFIG, RESULT)
    assert receipt["complete"] is True
    assert receipt["identity_digest"] == content_digest(IDENTITY)
    restored = CaseStore(tmp_path, IDENTITY).load("case", CONFIG)
    assert restored == RESULT
    restored["status"] = "changed"
    assert store.load("case", CONFIG) == RESULT
    assert store.save("case", CONFIG, RESULT) == receipt


def test_mismatched_config_misses_and_cannot_overwrite(tmp_path):
    store = CaseStore(tmp_path, IDENTITY)
    store.save("case", CONFIG, RESULT)
    with pytest.warns(RuntimeWarning, match="configuration"):
        assert store.load("case", {**CONFIG, "top_n": 10}) is None
    with pytest.raises(ValueError, match="conflicting"):
        store.save("case", {**CONFIG, "top_n": 10}, RESULT)
    with pytest.raises(ValueError, match="conflicting"):
        store.save("case", CONFIG, {**RESULT, "status": "complete"})


@pytest.mark.parametrize("corruption", [b'{"status":"other"}', b'{"status":', b'{"profit":NaN}', b'{"status":1,"status":2}', b'null'])
def test_tampered_or_nonportable_result_never_resumes(tmp_path, corruption):
    store = CaseStore(tmp_path, IDENTITY)
    store.save("case", CONFIG, RESULT)
    (tmp_path / "cases/case/result.json").write_bytes(corruption)
    with pytest.warns(RuntimeWarning):
        assert store.load("case", CONFIG) is None


@pytest.mark.parametrize("key,value", [
    ("identity_digest", "wrong"), ("config_digest", "wrong"),
    ("result_sha256", "wrong"), ("schema_version", 2),
    ("schema_version", True), ("complete", False), ("complete", 1),
    ("case_name", "another"), ("receipt_digest", "wrong"),
])
def test_receipt_requires_all_identity_digests_and_schema_flags(tmp_path, key, value):
    store = CaseStore(tmp_path, IDENTITY)
    receipt = store.save("case", CONFIG, RESULT)
    receipt[key] = value
    # Even a recomputed detached hash cannot bind a receipt to the wrong run.
    if key != "receipt_digest":
        receipt["receipt_digest"] = content_digest({k: v for k, v in receipt.items() if k != "receipt_digest"})
    (tmp_path / "cases/case/receipt.json").write_text(json.dumps(receipt))
    with pytest.warns(RuntimeWarning):
        assert store.load("case", CONFIG) is None


def test_receipt_missing_field_and_corrupt_receipt_never_resume(tmp_path):
    store = CaseStore(tmp_path, IDENTITY)
    receipt = store.save("case", CONFIG, RESULT)
    path = tmp_path / "cases/case/receipt.json"
    del receipt["receipt_digest"]
    for data in (json.dumps(receipt), "{partial"):
        path.write_text(data)
        with pytest.warns(RuntimeWarning):
            assert store.load("case", CONFIG) is None
        with pytest.raises(ValueError, match="conflicting"):
            store.save("case", CONFIG, RESULT)


def test_interrupted_receipt_write_is_not_complete_and_can_retry(tmp_path, monkeypatch):
    store = CaseStore(tmp_path, IDENTITY)
    original = os.replace

    def interrupt_receipt(source, destination):
        if destination.name == "receipt.json":
            raise OSError("simulated interruption")
        return original(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(cache.os, "replace", interrupt_receipt)
        with pytest.raises(OSError, match="simulated interruption"):
            store.save("case", CONFIG, RESULT)
    assert (tmp_path / "cases/case/result.json").exists()
    assert not list(tmp_path.rglob("*.tmp"))
    with pytest.warns(RuntimeWarning, match="incomplete"):
        assert store.load("case", CONFIG) is None
    with pytest.raises(ValueError, match="conflicting"):
        store.save("case", CONFIG, {"different": True})
    store.save("case", CONFIG, RESULT)
    assert store.load("case", CONFIG) == RESULT


def test_interrupted_result_write_has_no_completion_receipt(tmp_path, monkeypatch):
    store = CaseStore(tmp_path, IDENTITY)

    def interrupt_result(*args):
        raise OSError("simulated interruption")

    monkeypatch.setattr(cache.os, "replace", interrupt_result)
    with pytest.raises(OSError):
        store.save("case", CONFIG, RESULT)
    assert store.load("case", CONFIG) is None
    assert not list(tmp_path.rglob("*.tmp"))


def test_missing_result_cannot_be_implicitly_repaired(tmp_path):
    store = CaseStore(tmp_path, IDENTITY)
    store.save("case", CONFIG, RESULT)
    (tmp_path / "cases/case/result.json").unlink()
    with pytest.warns(RuntimeWarning):
        assert store.load("case", CONFIG) is None
    with pytest.raises(ValueError, match="without its result"):
        store.save("case", CONFIG, RESULT)


def test_immutable_identity_is_checked_on_every_operation(tmp_path):
    identity = {"version": 1}
    store = CaseStore(tmp_path, identity)
    identity["version"] = 2
    assert store.identity_digest == content_digest({"version": 1})
    store.save("case", CONFIG, RESULT)
    path = tmp_path / "identity.json"
    path.write_text("{broken")
    for action in (
        lambda: CaseStore(tmp_path, {"version": 1}),
        lambda: store.load("case", CONFIG),
        lambda: store.save("other", CONFIG, RESULT),
    ):
        with pytest.raises(ValueError, match="identity.*corrupt"):
            action()
    path.unlink()
    with pytest.raises(ValueError, match="identity is missing"):
        CaseStore(tmp_path, {"version": 1})


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "/absolute", "a/b", "a\\b", "abc..def", "a" * 129, "CON", "nul.txt"])
def test_case_names_cannot_traverse_paths(tmp_path, name):
    store = CaseStore(tmp_path, IDENTITY)
    with pytest.raises(ValueError, match="Case name"):
        store.load(name, CONFIG)
    with pytest.raises(ValueError, match="Case name"):
        store.save(name, CONFIG, RESULT)


def test_case_symlinks_and_store_symlinks_are_rejected(tmp_path):
    directory = tmp_path / "cache"
    store = CaseStore(directory, IDENTITY)
    outside = tmp_path / "outside"
    outside.mkdir()
    (directory / "cases").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        store.save("case", CONFIG, RESULT)
    with pytest.raises(ValueError, match="symbolic link"):
        store.load("case", CONFIG)
    link = tmp_path / "link"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        CaseStore(link, IDENTITY)
    assert not list(outside.iterdir())


def test_null_result_is_reserved_for_misses(tmp_path):
    store = CaseStore(tmp_path, IDENTITY)
    with pytest.raises(ValueError, match="Null is reserved"):
        store.save("case", CONFIG, None)
