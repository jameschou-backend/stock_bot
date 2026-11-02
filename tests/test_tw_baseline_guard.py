import json
from pathlib import Path

import pytest

from skills.tw_baseline_guard import capture, verify


def test_existing_sources_are_protected_but_new_us_files_are_allowed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "prices.parquet").write_bytes(b"original prices")
    output = tmp_path / "baseline"
    capture(repo, output, names=["prices.parquet"])
    (repo / "new-us-data").write_bytes(b"US")
    assert verify(repo, output)["unchanged"]
    (repo / "prices.parquet").write_bytes(b"changed prices")
    assert not verify(repo, output)["unchanged"]
    (repo / "prices.parquet").unlink()
    assert not verify(repo, output)["unchanged"]


def test_inventory_tampering_and_overwrite_are_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a").write_text("a")
    out = tmp_path / "baseline"
    capture(repo, out, names=["a"])
    with pytest.raises(ValueError, match="already exists"):
        capture(repo, out, names=["a"])
    (out / "files.jsonl").write_text("")
    with pytest.raises(ValueError, match="mismatch"):
        verify(repo, out)


def test_replaced_parent_link_cannot_escape_guard(tmp_path):
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "data/a").write_text("a")
    out = tmp_path / "baseline"
    capture(repo, out, names=["data"])
    (repo / "data").rename(repo / "moved")
    (repo / "data").symlink_to(repo / "moved", target_is_directory=True)
    assert not verify(repo, out)["unchanged"]


def test_missing_roots_do_not_produce_completion_receipt(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    out = tmp_path / "baseline"
    with pytest.raises(ValueError, match="Missing"):
        capture(repo, out, names=["missing"])
    assert not (out / "receipt.json").exists()
