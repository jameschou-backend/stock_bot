"""Exercise the actual frozen snapshot block with a nested donor source."""
from pathlib import Path
import hashlib
import textwrap
import pytest

from skills.poc_followup_snapshots import rewrite, seal_snapshot, install

ROOT = Path(__file__).resolve().parents[1]


def snapshot_block():
    code = rewrite((ROOT/'scripts/research_strategy_account_comparison.py').read_text())
    start = code.index(' source_snapshots={}\n')
    return textwrap.dedent(code[start:code.index(' cases={}\n', start)])


def test_nested_donor_sources_have_bounded_snapshot_paths_and_identical_bytes(tmp_path):
    source = '/'.join(['donor-'+('x'*80)]*8+['docs', 'volume_profile_account.md'])
    original = tmp_path/source
    original.parent.mkdir(parents=True)
    original.write_bytes(b'sealed source\n')
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    output = tmp_path/('o'*245)
    marked = []
    namespace = dict(ROOT=tmp_path, refs={source:digest}, output=output,
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(), mark=marked.append,
        _poc_seal_snapshot=seal_snapshot)
    exec(snapshot_block(), namespace)
    metadata = namespace['source_snapshots'][source]
    path = tmp_path/metadata['path']
    assert len(str(path.relative_to(output))) < 100
    assert path.read_bytes() == original.read_bytes()
    assert metadata['sha256'] == digest
    assert marked == [path]


def test_snapshot_deduplicates_identical_bytes_without_overwriting(tmp_path):
    for name in ('one.md', 'two.md'):
        (tmp_path/name).write_bytes(b'source')
    digest = hashlib.sha256(b'source').hexdigest()
    one = seal_snapshot(tmp_path, tmp_path/'output', 'one.md', digest)
    stamp = one.stat().st_mtime_ns
    two = seal_snapshot(tmp_path, tmp_path/'output', 'two.md', digest)
    assert one == two and two.stat().st_mtime_ns == stamp
    one.write_bytes(b'tampered snapshot')
    with pytest.raises(ValueError, match='Existing source snapshot differs'):
        seal_snapshot(tmp_path, tmp_path/'output', 'one.md', digest)
    assert one.read_bytes() == b'tampered snapshot'


def test_changed_source_fails_before_writing_snapshot(tmp_path):
    (tmp_path/'source.md').write_bytes(b'changed source')
    with pytest.raises(ValueError, match='Source changed'):
        seal_snapshot(tmp_path, tmp_path/'output', 'source.md', hashlib.sha256(b'original').hexdigest())
    assert not (tmp_path/'output').exists()


def test_installer_preserves_existing_audit_binding_and_financial_body():
    import ast
    from types import SimpleNamespace
    from scripts.research_all_entry_accounts import FROZEN
    source = FROZEN.read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_run')
    code = ast.get_source_segment(source, function)+'\n'
    code = code.replace('from skills.poc_range_audit import audit_range_execution',
                        'audit_range_execution = _all_entry_indexed_audit')
    runner = SimpleNamespace(END='2026-10-02', _all_entry_indexed_audit=object())
    exec(compile(code, str(FROZEN), 'exec'), runner.__dict__)
    install(runner)
    expected = dict(runner.__dict__)
    exec(compile(rewrite(code), str(FROZEN), 'exec'), expected)
    assert runner._run.__code__ == expected['_run'].__code__
    assert code.split(' cases={}\n', 1)[1] == rewrite(code).split(' cases={}\n', 1)[1]
    with pytest.raises(ValueError, match='Unexpected private runner transformation'):
        install(runner)
