"""Bound snapshot paths without changing financial code or source identities."""
import ast
import hashlib
from pathlib import Path

POLICY = 'content_addressed_source_snapshots_v1'
OLD = """   original=ROOT/source
   snapshot=output/'source-snapshots'/source
   snapshot.parent.mkdir(parents=True,exist_ok=True)
   snapshot.write_bytes(original.read_bytes())
   if sha(snapshot)!=refs[source]:raise ValueError('Source changed before execution snapshot: '+source)
"""
NEW = """   snapshot=_poc_seal_snapshot(ROOT,output,source,refs[source])
"""


def seal_snapshot(root, output, source, expected):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.relative_to(root)
    original = (root/source).resolve()
    original.relative_to(root)
    raw = original.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Source changed before execution snapshot: '+source)
    path = output/'source-snapshots'/(expected+'.snapshot')
    path.resolve().relative_to(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Existing source snapshot differs: '+str(path))
    else:
        with path.open('xb') as stream:
            stream.write(raw)
    return path


def rewrite(code):
    if code.count(OLD) != 1:
        raise ValueError('Frozen snapshot creation seam changed')
    return code.replace(OLD, NEW)


def install(runner):
    from scripts.research_all_entry_accounts import FROZEN, FROZEN_SHA
    if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != FROZEN_SHA:
        raise ValueError('Frozen runner changed before snapshot adapter')
    if runner._run.__globals__ is not runner.__dict__:
        raise ValueError('Snapshot adapter requires a private runner namespace')
    source = FROZEN.read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_run')
    code = ast.get_source_segment(source, function)+'\n'
    old = 'from skills.poc_range_audit import audit_range_execution'
    if code.count(old) != 1 or not hasattr(runner, '_all_entry_indexed_audit'):
        raise ValueError('Independent audit binding is missing')
    code = code.replace(old, 'audit_range_execution = _all_entry_indexed_audit')
    # The existing adapter only changes this audit binding. Reject any other
    # _run transformation rather than accidentally replacing financial logic.
    check = dict(runner.__dict__)
    exec(compile(code, str(FROZEN), 'exec'), check)
    if check['_run'].__code__ != runner._run.__code__:
        raise ValueError('Unexpected private runner transformation')
    runner._poc_seal_snapshot = seal_snapshot
    exec(compile(rewrite(code), str(FROZEN), 'exec'), runner.__dict__)
