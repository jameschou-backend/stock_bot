#!/usr/bin/env python3
"""Seed an independent exit-research cache using only verified local evidence.

Large sealed matrices remain read-only references. Execution/dividend files are
independent copies: never symlinks or hardlinks. This module imports no provider,
database, configuration or replay engine and makes no API requests.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
INPUT = Path('.cache/million-replay-inputs')
SIGNALS = Path('.cache/million-replay-signals')
REPORT = Path('.cache/million-replay')
DESTINATION = Path('.cache/exit-research-inputs')
SPEC = Path('docs/prereg_million_replay_20260910.md')
OVERRIDES = Path('docs/million_replay_corporate_overrides_20260910.json')
MANIFEST_NAMES = {'manifest.json', 'inputs.json', 'signal-inputs.json'}
CHUNK_SIZE = 4 * 1024 * 1024
REFERENCE_PATHS = {'quotes': INPUT/'quotes.parquet', 'calendar': INPUT/'calendar.parquet',
    'companies': INPUT/'companies.parquet', 'events': INPUT/'events.parquet',
    'close_official': SIGNALS/'close-official.parquet', 'signals': SIGNALS/'signals.json',
    'corporate_overrides': OVERRIDES}


class InputChanged(ValueError):
    """Missing, changed or ambiguous frozen evidence; never silently reseed."""


def _json(path):
    def invalid(value):
        raise InputChanged('Non-finite JSON value: ' + str(path))
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InputChanged('Duplicate JSON key: ' + str(path))
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(), parse_constant=invalid, object_pairs_hook=unique)
    except (OSError, json.JSONDecodeError) as exc:
        raise InputChanged('Cannot read manifest: ' + str(path)) from exc


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _relative(name):
    value = Path(name)
    if value.is_absolute() or '..' in value.parts or not value.parts:
        raise InputChanged('Unsafe source path: ' + str(name))
    return value


def _file(root, name):
    path = root / _relative(name)
    current = path
    while current != root:
        if current.is_symlink():
            raise InputChanged('Symlink is not frozen evidence: ' + str(path))
        current = current.parent
    if not path.is_file():
        raise InputChanged('Frozen file missing: ' + str(path))
    return path


def _stamp(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b''):
            digest.update(chunk)
    return digest.hexdigest()


class HashAudit:
    """One streaming SHA per distinct path per verification, not per case."""
    def __init__(self, root):
        self.root, self.hashes, self.stamps = root, {}, {}

    def check(self, name, expected=None):
        name = str(_relative(name))
        path = _file(self.root, name)
        if expected is not None and (not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected)):
            raise InputChanged('Invalid SHA256: ' + name)
        if name not in self.hashes:
            before = _stamp(path)
            self.hashes[name] = _sha(path)
            self.stamps[name] = _stamp(path)
            if before != self.stamps[name]:
                raise InputChanged('Source changed while hashing: ' + name)
        if expected is not None and self.hashes[name] != expected:
            raise InputChanged('Frozen source hash changed: ' + name)
        return self.hashes[name]

    def stable(self):
        # Re-stat at the boundary catches concurrent ordinary writes without
        # streaming the same large input a second time. Each new verify call
        # always streams fresh hashes; there is no persistent stat-only cache.
        for name, before in self.stamps.items():
            if _stamp(_file(self.root, name)) != before:
                raise InputChanged('Source changed during preparation: ' + name)


def _collect_parent(root):
    audit, manifests = HashAudit(root), {}

    def visit(name, expected=None):
        name = str(_relative(name))
        audit.check(name, expected)
        if name in manifests:
            return manifests[name]
        data = _json(root / name)
        if not isinstance(data, dict) or data.get('schema') != 1:
            raise InputChanged('Unsupported parent manifest: ' + name)
        manifests[name] = data
        for field in ('files_sha256', 'code_sha256', 'source_files_sha256', 'parent_files_sha256'):
            values = data.get(field, {})
            if not isinstance(values, dict):
                raise InputChanged('Invalid parent hash map: ' + name)
            for child, digest in values.items():
                child = _relative(child)
                path = str(Path(name).parent / child) if field == 'files_sha256' else str(child)
                audit.check(path, digest)
                # Follow actual manifest references, not arbitrary raw JSON
                # payloads or embedded historical provenance descriptions.
                if child.name in MANIFEST_NAMES:
                    visit(path, digest)
        if 'spec_sha256' in data:
            audit.check(data['spec_path'], data['spec_sha256'])
        for field, filename in (('plan_sha256', 'plan.json'), ('attempts_sha256', 'attempts.json')):
            if field in data:
                audit.check(Path(name).parent / filename, data[field])
        if 'parent_price_manifest_sha256' in data:
            visit('.cache/event-group-research/signal-inputs.json', data['parent_price_manifest_sha256'])
        return data

    report = visit(REPORT / 'manifest.json')
    required_sources = {str(INPUT/'manifest.json'), str(SIGNALS/'manifest.json'), str(SPEC), str(OVERRIDES)}
    if (report.get('offline_identical') is not True or report.get('live_qualified') is not False
            or set(report.get('source_files_sha256', {})) != required_sources
            or set(report.get('files_sha256', {})) != {'report.json', 'summary.json', 'prepared-accounts.json'}):
        raise InputChanged('Incomplete or unsealed parent replay')
    raw, signals = manifests[str(INPUT/'manifest.json')], manifests[str(SIGNALS/'manifest.json')]
    if not {'quotes.parquet','calendar.parquet','companies.parquet','events.parquet'} <= set(raw.get('files_sha256', {})):
        raise InputChanged('Parent raw inputs are incomplete')
    if not {'close-official.parquet','signals.json'} <= set(signals.get('files_sha256', {})):
        raise InputChanged('Parent signal inputs are incomplete')
    feed = report.get('execution_feeds', {})
    index_name = str(INPUT / 'execution-feeds/index.json')
    audit.check(index_name, feed.get('manifest_sha256'))
    state = _json(root / index_name)
    if (state.get('schema') != 1 or not isinstance(state.get('files_sha256'), dict)
            or any(feed.get(key) != value for key, value in state.items())):
        raise InputChanged('Parent execution index differs from sealed replay')
    audit.check('skills/replay_market_feeds.py', state['parser_sha256'])
    for name, digest in state['files_sha256'].items():
        audit.check(INPUT / 'execution-feeds' / _relative(name), digest)
    for entry in state['entries'].values():
        if not {entry['raw_file'], entry['rows_file']} <= set(state['files_sha256']):
            raise InputChanged('Unregistered execution entry')
    dividend_hashes = report.get('corporate_sources', {}).get('files_sha256')
    if not isinstance(dividend_hashes, dict) or not dividend_hashes:
        raise InputChanged('Parent dividend source inventory is missing')
    for name, digest in dividend_hashes.items():
        audit.check(INPUT / 'dividends' / _relative(name), digest)
    # Preparation may have downloaded unused policies (e.g. a rejected entry).
    # Preserve those existing files separately from the sealed report's set.
    dividends = {}
    for path in sorted((root / INPUT / 'dividends').iterdir()):
        if not re.fullmatch(r'\d{4}\.parquet', path.name):
            raise InputChanged('Unexpected dividend source file: ' + path.name)
        dividends[path.name] = audit.check(path.relative_to(root))
    audit.stable()
    return audit, manifests, state, dividends


def _destination(root, destination):
    target = Path(destination) if destination is not None else root / DESTINATION
    if not target.is_absolute():
        target = root / target
    target = target.absolute()
    # Fixed ownership boundary: an output cannot alias or contain any old cache.
    if not target.is_relative_to(root) or '..' in target.parts:
        raise InputChanged('Destination must be inside the project')
    for protected in (root / INPUT, root / SIGNALS, root / REPORT):
        if target == protected or target.is_relative_to(protected) or protected.is_relative_to(target):
            raise InputChanged('Destination overlaps immutable parent evidence')
    current = target
    while current != root:
        if current.is_symlink():
            raise InputChanged('Destination must not use symlinks')
        current = current.parent
    return target


def _verify_child(target, data, parent_audit):
    child = HashAudit(target)
    for name, digest in data['seed_files_sha256'].items():
        child.check(name, digest)
        parent_name = data['clone_parent_paths'].get(name)
        if parent_name and os.path.samefile(target/name, parent_audit.root/parent_name):
            raise InputChanged('Child evidence aliases the parent: ' + name)
    child.check('execution-feeds/index.json')
    if os.path.samefile(target/'execution-feeds/index.json', parent_audit.root/INPUT/'execution-feeds/index.json'):
        raise InputChanged('Child execution index aliases the parent')
    state = _json(_file(target, 'execution-feeds/index.json'))
    seed = data['seed_execution_index']
    if (state.get('schema') != seed['schema'] or state.get('parser_sha256') != seed['parser_sha256']
            or state.get('limitations') != seed['limitations']):
        raise InputChanged('Child execution parser or limitations changed')
    for key in ('entries', 'files_sha256'):
        if not isinstance(state.get(key), dict) or any(state[key].get(k) != v for k,v in seed[key].items()):
            raise InputChanged('Inherited execution evidence changed: ' + key)
    for key, value in seed['request_counters'].items():
        current = state.get('request_counters', {}).get(key)
        if type(current) is not int or current < value:
            raise InputChanged('Child request counters cannot reset inherited usage')
    for name, digest in state['files_sha256'].items():
        child.check(Path('execution-feeds') / _relative(name), digest)
    for entry in state['entries'].values():
        if not {entry['raw_file'], entry['rows_file']} <= set(state['files_sha256']):
            raise InputChanged('Child execution entry has unregistered files')
    child.stable()
    parent_audit.stable()


def verify(destination=None, *, root=ROOT):
    """Verify immutable parent/seed content while allowing child append only."""
    root = Path(root).resolve()
    target = _destination(root, destination)
    data = _json(_file(target, 'manifest.json'))
    if data.get('schema') != 1 or data.get('kind') != 'exit-research-inputs':
        raise InputChanged('Unsupported exit-input manifest')
    if data.get('preparation_code_sha256') != _sha(Path(__file__)):
        raise InputChanged('Exit input preparation code changed')
    audit, manifests, state, dividends = _collect_parent(root)
    if audit.hashes != data['parent_files_sha256'] or dividends != data['seed_dividend_hashes']:
        raise InputChanged('Parent chain or source inventory changed; refusing reseed')
    expected_clones = {str(Path('execution-feeds')/name): str(INPUT/'execution-feeds'/name)
                       for name in state['files_sha256']}
    expected_clones.update({str(Path('dividends')/name): str(INPUT/'dividends'/name) for name in dividends})
    expected_clones.update({str(Path('parent-manifests')/name): name for name in
        [*manifests, str(INPUT/'execution-feeds/index.json'), str(SPEC), str(OVERRIDES)]})
    if (data['seed_execution_index'] != state or data['clone_parent_paths'] != expected_clones
            or data['seed_files_sha256'] != {name: audit.hashes[parent] for name,parent in expected_clones.items()}):
        raise InputChanged('Inherited evidence manifest changed')
    expected_references = {logical:{'path':str(name),'sha256':audit.hashes[str(name)]}
                           for logical,name in REFERENCE_PATHS.items()}
    if data['references'] != expected_references:
        raise InputChanged('Read-only input reference changed')
    _verify_child(target, data, audit)
    return data


def prepare(*, root=ROOT, destination=None):
    """Clone once; subsequent calls verify without touching existing child data."""
    root = Path(root).resolve()
    target = _destination(root, destination)
    if target.exists():
        return verify(target, root=root)
    audit, manifests, state, dividends = _collect_parent(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.'+target.name+'-', dir=target.parent))
    try:
        seeds, clones = {}, {}
        def copy(source_name, child_name):
            source_name, child_name = str(source_name), str(child_name)
            source = _file(root, source_name)
            dest = stage / _relative(child_name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            digest = _sha(dest)
            if digest != audit.hashes[source_name]:
                raise InputChanged('Source changed while copying: ' + source_name)
            seeds[child_name], clones[child_name] = digest, source_name

        for name in state['files_sha256']:
            copy(INPUT/'execution-feeds'/name, Path('execution-feeds')/name)
        for name in dividends:
            copy(INPUT/'dividends'/name, Path('dividends')/name)
        for name in [*manifests, str(INPUT/'execution-feeds/index.json'), str(SPEC), str(OVERRIDES)]:
            copy(name, Path('parent-manifests')/name)
        # This independent mutable index initially retains byte-identical seed
        # counters. Incremental request counts are measured against that baseline.
        shutil.copyfile(root/INPUT/'execution-feeds/index.json', stage/'execution-feeds/index.json')
        references = {}
        for logical, name in REFERENCE_PATHS.items():
            references[logical] = {'path': str(name), 'sha256': audit.hashes[str(name)]}
        data = dict(schema=1, kind='exit-research-inputs', prepared_at=datetime.now(timezone.utc).isoformat(),
            preparation_code_sha256=_sha(Path(__file__)), parent_files_sha256=audit.hashes,
            parent_report_manifest=str(REPORT/'manifest.json'), references=references,
            seed_files_sha256=seeds, clone_parent_paths=clones, seed_execution_index=state,
            seed_dividend_hashes=dividends,
            parent_report_dividend_files=sorted(manifests[str(REPORT/'manifest.json')]['corporate_sources']['files_sha256']),
            finmind_requests=0, official_http_requests=0,
            boundary={'immutable': 'All parent files, read-only matrix references, manifest snapshots and inherited child files.',
                'appendable': 'Only new child execution entries/files and new per-stock dividend files; never reseed or overwrite inherited evidence.',
                'index': 'Child execution index may add entries and increase counters; inherited keys and parser identity stay fixed.',
                'quota': 'Subsequent fetches use app.finmind shared quota: 6000/hour plan, 10% reserve (5400/hour); copied counters are inherited, not new requests.',
                'hash_scope': 'Reachable named parent manifests and their file hash maps; embedded historical transformation/runtime claims are preserved verbatim, not recomputed.',
                'verification': 'One streaming hash per distinct parent path per invocation; verify once at run boundaries, never inside per-date/per-policy loops.'})
        _write(stage/'manifest.json', data)
        _verify_child(stage, data, audit)
        audit.stable()
        if target.exists():
            raise InputChanged('Destination appeared concurrently; refusing overwrite')
        stage.rename(target)
        return data
    finally:
        if stage.exists():
            shutil.rmtree(stage)


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InputChanged('Another exit input preparation is running') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Verify only; never create or append data')
    parser.add_argument('--destination', type=Path, default=ROOT/DESTINATION)
    args = parser.parse_args()
    destination = _destination(ROOT, args.destination)
    with _lock(destination.parent / ('.'+destination.name+'.prepare.lock')):
        data = verify(destination) if args.verify else prepare(destination=destination)
    print(json.dumps({'status': 'verified' if args.verify else 'ready', 'directory': str(destination),
        'parent_files': len(data['parent_files_sha256']), 'seed_files': len(data['seed_files_sha256']),
        'finmind_requests': 0, 'official_http_requests': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
