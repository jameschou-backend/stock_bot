#!/usr/bin/env python3
"""Independently copy sealed exit-research execution evidence; never fetch data."""
from datetime import datetime, timezone
import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import research_exit_scenarios as parent
from scripts.prepare_exit_inputs import HashAudit, InputChanged, _file, _json, _relative, _sha, _write


PARENT = Path('.cache/exit-research')
PARENT_INPUT = Path('.cache/exit-research-inputs')
DESTINATION = Path('.cache/cash-allocation-inputs')


def _destination(root, destination):
    target = Path(destination) if destination is not None else root / DESTINATION
    if not target.is_absolute():
        target = root / target
    target = target.absolute()
    if not target.is_relative_to(root) or '..' in target.parts:
        raise InputChanged('Destination must be inside the project')
    for name in (PARENT, PARENT_INPUT, Path('.cache/million-replay'),
                 Path('.cache/million-replay-inputs'), Path('.cache/million-replay-signals')):
        protected = root / name
        if target == protected or target.is_relative_to(protected) or protected.is_relative_to(target):
            raise InputChanged('Destination overlaps immutable parent evidence')
    current = target
    while current != root:
        if current.is_symlink():
            raise InputChanged('Destination must not use symlinks')
        current = current.parent
    return target


def _collect_parent(root):
    sealed = parent.verify_report(root / PARENT)
    # Parent verifier streams and validates its entire transitive source closure.
    inventory = dict(sealed['verification_files_sha256'])
    inventory[str(PARENT / 'manifest.json')] = _sha(_file(root, PARENT / 'manifest.json'))
    inputs = _json(_file(root, PARENT_INPUT / 'manifest.json'))
    index = _json(_file(root, PARENT_INPUT / 'execution-feeds/index.json'))
    clones = {str(Path('execution-feeds') / name): str(PARENT_INPUT / 'execution-feeds' / name)
              for name in index['files_sha256']}
    dividends = {}
    for path in sorted((root / PARENT_INPUT / 'dividends').iterdir()):
        if path.suffix != '.parquet' or len(path.stem) != 4 or not path.stem.isdigit():
            raise InputChanged('Unexpected parent dividend evidence: ' + path.name)
        source = str(path.relative_to(root))
        if source not in inventory:
            raise InputChanged('Unsealed parent dividend evidence: ' + source)
        dividends[path.name] = inventory[source]
        clones[str(Path('dividends') / path.name)] = source
    # Read-only references retain original identity, including the complete
    # entry pool and adjusted price matrix. No copying of large quote matrices.
    refs = inputs['references']
    if any(inventory.get(row['path']) != row['sha256'] for row in refs.values()):
        raise InputChanged('Parent input reference is outside the sealed closure')
    return inventory, index, dividends, clones, refs


def _verify_child(root, target, data, parent_values):
    inventory, index, dividends, clones, refs = parent_values
    expected_seeds = {name: inventory[source] for name, source in clones.items()}
    if (data.get('parent_files_sha256') != inventory or data.get('seed_execution_index') != index
            or data.get('seed_dividend_hashes') != dividends or data.get('clone_parent_paths') != clones
            or data.get('seed_files_sha256') != expected_seeds or data.get('references') != refs
            or data.get('parent_report_manifest') != str(PARENT / 'manifest.json')):
        raise InputChanged('Parent chain or inherited cache changed; refusing reseed')
    audit = HashAudit(target)
    for name, digest in expected_seeds.items():
        audit.check(name, digest)
        if os.path.samefile(_file(target, name), _file(root, clones[name])):
            raise InputChanged('Child evidence aliases the parent: ' + name)
    index_path = _file(target, 'execution-feeds/index.json')
    if os.path.samefile(index_path, _file(root, PARENT_INPUT / 'execution-feeds/index.json')):
        raise InputChanged('Child execution index aliases the parent')
    audit.check('execution-feeds/index.json')
    current = _json(index_path)
    if any(current.get(key) != index.get(key) for key in ('schema', 'parser_sha256', 'limitations')):
        raise InputChanged('Child execution identity changed')
    for key in ('entries', 'files_sha256'):
        if not isinstance(current.get(key), dict) or any(current[key].get(k) != v for k, v in index[key].items()):
            raise InputChanged('Inherited execution evidence changed: ' + key)
    for key, value in index['request_counters'].items():
        count = current.get('request_counters', {}).get(key)
        if type(count) is not int or count < value:
            raise InputChanged('Child request counter reset')
    for name, digest in current['files_sha256'].items():
        audit.check(Path('execution-feeds') / _relative(name), digest)
    for entry in current['entries'].values():
        if not {entry['raw_file'], entry['rows_file']} <= set(current['files_sha256']):
            raise InputChanged('Unregistered child execution entry')
    # Newly downloaded dividend files also become part of the final report's
    # closure, while existing seed bytes are immutable across resumptions.
    for path in sorted((target / 'dividends').iterdir()):
        if path.suffix != '.parquet' or len(path.stem) != 4 or not path.stem.isdigit():
            raise InputChanged('Unexpected child dividend evidence: ' + path.name)
        audit.check(path.relative_to(target))
    audit.stable()


def verify(destination=None, *, root=ROOT):
    root = Path(root).resolve()
    target = _destination(root, destination)
    data = _json(_file(target, 'manifest.json'))
    if (data.get('schema') != 1 or data.get('kind') != 'cash-allocation-inputs'
            or data.get('preparation_code_sha256') != _sha(Path(__file__))):
        raise InputChanged('Unsupported or changed cash-allocation input preparation')
    _verify_child(root, target, data, _collect_parent(root))
    return data


def prepare(*, root=ROOT, destination=None):
    root = Path(root).resolve()
    target = _destination(root, destination)
    if target.exists():
        return verify(target, root=root)
    values = _collect_parent(root)
    inventory, index, dividends, clones, refs = values
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + target.name + '-', dir=target.parent))
    try:
        seeds = {}
        for child_name, source_name in clones.items():
            dest = stage / _relative(child_name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_file(root, source_name), dest)
            seeds[child_name] = _sha(dest)
            if seeds[child_name] != inventory[source_name]:
                raise InputChanged('Parent changed during independent copy: ' + source_name)
        shutil.copyfile(_file(root, PARENT_INPUT / 'execution-feeds/index.json'),
                        stage / 'execution-feeds/index.json')
        data = dict(schema=1, kind='cash-allocation-inputs',
            prepared_at=datetime.now(timezone.utc).isoformat(), preparation_code_sha256=_sha(Path(__file__)),
            parent_report_manifest=str(PARENT / 'manifest.json'), parent_files_sha256=inventory,
            references=refs, seed_files_sha256=seeds, clone_parent_paths=clones,
            seed_execution_index=index, seed_dividend_hashes=dividends,
            finmind_requests=0, official_http_requests=0)
        _write(stage / 'manifest.json', data)
        _verify_child(root, stage, data, values)
        # Verify the source closure again at the mutation boundary, not once
        # per copied file or per simulated market session.
        if _collect_parent(root) != values:
            raise InputChanged('Parent source changed during preparation')
        if target.exists():
            raise InputChanged('Destination appeared concurrently; refusing overwrite')
        stage.rename(target)
        return data
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--destination', type=Path, default=ROOT / DESTINATION)
    args = parser.parse_args()
    data = verify(args.destination) if args.verify else prepare(destination=args.destination)
    print(parent.encoded(dict(status='verified' if args.verify else 'ready',
        parent_files=len(data['parent_files_sha256']), seed_files=len(data['seed_files_sha256']),
        finmind_requests=0, official_http_requests=0)))


if __name__ == '__main__':
    main()
