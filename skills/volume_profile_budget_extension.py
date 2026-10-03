"""Explicit data-only 8,000-attempt supplement to the sealed profile provider."""
import json
from pathlib import Path

from app.file_lock import file_lock
from scripts.research_early_signal_losses import digest
from skills.volume_profile_data import AccountProfileData

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/'.cache/volume-profile-account-20261003/profiles-v1'
EVIDENCE = ROOT/'.cache/red-volume-exit-20261003/data-extension-v1'
INITIAL_MAXIMUM = 4800
MAXIMUM = 8000
ADDENDUM = 'docs/prereg_candle_volume_data_extension_20261003.md'
ADDENDUM_SHA = '9d1274ee41cd034a2ca676ebe29200903a7b78e7a100b60e763480046661a1fc'
PREVIOUS_REPORT = '.cache/red-volume-exit-20261003/full-v1/report.json'
EXTENSION_SOURCES = (
    ADDENDUM, 'skills/volume_profile_budget_extension.py',
    'scripts/research_candle_volume_extension.py',
    'tests/test_volume_profile_budget_extension.py',
)
SEALED_BINDINGS = {
    ADDENDUM: ADDENDUM_SHA,
    'skills/volume_profile_data.py': '3d87d08ed7e807c434d785520c6af7307634a3bbd2cb29ffed14015432554349',
    'scripts/research_candle_volume_account.py': '43be2f40492c474475aa2ece1b0dcef4064fa1190e682edb20e87e42a0a00b89',
    'skills/candle_volume_rules.py': '470f17c0643faf7e436ca740a21b01b428ce6aa97704d2e5f46717dd9d8d7600',
    'docs/prereg_candle_volume_account_20261003.md': '052b231af61a09c619f15069cfdc18d2d4aa1356fe0a254547c64ddf8a553d68',
    PREVIOUS_REPORT: '2b33236460533746c4fab4f2e65a9dd3df04efb2ab0bd5031b61a1c4d8e37a55',
    '.cache/red-volume-exit-20261003/full-v1/profile-data/profile-features.json':
        'df79ce27c9896e6d936f4ec1dea5c3a9e0cdd83b8d0f70ec6e237787ba728e56',
}


def extension_sources(root=ROOT):
    """Verify the sealed parents and bind all five prior outcomes, including failures."""
    refs = dict(SEALED_BINDINGS)
    for name, expected in refs.items():
        if digest(root/name) != expected:
            raise ValueError('Data extension source changed: '+name)
    previous = json.loads((root/PREVIOUS_REPORT).read_text())
    expected = {'poc_red': True, 'poc_dry': True, 'poc_red_dry': False,
                'poc_dry_weak': False, 'poc_red_dry_weak': False}
    if {a: c['completed'] for a, c in previous['cases'].items()} != expected:
        raise ValueError('Data extension requires the fixed five prior attempts')
    for case in previous['cases'].values():
        path = (root/case['path']).resolve()
        path.relative_to(root.resolve())
        if digest(path) != case['sha256']:
            raise ValueError('Previous case changed: '+case['path'])
        refs[case['path']] = case['sha256']
    for name in EXTENSION_SOURCES:
        refs[name] = digest(root/name)
    return refs


def immutable_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError('Immutable extension evidence changed: '+str(path))
    else:
        with path.open('xb') as handle:
            handle.write(value)


def preserve_initial_ledger(directory, evidence, *, root=ROOT):
    """A missing/reset ledger must not reopen any of the first 4,800 attempts."""
    path = evidence/'initial-attempt-ledger.json'
    if not path.exists():
        attempts = sorted((directory/'attempts').glob('*.json'))
        if len(attempts) != INITIAL_MAXIMUM:
            raise ValueError('Initial extension ledger must contain exactly 4800 attempts')
        refs = {}
        for attempt in attempts:
            refs[str(attempt.relative_to(root))] = digest(attempt)
            receipt = directory/'receipts'/attempt.name
            # Preserve an orphan reservation without inventing a receipt.
            if receipt.exists():
                refs[str(receipt.relative_to(root))] = digest(receipt)
        value = dict(schema='profile_attempt_ledger_extension_v1',
                     initial_attempts=INITIAL_MAXIMUM, maximum_attempts=MAXIMUM,
                     addendum_sha256=ADDENDUM_SHA, source_sha256=refs)
        immutable_bytes(path, (json.dumps(value, sort_keys=True, indent=2)+'\n').encode())
    value = json.loads(path.read_text())
    if (value.get('initial_attempts') != INITIAL_MAXIMUM or
            value.get('maximum_attempts') != MAXIMUM or
            value.get('addendum_sha256') != ADDENDUM_SHA):
        raise ValueError('Data extension ledger binding changed')
    refs = value['source_sha256']
    prefix = str((directory/'attempts').relative_to(root))+'/'
    if sum(name.startswith(prefix) for name in refs) != INITIAL_MAXIMUM:
        raise ValueError('Data extension lost original attempt identities')
    for name, expected in refs.items():
        p = (root/name).resolve(); p.relative_to(root.resolve())
        if not p.is_file() or digest(p) != expected:
            raise ValueError('Original attempt or receipt changed: '+name)
    if len(list((directory/'attempts').glob('*.json'))) > MAXIMUM:
        raise ValueError('Persistent profile attempts exceed the fixed total 8000 cap')
    return dict(refs, **{str(path.relative_to(root)): digest(path)})


class ExtendedAccountProfileData(AccountProfileData):
    """Change only this instance's cap; inherit raw requests and profile rules."""
    def __init__(self, bundle, *, online=False):
        bindings = extension_sources()
        super().__init__(bundle, online=online, maximum_requests=INITIAL_MAXIMUM,
                         directory=DIRECTORY)
        with file_lock(self.directory/'.run.lock', timeout=0):
            bindings.update(preserve_initial_ledger(self.directory, EVIDENCE))
            for name in EXTENSION_SOURCES:
                source = ROOT/name
                snapshot = EVIDENCE/'source-snapshots'/bindings[name]/name
                immutable_bytes(snapshot, source.read_bytes())
                if digest(snapshot) != bindings[name]:
                    raise ValueError('Extension source changed during snapshot: '+name)
                bindings[str(snapshot.relative_to(ROOT))] = bindings[name]
            for name, expected in bindings.items():
                self._mark(ROOT/name, expected)
        self.maximum = MAXIMUM
