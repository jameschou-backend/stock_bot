"""Explicit one-coordinate completion; immutable provider failure remains bound."""
import json
from pathlib import Path

from skills.all_entry_accounts import digest
from skills.poc_executable_data import TickReceipts
from skills.poc_latest_data import merge_refs

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT/'docs/poc_split_profile_completion_20261015.json'
DIRECTORY = ROOT/'.cache/poc-split-accounts-20261015/profile-completion-v1'
QUERY = dict(dataset='TaiwanStockPriceTick', data_id='4561', start_date='2024-05-21')
KEY = '4561-2024-05-21'


def document(root=ROOT):
    root = Path(root)
    path = root/DOCUMENT.relative_to(ROOT)
    data = json.loads(path.read_text())
    if (data['schema'] != 'poc_split_exact_profile_completion_v1' or data['query'] != QUERY
            or data['maximum_new_attempts'] != 1 or data['max_retries'] != 0
            or data['immutable_failure_preserved'] is not True or data['rerun_completed'] is not False):
        raise ValueError('Profile completion scope changed')
    old = (root/data['previous_failure']['path']).resolve()
    old.relative_to(root.resolve())
    if digest(old) != data['previous_failure']['sha256']:
        raise ValueError('Original failed receipt changed')
    previous = json.loads(old.read_text())
    if previous['query'] != QUERY or previous['status'] != 'provider_error':
        raise ValueError('Only the exact provider failure can be completed')
    return data, path, old


def acquire():
    data, path, old = document()
    store = TickReceipts(ROOT, DIRECTORY, maximum=1, online=False)
    dest, attempt = DIRECTORY/'receipts'/(KEY+'.json'), DIRECTORY/'attempts'/(KEY+'.json')
    if dest.exists():
        item = store._validate(dest, QUERY)
    else:
        # Deliberately registered retry in its own directory. The normal get()
        # continues to refuse the old error; no earlier receipt is overwritten.
        item = store._fetch(QUERY, dest, attempt)
    if item.get('status') != 'received' or not item.get('rows'):
        raise ValueError('Exact profile completion remains unavailable: '+item['status'])
    if digest(old) != data['previous_failure']['sha256']:
        raise ValueError('Original failure changed during completion')
    return item


class ExactCompletion:
    def __init__(self, previous, *, root=ROOT):
        self.previous, self.root = previous, Path(root)
        self.data, self.doc, self.old = document(self.root)
        self.store = TickReceipts(self.root, self.root/DIRECTORY.relative_to(ROOT), maximum=1, online=False)
        self.receipt = self.store.directory/'receipts'/(KEY+'.json')
        self.expected_receipt = digest(self.receipt)

    def __getattr__(self, name):
        return getattr(self.previous, name)

    @property
    def refs(self):
        refs = dict(self.previous.refs)
        merge_refs(refs, self.store.refs)
        return refs

    def get(self, sid, day):
        if (sid, day) != (QUERY['data_id'], QUERY['start_date']):
            return self.previous.get(sid, day)
        self.store._mark(self.doc)
        self.store._mark(self.old, self.data['previous_failure']['sha256'])
        self.store._mark(self.receipt, self.expected_receipt)
        item = self.store._validate(self.receipt, QUERY)
        if item.get('status') != 'received' or not item.get('rows'):
            raise ValueError('Completed profile is not a nonempty successful receipt')
        return item
