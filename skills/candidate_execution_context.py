"""Read sealed execution routing without rebuilding unrelated legacy feature matrices."""
import hashlib
import json
from pathlib import Path
import pandas as pd


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_context(root, days):
    root = Path(root).resolve()
    publication = root/'artifacts/forward_simulation/historical_selector_replay_20260925.json'
    sidecar = publication.with_suffix('.sha256')
    if _sha(publication) != sidecar.read_text().strip():
        raise ValueError('Execution publication hash differs')
    pub = json.loads(publication.read_text())
    refs = dict(pub['source_sha256'])
    refs[str(publication.relative_to(root))] = _sha(publication)
    refs[str(sidecar.relative_to(root))] = _sha(sidecar)
    manifest = root/pub['run_manifest']['path']
    refs[str(manifest.relative_to(root))] = pub['run_manifest']['sha256']
    route = manifest.parent/'execution-source.json'
    if str(route.relative_to(root)) not in refs:
        raise ValueError('Execution routing lacks a sealed identity')
    for name, digest in refs.items():
        p = (root/name).resolve()
        if not p.is_relative_to(root) or _sha(p) != digest:
            raise ValueError('Changed execution context source: '+name)
    routing = json.loads(route.read_text())
    inputs = (root/routing['path']).resolve()
    if routing.get('offline') is not True or not inputs.is_relative_to(root) or not inputs.is_dir():
        raise ValueError('Invalid offline execution route')
    split_path = root/'docs/benchmark_split_evidence_20260914.json'
    split = json.loads(split_path.read_text())
    schedule, terms = split['schedule_source'], split['verified_terms']
    notice = root/schedule['local_path']
    if (_sha(notice) != schedule['sha256']
        or schedule['conservative_known_by'] >= terms['suspension_start']):
        raise ValueError('Split suspension source is unverified')
    refs[str(split_path.relative_to(root))] = _sha(split_path)
    refs[str(notice.relative_to(root))] = _sha(notice)
    rows = [dict(stock_id='0050', date=day, open=0., high=0., low=0., close=0., volume=0.)
            for day in days if terms['suspension_start'] <= str(day.date()) < terms['new_units_listing_date']]
    exclusion = dict(stock_id='0050', kind='trading_suspension', start=terms['suspension_start'],
                     end=terms['new_units_listing_date'], source_path=schedule['local_path'])
    return inputs, pd.DataFrame(rows), exclusion, refs
