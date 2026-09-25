#!/usr/bin/env python3
"""Capture dated issuer releases without backdating their currently observed content."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.publication_versions import (INDEX_URL, digest, encoded, index_links,
    parse_release, extend_versions, verify_archive, as_of)


def capture(output, parent=None):
    import requests
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / '.cache') or output.exists():
        raise ValueError('Use a new immutable directory inside the project cache')
    prior = verify_archive(parent) if parent else None
    output.mkdir(parents=True)
    files, attempts = {}, []
    def get(url, name):
        if len(attempts) >= 31:
            raise ValueError('Publication capture request budget exhausted')
        # Persist before transport; a failed run is retained, never silently retried.
        attempts.append(dict(url=url, status='started'))
        (output / 'attempts.json').write_bytes(encoded(attempts))
        response = requests.get(url, timeout=30, allow_redirects=False)
        (output / name).write_bytes(response.content)
        row = dict(path=name, url=url, observed_at=datetime.now(timezone.utc).isoformat(),
                   http_status=response.status_code, sha256=digest(response.content))
        attempts[-1].update(status='received', http_status=response.status_code)
        (output / 'attempts.json').write_bytes(encoded(attempts))
        (output / (name+'.source.json')).write_bytes(encoded(row))
        files[name], files[name+'.source.json'] = row['sha256'], digest((output/(name+'.source.json')).read_bytes())
        if response.status_code != 200:
            raise ValueError('Official publication source returned non-200 response; retained without retry')
        return response.content, row
    raw, index = get(INDEX_URL, 'index.html')
    observations = []
    for i, url in enumerate(index_links(raw)):
        raw, receipt = get(url, f'release-{i:02}.html')
        observations.append(dict(parse_release(raw, url), observed_at=receipt['observed_at'], path=receipt['path']))
    files['attempts.json'] = digest((output/'attempts.json').read_bytes())
    archive = dict(format='issuer_publication_versions_v1', issuer='2492',
        historical_complete=False, live_qualified=False, network_requests=len(attempts), finmind_requests=0,
        files_sha256=files, index=index, observations=observations,
        parent=dict(path=str(Path(parent).resolve()), sha256=digest(Path(parent).read_bytes())) if parent else None,
        versions=extend_versions(prior['versions'] if prior else [], observations))
    path = output/'archive.json'
    path.write_bytes(encoded(archive))
    path.with_suffix('.sha256').write_text(digest(path.read_bytes())+'\n')
    verify_archive(path)
    print(encoded(dict(archive=str(path.relative_to(ROOT)), documents=len(observations),
        network_requests=len(attempts), finmind_requests=0,
        usable_observed_versions_on_20260402=len(as_of(archive['versions'],'2026-04-02T23:59:59+08:00')))).decode())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        value = verify_archive(args.verify)
        print(encoded(dict(verified=True, observations=len(value['observations']), network_requests=0)).decode())
    elif args.fetch and args.output:
        capture(args.output, args.parent)
    else:
        parser.error('Use --fetch --output <new cache directory>, or --verify <archive.json>')
