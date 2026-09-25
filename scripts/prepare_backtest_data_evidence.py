#!/usr/bin/env python3
"""Explicit bounded capture of public documentation, never a market-data API."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
import subprocess

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / '.cache/backtest-data-completion-20260925/official'
URLS = {
    'twse-h4.html': 'https://eshop.twse.com.tw/zh/product/detail/0000000080da7fa70182334eb932009d',
    'tpex-mth.html': 'https://eshop.tpex.org.tw/zh/product/detail/2c92e0139984eab70199892c78bf0004',
    'finmind-technical.html': 'https://finmind.github.io/tutor/TaiwanMarket/Technical/',
    'finmind-fundamental.html': 'https://finmind.github.io/tutor/TaiwanMarket/Fundamental/',
    'twse-industry-announcement.html': 'https://www.twse.com.tw/rwd/zh/announcement/announcement_detail?id=346FAB95F87B11EDB2DA005056BE380E&response=html',
    'twse-industry-1121802250-1.pdf': 'https://www.twse.com.tw/staticFiles/announcement/announcement/1121802250-1.pdf',
    'mops-correction-query.html': 'https://mops.twse.com.tw/mops/web/t120sb02_q10',
}
EXECUTION_URLS = {
    'twse-h2.html': 'https://eshop.twse.com.tw/zh/product/detail/0000000063ce6ab00163d860b694000a',
    'twse-h1.html': 'https://eshop.twse.com.tw/zh/product/detail/00000000639057100163905e1d7c0001',
}


def run(output, execution_only=False):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Evidence capture is immutable; use a new empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema='bounded_official_data_source_capture_v1', attempts=0,
                    finmind_api_requests=0, paid_requests=0, files={})
    for name, url in (EXECUTION_URLS if execution_only else URLS).items():
        manifest['attempts'] += 1
        response = requests.get(url, timeout=25)
        response.raise_for_status()
        (output / name).write_bytes(response.content)
        manifest['files'][name] = dict(requested_url=url, response_url=response.url,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            sha256=sha256(response.content).hexdigest(), bytes=len(response.content),
            http_status=response.status_code)
        print(name, response.status_code, len(response.content), flush=True)
    manifest['derived'] = {}
    if not execution_only:
        name = 'twse-industry-1121802250-1.txt'
        try:
            subprocess.run(['pdftotext', '-layout', str(output / name.replace('.txt', '.pdf')),
                            str(output / name)], check=True)
        except FileNotFoundError as exc:
            raise RuntimeError('Install Poppler (pdftotext) to extract the official attachment') from exc
        manifest['derived'][name] = dict(sha256=sha256((output / name).read_bytes()).hexdigest(),
            source=name.replace('.txt', '.pdf'), command=['pdftotext', '-layout'],
            tool_version=subprocess.run(['pdftotext', '-v'], capture_output=True, text=True).stderr.strip())
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help='Explicitly request seven (or two execution-only) document GETs')
    parser.add_argument('--execution-only', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.fetch:
        parser.error('This network operation requires the explicit --fetch flag')
    run(args.output, args.execution_only)
