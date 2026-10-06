#!/usr/bin/env python3
"""Move the existing local scanner bookmark to the API-backed application."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="0;url=http://127.0.0.1:8000/terminal/">
<title>股票研究工作台</title><body>
<h1>股票研究工作台已整合</h1>
<p><a href="http://127.0.0.1:8000/terminal/">開啟新版：訊號、走勢圖、回測與研究</a></p>
<p>若無法開啟，請於專案執行 make api。</p>
<a href="multi_strategy_scanner_snapshot_20261006.html">查看原始封存掃描報告</a>
</body></html>\n'''


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def publish(root=ROOT):
    root = Path(root)
    source = root / '.cache/multi-strategy-scanner/20261006-oct05-v2'
    receipt = json.loads((source / 'receipt.json').read_text())
    expected = receipt['files_sha256']['index.html']
    if digest(source / 'index.html') != expected:
        raise ValueError('原始掃描頁指紋不符，停止發布')
    output = root / 'artifacts/reports'
    output.mkdir(parents=True, exist_ok=True)
    current = output / 'multi_strategy_scanner.html'
    if current.exists() and digest(current) not in (expected, hashlib.sha256(BRIDGE.encode()).hexdigest()):
        raise ValueError('目前頁面有其他變更；停止覆寫')
    archived = output / 'multi_strategy_scanner_snapshot_20261006.html'
    if archived.exists() and digest(archived) != expected:
        raise ValueError('封存頁面指紋不符')
    if not archived.exists():
        shutil.copyfile(source / 'index.html', archived)
    temporary = current.with_suffix('.tmp')
    temporary.write_text(BRIDGE)
    temporary.replace(current)
    return {'application_url': 'http://127.0.0.1:8000/terminal/',
            'legacy_url': 'http://127.0.0.1:8768/multi_strategy_scanner.html',
            'archived_sha256': expected}


if __name__ == '__main__':
    print(json.dumps(publish(), ensure_ascii=False))
