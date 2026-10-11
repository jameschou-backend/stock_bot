"""Add only the issuer-evidenced 2352 full-session reduction suspension."""
from copy import deepcopy
import json
from pathlib import Path
from bs4 import BeautifulSoup

from skills.all_entry_capital_completion import load_capital_completion


def load_capital_halt(root, document):
    root, document = Path(root).resolve(), Path(document).resolve()
    _, _, refs = load_capital_completion(root, document)
    value = json.loads(document.read_text())
    notice = value['sources']['mops_republication']
    body = ''.join(BeautifulSoup((root/notice['path']).read_bytes(), 'html.parser').get_text().split())
    required = ('減資股票停止在交易市場買賣日期：民國114年9月25日至民國114年10月3日',
                '換發新股暨上市買賣日期(即舊股票終止上市日)：民國114年10月7日')
    if any(text not in body for text in required):
        raise ValueError('Capital suspension lacks the reviewed full-session dates')
    return dict(stock_id='2352', market='TWSE', kind='trading_suspension',
        start='2025-09-25', end='2025-10-07', announcement_date='2025-08-04',
        source_path=notice['path'], source_sha256=notice['sha256'],
        source='Issuer MOPS announcement reproduced by Refinitiv; TAIFEX notice corroborated'), refs


def install_capital_halt(runner, document, root):
    halt, refs = load_capital_halt(root, document)
    target = Path(root).resolve()/'docs/stock_universe_2019_corporate_terms.json'
    original = runner.read
    def read(path):
        value = original(path)
        if Path(path).resolve() != target:
            return value
        value = deepcopy(value)
        if any(h['stock_id'] == halt['stock_id'] and h['start'] < halt['end'] and halt['start'] < h['end']
               for h in value['verified_halts']):
            raise ValueError('Capital suspension overlaps an existing completion')
        value['verified_halts'].append(deepcopy(halt))
        for name, sha in refs.items():
            if name in value['source_sha256'] and value['source_sha256'][name] != sha:
                raise ValueError('Capital halt source conflicts with prior evidence')
        value['source_sha256'].update(refs)
        return value
    runner.read = read
    return refs
