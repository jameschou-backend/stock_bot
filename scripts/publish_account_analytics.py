"""Publish derived cohort metrics without changing or rerunning sealed accounts."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import account_analytics as analytics
from app.research_terminal_archive import _publication, case_detail


def build(root=ROOT):
    root = Path(root)
    result = dict(schema='account_analytics_v1', live_qualified=False,
        calculator_sha256=hashlib.sha256(Path(analytics.__file__).read_bytes()).hexdigest(),
        method='按原進場批次合併分批成交、扣費稅滑價並含實收分配；排除未平倉及未結清應收。',
        studies={})
    for identifier in analytics.STUDIES:
        report, _, sha = _publication(identifier, root)
        cases = {}
        for key, row in report['cases'].items():
            if not row['completed']:
                continue
            # Existing reader verifies account bytes, scope and original summary.
            detail = case_detail(identifier, f"{row['id']}_{row['mode']}", root)
            raw = (root / row['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != detail['source_sha256']:
                raise ValueError('帳戶在計算期間改變')
            cases[key] = dict(source_sha256=row['sha256'],
                **analytics.realized_metrics(json.loads(raw)['account']))
        result['studies'][identifier] = dict(publication_sha256=sha, cases=cases)
    return result


if __name__ == '__main__':
    result = build()
    path = ROOT / analytics.SUPPLEMENT
    raw = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    if path.exists() and path.read_bytes() != raw:
        raise ValueError('已發布補充不可覆寫；請使用新版本')
    path.write_bytes(raw)
    path.with_suffix('.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
    print(json.dumps(dict(path=str(path), cases=sum(len(s['cases']) for s in result['studies'].values()))))
