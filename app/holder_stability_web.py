"""Read-only publication of one frozen study; independent of scanner rules."""
import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from app.research_terminal_web import HEADERS
from app.workbench_api import require_local

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'artifacts/forward_simulation/holder_stability_600_20261102.json'
ASSETS = ROOT / 'ui/research_terminal/holder-stability'
EXPECTED_SHA256 = '89f188fa78d29c50e31366f8b360ff4752e537315ea7d692aa08f6065e7e8003'
router = APIRouter(dependencies=[Depends(require_local)])

CANDIDATE_FIELDS = ('stock_id', 'name', 'close', 'drawdown60', 'amount20',
                    'large_pct', 'large_pct_delta4', 'large_units_relative4',
                    'observed_date', 'available_date', 'named_case', 'price_confirmed')


def study_view():
    """Expose display fields only, never local paths, source receipts or live state."""
    raw = REPORT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Study integrity mismatch')
    report = json.loads(raw)
    if (report['schema'] != 'holder_stability_600_v1' or not report['completed']
            or report['end'] != '2026-10-30' or report['live_qualified']
            or report['adopted']):
        raise ValueError('Unexpected study identity or qualification')
    latest = report['latest']
    main = [row for row in latest if row['holder_lag'] == 8 and row['stable']]
    delayed = {row['stock_id']: row for row in latest if row['holder_lag'] == 15}
    if len(main) != 112 or len({row['stock_id'] for row in main}) != 112:
        raise ValueError('Unexpected watchlist membership')
    candidates = []
    for row in sorted(main, key=lambda r: (-r['amount20'], r['stock_id'])):
        item = {key: row[key] for key in CANDIDATE_FIELDS}
        item['delayed_state'] = delayed[row['stock_id']]['holder_state']
        item['delayed_observed_date'] = delayed[row['stock_id']]['observed_date']
        candidates.append(item)
    return dict(study_id='holder_stability_600_20261102', start=report['start'],
                end=report['end'], universe_count=report['universe_count'],
                snapshot_weeks=report['snapshot_weeks'], counts=report['counts'],
                latest_counts=report['latest_counts'], candidates=candidates,
                confirmed_count=sum(row['price_confirmed'] for row in main),
                statistics=report['statistics'], paired_statistics=report['paired_statistics'],
                costs=report['costs'], live_qualified=False, adopted=False,
                original_strategy_changed=False, study_type=report['study_type'])


@router.get('/research-terminal/api/holder-stability')
def holder_stability():
    try:
        return study_view()
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(503, '籌碼穩定研究尚未通過封存核對，請確認研究檔案完整後重試。') from None


@router.get('/terminal/holder-stability', include_in_schema=False)
def redirect():
    return RedirectResponse('/terminal/holder-stability/', status_code=307)


@router.get('/terminal/holder-stability/', include_in_schema=False)
def index():
    return asset('index.html')


@router.get('/terminal/holder-stability/{name}', include_in_schema=False)
def asset(name: str):
    types = {'index.html': 'text/html', 'app.css': 'text/css', 'app.js': 'text/javascript'}
    if name not in types:
        raise HTTPException(404, '找不到介面檔案')
    path = ASSETS / name
    if not path.is_file():
        raise HTTPException(503, '籌碼穩定研究介面尚未安裝完成')
    return FileResponse(path, media_type=types[name], headers=HEADERS)
