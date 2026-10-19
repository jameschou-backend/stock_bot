"""Serve the local research application without exposing research files or secrets."""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from app.workbench_api import require_local


ASSETS = Path(__file__).resolve().parents[1] / 'ui' / 'research_terminal'
router = APIRouter(dependencies=[Depends(require_local)], include_in_schema=False)
HEADERS = {
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'no-referrer',
    'Content-Security-Policy': (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
}


@router.get('/terminal')
def terminal_redirect():
    return RedirectResponse('/terminal/', status_code=307)


@router.get('/terminal/')
def terminal_index():
    return terminal_asset('index.html')


@router.get('/terminal/{asset}')
def terminal_asset(asset: str):
    types = {'index.html': 'text/html', 'app.css': 'text/css',
             'app.js': 'text/javascript', 'poc-market-sizing.js': 'text/javascript', 'poc-cap40.js': 'text/javascript', 'poc-trailing.js': 'text/javascript', 'poc-paper-preparation.js': 'text/javascript', 'poc-split-accounts.js': 'text/javascript', 'poc-risk-controls.js': 'text/javascript', 'poc-followup.js': 'text/javascript', 'poc-optimization.js': 'text/javascript', 'comparison-overview.js': 'text/javascript', 'account-comparison.js': 'text/javascript', 'top3-validation.js': 'text/javascript', 'legacy-comparison.js': 'text/javascript'}
    if asset not in types:
        raise HTTPException(404, '找不到介面檔案')
    path = ASSETS / asset
    if not path.is_file():
        raise HTTPException(503, '股票工作台檔案尚未安裝完成')
    return FileResponse(path, media_type=types[asset], headers=HEADERS)
