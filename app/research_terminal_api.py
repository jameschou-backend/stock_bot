"""Small bounded JSON API for the local research terminal."""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, model_validator

from app.research_terminal_service import EvidenceError, get_terminal
from app.workbench_api import require_local


def require_same_origin(request: Request):
    """Do not let an unrelated browser origin start local research workers."""
    origin = request.headers.get('origin')
    if request.method not in ('GET', 'HEAD', 'OPTIONS') and origin:
        if origin.rstrip('/') != str(request.base_url).rstrip('/'):
            raise HTTPException(status_code=403, detail='研究工作只能由本機工作台啟動')


router = APIRouter(prefix='/research-terminal/api', tags=['research-terminal'],
                   dependencies=[Depends(require_local), Depends(require_same_origin)])


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    mode: Literal['signal_study', 'account_replay'] = 'signal_study'
    strategy_id: str | None = None
    start: str | None = None
    end: str | None = None
    horizon: Literal[5, 20, 60] | None = None
    replay_mode: Literal['daily', 'strict'] = 'daily'
    replay_policy: Literal['mixed', 'board_only', 'all'] = 'mixed'
    replay_stress: Literal['control', 'combined', 'all'] = 'control'
    preflight: bool = False
    fresh: bool = False

    @model_validator(mode='after')
    def separate_modes(self):
        signal_fields = ('strategy_id', 'start', 'end', 'horizon')
        if self.mode == 'signal_study':
            if any(getattr(self, k) is None for k in signal_fields):
                raise ValueError('訊號研究需要策略、起迄日及持有期間')
            account_fields = {'replay_mode', 'replay_policy', 'replay_stress', 'preflight', 'fresh'}
            if self.model_fields_set & account_fields:
                raise ValueError('訊號研究不能帶入帳戶重播參數')
        elif any(getattr(self, k) is not None for k in signal_fields):
            raise ValueError('原封存帳戶使用固定策略與期間，不能套用訊號研究選項')
        return self


def _call(method, *args, **kwargs):
    try:
        return getattr(get_terminal(), method)(*args, **kwargs)
    except EvidenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TimeoutError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, KeyError) as exc:
        raise HTTPException(status_code=503, detail='本機研究資料不完整，請核對封存來源後再試。') from exc


@router.get('/overview')
def overview():
    return _call('overview')


@router.get('/signals')
def signals(date: str | None = None, strategy_id: str = 'poc_up_red',
            first_only: bool = False, search: str = Query(default='', max_length=64)):
    return _call('signals', date, strategy_id, first_only, search)


@router.get('/stocks/{stock_id}')
def stock(stock_id: str, date: str | None = None, sessions: int = Query(default=120, ge=20, le=500)):
    return _call('stock', stock_id, date, sessions)


@router.get('/strategies')
def strategies():
    return _call('strategies')


@router.get('/studies')
def studies():
    return _call('studies')


@router.get('/research')
def research():
    return _call('research')


@router.get('/research/episodes')
@router.get('/research/examples')
def research_episodes(stock_id: str | None = None, year: str | None = None,
                      horizon: int | None = Query(default=None, ge=20, le=60),
                      offset: int = Query(default=0, ge=0, le=1000000),
                      limit: int = Query(default=50, ge=1, le=200)):
    return _call('research_episodes', stock_id, year, horizon, offset, limit)


@router.get('/archive')
def archive():
    from app.research_terminal_archive import overview
    return overview()


@router.get('/archive/{identifier}')
def archive_publication(identifier: str):
    from app.research_terminal_archive import publication_bytes
    try:
        raw = publication_bytes(identifier)
        return Response(raw, media_type='application/json', headers={
            'Content-Disposition': f'attachment; filename="{identifier}.json"',
            'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(422, '這份研究紀錄不存在或來源核對失敗') from exc


@router.get('/archive/{identifier}/cases/{case_id}')
def archive_case(identifier: str, case_id: str):
    from app.research_terminal_archive import case_detail
    try:
        return case_detail(identifier, case_id)
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(422, '這個帳戶案例不存在或來源核對失敗') from exc


@router.post('/backtests')
def backtests(request: BacktestRequest):
    return _call('backtest', request.model_dump())


@router.get('/backtests/{job_id}')
def backtest(job_id: str):
    return _call('get_backtest', job_id)


@router.get('/backtests/{job_id}/events')
def backtest_events(job_id: str, offset: int = Query(default=0, ge=0, le=1000000),
                    limit: int = Query(default=200, ge=1, le=2000)):
    return _call('backtest_events', job_id, offset, limit)
