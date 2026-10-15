"""POC original preparation and prospective observations, with no fill inference.

This does not initialize a money account, place orders, or relax the historical
qualification gate. Existing research and other forward journals stay immutable.
"""
from datetime import time
import hashlib
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from app import forward_journal as journal

ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo('Asia/Taipei')
PROTOCOL = ROOT/'docs/poc_original_paper_preparation_20261015.json'
PATH = ROOT/'.cache/poc-original-paper-v1/observations.sqlite3'
IDENTIFIER = 'poc_original_paper_preparation_20261015'
HISTORY = 'artifacts/forward_simulation/poc_risk_controls_20261014.json'
HISTORY_SHA = '73b8c39674df42525e2f65e43673f074d849bbaa916145ecd08ad81a6a6889c4'
EXECUTION = 'artifacts/forward_simulation/poc_execution_readiness_20261014.json'
EXECUTION_SHA = '22fc04045ed353a023952721df73849dc75c329a8ae52618c91755dd0e359d6a'
CODE = ('app/poc_paper_preparation.py', 'scripts/prepare_poc_paper.py',
        'app/forward_journal.py', 'skills/all_entry_accounts.py',
        'skills/strategy_scanner/engine.py', 'skills/strategy_scanner/data.py',
        'skills/independent_three_black.py', 'skills/exit_policy.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def read_ref(ref, root=ROOT):
    root = Path(root).resolve()
    path = (root/ref['path']).resolve()
    if not path.is_relative_to(root) or sha(path) != ref['sha256']:
        raise ValueError('POC preparation source hash/path mismatch')
    return json.loads(path.read_text())


def protocol_payload(root=ROOT):
    history = read_ref(dict(path=HISTORY, sha256=HISTORY_SHA), root)
    refs = {k:{n:v[n] for n in ('path','sha256')} for k,v in history['cases'].items()
            if k.startswith(('poc_up_red:original:', 'benchmark:original:'))}
    original = read_ref(refs['poc_up_red:original:baseline'], root)
    settings = original['account']['settings']
    if (settings['slots'], settings['initial_cash'], settings['idle_capital'], settings['drawdown_arm']) != (3,1000000,'cash','three_black'):
        raise ValueError('Original strategy funding/exit scope differs')
    return dict(schema='poc_original_preparation_protocol_v1', strategy_id='poc_up_red',
        name='POC 硬篩＋紅K 原版', phase='observation_preparation_only',
        planning_capital=1000000, cash_account_initialized=False,
        capital_basis='same capital as the user-selected original comparison; not a ledger deposit',
        rules=dict(slots=3, idle_capital='cash', signal_policy='daily_matched_not_first_only',
            ranking='signal_amount20_desc_stock_id_asc', poc='exact_previous_20_market_sessions_up',
            red='signal_raw_close_above_raw_open', signal_lag_sessions=1,
            buy_limit='preknown_official_daily_upper', sell_limit='preknown_official_daily_lower',
            exit_rules=['12pct_close_stop','63_market_sessions','post_entry_three_black'],
            exit_authority='unchanged_original_engine_not_the_older_leader_paper_portfolio',
            observation_cutoff='same_signal_day_after_18_Asia_Taipei_before_next_session',
            missing_data='explicit_unknown_never_a_fill_or_no_signal_day'),
        original_settings=settings, historical_publication=dict(path=HISTORY,sha256=HISTORY_SHA),
        completed_controls=refs, execution_audit=dict(path=EXECUTION,sha256=EXECUTION_SHA),
        source_code_sha256={name:sha(Path(root)/name) for name in CODE},
        fill_policy='no_automatic_fills_no_daily_HL2_or_quote_to_fill_inference',
        existing_other_strategy_journals='not_reused_or_modified',
        minimum_forward_sessions=126, minimum_settled_cohorts=30,
        qualification_note='These are later evaluation thresholds, not evidence already obtained or a reason to backfill prospective observations.',
        paper_enabled=False, actual_fill_verified=False, unseen_validation=False, live_qualified=False)


def load_protocol(root=ROOT):
    path = Path(root)/PROTOCOL.relative_to(ROOT)
    p = json.loads(path.read_text())
    if p != protocol_payload(root):
        raise ValueError('Frozen POC preparation protocol/code changed; retain this version and register a new one')
    return p, dict(path=str(PROTOCOL.relative_to(ROOT)),sha256=sha(path))


def initialize(con, protocol, ref, clock):
    body = dict(protocol=protocol, protocol_ref=ref)
    rows = journal.read_events(con)
    if rows and (rows[0]['kind'] != 'poc_protocol' or rows[0]['body'] != body):
        raise ValueError('Different POC strategy/version cannot share an observation journal')
    return journal.append(con,'poc_protocol','poc_protocol',body,clock)


def observe(con, protocol, ref, snapshot, status, sessions, clock=journal.now):
    at = clock()
    if at.tzinfo is None: raise ValueError('Observation clock needs a timezone')
    if snapshot.get('schema') != 'poc_original_observation_inputs_v1' or snapshot.get('first_only') is not False:
        raise ValueError('Observation requires complete original daily candidates')
    if snapshot.get('ranking') != protocol['rules']['ranking']:
        raise ValueError('Observation ranking differs from the original strategy')
    candidates = snapshot['candidates']
    if len(candidates)+snapshot['rank_unknown_count'] != snapshot['matched_count']:
        raise ValueError('Original candidates were dropped')
    if candidates != sorted(candidates,key=lambda r:(-r['priority'],r['members'][0])):
        raise ValueError('Original candidate order differs')
    if any(c['strategy_id'] != 'poc_up_red' or c['signal_date'] != snapshot['signal_date'] for c in candidates):
        raise ValueError('Different strategy/date in observation')
    existing = journal.read_events(con)
    if existing and at < journal.timestamp(existing[-1]['recorded_at']):
        raise ValueError('Observation cannot precede an already recorded event')
    first = initialize(con,protocol,ref,lambda:at)
    local = at.astimezone(TZ)
    day = snapshot['signal_date']
    if sessions != sorted(set(sessions)):
        raise ValueError('Market sessions must be unique and ordered')
    next_day = next((d for d in sessions if d>day),None) if day in sessions else None
    reasons=[]
    if day != str(local.date()): reasons.append('signal_not_observed_on_its_own_day')
    if day < str(journal.timestamp(first['recorded_at']).astimezone(TZ).date()): reasons.append('signal_predates_protocol_lock')
    if local.time() < time(18): reasons.append('daily_publication_window_not_finished')
    if not status.get('data_ready') or status.get('price_date') != day: reasons.append('daily_market_data_not_ready')
    if next_day is None or next_day <= str(local.date()): reasons.append('next_market_session_not_future_or_unknown')
    if snapshot.get('rank_unknown_count'): reasons.append('original_candidate_ranking_incomplete')
    body = dict(snapshot=snapshot,next_market_session=next_day,
        calendar=dict(source='DB/trading_calendar',sessions=sessions),
        data_status={k:status.get(k) for k in ('price_date','feature_date','data_ready','problems')},
        eligible_observation=not reasons, reasons=reasons,
        orders_created=0,fills_created=0,paper_enabled=False,live_qualified=False)
    kind = 'poc_signal_observation' if not reasons else 'poc_blocked_observation'
    key = kind+':'+day+((':'+journal.digest(body)) if reasons else '')
    return journal.append(con,key,kind,body,lambda:at)


def observation_summary(path=PATH):
    if not Path(path).exists():
        return dict(initialized=False,prospective_signal_days=0,blocked_observations=0,orders=0,fills=0)
    with sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True) as con:
        rows=journal.read_events(con)
    if any(r['kind'] not in ('poc_protocol','poc_signal_observation','poc_blocked_observation') for r in rows):
        raise ValueError('Unexpected money/order event in observation-only journal')
    return dict(initialized=bool(rows),protocol_locked_at=rows[0]['recorded_at'] if rows else None,
        prospective_signal_days=sum(r['kind']=='poc_signal_observation' for r in rows),
        blocked_observations=sum(r['kind']=='poc_blocked_observation' for r in rows),
        latest_hash=rows[-1]['hash'] if rows else None,orders=0,fills=0)


def scanner_snapshot(terminal):
    """Reuse published signals; compute only their original T-known cash ranking."""
    import pandas as pd
    from skills.all_entry_accounts import event
    from skills.strategy_scanner.data import load_bundle
    from skills.strategy_scanner.engine import _prepare
    terminal._load()
    day=max(terminal._days)
    source=terminal._days[day]['stocks']
    matched=[r for r in source if r['results']['poc_up_red']['status']=='matched']
    unknown=[dict(stock_id=r['stock_id'],name=r['name'],reasons=r['results']['poc_up_red']['reasons'])
             for r in source if r['results']['poc_up_red']['status']=='unknown']
    data=load_bundle(terminal.bundle,day,day)
    wanted={r['stock_id'] for r in matched}|{'0050'}
    f,dates,ids=_prepare(data['bars'].loc[data['bars'].stock_id.isin(wanted)],data['calendar'],pd.Timestamp(day))
    rank=f['a'].rolling(20,min_periods=20).mean().loc[day]
    candidates=[];unranked=[]
    for row in matched:
        sid=row['stock_id'];amount=rank[sid]
        if pd.isna(amount) or not float('-inf')<float(amount)<float('inf') or amount<=0:
            unranked.append(sid);continue
        c=event('poc_up_red',sid,day,None,amount)
        c.update(name=row['name'],first_signal=row['results']['poc_up_red']['first_signal'],
            poc=row['results']['poc_up_red']['metrics'])
        candidates.append(c)
    candidates.sort(key=lambda r:(-r['priority'],r['members'][0]))
    terminal._unchanged()
    return dict(schema='poc_original_observation_inputs_v1',signal_date=day,
        candidates=candidates,rank_unknown_count=len(unranked),unranked=unranked,
        unknown=unknown,matched_count=len(matched),unknown_count=len(unknown),
        source_audit=terminal.daily_descriptor,source_hashes=terminal.source_hashes,
        ranking='signal_amount20_desc_stock_id_asc',first_only=False,
        historical_backtests_replayed=0,source_scope='published_signals_and_direct_bundle_hashes',
        current_security_master_provisional=True,paper_enabled=False,live_qualified=False)


def readiness(protocol, ref, snapshot, observation, status, *, root=ROOT, clock=journal.now):
    history=read_ref(protocol['historical_publication'],root)
    execution=read_ref(protocol['execution_audit'],root)
    cases={}
    for key,source in protocol['completed_controls'].items():
        case=read_ref(source,root)
        if case.get('completed') is not True: raise ValueError('Reuse requires a completed control')
        row=history['cases'][key]
        if any(case['summary'][k]!=row['metrics'][k] for k in ('total_return','max_drawdown','initial_cash')):
            raise ValueError('Completed control metrics differ')
        cases[key]=dict(source=source,total_return=row['metrics']['total_return'],max_drawdown=row['metrics']['max_drawdown'])
    audit=execution['cases']['poc_up_red:original:baseline']
    gaps=[
        dict(id='fill_source',title='盤中行情或券商模擬成交回報尚未接入',
             detail='沿用已查核的 FinMind 結論：未取得可辨識通路的盤中零股撮合時序。報價、日高低價與零股日彙總不能當作成交。'),
        dict(id='causal_execution',title='原版成交估算尚未轉成經驗證的事前委託／成交流程',
             detail=f"原成本封存買賣單含 {audit['missing_odd_child_orders']} 筆缺零股時序；普通盤 {audit['board_child_orders']} 筆中只有 {audit['board_total_reconciled_orders']} 筆日總量相符，時序完整性仍未通過。這些是歷史限制，不是新紙上委託。"),
        dict(id='poc_account_integration',title='POC 專用資金帳戶與回報轉接尚未啟用',
             detail='既有 leader 帳戶會把閒置資金買0050且出場規則不同，不能冒充本原版。待確定成交來源後，須核對3名額、殘餘零股、部分成交、三黑出場及公司行動的完整帳戶流程。'),
        dict(id='daily_pit',title='每日事前資料與證券身分仍須逐日核對',
             detail=f"本次沿用 {snapshot['signal_date']} 已完成掃描：{snapshot['matched_count']} 檔符合、{snapshot['unknown_count']} 檔未知。未知保留原因；不把品質衝突改成通過。"),
        dict(id='forward_evidence',title='鎖版後的未見追蹤尚未累積',
             detail='126個交易日、30批結清及成本後相對0050的評估需要未來資料；過去回測與過期訊號不能補算。')]
    if observation['body']['reasons']:
        gaps.insert(0,dict(id='timely_signal',title='本次訊號只能作準備核對，不能回填事前委託',
                          detail='；'.join(observation['body']['reasons'])))
    return dict(schema='poc_original_paper_preparation_v1',checked_at=clock().isoformat(),
        strategy_id='poc_up_red',name=protocol['name'],status='prepared_observation_only',
        initial_cash=1000000,capital_initialized=False,start='2024-01-02',end='2026-10-02',
        protocol=ref,rules=protocol['rules'],completed_controls=cases,reused_accounts=len(cases),
        historical_backtests_replayed=0,finmind_requests=0,
        completed=['鎖定POC硬篩＋紅K原版與100萬元準備設定，保留3名額與現金閒置。',
                   '沿用原版及同成本0050的4種成本結果，核對8份帳戶指紋。',
                   '接入原版20日平均成交值排序，保留持續符合訊號及未知原因。',
                   '建立獨立、不可回填的訊號封存；同內容重送不重複，來源修改會拒絕。'],
        latest_inputs=snapshot,observation={k:observation[k] for k in ('kind','recorded_at','hash','body')},
        data_status={k:status.get(k) for k in ('price_date','feature_date','data_ready','problems')},
        remaining=gaps,execution_audit_reused=protocol['execution_audit'],
        forward_thresholds=dict(sessions=126,settled_cohorts=30),
        historical_gate_pass=False,observation_ready=True,orders_ready=False,
        paper_enabled=False,actual_fill_verified=False,unseen_validation=False,live_qualified=False)


def validate(report):
    expected = {f'{arm}:original:{mode}' for arm in ('poc_up_red','benchmark')
                for mode in ('baseline','fee2','slip2','combined')}
    if (report.get('schema')!='poc_original_paper_preparation_v1' or report.get('strategy_id')!='poc_up_red'
            or report.get('status')!='prepared_observation_only' or report.get('initial_cash')!=1000000
            or report.get('reused_accounts')!=8 or set(report.get('completed_controls',{}))!=expected
            or report.get('historical_backtests_replayed')!=0
            or any(report.get(k) is not False for k in ('capital_initialized','orders_ready','paper_enabled','actual_fill_verified','unseen_validation','live_qualified'))
            or report.get('rules',{}).get('idle_capital')!='cash'
            or not report.get('remaining')):
        raise ValueError('POC紙上準備不能冒充成交或啟用資格')
    body=report['observation']['body']
    if (body['orders_created']!=0 or body['fills_created']!=0 or body['paper_enabled'] is not False
            or body['live_qualified'] is not False
            or bool(body['eligible_observation']) != (not bool(body['reasons']))):
        raise ValueError('POC observation cannot contain orders/fills or hide gaps')
    return report
