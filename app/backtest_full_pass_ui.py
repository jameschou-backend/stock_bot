"""Show one verified full-pass inventory instead of successive partial counts."""
import json
import os
import stat
import threading
from copy import deepcopy
from pathlib import Path
from collections import Counter

from app.backtest_tool_ui import verified_bytes
from skills.publication_versions import digest

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/forward_simulation/backtest_full_pass_v2_20260925.json'
KINDS={'ordinary','identity','odd_daily','external','provider_refresh','dependencies','cohort_prices'}
CODE=('app/backtest_full_pass_ui.py','scripts/publish_backtest_full_pass.py')
_VERIFIED_CACHE={}
_CACHE_LOCK=threading.RLock()
_HASH_MAPS={'input_sha256','source_sha256','code_sha256'}
_SCHEMAS={
    'ordinary':'board_tape_reconciliation_v2',
    'identity':'historical_universe_completion_v2',
    'odd_daily':'odd_lot_daily_gap_evidence_v1',
    'external':'external_source_followup_v1',
    'provider_refresh':'provider_tape_conflict_resolution_v1',
    'dependencies':'backtest_case_dependencies_v1',
    'cohort_prices':'historical_cohort_local_prices_v1',
}


def _file_signature(path,root,optional=False):
    """Keep lexical paths so replacing a symlink cannot retain its old target."""
    if not path.resolve().is_relative_to(root):
        raise ValueError('快取來源超出專案範圍')
    try:
        value=path.stat()
        link=path.lstat()
    except FileNotFoundError:
        if optional:
            return None
        raise
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('快取來源不是一般檔案：'+str(path))
    fields=('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns')
    return tuple(getattr(value,key) for key in fields),tuple(getattr(link,key) for key in fields)


def _source_signatures(path,root):
    """Guard the fixed seven-report contract, whose maps contain flattened refs.

    Only index, parent, base and the seven reports are decoded. Account/tape
    leaves are stat-only, even when JSON. Do not guess bases for nested copied
    manifests or receipt paths. A new report schema requires a closure review.
    """
    signatures={}

    def add(name,optional=False):
        if not isinstance(name,(str,Path)):
            raise ValueError('快取來源路徑格式不符')
        source=Path(os.path.abspath(root/name))
        if source not in signatures:
            signatures[source]=(optional,_file_signature(source,root,optional))
            if source.suffix=='.json':
                add(source.with_suffix('.sha256'),True)
        elif not optional and signatures[source][1] is None:
            raise FileNotFoundError(source)
        return source

    def document(descriptor):
        return json.loads(add(descriptor['path']).read_text())

    def maps(value):
        for key in _HASH_MAPS:
            mapping=value.get(key)
            if isinstance(mapping,dict):
                for name in mapping:
                    add(name)

    index=json.loads(add(path).read_text())
    if index.get('schema')!='backtest_full_pass_v1' or set(index['reports'])!=KINDS:
        raise ValueError('快取不支援這個整批核對範圍')
    maps(index)
    parent=document(index['parent_followup'])
    if parent.get('schema')!='backtest_data_followup_v1':
        raise ValueError('快取不支援這個前輪報告')
    base=document(parent['base_data_report'])
    if base.get('schema')!='backtest_data_completion_v1':
        raise ValueError('快取不支援這個回測資料範圍')
    maps(parent);maps(base)
    for descriptor in parent['reports'].values():
        add(descriptor['path'])
    for descriptor in base.get('case_sources',{}).values():
        add(descriptor['path'])
    for kind,descriptor in index['reports'].items():
        report=document(descriptor)
        if report.get('schema')!=_SCHEMAS[kind]:
            raise ValueError('快取尚未審核這個來源報告版本：'+kind)
        maps(report)
        if kind=='cohort_prices':
            maps(report['plan'])
            add((root/descriptor['path']).parent/'quotes.parquet')
            add('scripts/prepare_historical_cohort_supplement.py')
    return signatures


def _same_signatures(signatures,root):
    return all(_file_signature(path,root,optional)==expected
               for path,(optional,expected) in signatures.items())


def load_for_ui(path=None,root=None):
    """Reuse successful verification only while the entire evidence is intact.

    There is no TTL and failures are never cached. The lock makes verification
    and replacement single-flight across Streamlit sessions in this process.
    ``load`` remains uncached for CLI/publication and all seven validators run
    on every miss. Filesystem signatures include ctime to catch restored mtime.
    """
    root=Path(ROOT if root is None else root).resolve()
    path=Path(os.path.abspath(root/Path(REPORT if path is None else path)))
    key=(root,path)
    with _CACHE_LOCK:
        entry=_VERIFIED_CACHE.get(key)
        if entry is not None:
            signatures,result=entry
            try:
                unchanged=_same_signatures(signatures,root)
            except Exception:
                _VERIFIED_CACHE.pop(key,None)
                raise
            if unchanged:
                result=deepcopy(result)
                try:
                    stable=_same_signatures(signatures,root)
                except Exception:
                    _VERIFIED_CACHE.pop(key,None)
                    raise
                if not stable:
                    _VERIFIED_CACHE.pop(key,None)
                    raise ValueError('快取讀取期間來源檔案變動')
                return result
            _VERIFIED_CACHE.pop(key,None)
        signatures=_source_signatures(path,root)
        result=load(path,root)
        stored=deepcopy(result)
        if not _same_signatures(signatures,root):
            raise ValueError('完整驗證期間來源檔案變動，未建立快取')
        _VERIFIED_CACHE[key]=(signatures,stored)
        return result


def _validators(root):
    """Each report must revalidate its raw sources before counts can be shown."""
    from skills.board_tape_reconciliation_v2 import verify_report as ordinary
    from scripts.audit_historical_universe_completion import verify_report as identity
    from scripts.audit_odd_lot_daily_gaps import verify_report as odd_daily
    from scripts.audit_external_source_followup import verify_report as external
    from scripts.reconcile_provider_tape_conflicts import verify as provider_refresh
    from scripts.audit_backtest_case_dependencies import verify as dependencies
    from scripts.prepare_historical_cohort_supplement import verify as cohort_prices
    return dict(ordinary=lambda p:ordinary(p,root),identity=identity,odd_daily=odd_daily,
        external=external,provider_refresh=provider_refresh,dependencies=dependencies,
        cohort_prices=lambda p:cohort_prices(p.parent))


def _count(value):
    if type(value) is not int or value<0:
        raise ValueError('核對數量必須是非負整數')
    return value


def _validate_display(reports):
    board=reports['ordinary']['summary']
    fields=('same_scope_aggregate_matched','aggregate_conflicts','independent_daily_source_missing')
    if sum(_count(board[key]) for key in fields)!=_count(board['required_sessions']):
        raise ValueError('普通盤核對數量未涵蓋同一股日集合')
    for key in ('remaining_unknown_starts','unconfirmed_categories','unresolved_current_date_discrepancies'):
        _count(reports['identity'][key])
    daily=reports['odd_daily']
    allowed={'official_daily_positive_trade','official_daily_zero_trades','stock_row_absent_unproven'}
    if (not set(daily['statuses']).issubset(allowed)
            or sum(_count(v) for v in daily['statuses'].values())!=_count(daily['required_stock_days'])):
        raise ValueError('零股日表結果有未核對的股日')
    external=reports['external']
    for key in ('historical_auction_sessions_acquired','missing_historical_auction_sessions'):
        _count(external[key])
    _count(external['issuer']['current_observed_historical_documents_received'])
    cohort=reports['cohort_prices']['summary']
    if (_count(cohort['warmup_rows'])+_count(cohort['research_period_rows'])!=_count(cohort['quote_rows'])
            or _count(cohort['quarantined_rows'])>cohort['quote_rows']):
        raise ValueError('歷史母集合行情列數不一致')
    _count(cohort['stock_count'])
    refresh=reports['provider_refresh']['summary']
    if (_count(refresh['refreshed'])>_count(refresh['required'])
            or _count(refresh['aggregate_matched'])+_count(refresh['conflicts'])!=refresh['refreshed']):
        raise ValueError('來源差異重查數量不一致')
    dependency=reports['dependencies']
    if _count(dependency['summary']['case_count'])!=20 or len(dependency['cases'])!=20:
        raise ValueError('策略依賴必須對應原 20 案')
    counts=Counter(row['code'] for case in dependency['cases'].values()
                   for row in case['dependencies'] if row['required'])
    if dict(counts)!=dependency['summary']['required_case_counts']:
        raise ValueError('策略依賴摘要與各案例不一致')
    # This also checks the per-case flags used by the table below.
    dependency_rows(dependency)


def dependency_rows(report):
    """Show trading-channel requirements separately from selection factors."""
    rows=[]
    for name,case in sorted(report['cases'].items()):
        config=case['config']
        if case.get('live_qualified') is not False or case.get('strict_data_ready') is not False:
            raise ValueError('資料依賴不能提升策略資格')
        required={row['code']:row['required'] for row in case['dependencies']}
        if any(type(value) is not bool for value in required.values()):
            raise ValueError('資料依賴缺少明確的必要性判斷')
        if not required['daily_price_volume_and_adjustments'] or not required['corporate_event_terms_and_delivery']:
            raise ValueError('不能略過價格或公司行動證據')
        if config['board_only'] and required['odd_lot_complete_authenticated_sessions']:
            raise ValueError('整張案例含未釐清的零股成交需求')
        label=('0050 持有基準' if config['benchmark'] else
               '價量擴散' if name.startswith('corporate:') else
               '相對強勢＋同業成交升溫' if config['arm']=='strength_with_turnover' else '相對強勢')
        row={'策略':label,'交易方式':'只下整張' if config['board_only'] else '整張＋零股',
            '歷史股票全集':'需要' if required['historical_universe_and_eligibility'] else '僅核對0050自身',
            '歷史產業成員':'需要' if required['historical_industry_membership'] else '未使用',
            '完整零股時序':'需要' if required['odd_lot_complete_authenticated_sessions'] else '不下零股單',
            '營收／新聞版本':'需要' if required['financial_and_news_publication_versions'] else '未使用',
            '公司行動證據':'需要'}
        if row not in rows:
            rows.append(row)
    return rows


def load(path=None,root=None):
    root=Path(ROOT if root is None else root).resolve()
    path=(root/Path(REPORT if path is None else path)).resolve()
    if not path.is_relative_to(root) or path.suffix!='.json':
        raise ValueError('整批核對摘要超出允許範圍')
    raw=path.read_bytes()
    if digest(raw)!=path.with_suffix('.sha256').read_text().strip():
        raise ValueError('整批核對摘要的指紋不一致')
    index=json.loads(raw)
    if (index.get('schema')!='backtest_full_pass_v1' or set(index['reports'])!=KINDS
            or index.get('strict_data_ready') is not False
            or index.get('live_qualified') is not False or index.get('performance_recomputed') is not False):
        raise ValueError('資料核對範圍或資格標記不符')
    if set(index.get('code_sha256',{}))!=set(CODE):
        raise ValueError('整批核對缺少介面與發布程式指紋')
    for name,expected in index['code_sha256'].items():
        verified_bytes(dict(path=name,sha256=expected),root,'.py')
    parent=json.loads(verified_bytes(index['parent_followup'],root,'.json'))
    if (parent['schema']!='backtest_data_followup_v1' or parent['live_qualified'] is not False
            or parent['performance_recomputed'] is not False):
        raise ValueError('前輪資料範圍不符')
    verified_bytes(parent['base_data_report'],root,'.json')
    reports={k:json.loads(verified_bytes(v,root,'.json')) for k,v in index['reports'].items()}
    validators=_validators(root)
    for key,validate in validators.items():
        if validate(root/index['reports'][key]['path'])!=reports[key]:
            raise ValueError('來源核對期間資料變動：'+key)
    # Tie new evidence to the same demand/case scope rather than lending counts
    # from another experiment with different orders or execution channels.
    for key,descriptor in [('ordinary',parent['reports']['ordinary']),
                           ('provider_refresh',parent['reports']['ordinary']),
                           ('odd_daily',parent['reports']['odd_lot']),
                           ('dependencies',parent['base_data_report'])]:
        if reports[key]['input_sha256'].get(descriptor['path'])!=descriptor['sha256']:
            raise ValueError('補件未對應相同回測範圍：'+key)
    descriptor=parent['reports']['identity']
    if reports['identity']['source_sha256'].get(descriptor['path'])!=descriptor['sha256']:
        raise ValueError('歷史身分補件未對應相同前輪範圍')
    if reports['cohort_prices']['plan']['input_sha256'].get(descriptor['path'])!=descriptor['sha256']:
        raise ValueError('歷史行情補件未對應相同前輪範圍')
    descriptor=index['reports']['odd_daily']
    if reports['external']['input_sha256'].get(descriptor['path'])!=descriptor['sha256']:
        raise ValueError('外部資料核對未使用本輪零股日表')
    _validate_display(reports)
    return index,reports


def render():
    import pandas as pd
    import streamlit as st
    if not REPORT.exists():
        return
    st.subheader('資料補件總檢查')
    try:
        index,reports=load_for_ui()
    except (OSError,ValueError,KeyError,TypeError,ImportError,RuntimeError) as exc:
        st.error(f'整批核對結果無法驗證：{exc}')
        return
    board=reports['ordinary']['summary']
    identities=reports['identity']
    daily=reports['odd_daily']
    external=reports['external']
    cohort=reports['cohort_prices']['summary']
    st.warning('尚未補齊全部實戰所需資料。這次資料核對沒有重算或認證策略報酬。')
    rows=[
        {'項目':'普通盤獨立對帳','結果':f"{board['same_scope_aggregate_matched']} 股日一致；{board['aggregate_conflicts']} 股日衝突；{board['independent_daily_source_missing']} 股日缺完整來源"},
        {'項目':'已知證券身分差異','結果':f"上市日未知 {identities['remaining_unknown_starts']}；類別待核 {identities['unconfirmed_categories']}；日期差異待核 {identities['unresolved_current_date_discrepancies']}"},
        {'項目':'先前缺少的零股日表','結果':f"{daily['statuses'].get('official_daily_positive_trade',0)}／{daily['required_stock_days']} 股日查得實際成交"},
        {'項目':'完整歷史零股時序','結果':f"本輪取得 {external['historical_auction_sessions_acquired']} 股日；仍缺 {external['missing_historical_auction_sessions']:,} 股日"},
        {'項目':f"重新查詢 {reports['provider_refresh']['summary']['required']} 個來源差異",'結果':f"{reports['provider_refresh']['summary']['conflicts']} 股日仍不一致，保留隔離"},
        {'項目':'歷史公告原文','結果':f"本輪取得 {external['issuer']['current_observed_historical_documents_received']} 篇；仍未取得當年完整修訂鏈"},
        {'項目':'舊選股全集漏列的歷史行情','結果':f"已封存 {cohort['stock_count']} 檔、{cohort['quote_rows']:,} 列；{cohort['quarantined_rows']} 列價格缺漏／矛盾隔離，尚未改算策略"},
    ]
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.caption('日成交總量與逐筆時序是不同證據。日表補齊或日總量相符，都不能直接當成委託成交證明。')
    st.write('身分差異釐清仍不等於已取得全市場、含下市股票的完整歷史投資資格。')
    with st.expander('哪些資料是目前策略真正需要的？'):
        counts=reports['dependencies']['summary']['required_case_counts']
        st.write(f"原 20 案中：{counts['historical_universe_and_eligibility']} 案需要歷史選股全集，"
                 f"{counts['historical_industry_membership']} 案需要歷史產業成員，"
                 f"{counts['odd_lot_complete_authenticated_sessions']} 案需要零股時序。")
        news=counts.get('financial_and_news_publication_versions',0)
        st.write('這 20 案沒有直接使用營收或新聞挑股；全市場新聞修訂史不列為共同必要輸入。'
                 if not news else f'{news} 案需要核對營收或新聞的歷史版本。')
        st.write('除權息、股票交付與現金入帳時間仍須逐項核對。')
        st.dataframe(pd.DataFrame(dependency_rows(reports['dependencies'])),hide_index=True,use_container_width=True)
        st.caption('只下整張可免除零股下單時序需求；公司行動造成的殘股估值與股票交付仍要核對。')
        st.write('本次保留原帳本及報酬；資料核對沒有啟動排程、下單或採購。')
    with st.expander('下載來源與核對紀錄'):
        labels={'ordinary':'普通盤','identity':'歷史身分','odd_daily':'零股日表',
                'external':'外部來源限制','provider_refresh':'FinMind 差異重查','dependencies':'策略資料依賴',
                'cohort_prices':'歷史股票行情補包'}
        for key,label in labels.items():
            st.download_button('下載'+label,json.dumps(reports[key],ensure_ascii=False,indent=2),
                key+'-full-pass.json','application/json',key='full_pass_'+key)
