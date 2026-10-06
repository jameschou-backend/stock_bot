"""Curated, hash-checked research history; unlike a leaderboard, scopes stay separate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PUBLICATIONS = (
    ('strategy_scanner_20261006', '10/5 全市場策略掃描', '2026-10-06', 'data', '行情與每日訊號'),
    ('strategy_scanner_expansion_20261005', '29 套進場規則：訊號後續比較', '2026-10-05', 'signal_research', '固定 5／20／60 日、同期間 0050'),
    ('poc_range_first_account_20261005', 'POC＋紅 K：首日訊號與成交價比較', '2026-10-05', 'account', '三檔帳戶、現金與零股、六個固定比較'),
    ('poc_intraday_account_20261005', '盤中整股與零股成交研究', '2026-10-05', 'account', '逐筆普通盤與零股日線估算分開'),
    ('poc_broker_account_20261004', '分點籌碼加入 POC 帳戶', '2026-10-04', 'account', '分點資料覆蓋與六組對照'),
    ('broker_branch_diagnostics_20261004', '券商分點連買與籌碼集中', '2026-10-04', 'signal_research', '分點覆蓋有限，未知不算買超'),
    ('volume_profile_pilot_20261003', '成交量分布 POC／VAH／VAL', '2026-10-03', 'signal_research', '逐筆量價分布與進出條件'),
    ('signal_rank_20261003', '高分訊號是否比較會上漲', '2026-10-03', 'signal_research', '排名分組與配對比較'),
    ('sector_participation_20261003', '族群成交熱度與上漲廣度', '2026-10-03', 'signal_research', '族群資金參與條件'),
    ('sequential_research_20261003', '進場條件依序交叉比較', '2026-10-03', 'signal_research', '保留全部比較與未成立的假設'),
)
CASE_NAMES = {
    'benchmark_range50': '0050 基準・中間價',
    'benchmark_range70_30': '0050 基準・買 70%／賣 30% 區間價',
    'poc_range50_all': 'POC 優先＋紅 K・全部訊號・中間價',
    'poc_range50_first': 'POC 優先＋紅 K・紅 K 候選首日・中間價',
    'poc_range70_30_all': 'POC 優先＋紅 K・全部訊號・買 70%／賣 30%',
    'poc_range70_30_first': 'POC 優先＋紅 K・紅 K 候選首日・買 70%／賣 30%',
}


def _decode(raw):
    def reject(value):
        raise ValueError('研究紀錄含有無效數字：' + value)
    return json.loads(raw, parse_constant=reject)


def _publication(identifier, root=ROOT):
    if identifier not in {item[0] for item in PUBLICATIONS}:
        raise ValueError('未知研究紀錄')
    path = Path(root) / 'artifacts' / 'forward_simulation' / (identifier + '.json')
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    # Older publications use sha256sum's "digest  filename" receipt format.
    receipt = path.with_suffix('.sha256').read_text().strip()
    match = re.fullmatch(r'([a-f0-9]{64})(?:  (.+))?', receipt)
    if (match is None or sha != match.group(1)
            or match.group(2) not in (None, path.name)):
        raise ValueError('研究紀錄指紋不符；不顯示舊報酬')
    report = _decode(raw)
    if report.get('live_qualified') is not False:
        raise ValueError('研究紀錄缺少明確資格標示')
    return report, raw, sha


def _period(report):
    period = report.get('period')
    period = period if isinstance(period, dict) else {}
    nested = report.get('summary') or report.get('study') or {}
    nested = nested if isinstance(nested, dict) else {}
    return {'start': report.get('start') or period.get('start') or period.get('signal_start') or nested.get('start'),
            'end': report.get('end') or period.get('end') or period.get('signal_end') or report.get('data_end') or nested.get('end')}


def _case(identifier, row):
    summary = row['summary']
    return dict(id=identifier, name=CASE_NAMES.get(identifier, identifier),
                **{key: summary.get(key) for key in ('total_return', 'max_drawdown', 'final_nav', 'buy_count', 'sell_count')})


def overview(root=ROOT):
    items = []
    for identifier, title, day, kind, topic in PUBLICATIONS:
        row = dict(id=identifier, title=title, date=day, kind=kind, topic=topic,
                   live_qualified=False, cases=[], period={'start': None, 'end': None})
        try:
            report, _, sha = _publication(identifier, root)
            row.update(status='verified_publication', source_sha256=sha,
                       period=_period(report), initial_cash=report.get('initial_cash'),
                       position_count=report.get('maximum_active_stock_slots', report.get('position_count', report.get('active_slots'))),
                       note='已核對發布紀錄指紋；未於開啟頁面時重跑帳戶或核對全部歷史來源。',
                       download_url='/research-terminal/api/archive/' + identifier)
            if identifier == 'poc_range_first_account_20261005':
                row['execution_note'] = 'T 收盤訊號、T+1 日內區間價估算；中間價=(H+L)/2，另一組買 L+70%×(H−L)、賣 L+30%×(H−L)。日內高低事前未知，並非可預掛的成交保證。'
                row['note'] += '「首日」指原紅 K 候選首次連續成立，與 POC 硬篩首次成立不同。'
                for name, case in report['cases'].items():
                    if case.get('completed') is True:
                        value = _case(name, case)
                        value['detail_url'] = '/research-terminal/api/archive/' + identifier + '/cases/' + name
                        row['cases'].append(value)
            elif identifier == 'poc_intraday_account_20261005':
                row['execution_note'] = '普通盤採逐筆穿價參與率；盤中零股仍為當日高低中間價估算，未核實排隊或成交。'
                if report.get('completed') is True:
                    row['cases'] = [_case('strategy', {'summary': report['strategy_summary']}),
                                    _case('0050', {'summary': report['benchmark_summary']})]
            elif identifier == 'poc_broker_account_20261004':
                row['execution_note'] = '普通盤／零股日內高低中間價估算；POC 優先排序，分點資料未知依原實驗規則保留。'
            else:
                row['execution_note'] = '詳細設定與資料限制請看研究紀錄；單筆訊號結果不能當帳戶累積報酬。'
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status='unavailable', note=str(exc), cases=[])
        items.append(row)
    return dict(available=True, items=items, live_qualified=False,
                note='不同研究的期間、名額、進出價格與資料品質不同，不能混成同一個報酬排行。')


def publication_bytes(identifier, root=ROOT):
    return _publication(identifier, root)[1]


def case_detail(identifier, case_id, root=ROOT):
    if identifier != 'poc_range_first_account_20261005' or case_id not in CASE_NAMES:
        raise ValueError('這份研究尚未提供帳戶曲線')
    publication, _, _ = _publication(identifier, root)
    descriptor = publication['cases'][case_id]
    path = (Path(root) / descriptor['path']).resolve()
    if not path.is_relative_to(Path(root).resolve()) or path.suffix != '.json':
        raise ValueError('帳戶檔案路徑錯誤')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != descriptor['sha256']:
        raise ValueError('帳戶指紋不符')
    report = _decode(raw)
    if (report.get('completed') is not True or report.get('live_qualified') is not False
            or report.get('summary') != descriptor.get('summary')):
        raise ValueError('帳戶結果與發布摘要不同')
    account = report['account']
    daily = [{key: row.get(key) for key in ('date', 'nav', 'cash', 'market_value', 'receivable', 'drawdown', 'holdings')}
             for row in account['daily']]
    trades = [{key: row.get(key) for key in ('date', 'signal_date', 'stock_id', 'name', 'side', 'qty', 'price',
                                            'proxy_price', 'reason', 'commission', 'tax', 'slippage', 'gross',
                                            'cash_change', 'total_cost', 'cash_after', 'channel')}
              for row in account['trades']]
    benchmark = case_id.startswith('benchmark_')
    return dict(id=case_id, name=CASE_NAMES[case_id], summary=report['summary'], daily=daily,
                trades=trades, initial_cash=publication['initial_cash'], position_count=1 if benchmark else 3,
                actual_fill_verified=False, live_qualified=False, source_sha256=descriptor['sha256'],
                note=('0050 基準帳戶；' if benchmark else '封存三檔個股帳戶；') +
                     'T+1 日內區間價假設，不是每日開盤價成交。曲線包含持倉與應收股利市值。')
