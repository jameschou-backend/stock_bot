"""Curated, hash-checked research history; unlike a leaderboard, scopes stay separate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PUBLICATIONS = (
    ('strategy_scanner_20261028_close', '10/28 POC訊號與當日掃描', '2026-10-28', 'data', '補上10/27～10/28及280組逐筆；重用17天訊號與4926筆POC紀錄，原策略不變'),
    ('strategy_scanner_20261022_close', '10/22 POC訊號與當日掃描', '2026-10-22', 'data', '補上10/20～10/22及238組逐筆；重用14天訊號與4887筆POC紀錄，10/23颱風休市'),
    ('strategy_scanner_20261019_close', '10/19 POC訊號與當日掃描', '2026-10-19', 'data', '補上10/16與10/19及484組缺漏逐筆；重用12天訊號與4841筆POC紀錄'),
    ('poc_weakness_20261019', 'POC原版：三黑K之前的轉弱與部分減碼候選', '2026-10-19', 'data', '既有73個持股批次與五段回撤診斷；原版不變，未回測減碼績效'),
    ('poc_drawdown_attribution_20261019', 'POC硬篩＋紅K原版：五次主要回撤拆解', '2026-10-19', 'account', '只用既有帳本拆解舊贏家跌價、新買虧損與成本，原版不變'),
    ('poc_market_sizing_20261019', 'POC硬篩＋紅K：大盤轉弱新買減半', '2026-10-19', 'account', '0050低於MA120只縮小新買，重用原版與同成本0050'),
    ('poc_cap40_20261019', 'POC硬篩＋紅K：單股40%減至1/3', '2026-10-19', 'account', '只新增比重減碼，重用原版與同成本0050'),
    ('poc_trailing_20261019', 'POC硬篩＋紅K：移動停利20/12', '2026-10-19', 'account', '只新增移動停利，重用原版與同成本0050'),
    ('poc_original_signal_preparation_20261015_close', '10/15 POC硬篩＋紅K原版：訊號準備', '2026-10-15', 'data', '8檔符合，按原版成交值排序；僅封存訊號，成交待使用者回報'),
    ('strategy_scanner_20261015_close', '10/15 POC訊號與當日掃描', '2026-10-15', 'data', '只補當日結果與225組缺漏逐筆，重用11天訊號及4820筆POC紀錄'),
    ('poc_original_paper_preparation_20261015', 'POC硬篩＋紅K原版：訊號準備', '2026-10-15', 'data', '沿用已完成驗證，只準備訊號；使用者回報買進後才記錄'),
    ('poc_split_accounts_20261015', '兩套原版各50萬：不再平衡', '2026-10-15', 'account', '總本金100萬，兩個獨立帳戶；重用原版與同成本0050'),
    ('poc_risk_controls_20261014', '兩套POC：限制追高與波動部位', '2026-10-14', 'account', '兩項獨立實驗，重用原版與同成本0050；以原成本報酬和回撤判斷改善'),
    ('strategy_scanner_20261014_close', '10/14 全市場策略掃描', '2026-10-14', 'data', '只追加今日行情、POC與情境訊號；歷史資料不變'),
    ('poc_execution_readiness_20261014', '兩套POC原版：FinMind成交資料缺口', '2026-10-14', 'data', '重用已完成帳戶，核對FinMind逐筆與零股時序；缺資料時不產生可執行報酬'),
    ('poc_followup_20261014', 'POC 後續驗證：成交、出場、強度與相關性', '2026-10-14', 'account', '保留兩套原版，三個獨立變因與同成本0050；完整驗證前不啟用紙上交易'),
    ('poc_optimization_20261013', 'POC 優化：冷卻5日與成本壓力', '2026-10-13', 'account', '兩套POC原版與冷卻版、相同成本0050及成交缺件'),
    ('legacy_return_comparison_20261012', '949%與576%：同資金、期間與成本比較', '2026-10-12', 'account', '兩套原策略與0050、四組成本、買HL2／賣30%'),
    ('top3_validation_20261012', '前三名：HL2買價、成本壓力與樣本外資格', '2026-10-12', 'account', '固定前三名與0050；紙上交易採資格關卡'),
    ('all_entry_accounts_20261011', '31 組進場條件：同資金帳戶比較', '2026-10-11', 'account', '共同本金、名額、成交與出場；含0050基準'),
    ('strategy_scanner_20261014', '10/13 全市場策略掃描', '2026-10-14', 'data', '已補351組逐筆資料與還原價錨點；29套進場規則與2組情境更新'),
    ('strategy_scanner_20261013', '10/12 全市場策略掃描', '2026-10-13', 'data', '29 套進場規則與2組情境研究更新'),
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
            if identifier in ('poc_original_paper_preparation_20261015', 'poc_original_signal_preparation_20261015_close'):
                from app.poc_paper_preparation import validate
                row['paper_preparation'] = validate(report)
                row['execution_note'] = '目前只準備訊號；使用者回報買進後才依實際資料記錄，未建立委託或成交。'
            elif identifier == 'poc_weakness_20261019':
                from app.poc_weakness_ui import validate
                row['weakness'] = validate(report)
            elif identifier == 'poc_drawdown_attribution_20261019':
                from app.poc_drawdown_ui import validate
                row['drawdown_attribution'] = validate(report)
            elif identifier == 'poc_market_sizing_20261019':
                from app.poc_market_sizing_ui import validate
                row['market_sizing'] = validate(report)
            elif identifier == 'poc_cap40_20261019':
                from app.poc_cap40_ui import validate
                row['cap40'] = validate(report)
            elif identifier == 'poc_trailing_20261019':
                from app.poc_trailing_ui import validate
                row['trailing'] = validate(report)
                row['execution_note'] = '還原收盤漲20%後啟動、從最高收盤回落12%後次日起出場；未成交保留指令，原版出場規則優先。日區間價仍屬成交估算。'
            elif identifier == 'poc_split_accounts_20261015':
                from app.poc_split_accounts_ui import validate
                row['split_accounts'] = validate(report)
            elif identifier == 'poc_risk_controls_20261014':
                from app.poc_risk_controls_ui import validate
                row['risk_controls'] = validate(report)
            elif identifier == 'poc_followup_20261014':
                from app.poc_followup_ui import validate
                row['followup'] = validate(report)
                row['execution_note'] = '凍結委託診斷與歷史敏感度分開呈現；成交驗證未通過，紙上交易關閉。'
                try:
                    from app.poc_execution_readiness import IDENTIFIER, validate as validate_execution
                    execution, _, execution_sha = _publication(IDENTIFIER, root)
                    row['execution_readiness'] = dict(status='verified_publication',
                        report=validate_execution(execution, sha), source_sha256=execution_sha,
                        download_url='/research-terminal/api/archive/'+IDENTIFIER)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row['execution_readiness'] = dict(status='unavailable', reason=str(exc))
            elif identifier == 'poc_execution_readiness_20261014':
                from app.poc_execution_readiness import validate as validate_execution
                validate_execution(report)
                row['execution_note'] = 'FinMind介面與實際回傳已查核；原版帳戶未重跑，可執行成交驗證仍缺盤中零股時序。'
            elif identifier == 'poc_optimization_20261013':
                from app.poc_optimization_ui import validate
                row['optimization'] = validate(report)
                row['execution_note'] = '同檔候選訊號冷卻5個市場交易日；原版沿用已核對的封存帳戶，冷卻版另跑四種成本。買HL2仍為事後代理價格。'
            elif identifier == 'legacy_return_comparison_20261012':
                from app.legacy_return_comparison_ui import validate
                row['legacy_comparison'] = validate(report)
                row['execution_note'] = '兩套原策略與0050同期間、100萬、買HL2／賣30%；保留各自原選股及出場。'
            elif identifier == 'top3_validation_20261012':
                from app.top3_validation_ui import validate
                row['validation'] = validate(report)
                row['execution_note'] = 'T+1 買HL2／賣30%；固定四組成本。已見歷史不可視為未見樣本外。'
            elif identifier == 'all_entry_accounts_20261011':
                from app.all_entry_account_ui import validate
                row['comparison'] = validate(report)
                row['execution_note'] = '本金100萬、3個活躍名額、共同12%收盤停損／63日期限／進場後三黑K；T+1買70%／賣30%日區間價代理。'
            elif identifier == 'poc_range_first_account_20261005':
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
            if identifier in ('legacy_return_comparison_20261012', 'top3_validation_20261012'):
                from app.account_analytics import supplement
                try:
                    row['analytics'] = dict(status='verified_supplement',
                        cases=supplement(identifier, report, sha, root))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row['analytics'] = dict(status='unavailable', reason=str(exc), cases={})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status='unavailable', note=str(exc), cases=[])
        items.append(row)
    return dict(available=True, items=items, live_qualified=False,
                note='不同研究的期間、名額、進出價格與資料品質不同，不能混成同一個報酬排行。')


def publication_bytes(identifier, root=ROOT):
    return _publication(identifier, root)[1]


def case_detail(identifier, case_id, root=ROOT):
    if identifier == 'poc_market_sizing_20261019':
        from app.poc_market_sizing_ui import case_detail as detail
        publication, _, _ = _publication(identifier, root)
        return detail(publication, case_id, root)
    if identifier == 'poc_cap40_20261019':
        from app.poc_cap40_ui import case_detail as detail
        publication, _, _ = _publication(identifier, root)
        return detail(publication, case_id, root)
    if identifier == 'poc_trailing_20261019':
        from app.poc_trailing_ui import case_detail as detail
        publication, _, _ = _publication(identifier, root)
        return detail(publication, case_id, root)
    if identifier == 'poc_split_accounts_20261015':
        from app.poc_split_accounts_ui import case_detail as detail
        publication, _, _ = _publication(identifier, root)
        return detail(publication, case_id, root)
    if identifier == 'poc_risk_controls_20261014':
        from app.poc_risk_controls_ui import case_detail as detail
        publication, _, _ = _publication(identifier, root)
        return detail(publication, case_id, root)
    if identifier == 'poc_followup_20261014':
        from app.poc_followup_ui import case_detail as detail
        publication, _, _ = _publication(identifier, root)
        return detail(publication, case_id, root)
    if identifier == 'poc_optimization_20261013':
        from app.poc_optimization_ui import case_detail as detail
        publication, _, _ = _publication(identifier, root)
        return detail(publication, case_id, root)
    if identifier == 'legacy_return_comparison_20261012':
        from app.legacy_return_comparison_ui import case_detail as detail
        publication, _, _ = _publication(identifier, root)
        return detail(publication, case_id, root)
    if identifier == 'top3_validation_20261012':
        from app.top3_validation_ui import case_detail as detail
        publication, _, _ = _publication(identifier, root)
        return detail(publication, case_id, root)
    if identifier == 'all_entry_accounts_20261011':
        from app.all_entry_account_ui import case_detail as comparison_detail
        publication, _, _ = _publication(identifier, root)
        return comparison_detail(publication, case_id, root)
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
