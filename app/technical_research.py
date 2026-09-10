"""Read sealed technical experiments without triggering research or data fetches."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib
from importlib.metadata import version
from pathlib import Path
import platform
import re

import pandas as pd
import streamlit as st

from app.exit_research import _account, _decode, _file, _finite, _read, _stamp, _trades
from app.cash_allocation_research import _exposure

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('.cache/technical-research')
DRIVER = 'scripts/research_technical.py'
SPEC = 'docs/prereg_technical_20260910.md'
INPUT_MANIFEST = '.cache/technical-inputs/manifest.json'
INPUT_PREPARER = 'scripts/prepare_technical_inputs.py'
VERIFICATION_DIRECTORIES = ('.cache/technical-inputs/dividends',)
START, END = '2022-01-03', '2026-09-09'
MODES = ('control', 'support20', 'risk2', 'support_risk2', 'support_risk2_add', 'support_risk2_pattern')
MODE_LABELS = {'control': '原策略對照', 'support20': '加入支撐出場', 'risk2': '加入2%配置',
    'support_risk2': '支撐出場＋2%配置', 'support_risk2_add': '再加入一次強勢加碼',
    'support_risk2_pattern': '再加入收斂突破篩選'}
BASES = {mode: ('support_risk2' if mode in MODES[-2:] else 'control') for mode in MODES}
RULES = {
    'control': '原本的12%停損／63日到期出場，單次買股預算最多前日淨資產三分之一。',
    'support20': '在原策略上加入20日低點支撐；支撐只上移，前日收盤跌破後開始賣出。',
    'risk2': '維持原出場規則，按12%計畫停損距離及交易成本，限制新增個股部位的計畫風險。',
    'support_risk2': '同時採用支撐出場與2%配置，計畫停損取支撐及12%價格門檻中較高者。',
    'support_risk2_add': '固定在「支撐出場＋2%配置」上測加碼：原買入日起漲至少10%，且突破之前20日高點，每批持股最多一次有成交的加碼。',
    'support_risk2_pattern': '固定在「支撐出場＋2%配置」上測篩選：20日突破、10日區間收斂至前10日的75%以下，且成交量至少為前20日均量1.5倍。',
}
REASONS = {'support20': '跌破只升不降的支撐', 'pyramid_add': '強勢突破，加碼一次',
    'fund_stock_add': '賣0050準備加碼', 'loss12': '入場價格訊號下跌12%',
    'time63': '持有滿63日', 'scheduled_exit': '依已觸發出場指令賣出',
    'missing_prior_adjusted_close': '前日還原收盤不足',
    'initial_support_missing_or_not_below_price': '初始支撐不足，或不低於股價',
    'pattern_not_confirmed': '未同時符合三項型態條件', 'pattern_data_missing': '型態資料不足',
    'risk_or_capital_budget_zero': '風險或資金預算不足', 'add_risk_or_capital_budget_zero': '加碼後風險或部位上限不足',
    'add_missing_price_anchor': '加碼價格起點不足', 'add_gain_below_10pct': '未自原買入日起漲10%',
    'add_breakout_not_confirmed': '未確認突破20日高點',
    'prior_liquidity_below_50m_or_missing': '前20日均成交額不足5,000萬元或缺資料',
    'no_prior_price': '缺少事前價格', 'overlapping_member': '已持有同一股票', 'slots_full': '已滿3個個股名額',
    'partial_or_unfilled_execution': '成交不足請求股數',
    'missing_or_zero_quote_volume': '缺少有效報價或當日沒有成交量', 'single_price_session': '當日只有單一價格',
    'missing_price_limits': '缺少當日漲跌停資料', 'unsupported_no_price_limit_session': '無漲跌幅限制日尚不支援',
    'at_upper_limit': '漲停，買進受限', 'at_lower_limit': '跌停，賣出受限', 'missing_adv20': '缺少前20日平均量',
    'no_odd_lot_trade': '當日沒有有效零股成交', 'invalid_odd_lot_quote': '零股買賣報價無效',
    'no_odd_lot_opposing_quote_quantity': '零股對手報價沒有股數', 'odd_lot_at_price_limit': '零股價格觸及漲跌停',
    'proceeds_below_costs': '賣出所得不足支付成本', 'partial_capacity_or_cash': '成交量或現金只夠部分成交',
    'capacity_or_cash_zero': '當日可成交量或現金不足',
    'proceeds_below_costs_insufficient_cash': '賣出所得不足支付成本，現金也不足補差額',
}
DIAGNOSTICS = {'no_prior_market_session': '沒有前一市場日', 'raw_bar_missing': '缺原始日行情',
    'raw_ohlc_missing': '開高低收缺值', 'raw_ohlc_invalid': '開高低收關係或數值無效',
    'adjusted_close_missing_or_invalid': '還原收盤缺值或無效', 'adjusted_bar_invalid': '還原行情無效',
    'support_history_incomplete': '20日支撐窗口不完整', 'support_distance_not_positive': '股價不高於支撐',
    'signal_volume_missing_or_invalid': '訊號日成交量缺值或無效', 'volume_history_incomplete': '20日成交量窗口不完整',
    'volume_baseline_not_positive': '20日平均成交量為零', 'preceding_range_not_positive': '較早10日區間為零'}


def _signature(root, output):
    """Stat all declared sources on a rerender; hash them only after a change."""
    manifest_path = output / 'manifest.json'
    before = _stamp(manifest_path)
    payload = manifest_path.read_bytes()
    manifest = _decode(payload)
    inventory = manifest['verification_files_sha256']
    required = {DRIVER, SPEC, INPUT_MANIFEST, INPUT_PREPARER, str((output / 'report.json').relative_to(root))}
    if (not isinstance(inventory, dict) or not required.issubset(inventory)
            or any(not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)
                   for digest in inventory.values())):
        raise ValueError('Incomplete technical source verification inventory')
    files = tuple((name, digest, _stamp(_file(root, name))) for name, digest in sorted(inventory.items()))
    watched = manifest['verification_directories']
    if (not isinstance(watched, list) or len(watched) != len(set(watched))
            or set(watched) != set(VERIFICATION_DIRECTORIES)):
        raise ValueError('Incomplete technical source-directory verification inventory')
    parents = {_file(root, name).parent for name in inventory}
    for name in watched:
        directory = root / name
        if directory.is_symlink() or not directory.is_dir() or not directory.resolve().is_relative_to(root):
            raise ValueError('Invalid technical verification directory')
        parents.add(directory.resolve())
    directories = tuple((str(parent.relative_to(root)), _stamp(parent)) for parent in sorted(parents))
    if before != _stamp(manifest_path):
        raise ValueError('Technical manifest changed while reading')
    runtime_names = manifest.get('context', {}).get('runtime_versions',
        manifest.get('runtime_versions', {'python': '', 'pandas': ''}))
    runtime = tuple((name, platform.python_version() if name == 'python' else version(name))
                    for name in sorted(runtime_names))
    return before, runtime, files, hashlib.sha256(payload).hexdigest(), directories


_driver_stamp = None


def _verify_driver(output):
    global _driver_stamp
    current = _stamp(ROOT / DRIVER)
    if _driver_stamp is not None and _driver_stamp != current:
        raise ValueError('Verification code changed; restart the workbench to verify a new seal')
    module = importlib.import_module('scripts.research_technical')
    _driver_stamp = current
    return module.verify_report(output=output)


def _validate_decisions(account, mode):
    dates = {row['date'] for row in account['daily']}
    for kind in ('exit', 'sizing', 'add', 'pattern'):
        rows = account[kind + '_decisions']
        if not isinstance(rows, list):
            raise ValueError('Missing technical decision journal')
        for row in rows:
            if (not isinstance(row, dict) or row['date'] not in dates
                    or not isinstance(row['stock_id'], str) or not re.fullmatch(r'\d{4}', row['stock_id'])
                    or not isinstance(row['signal_date'], str) or row['signal_date'] >= row['date']):
                raise ValueError('Invalid or unlagged technical decision')
            if kind == 'sizing' and row['mode'] != mode:
                raise ValueError('Sizing decision belongs to a different method')
            if kind in ('sizing', 'add'):
                if (any(type(row[key]) is not int or row[key] < 0 for key in ('requested_qty', 'filled_qty'))
                        or row['filled_qty'] > row['requested_qty']
                        or any(row.get(key) is not None and not _finite(row[key]) for key in
                               ('prior_nav', 'previous_price', 'adjusted_close', 'support20', 'support_floor',
                                'risk_cap', 'planned_stop', 'raw_planned_stop', 'planned_risk'))):
                    raise ValueError('Invalid technical quantity or planned risk')
            if kind == 'pattern':
                fields = ('pattern_pass', 'breakout20', 'contraction10', 'volume_expansion')
                if type(row['pattern_available']) is not bool or any(row[key] is not None and type(row[key]) is not bool for key in fields):
                    raise ValueError('Invalid pattern decision')
                components = [row[key] for key in fields[1:]]
                available = all(value is not None for value in components)
                if row['pattern_available'] != available or row['pattern_pass'] != (all(components) if available else None):
                    raise ValueError('Pattern pass disagrees with components')


def _normalize(report):
    if (report['schema'] != 1 or report['mode_order'] != list(MODES)
            or report['research_kind'] != 'technical' or report['control_exit_mode'] != 'loss12'
            or report['live_qualified'] is not False or report['unseen_validation'] is not False
            or report['auto_promote'] is not False or report['candidate_count'] != 458
            or report['execution_signal_lag_market_sessions'] != 1
            or report['benchmark']['mode'] != 'benchmark' or set(report['cases']) != set(MODES)):
        raise ValueError('Incomplete or unsupported technical comparison')
    candidates = report['candidate_features']
    candidate_rows = candidates['rows']
    if (candidates['schema'] != 1 or candidates['counts_are_orders_or_fills'] is not False
            or not isinstance(candidate_rows, list) or len(candidate_rows) != 458
            or len({row['event_id'] for row in candidate_rows}) != 458
            or candidates['summary']['scope'] != 'complete_original_candidates_before_portfolio_eligibility'
            or candidates['summary']['candidate_count'] != 458):
        raise ValueError('Complete pre-eligibility candidate feature audit required')
    for row in candidate_rows:
        if (row['signal_date'] != row['original_signal_date'] or row['signal_date'] >= row['entry_date']
                or row['pattern_pass'] is not None and type(row['pattern_pass']) is not bool):
            raise ValueError('Invalid original-candidate feature audit')
    for flag, suffix in ((True, 'true'), (False, 'false'), (None, 'unknown')):
        if candidates['summary']['pattern_pass_' + suffix + '_count'] != sum(row['pattern_pass'] is flag for row in candidate_rows):
            raise ValueError('Candidate pattern count differs from source rows')
    methods = {}
    for mode in MODES:
        case = report['cases'][mode]
        if case['mode'] != mode or case['research_kind'] != 'technical' or case['control_exit_mode'] != 'loss12':
            raise ValueError('Technical case identity mismatch')
        account = _account(case['account'], case['summary'])
        methods[mode] = account | {'exposure': _exposure(account),
            **{kind + '_decisions': case[kind + '_decisions'] for kind in ('exit', 'sizing', 'add', 'pattern')}}
        _validate_decisions(methods[mode], mode)
        _exit_cash_costs(methods[mode])
    benchmark = _account(report['benchmark']['account'], report['benchmark']['summary'])
    dates = [row['date'] for row in benchmark['daily']]
    if any([row['date'] for row in account['daily']] != dates for account in methods.values()):
        raise ValueError('Technical and benchmark dates differ')
    return {'available': True, 'start': START, 'end': END, 'initial_cash': 1_000_000,
        'methods': methods, 'benchmark': benchmark, 'limitations': report.get('limitations', []),
        'performance': report.get('performance', {}), 'candidate_features': candidates}


@lru_cache(maxsize=2)
def _verified(root_name, output_name, signature):
    root, output = Path(root_name), Path(output_name)
    manifest = _verify_driver(output)
    if manifest.get('offline_identical') is not True or manifest.get('live_qualified') is not False:
        raise ValueError('Technical offline reproduction has not completed')
    payload = (output / 'report.json').read_bytes()
    expected = {name: digest for name, digest, _ in signature[2]}[str((output / 'report.json').relative_to(root))]
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Technical report changed while reading')
    result = _normalize(_decode(payload))
    if _signature(root, output) != signature:
        raise ValueError('Technical sources changed during verification')
    result['source_verification'] = {'status': 'verified', 'file_count': len(signature[2]),
                                    'checked_at': datetime.now(timezone.utc).isoformat()}
    return result


def overview():
    root = Path(ROOT).resolve()
    output = root / CACHE
    if not (output / 'report.json').is_file() or not (output / 'manifest.json').is_file():
        return {'available': False, 'source_verification': {'status': 'pending'},
            'note': '支撐、風險配置、加碼與型態研究尚未完成；全部帳本封存並通過驗證後才顯示績效。'}
    try:
        return _verified(str(root), str(output), _signature(root, output))
    except (OSError, ValueError, KeyError, TypeError, ImportError, AttributeError) as exc:
        return {'available': False, 'source_verification': {'status': 'invalid'},
            'note': '這份技術策略結果未通過完整封存驗證，暫不顯示報酬。', 'verification_error': str(exc)}


def _comparison_rows(report):
    methods, benchmark = report['methods'], report['benchmark']
    rows = []
    for mode in (*MODES, 'benchmark'):
        summary = (benchmark if mode == 'benchmark' else methods[mode])['summary']
        base = None if mode == 'benchmark' else BASES[mode]
        delta = None if base is None else summary['total_return'] - methods[base]['summary']['total_return']
        rows.append({'方法': '0050持有' if mode == 'benchmark' else MODE_LABELS[mode],
            '本項比較對象': '—' if mode in ('control', 'benchmark') else MODE_LABELS[base],
            '期末資產（元）': round(summary['final_nav'], 2), '累積淨報酬': f"{summary['total_return']:.2%}",
            '比比較對象增減（百分點）': '—' if mode in ('control', 'benchmark') else f'{delta*100:+.2f}',
            '年化報酬': f"{summary['cagr']:.2%}", '最大回撤': f"{summary['max_drawdown']:.2%}",
            '全部成本（元）': round(summary['costs']['total_cost'], 2), '成交筆數': summary['trade_count']})
    return pd.DataFrame(rows)


def _ledger(account):
    frame = _trades(account)
    if not frame.empty:
        frame['原因'] = [REASONS.get(row['reason'], current) for row, current in zip(account['trades'], frame['原因'])]
    return frame


def _why(value):
    return REASONS.get(value, value) if value else '—'


def _flag(value):
    return '符合' if value is True else '不符合' if value is False else '資料不足'


def _diagnostics(values):
    return '、'.join(DIAGNOSTICS.get(value, value) for value in values)


def _support_rows(account):
    return pd.DataFrame([{'執行日': row['date'], '訊號日': row['signal_date'], '代號': row['stock_id'],
        '還原收盤（訊號尺度）': row.get('signal_close'), '20日低點（還原尺度）': row.get('support20'),
        '只升不降支撐（還原尺度）': row.get('support_floor'),
        '支撐失效': _flag(row.get('support_failure')), '出場判斷': _why(row.get('reason')),
        '首次出場訊號日': row.get('first_signal_date'),
        '資料診斷': _diagnostics(row.get('technical_diagnostics', [])), '事件': row.get('event_id')}
        for row in account['exit_decisions'] if 'support_floor' in row])


def _sizing_rows(account):
    return pd.DataFrame([{'執行日': row['date'], '訊號日': row['signal_date'], '代號': row['stock_id'],
        '請求股數': row['requested_qty'], '成交股數': row['filled_qty'],
        '事前淨資產（元）': row['prior_nav'], '事前規劃價格（原始尺度）': row['previous_price'],
        '計畫停損（原始尺度）': row.get('raw_planned_stop'),
        '初始20日支撐（還原尺度）': row.get('support20'), '計畫風險（元）': row.get('planned_risk'),
        '2%參考金額（元）': row.get('risk_cap'), '拒絕或執行狀況': _why(row.get('failure')),
        '資料診斷': _diagnostics(row.get('diagnostics', [])), '事件': row.get('event_id')}
        for row in account['sizing_decisions']])


def _add_rows(account):
    return pd.DataFrame([{'執行日': row['date'], '訊號日': row['signal_date'], '代號': row['stock_id'],
        '請求加碼股數': row['requested_qty'], '加碼成交股數': row['filled_qty'],
        '突破20日高點': _flag(row.get('breakout20')),
        '原買入還原收盤': row.get('entry_price'), '訊號還原收盤': row.get('adjusted_close'),
        '加碼後計畫風險（元）': row.get('planned_risk'), '2%參考金額（元）': row.get('risk_cap'),
        '拒絕或執行狀況': _why(row.get('failure')), '事件': row.get('event_id')}
        for row in account['add_decisions']])


def _pattern_rows(account):
    return pd.DataFrame([{'執行日': row['date'], '訊號日': row['signal_date'], '代號': row['stock_id'],
        '突破20日高點': _flag(row['breakout20']), '前10日區間收斂': _flag(row['contraction10']),
        '成交量達1.5倍': _flag(row['volume_expansion']), '三條件全部通過': _flag(row['pattern_pass']),
        '資料診斷': _diagnostics(row.get('diagnostics', [])), '事件': row.get('event_id')}
        for row in account['pattern_decisions']])


def _candidate_rows(candidates):
    return pd.DataFrame([{'原候選進場日': row['entry_date'], '訊號日': row['signal_date'], '代號': row['stock_id'],
        '訊號原始收盤': row.get('raw_close'), '訊號還原收盤': row.get('adjusted_close'),
        '20日低點（還原尺度）': row.get('support20'), '20日高點（還原尺度）': row.get('resistance20'),
        '前10日區間（還原尺度）': row.get('range_recent10'), '再前10日區間（還原尺度）': row.get('range_preceding10'),
        '訊號日成交量（股）': row.get('raw_volume'), '前20日平均量（股）': row.get('volume20'),
        '突破20日高點': _flag(row.get('breakout20')), '區間收斂': _flag(row.get('contraction10')),
        '成交量達1.5倍': _flag(row.get('volume_expansion')), '三條件全部通過': _flag(row['pattern_pass']),
        '資料診斷': _diagnostics(row.get('diagnostics', [])), '事件': row['event_id']}
        for row in candidates['rows']])


def _rejected_rows(account):
    return pd.DataFrame([{'執行日': row['date'], '代號': row['stock_id'],
        '請求股數': row['requested_qty'], '成交股數': row['filled_qty'],
        '原因': _why(row['failure']), '事件': row.get('event_id')}
        for row in account.get('orders', []) if row.get('failure')])


def _exit_cash_costs(account):
    """Describe net sale cash deficits already present in the trade ledger."""
    sales = [row for row in account['trades'] if row['side'] == 'sell']
    negative = [row for row in sales if row['cash_change'] < 0]
    result = dict(negative_count=len(negative), zero_count=sum(row['cash_change'] == 0 for row in sales),
                  cash_paid=-sum(row['cash_change'] for row in negative))
    technical = account.get('summary', {}).get('technical', {})
    expected = {'nonpositive_proceeds_exit_trade_count': result['negative_count'] + result['zero_count'],
                'fee_funded_exit_trade_count': result['negative_count'],
                'zero_proceeds_exit_trade_count': result['zero_count'],
                'fee_funded_exit_cash_paid': result['cash_paid']}
    if any(key in technical and (not _finite(technical[key]) or abs(technical[key] - value) > .011)
           for key, value in expected.items()):
        raise ValueError('Exit net-cash summary differs from actual trades')
    return result


def render():
    with st.container(border=True):
        st.subheader('支撐出場、風險配置、加碼，哪一項有幫助？')
        report = overview()
        if not report.get('available') or report.get('source_verification', {}).get('status') != 'verified':
            method = st.warning if report.get('source_verification', {}).get('status') == 'invalid' else st.info
            method(report.get('note', '技術策略研究尚未完成。'))
            return
        methods, benchmark = report['methods'], report['benchmark']
        st.caption(f'本金 NT$1,000,000｜{START}～{END}｜已扣手續費、交易稅與滑價｜歷史模擬，非實盤')
        st.write('先把支撐出場、2%配置分別與原策略比較，再看合併效果；加碼與型態篩選都固定比較「支撐出場＋2%配置」。')
        st.dataframe(_comparison_rows(report), hide_index=True, use_container_width=True)
        st.caption('表中的增減是各自對照的帳戶報酬差。資金、出場與名額改變會影響之後買到的股票，不能直接加總成每條規則的獨立貢獻。')
        mode = st.selectbox('查看哪項技術實驗', list(MODES), format_func=lambda value: MODE_LABELS[value], key='technical_method')
        selected = methods[mode]
        st.write(RULES[mode])
        st.caption('所有判斷只用前一市場日及更早的訊號，次日成交仍受漲跌停、成交量與可用現金限制。原12%停損與63日到期保留；出場未成交會繼續嘗試。')
        st.caption('2%是單檔部位的計畫風險，含預估交易成本，不是實際損失保證或全帳戶風險上限。所有組別剩餘資金仍買0050，因此仍有大盤曝險。')
        compared = [(MODE_LABELS[mode], selected)]
        if mode != BASES[mode]:
            compared.append((MODE_LABELS[BASES[mode]], methods[BASES[mode]]))
        compared.append(('0050持有', benchmark))
        chart = pd.concat([pd.Series({row['date']: row['nav'] for row in account['daily']}, name=name)
                           for name, account in compared], axis=1)
        chart.index = pd.to_datetime(chart.index)
        st.line_chart(chart, y_label='淨資產（NT$）')
        st.caption('資產包含持股與尚未收取的股利／新股權利估值，並非全部可提領現金。')
        with st.expander('每年表現與0050曝險'):
            st.dataframe(pd.DataFrame([{'方法': name, **{str(row['year'])+('截至09/09' if str(row['year']) == '2026' else ''):
                f"{row['total_return']:.2%}" for row in account['summary']['annual']}} for name, account in compared]),
                hide_index=True, use_container_width=True)
            exposure = pd.DataFrame(selected['exposure']).set_index('date')
            labels = {'cash': '現金', 'etf': '0050', 'stocks': '個股', 'receivable': '應收款與新股權利'}
            st.dataframe(pd.DataFrame([{'資產': label, '每日平均占比': f'{exposure[key].mean():.2%}',
                '期末占比': f'{exposure[key].iloc[-1]:.2%}'} for key, label in labels.items()]), hide_index=True, use_container_width=True)
        with st.expander('支撐、買進股數與拒絕原因'):
            st.caption('支撐表使用還原價格尺度；成交與計畫停損的原始尺度另有標示，兩者不可直接比價。20日低點排除訊號日，持股支撐只升不降；缺窗口不補值。')
            for title, frame in [('每天支撐與出場判斷', _support_rows(selected)), ('每次買入的事前配置', _sizing_rows(selected)),
                                 ('未全部成交或未進場的指令', _rejected_rows(selected))]:
                if not frame.empty:
                    st.write(title)
                    st.dataframe(frame, hide_index=True, use_container_width=True)
            if not selected['sizing_decisions'] and not _support_rows(selected).shape[0]:
                st.info('原策略對照沒有新增支撐或風險配置規則；實際買賣請看下方成交帳本。')
            st.caption('2%配置只在名稱含「2%配置」的實驗啟用；支撐單獨組顯示的2%金額僅供參考。請求與成交股數分開列出，0股代表尚未成交。')
        if mode == 'support_risk2_add':
            with st.expander('強勢加碼：條件與實際成交'):
                filled = [row for row in selected['add_decisions'] if row['filled_qty'] > 0]
                st.write(f"有成交的加碼共 {len(filled)} 次，共 {sum(row['filled_qty'] for row in filled):,} 股。")
                add_cost = sum(row['total_cost'] for row in selected['trades'] if row['reason'] == 'pyramid_add')
                funding_cost = sum(row['total_cost'] for row in selected['trades'] if row['reason'] == 'fund_stock_add')
                st.caption(f'加碼個股買進成本 NT${add_cost:,.0f}，賣0050籌加碼資金成本 NT${funding_cost:,.0f}；均已包含於總報酬。')
                st.caption('單次預算最多前日淨資產10%，加碼後單檔最多三分之一，並檢查整檔剩餘部位的計畫風險。部分成交也算已用掉一次；不重設原始起點與63日期限。')
                frame = _add_rows(selected)
                if not frame.empty:
                    st.dataframe(frame, hide_index=True, use_container_width=True)
                else:
                    st.info('沒有符合檢查資格的加碼紀錄。')
        if mode == 'support_risk2_pattern':
            with st.expander('收斂突破：三項條件逐一檢查'):
                candidates = report['candidate_features']
                counts = candidates['summary']
                st.write(f"原始458個候選中，型態符合 {counts['pattern_pass_true_count']} 個、不符合 {counts['pattern_pass_false_count']} 個、資料不足 {counts['pattern_pass_unknown_count']} 個。")
                st.caption('這是進場名額與資金限制之前的完整候選檢查，不是下單或成交次數。')
                st.dataframe(_candidate_rows(candidates), hide_index=True, use_container_width=True)
                passed = sum(row['pattern_pass'] is True for row in selected['pattern_decisions'])
                st.write(f"接受型態檢查 {len(selected['pattern_decisions'])} 次，三項條件全部通過 {passed} 次；通過不等於一定買到。")
                st.caption('先通過原持股名額與流動性限制，才做型態檢查。這裡只測一種收斂突破規則，沒有辨識W底、頭肩底或所有三角形型態。')
                frame = _pattern_rows(selected)
                if not frame.empty:
                    st.dataframe(frame, hide_index=True, use_container_width=True)
        with st.expander('全部買賣與CSV下載'):
            exit_cash = _exit_cash_costs(selected)
            if exit_cash['negative_count']:
                st.caption(f"有 {exit_cash['negative_count']} 筆賣出所得不足支付交易成本，使用帳戶現金補付差額共 NT${exit_cash['cash_paid']:,.2f}。"
                    '這是成交帳本中的淨現金流出，已計入資產與報酬，不是再加扣一次的成本。')
            if exit_cash['zero_count']:
                st.caption(f"另有 {exit_cash['zero_count']} 筆賣出所得恰好支付成本，淨現金收付為0，股數仍已賣出。")
            frame = _ledger(selected)
            st.dataframe(frame, hide_index=True, use_container_width=True)
            st.download_button('下載所選實驗全部買賣 CSV', frame.to_csv(index=False).encode('utf-8-sig'),
                file_name=f'technical-{mode}-trades.csv', mime='text/csv', key='technical_trades_csv')
            st.caption('整張參考普通收盤、零股參考最後價格，滑價另列成本；「當日期末資產」不是成交瞬間淨資產。零股雖受日量限制，仍可能超過最後揭示的對手量，欄位已逐筆標示。')
        with st.expander('來源驗證與研究限制'):
            st.write('同一段歷史已反覆研究，沒有未見測試；不會自動取代正式策略。')
            for note in report['limitations']:
                st.caption(note)
            verified = report['source_verification']
            st.caption(f"完整來源驗證已通過，共 {verified['file_count']:,} 個檔案；切換畫面只讀封存結果，不會啟動回測或抓取資料。來源變動會重新驗證。")
