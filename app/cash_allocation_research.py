"""Display a sealed idle-cash comparison without launching a replay or API call."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib
from importlib.metadata import version
import math
from pathlib import Path
import platform
import re

import pandas as pd
import streamlit as st

from app.exit_research import _account, _decode, _file, _finite, _read, _stamp, _trades


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('.cache/cash-allocation')
DRIVER = 'scripts/research_cash_allocation.py'
SPEC = 'docs/prereg_cash_allocation_20260910.md'
INPUT_MANIFEST = '.cache/cash-allocation-inputs/manifest.json'
INPUT_PREPARER = 'scripts/prepare_cash_allocation_inputs.py'
VERIFICATION_DIRECTORIES = ('.cache/cash-allocation-inputs/dividends',)
START, END = '2022-01-03', '2026-09-09'
MODES = ('always_0050', 'cash', 'trend_0050')
MODE_LABELS = {'always_0050': '剩餘資金買0050', 'cash': '剩餘資金留現金',
               'trend_0050': '趨勢向上才買0050'}
RULES = {
    'always_0050': '剩餘資金依原規則買0050；個股有買入訊號時，賣出0050籌資。',
    'cash': '剩餘資金保留現金，不買0050；有個股買入訊號時動用現金。',
    'trend_0050': '前一交易日0050高於120日均線時，剩餘資金才買0050；不高於均線時開始賣出0050。',
}
ALLOCATION_REASONS = {
    'initial_allocation': '初始配置0050', 'idle_cash': '剩餘資金投入0050',
    'fund_stock': '賣0050準備買股', 'always_0050': '依原規則持有0050',
    'cash': '剩餘資金保留現金', 'trend_on': '大盤趨勢向上',
    'trend_off': '大盤趨勢轉弱', 'trend_unknown': '大盤訊號不足，暫停新增0050',
    'allocation_trend_off': '大盤趨勢轉弱，賣出0050',
    'parking_trend_off': '大盤趨勢轉弱，賣出0050',
    'market_off': '大盤趨勢轉弱，賣出0050',
}


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
        raise ValueError('Incomplete allocation source verification inventory')
    files = tuple((name, digest, _stamp(_file(root, name))) for name, digest in sorted(inventory.items()))
    watched = manifest['verification_directories']
    if (not isinstance(watched, list) or len(watched) != len(set(watched))
            or set(watched) != set(VERIFICATION_DIRECTORIES)):
        raise ValueError('Incomplete allocation source-directory verification inventory')
    parents = {_file(root, name).parent for name in inventory}
    for name in watched:
        directory = root / name
        if directory.is_symlink() or not directory.is_dir() or not directory.resolve().is_relative_to(root):
            raise ValueError('Invalid allocation verification directory')
        parents.add(directory.resolve())
    directories = tuple((str(parent.relative_to(root)), _stamp(parent)) for parent in sorted(parents))
    if before != _stamp(manifest_path):
        raise ValueError('Allocation manifest changed while reading')
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
    module = importlib.import_module('scripts.research_cash_allocation')
    _driver_stamp = current
    return module.verify_report(output=output)


def _exposure(account):
    """Include unpaid corporate rights separately so NAV weights reconcile."""
    dates = {row['date'] for row in account['daily']}
    values = {day: {'0050': 0., 'stocks': 0.} for day in dates}
    seen = set()
    for row in account['holdings']:
        key = row['date'], row['stock_id']
        if (row['date'] not in dates or key in seen
                or not isinstance(row['stock_id'], str) or not re.fullmatch(r'\d{4}', row['stock_id'])
                or not _finite(row['market_value']) or row['market_value'] < 0):
            raise ValueError('Invalid allocation holding valuation')
        seen.add(key)
        values[row['date']]['0050' if row['stock_id'] == '0050' else 'stocks'] += row['market_value']
    result = []
    for row in account['daily']:
        if any(not _finite(row[key]) or row[key] < -.011 for key in ('cash', 'market_value', 'receivable')):
            raise ValueError('Invalid allocation daily asset component')
        etf, stocks = values[row['date']]['0050'], values[row['date']]['stocks']
        if (not math.isclose(etf + stocks, row['market_value'], abs_tol=.011)
                or not math.isclose(row['cash'] + etf + stocks + row['receivable'], row['nav'], abs_tol=.011)):
            raise ValueError('Allocation exposure does not reconcile with NAV')
        result.append({'date': row['date'], 'cash': row['cash']/row['nav'],
            'etf': etf/row['nav'], 'stocks': stocks/row['nav'], 'receivable': row['receivable']/row['nav']})
    return result


def _normalize(report):
    if (report['schema'] != 1 or report['mode_order'] != list(MODES)
            or report['live_qualified'] is not False or report['unseen_validation'] is not False
            or report['auto_promote'] is not False or report['exit_mode'] != 'loss12'
            or report['candidate_count'] != 458 or report['execution_signal_lag_market_sessions'] != 1
            or report['benchmark']['mode'] != 'benchmark' or set(report['cases']) != set(MODES)):
        raise ValueError('Incomplete or unsupported allocation comparison')
    methods = {}
    for mode in MODES:
        case = report['cases'][mode]
        if case['mode'] != mode or not isinstance(case['allocation_decisions'], list):
            raise ValueError('Allocation case mode or decision records mismatch')
        account = _account(case['account'], case['summary'])
        methods[mode] = account | {'exposure': _exposure(account),
                                  'allocation_decisions': case['allocation_decisions']}
        if any(row['allocation_mode'] != mode for row in case['allocation_decisions']):
            raise ValueError('Allocation decision belongs to a different method')
        _decision_rows(methods[mode])
    benchmark = _account(report['benchmark']['account'], report['benchmark']['summary'])
    dates = [row['date'] for row in benchmark['daily']]
    if any([row['date'] for row in account['daily']] != dates for account in methods.values()):
        raise ValueError('Allocation and benchmark dates differ')
    return {'available': True, 'start': START, 'end': END, 'initial_cash': 1_000_000,
        'methods': methods, 'benchmark': benchmark, 'limitations': report.get('limitations', []),
        'performance': report.get('performance', report.get('summary', {}).get('performance', {}))}


@lru_cache(maxsize=2)
def _verified(root_name, output_name, signature):
    root, output = Path(root_name), Path(output_name)
    manifest = _verify_driver(output)
    if manifest.get('offline_identical') is not True or manifest.get('live_qualified') is not False:
        raise ValueError('Allocation offline reproduction has not completed')
    payload = (output / 'report.json').read_bytes()
    expected = {name: digest for name, digest, _ in signature[2]}[str((output / 'report.json').relative_to(root))]
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Allocation report changed while reading')
    result = _normalize(_decode(payload))
    if _signature(root, output) != signature:
        raise ValueError('Allocation sources changed during verification')
    result['source_verification'] = {'status': 'verified', 'file_count': len(signature[2]),
                                    'checked_at': datetime.now(timezone.utc).isoformat()}
    return result


def overview():
    root = Path(ROOT).resolve()
    output = root / CACHE
    if not (output / 'report.json').is_file() or not (output / 'manifest.json').is_file():
        return {'available': False, 'source_verification': {'status': 'pending'},
            'note': '剩餘資金配置研究尚未完成；完整帳本封存並通過驗證後才會顯示。'}
    try:
        return _verified(str(root), str(output), _signature(root, output))
    except (OSError, ValueError, KeyError, TypeError, ImportError, AttributeError) as exc:
        return {'available': False, 'source_verification': {'status': 'invalid'},
            'note': '這份資金配置結果尚未通過完整封存驗證，暫不顯示報酬。', 'verification_error': str(exc)}


def _ledger(account):
    frame = _trades(account)
    if not frame.empty:
        frame['原因'] = [ALLOCATION_REASONS.get(row['reason'], current)
                         for row, current in zip(account['trades'], frame['原因'])]
    return frame


def _allocation_rows(account):
    """Actual ETF fills, including partial execution, carry the engine reason."""
    frame = _ledger(account)
    return frame[frame['代號'] == '0050'].copy() if not frame.empty else frame


def _decision_rows(account):
    labels = {'buy_0050': '嘗試買0050', 'sell_0050': '嘗試賣0050',
              'hold_cash': '保留現金', 'retain_unknown': '訊號不足，維持配置'}
    rows = []
    dates = {row['date'] for row in account['daily']}
    for row in account['allocation_decisions']:
        if (row['date'] not in dates or row['action'] not in labels or row['market_state'] not in {'ON', 'OFF', 'UNKNOWN'}
                or row['allocation_mode'] not in MODES
                or any(type(row[field]) is not int or row[field] < 0 for field in
                       ('etf_qty_before', 'etf_qty_after', 'requested_qty', 'filled_qty'))
                or any(not _finite(row[field]) for field in ('cash_before', 'cash_after'))):
            raise ValueError('Invalid allocation decision record')
        rows.append({'執行日': row['date'], '訊號日': row['signal_date'],
            '大盤趨勢': {'ON': '高於120日均線', 'OFF': '不高於120日均線', 'UNKNOWN': '資料不足'}[row['market_state']],
            '配置指令': labels[row['action']], '請求股數': row['requested_qty'], '成交股數': row['filled_qty'],
            '指令前0050股數': row['etf_qty_before'], '指令後0050股數': row['etf_qty_after'],
            '指令前現金': row['cash_before'], '指令後現金': row['cash_after']})
    return pd.DataFrame(rows)


def render():
    with st.container(border=True):
        st.subheader('剩餘資金：買0050，還是留現金？')
        report = overview()
        if not report.get('available') or report.get('source_verification', {}).get('status') != 'verified':
            method = st.warning if report.get('source_verification', {}).get('status') == 'invalid' else st.info
            method(report.get('note', '資金配置研究尚未完成。'))
            return
        methods, benchmark = report['methods'], report['benchmark']
        st.caption(f"本金 NT$1,000,000｜{START}～{END}｜扣除手續費、交易稅與滑價｜歷史模擬，非實盤")
        st.write('三種方法使用相同選股候選與個股12%停損／63日出場，只改剩餘資金的去處。資產與可用現金不同，後來的成交股數、買到的股票也可能不同。')
        st.write('0050用來參與大盤報酬，不是避險。留現金可以降低股票曝險，也會錯過空手期間的漲幅。')
        rows = []
        for mode in (*MODES, 'benchmark'):
            summary = (benchmark if mode == 'benchmark' else methods[mode])['summary']
            rows.append({'方法': '0050持有' if mode == 'benchmark' else MODE_LABELS[mode],
                '期末資產（元）': round(summary['final_nav'], 2), '累積淨報酬': f"{summary['total_return']:.2%}",
                '相對0050（百分點）': f"{(summary['total_return']-benchmark['summary']['total_return'])*100:+.2f}",
                '年化報酬': f"{summary['cagr']:.2%}", '最大回撤': f"{summary['max_drawdown']:.2%}",
                '全部成本（元）': round(summary['costs']['total_cost'], 2), '成交筆數': summary['trade_count']})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        mode = st.selectbox('查看哪種資金配置', list(MODES), format_func=lambda value: MODE_LABELS[value], key='cash_allocation_method')
        selected = methods[mode]
        st.write(RULES[mode])
        st.caption('均線訊號只用前一交易日及更早的資料。訊號不足時不新增0050，也不因資料缺失強制賣出；賣出仍受成交量與漲跌停限制。')
        st.caption('個股停損以買入日還原收盤為起點，翌日開始依實際價格執行；12%不是保證成交損失。最多3檔個股，單次預算約當時淨資產的1/3，沒有強勢加碼。')
        compared = [(MODE_LABELS[mode], selected)]
        if mode != 'always_0050':
            compared.append((MODE_LABELS['always_0050'], methods['always_0050']))
        compared.append(('0050持有', benchmark))
        chart = pd.concat([pd.Series({point['date']: point['nav'] for point in account['daily']}, name=name)
                           for name, account in compared], axis=1)
        chart.index = pd.to_datetime(chart.index)
        st.line_chart(chart, y_label='淨資產（NT$）')
        st.caption('淨資產包含期末持股與應收股利／新股權利估值，並非全部可立即提領。')
        with st.expander('每年報酬與資金分配'):
            st.dataframe(pd.DataFrame([{'方法': name, **{str(row['year'])+('截至09/09' if str(row['year']) == '2026' else ''):
                f"{row['total_return']:.2%}" for row in account['summary']['annual']}} for name, account in compared]),
                hide_index=True, use_container_width=True)
            exposure = pd.DataFrame(selected['exposure']).set_index('date')
            exposure.index = pd.to_datetime(exposure.index)
            exposure = exposure.rename(columns={'cash': '現金', 'etf': '0050', 'stocks': '個股', 'receivable': '應收款與新股權利'})
            st.area_chart(exposure*100, y_label='占每日淨資產（%）')
            st.dataframe(pd.DataFrame([{'資產': name, '每日平均占比': f'{exposure[name].mean():.2%}',
                '期末占比': f'{exposure[name].iloc[-1]:.2%}'} for name in exposure]), hide_index=True, use_container_width=True)
            st.caption('占比按每日收盤估值計算；應收款與尚未交付的新股權利分開列出，不當作可用現金。')
        with st.expander('0050買賣與切換原因'):
            allocation = _allocation_rows(selected)
            if allocation.empty:
                st.info('這個方法沒有0050成交。')
            else:
                st.dataframe(allocation, hide_index=True, use_container_width=True)
                st.caption('每列是實際模擬成交；賣0050買個股與因趨勢轉弱賣0050分開標示。未成交指令不算已切換。')
            decisions = _decision_rows(selected)
            if not decisions.empty:
                st.write('每天是否買0050、賣0050或保留現金')
                st.dataframe(decisions, hide_index=True, use_container_width=True)
                st.caption('指令與成交分開列出；成交0股不代表已完成配置。未完成的0050賣出隔日重新看趨勢，轉強或訊號不足就不延續賣出。')
        with st.expander('全部買賣與CSV下載'):
            frame = _ledger(selected)
            if frame.empty:
                st.info('這個方法沒有成交紀錄。')
            else:
                st.dataframe(frame, hide_index=True, use_container_width=True)
                st.download_button('下載所選配置全部買賣 CSV', frame.to_csv(index=False).encode('utf-8-sig'),
                    file_name=f'cash-allocation-{mode}-trades.csv', mime='text/csv', key='cash_allocation_trades_csv')
            st.caption('整張參考普通收盤、零股參考最後價格，滑價另列成本；成交受日量限制，但最後揭示深度仍可能不足。「當日期末資產」不是盤中成交瞬間資產。')
        with st.expander('研究限制與來源驗證'):
            st.write('同一段歷史已反覆研究，沒有未見測試；這次比較不會自動替換正式策略。')
            for note in report['limitations']:
                st.caption(note)
            verified = report['source_verification']
            st.caption(f"完整來源驗證已通過，共 {verified['file_count']:,} 個檔案；選單只讀封存結果，不會啟動回測或抓取資料。檔案變動時重新驗證。")
