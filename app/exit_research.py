"""Read and display sealed scenario-exit research; never launch research work."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import re

import pandas as pd
import streamlit as st

from skills.exit_policy import MODE_LABELS, MODES, REASON_LABELS


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('.cache/exit-research')
DRIVER = 'scripts/research_exit_scenarios.py'
SPEC = 'docs/prereg_exit_scenarios_20260910.md'
INPUT_MANIFEST = '.cache/exit-research-inputs/manifest.json'
VERIFICATION_DIRECTORIES = ('.cache/exit-research-inputs/dividends',)
START, END = '2022-01-03', '2026-09-09'
RULES = {
    'fixed63': '持有滿63個市場交易日後開始賣出。',
    'loss12': '價格訊號較買入日跌12%開始賣；未觸發則63日賣。',
    'trail20_12': '曾上漲20%後，從持有期高點回落12%開始賣；未觸發則63日賣。',
    'weak20': '持股後連續兩日收盤低於20日均線，且20日報酬落後0050時開始賣；否則63日賣。',
    'market_weak': '0050連續兩天未站上120日均線，且個股20日報酬落後0050時開始賣；否則63日賣。',
    'trend126': '63日前不提早賣；之後只有趨勢強才逐日延長，最晚126日開始賣。',
    'adaptive': '先檢查126日上限，再依序檢查12%停損、移動停利、大盤與個股轉弱、個股破線；63日後強勢才延長。',
}
TRADE_REASONS = REASON_LABELS | {
    'initial_allocation': '初始配置0050', 'event_entry': '族群領先訊號進場',
    'leader_entry': '族群領先訊號進場', 'fund_stock': '賣0050準備買股',
    'idle_cash': '閒置資金投入0050',
    'fund_event': '賣0050準備買股', 'scheduled_exit': '持有滿63日開始賣出',
    'reinvest_exit': '出場資金投入0050', 'reinvest_cash': '剩餘現金投入0050',
    'reinvest_dividend': '股利投入0050', 'cash_reinvestment': '現金投入0050',
}


def _decode(payload):
    def invalid(value):
        raise ValueError('Non-finite research JSON: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate research JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(payload, parse_constant=invalid, object_pairs_hook=unique)


def _read(path):
    return _decode(Path(path).read_bytes())


def _file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Invalid verification source path')
    root = root.resolve(strict=True)
    source = root / relative
    resolved = source.resolve(strict=True)
    if not resolved.is_relative_to(root) or source.is_symlink() or not resolved.is_file():
        raise ValueError('Invalid verification source file: ' + relative)
    return resolved


def _stamp(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _signature(root, output):
    """Cheap complete inventory stamp; no large price-file hashing on rerender."""
    manifest_path = output / 'manifest.json'
    before = _stamp(manifest_path)
    payload = manifest_path.read_bytes()
    manifest = _decode(payload)
    inventory = manifest['verification_files_sha256']
    required = {DRIVER, SPEC, INPUT_MANIFEST, str((output / 'report.json').relative_to(root))}
    if (not isinstance(inventory, dict) or not required.issubset(inventory)
            or any(not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)
                   for digest in inventory.values())):
        raise ValueError('Incomplete source verification inventory')
    files = tuple((name, digest, _stamp(_file(root, name))) for name, digest in sorted(inventory.items()))
    # The driver also checks source-directory inventories (for example newly
    # added dividend evidence). Watch containing directories so additions and
    # renames cannot reuse a previous success while all old files stay intact.
    watched = manifest['verification_directories']
    if not isinstance(watched, list) or len(watched) != len(set(watched)) or set(watched) != set(VERIFICATION_DIRECTORIES):
        raise ValueError('Incomplete source-directory verification inventory')
    parents = {_file(root, name).parent for name in inventory}
    for name in watched:
        directory = root / name
        if directory.is_symlink() or not directory.is_dir() or not directory.resolve().is_relative_to(root):
            raise ValueError('Invalid verification source directory: ' + name)
        parents.add(directory.resolve())
    parents = sorted(parents)
    directories = tuple((str(parent.relative_to(root)), _stamp(parent)) for parent in parents)
    after = _stamp(manifest_path)
    if before != after:
        raise ValueError('Research manifest changed while reading')
    # Runtime changes must not reuse a previous verification success.
    runtime_names = manifest.get('context', {}).get('runtime_versions',
        manifest.get('runtime_versions', {'python': '', 'pandas': ''}))
    runtime = tuple((name, platform.python_version() if name == 'python' else version(name))
                    for name in sorted(runtime_names))
    return (before, runtime, files, hashlib.sha256(payload).hexdigest(), directories)


_driver_stamp = None


def _verify_driver(output):
    """The driver verifier is read-only; never call its run/main functions."""
    global _driver_stamp
    current = _stamp(ROOT / DRIVER)
    if _driver_stamp is not None and _driver_stamp != current:
        raise ValueError('Verification code changed; restart the workbench before verifying a new seal')
    module = importlib.import_module('scripts.research_exit_scenarios')
    _driver_stamp = current
    return module.verify_report(output=output)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _account(account, summary):
    """Display validation in addition to the driver's full source/account audit."""
    fields = ('initial_cash', 'final_nav', 'profit', 'total_return', 'cagr', 'max_drawdown',
              'cash', 'market_value', 'receivable')
    if (any(not _finite(summary[key]) for key in fields)
            or summary['initial_cash'] != 1_000_000 or summary['start'] != START or summary['end'] != END
            or summary['final_nav'] <= 0 or not -1 <= summary['max_drawdown'] <= 0
            or not math.isclose(summary['final_nav'], summary['initial_cash']*(1+summary['total_return']), abs_tol=.011)
            or not math.isclose(summary['final_nav'], summary['cash']+summary['market_value']+summary['receivable'], abs_tol=.011)):
        raise ValueError('Invalid scenario account summary')
    daily, trades = account['daily'], account['trades']
    if (not isinstance(daily, list) or not daily or not isinstance(trades, list)
            or summary['trade_count'] != len(trades) or summary['trading_days'] != len(daily)):
        raise ValueError('Incomplete scenario account records')
    dates = [row['date'] for row in daily]
    if dates != sorted(set(dates)) or dates[0] != START or dates[-1] != END:
        raise ValueError('Invalid scenario date coverage')
    if any(not _finite(row['nav']) or row['nav'] <= 0 for row in daily):
        raise ValueError('Invalid scenario NAV observations')
    if not math.isclose(daily[-1]['nav'], summary['final_nav'], abs_tol=.011):
        raise ValueError('Summary and daily asset values differ')
    allowed = set(dates)
    for row in trades:
        if (row['date'] not in allowed or row['side'] not in {'buy', 'sell'}
                or not isinstance(row['stock_id'], str) or not re.fullmatch(r'\d{4}', row['stock_id'])
                or type(row['qty']) is not int or row['qty'] <= 0
                or any(not _finite(row[k]) for k in ('reference_price', 'total_cost', 'cash_change', 'cash_after'))):
            raise ValueError('Invalid scenario trade record')
    annual = summary['annual']
    if (not isinstance(annual, list) or [str(row['year']) for row in annual] != ['2022', '2023', '2024', '2025', '2026']
            or any(not _finite(row['total_return']) for row in annual)):
        raise ValueError('Incomplete scenario annual observations')
    return account | {'summary': summary}


def _normalize(report):
    """Adapt the driver's sealed report to a small, stable display contract."""
    if (report['schema'] != 1 or report['mode_order'] != list(MODES)
            or report['live_qualified'] is not False or report['unseen_validation'] is not False
            or report['auto_promote'] is not False or report['remaining_cash_asset'] != '0050'
            or report['candidate_count'] != 458 or report['execution_signal_lag_market_sessions'] != 1
            or report['benchmark']['mode'] != 'benchmark' or set(report['cases']) != set(MODES)):
        raise ValueError('Incomplete or unsupported scenario comparison')
    methods = {}
    for mode in MODES:
        case = report['cases'][mode]
        if case['mode'] != mode:
            raise ValueError('Scenario case mode mismatch')
        counts = case['summary']['reason_counts']
        if (not isinstance(counts, dict) or any(reason not in REASON_LABELS or type(count) is not int or count < 0
                                              for reason, count in counts.items())):
            raise ValueError('Invalid exit reason counts')
        methods[mode] = _account(case['account'], case['summary']) | {'exit_reason_counts': counts}
    benchmark = _account(report['benchmark']['account'], report['benchmark']['summary'])
    dates = [r['date'] for r in benchmark['daily']]
    if any([r['date'] for r in account['daily']] != dates for account in methods.values()):
        raise ValueError('Scenario and benchmark dates differ')
    return {'available': True, 'start': START, 'end': END, 'initial_cash': 1_000_000,
            'methods': methods, 'benchmark': benchmark, 'limitations': report.get('limitations', [])}


@lru_cache(maxsize=2)
def _verified(root_name, output_name, signature):
    root, output = Path(root_name), Path(output_name)
    manifest = _verify_driver(output)
    if manifest.get('offline_identical') is not True or manifest.get('live_qualified') is not False:
        raise ValueError('Offline reproduction has not completed')
    payload = (output / 'report.json').read_bytes()
    expected = dict((name, digest) for name, digest, _ in signature[2])[str((output / 'report.json').relative_to(root))]
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Scenario report changed while reading')
    result = _normalize(_decode(payload))
    if _signature(root, output) != signature:
        raise ValueError('Scenario sources changed during verification')
    result['source_verification'] = {'status': 'verified', 'file_count': len(signature[2]),
        'checked_at': datetime.now(timezone.utc).isoformat()}
    return result


def overview():
    """Fail closed for missing, incomplete, changed or unreadable research."""
    root = Path(ROOT).resolve()
    output = root / CACHE
    missing = {'available': False, 'source_verification': {'status': 'pending'},
               'note': '情境出場研究尚未完成；完整帳本封存並通過驗證後才會顯示。'}
    if not (output / 'report.json').is_file() or not (output / 'manifest.json').is_file():
        return missing
    try:
        signature = _signature(root, output)
        return _verified(str(root), str(output), signature)
    except (OSError, ValueError, KeyError, TypeError, ImportError, AttributeError) as exc:
        return {'available': False, 'source_verification': {'status': 'invalid'},
                'note': '這份情境出場結果尚未通過完整封存驗證，暫不顯示報酬。',
                'verification_error': str(exc)}


def _pct(value):
    return f'{value:.2%}'


def _trades(account):
    nav = {row['date']: row['nav'] for row in account['daily']}
    return pd.DataFrame([{'成交日': row['date'], '買賣': {'buy': '買入', 'sell': '賣出'}[row['side']],
        '代號': row['stock_id'], '名稱': row.get('name', ''), '股數': row['qty'],
        '交易別': {'board': '整張', 'regular': '整張', 'odd': '零股'}.get(row.get('channel'), row.get('channel', '')),
        '成交參考價': row['reference_price'], '手續費': row.get('commission'), '交易稅': row.get('tax'),
        '滑價現金成本': row.get('slippage'), '合計成本': row['total_cost'],
        '現金收付': row['cash_change'], '成交後現金': row['cash_after'], '當日期末資產': nav[row['date']],
        '原因': TRADE_REASONS.get(row['reason'], row['reason']), '訊號日': row.get('signal_date'),
        '零股最後買量': row.get('odd_bid_qty'), '零股最後賣量': row.get('odd_ask_qty'),
        '零股對手量檢查': ('' if row.get('channel') != 'odd' else '無最後揭示量'
            if row.get('odd_ask_qty' if row['side']=='buy' else 'odd_bid_qty') is None else '高於：僅日量假設'
            if row['qty'] > row['odd_ask_qty' if row['side']=='buy' else 'odd_bid_qty'] else '未高於'),
        '事件': row.get('event_id')} for row in account['trades']])


def render():
    with st.container(border=True):
        st.subheader('情境出場實測')
        report = overview()
        if not report.get('available') or report.get('source_verification', {}).get('status') != 'verified':
            if report.get('source_verification', {}).get('status') == 'invalid':
                st.warning(report.get('note', '這份情境出場結果尚未通過完整封存驗證，暫不顯示報酬。'))
            else:
                st.info(report.get('note', '情境出場研究尚未完成。'))
            return
        methods, benchmark = report['methods'], report['benchmark']
        st.caption(f"本金 NT$1,000,000｜{report['start']}～{report['end']}｜已扣手續費、交易稅與滑價｜歷史模擬，非實盤")
        st.write('進場候選相同，只改出場規則。提早賣出會釋放名額，後來買到的股票與投入金額也可能不同。')
        rows = []
        for mode in (*MODES, 'benchmark'):
            summary = (benchmark if mode == 'benchmark' else methods[mode])['summary']
            rows.append({'方法': '0050持有' if mode == 'benchmark' else MODE_LABELS[mode],
                '期末資產（元）': round(summary['final_nav'], 2), '累積淨報酬': _pct(summary['total_return']),
                '相對0050（百分點）': f"{(summary['total_return']-benchmark['summary']['total_return'])*100:+.2f}",
                '年化報酬': _pct(summary['cagr']), '最大回撤': _pct(summary['max_drawdown']),
                '全部成本（元）': round(summary['costs']['total_cost'], 2), '成交筆數': summary['trade_count'],
                '公司行動限制': '含權證延後出售假設' if summary.get('restricted_certificate_actions') else '—'})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        mode = st.selectbox('查看哪個出場方法', list(MODES), format_func=lambda value: MODE_LABELS[value], key='exit_method')
        selected = methods[mode]
        st.write(RULES[mode])
        if selected['summary'].get('restricted_certificate_actions'):
            st.warning('此方法曾持有已發放增資權利證書的部位；本研究等換成普通股才賣。'
                       '等待期間以普通股價代理估值，並持續占用持股名額；結果不是實際交易的保守下限。')
        if mode in ('trend126', 'adaptive'):
            st.caption('強勢條件：0050在120日均線上、個股不低於20日均線且均線比5個交易日前高、20日漲幅贏0050；每一項都要符合。')
        st.caption('使用前一交易日及以前的訊號，成交另受漲跌停與成交量限制；門檻不是保證成交價。賣股後資金依原規則投入0050，不代表整戶轉為現金。')
        st.caption('停損與停利百分比以買入日還原收盤為起點；實際成交價格、費用及公司行動後的帳戶損益另計。')
        curves = {MODE_LABELS[mode]: selected['daily']}
        if mode != 'fixed63':
            curves[MODE_LABELS['fixed63']] = methods['fixed63']['daily']
        curves['0050持有'] = benchmark['daily']
        chart = pd.concat([pd.Series({point['date']: point['nav'] for point in data}, name=name)
                           for name, data in curves.items()], axis=1)
        chart.index = pd.to_datetime(chart.index)
        st.line_chart(chart, y_label='淨資產（NT$）')
        st.caption('完整每日淨資產；期末含持股市值與應收股利／新股權利估值，未假設全部可立即提領。')
        fraction_amount = selected['summary'].get('unverified_fractional_cash_amount', 0)
        if fraction_amount:
            st.caption(f'其中 NT${fraction_amount:,.0f} 為付款日與淨額未核實的畸零股毛額應收，沒有當成可用現金。')
        with st.expander('每年報酬與退出原因', expanded=False):
            annual_rows = []
            compared = [(MODE_LABELS[mode], selected)]
            if mode != 'fixed63':
                compared.append((MODE_LABELS['fixed63'], methods['fixed63']))
            compared.append(('0050持有', benchmark))
            for name, account in compared:
                annual_rows.append({'方法': name, **{str(row['year'])+('截至09/09' if str(row['year'])=='2026' else ''):
                    _pct(row['total_return']) for row in account['summary']['annual']}})
            st.dataframe(pd.DataFrame(annual_rows), hide_index=True, use_container_width=True)
            counts = selected.get('exit_reason_counts', {})
            if counts:
                st.dataframe(pd.DataFrame([{'首次退出原因': TRADE_REASONS.get(reason, reason), '部位數': count}
                    for reason, count in counts.items()]), hide_index=True, use_container_width=True)
                st.caption('按部位的首次退出指令計數；成交受阻可能分多天、分整張與零股賣出。')
            else:
                st.info('這個方法沒有觸發退出指令。')
            waits = selected['summary'].get('exit_waits', [])
            if waits:
                st.dataframe(pd.DataFrame([{'代號': row['stock_id'], '原因': TRADE_REASONS.get(row['reason'], row['reason']),
                    '訊號日': row['signal_date'], '開始賣出日': row['target_date'], '首次成交日': row['first_fill_date'],
                    '部位含配股全數結束': row['complete_exit_date'] or '期末仍未結束',
                    '開始賣到首次成交（交易日）': row['target_to_first_fill_sessions']}
                    for row in waits]), hide_index=True, use_container_width=True)
                st.caption('「部位含配股全數結束」包含晚交付新股；可能晚於原持股售完日期。')
        with st.expander('全部買賣與CSV下載'):
            frame = _trades(selected)
            if frame.empty:
                st.info('這個方法沒有成交紀錄。')
            else:
                st.dataframe(frame, hide_index=True, use_container_width=True)
                st.download_button('下載所選方法全部買賣 CSV', frame.to_csv(index=False).encode('utf-8-sig'),
                    file_name=f'exit-{mode}-trades.csv', mime='text/csv', key='exit_trades_csv')
            st.caption('整張參考普通收盤、零股參考最後價格；滑價另列現金成本。「當日期末資產」不是盤中成交瞬間資產。')
            depth = selected['summary'].get('depth_audit', {})
            if depth.get('exceeds_last_opposing_depth_count') or depth.get('missing_opposing_depth_count'):
                st.caption(f"零股共 {depth['odd_trade_count']} 筆，其中 {depth.get('exceeds_last_opposing_depth_count', 0)} 筆超過最後揭示對手量、{depth.get('missing_opposing_depth_count', 0)} 筆缺最後揭示量；日量限制不代表該時點保證成交。")
        with st.expander('研究限制與來源驗證'):
            st.write('這段歷史已反覆研究，沒有未見測試；不會因為這次排名而自動替換正式策略。')
            for note in report['limitations']:
                st.caption(note)
            verified = report['source_verification']
            st.caption(f"完整來源驗證已通過，共 {verified['file_count']:,} 個檔案；選單只讀封存結果，不會啟動回測或抓取資料。檔案變動時重新驗證。")
