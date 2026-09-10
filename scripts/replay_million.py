#!/usr/bin/env python
"""Replay the predeclared NT$1m account, then reproduce it without the network."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.config import load_config
from app.file_lock import file_lock
from scripts.prepare_million_signals import verify_signals
from skills.million_replay import Replay
from skills.replay_market_feeds import ReplayMarketFeeds
from skills.replay_corporate_actions import CorporateActions

INPUT = ROOT / '.cache/million-replay-inputs'
SIGNALS = ROOT / '.cache/million-replay-signals'
OUTPUT = ROOT / '.cache/million-replay'
SPEC = ROOT / 'docs/prereg_million_replay_20260910.md'
OVERRIDES = ROOT / 'docs/million_replay_corporate_overrides_20260910.json'
CODE = ['scripts/replay_million.py', 'skills/million_replay.py',
        'skills/replay_market_feeds.py', 'skills/replay_corporate_actions.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def annual(daily):
    rows = []
    for year in sorted({r['date'][:4] for r in daily}):
        data = [r for r in daily if r['date'].startswith(year)]
        start = data[0]['opening_nav']
        peak, mdd = start, 0.
        for row in data:
            peak = max(peak, row['nav'])
            mdd = min(mdd, row['nav']/peak-1)
        rows.append(dict(year=year, start_nav=start, end_nav=data[-1]['nav'],
                         profit=data[-1]['nav']-start, total_return=data[-1]['nav']/start-1,
                         max_drawdown=mdd, partial_year=year=='2026'))
    return rows


def summarize(account):
    rows, trades = account['daily'], account['trades']
    last = rows[-1]
    years = (pd.Timestamp(last['date'])-pd.Timestamp(rows[0]['date'])).days/365.25
    return dict(start=rows[0]['date'], end=last['date'], initial_cash=account['settings']['initial_cash'],
        final_nav=last['nav'], profit=last['nav']-account['settings']['initial_cash'],
        total_return=last['total_return'], cagr=(1+last['total_return'])**(1/years)-1,
        max_drawdown=min(r['drawdown'] for r in rows), cash=last['cash'],
        market_value=last['market_value'], receivable=last['receivable'],
        trading_days=len(rows), trade_count=len(trades),
        buy_count=sum(t['side']=='buy' for t in trades), sell_count=sum(t['side']=='sell' for t in trades),
        stock_cohorts=len(account['cohorts']), costs={k:sum(t[k] for t in trades) for k in ('commission','tax','slippage','total_cost')},
        minimum_cash=min(r['cash'] for r in rows),
        partial_or_rejected_orders=dict(Counter(r['failure'] for r in account['orders'] if r.get('failure'))),
        final_holdings=[r for r in account['holdings'] if r['date']==last['date']],
        final_receivables=account['receivables'], annual=annual(rows),
        stale_final_holdings=last['stale_holdings'])


def audit(account):
    """Independently rebuild fees, daily cash and integer units from journal rows."""
    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Non-finite or invalid account number')
        return value

    def same(a, b, label, tolerance=.011):
        if not math.isclose(number(a), number(b), abs_tol=tolerance, rel_tol=0):
            raise ValueError(label)

    def integer(value, label, minimum=0):
        if type(value) is not int or value < minimum:
            raise ValueError(label)
        return value

    daily = account['daily']
    dates = [row['date'] for row in daily]
    if not dates or dates != sorted(set(dates)):
        raise ValueError('Daily dates must be unique and increasing')
    allowed = set(dates)
    for day in dates:
        if pd.Timestamp(day).strftime('%Y-%m-%d') != day:
            raise ValueError('Daily dates must be ISO dates')
    initial = number(account['settings']['initial_cash'])
    if initial <= 0:
        raise ValueError('Initial account cash must be positive')
    journals = {}
    for name in ('cash_ledger', 'trades', 'holdings', 'corporate_actions'):
        rows = account[name]
        if [row['date'] for row in rows] != sorted(row['date'] for row in rows):
            raise ValueError('Journal is not chronological: '+name)
        grouped = defaultdict(list)
        for row in rows:
            if row['date'] not in allowed:
                raise ValueError('Journal date outside account: '+name)
            grouped[row['date']].append(row)
        journals[name] = grouped
    cash, previous, peak = 0., initial, initial
    units = defaultdict(int)
    pending_shares = {}
    trade_sequence = 0
    initial_seen = False
    for row in daily:
        day = row['date']
        same(row['opening_nav'], previous, 'Opening NAV is not previous closing NAV')
        ledger_trades = []
        for movement in journals['cash_ledger'][day]:
            if movement['kind'] == 'initial_deposit':
                if initial_seen or day != dates[0] or cash != 0:
                    raise ValueError('Invalid initial cash deposit')
                same(movement['cash_change'], initial, 'Initial deposit differs from settings')
                initial_seen = True
            cash = float((Decimal(str(cash))+Decimal(str(number(movement['cash_change'])))).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
            same(cash, movement['cash_after'], 'Cash ledger does not reconcile')
            if cash < 0:
                raise ValueError('Cash ledger overdraft')
            if movement['kind'] in ('buy', 'sell'):
                ledger_trades.append(movement)
        if not initial_seen:
            raise ValueError('Initial deposit missing')
        same(cash, row['cash'], 'Daily cash differs from ledger')
        # Entitlements and deliveries occur before this day's trading.
        for action in journals['corporate_actions'][day]:
            sid, kind = action['stock_id'], action['kind']
            if 'entitled_qty' in action:
                integer(action['entitled_qty'], 'Invalid entitled shares', 1)
                if action['entitled_qty'] != units[sid]:
                    raise ValueError('Corporate entitlement differs from opening shares')
            if kind == 'split':
                multiplier = number(action['multiplier'])
                transformed = Decimal(units[sid])*Decimal(str(multiplier))
                if multiplier <= 0 or transformed != transformed.to_integral_value():
                    raise ValueError('Split does not produce exact integer shares')
                units[sid] = int(transformed)
                if integer(action['qty_after'], 'Invalid split shares') != units[sid]:
                    raise ValueError('Split share journal disagrees')
            elif kind == 'stock_dividend':
                key = (sid, action['action_id'])
                if key in pending_shares:
                    raise ValueError('Duplicate stock entitlement')
                quantity = Decimal(units[sid])*Decimal(str(number(action['shares_per_share'])))
                whole = int(quantity.to_integral_value(rounding=ROUND_FLOOR))
                if integer(action['whole_new_shares'], 'Invalid stock entitlement') != whole:
                    raise ValueError('Stock entitlement quantity disagrees')
                same(action['fractional_right'], float(quantity-whole), 'Fractional entitlement disagrees', 1e-10)
                pending_shares[key] = (whole, action['event_id'])
            elif kind == 'share_delivery':
                key = (sid, action['action_id'])
                expected = pending_shares.pop(key, None)
                if expected != (integer(action['qty'], 'Invalid delivered shares'), action['event_id']):
                    raise ValueError('Share delivery differs from locked entitlement')
                units[sid] += action['qty']
        trades = journals['trades'][day]
        if len(trades) != len(ledger_trades):
            raise ValueError('Trade and cash journals have different lengths')
        participation, daily_cost = defaultdict(int), 0.
        for trade, movement in zip(trades, ledger_trades):
            sid, side, channel = trade['stock_id'], trade['side'], trade['channel']
            if side not in ('buy', 'sell') or channel not in ('board', 'odd'):
                raise ValueError('Invalid trade side/channel')
            qty = integer(trade['qty'], 'Trade must contain positive integer shares', 1)
            if channel == 'board' and qty % 1000:
                raise ValueError('Board trade is not a full lot')
            if channel == 'odd' and qty >= 1000:
                raise ValueError('Odd trade exceeds one lot')
            trade_sequence += 1
            if trade['sequence'] != trade_sequence:
                raise ValueError('Trade sequence is inconsistent')
            price = number(trade['reference_price'])
            if price <= 0:
                raise ValueError('Trade price must be positive')
            raw_gross = Decimal(str(price))*qty
            gross = raw_gross.quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
            fee = max(Decimal(20), (raw_gross*Decimal('.001425')).quantize(Decimal(1), rounding=ROUND_HALF_UP))
            tax = ((raw_gross*Decimal('.001' if sid == '0050' else '.003')).quantize(Decimal(1), rounding=ROUND_FLOOR) if side == 'sell' else Decimal(0))
            slip = (raw_gross*Decimal('.0045')).quantize(Decimal(1), rounding=ROUND_CEILING)
            expected_cost = float(fee+tax+slip)
            expected_cash = float((gross if side == 'sell' else -gross)-fee-tax-slip)
            for field, expected in [('gross',float(gross)),('commission',float(fee)),('tax',float(tax)),('slippage',float(slip)),('total_cost',expected_cost),('cash_change',expected_cash)]:
                same(trade[field], expected, 'Trade fee/cash calculation differs: '+field)
            if any(trade[key] != movement[key] for key in ('stock_id','event_id','channel')) or movement['kind'] != side:
                raise ValueError('Trade and cash journal identity disagree')
            same(expected_cash, movement['cash_change'], 'Trade cash movement differs')
            same(trade['cash_after'], movement['cash_after'], 'Trade cash balance differs')
            participation[(sid, channel)] += qty
            if qty > integer(trade['requested_qty'], 'Invalid requested shares', 1) or participation[(sid,channel)] > integer(trade['capacity_qty'], 'Invalid capacity', 1):
                raise ValueError('Trade exceeds order/capacity')
            if side == 'buy' and sid != '0050' and number(trade['prior_avg_amount20']) < 50e6:
                raise ValueError('Entry violates liquidity filter')
            units[sid] += qty if side == 'buy' else -qty
            if units[sid] < 0 or units[sid] != integer(trade['remaining_shares'], 'Invalid remaining shares'):
                raise ValueError('Trade shares do not reconcile')
            daily_cost += expected_cost
        actual, assets = {}, 0.
        for holding in journals['holdings'][day]:
            sid = holding['stock_id']
            if sid in actual:
                raise ValueError('Duplicate daily holding')
            actual[sid] = integer(holding['qty'], 'Holding must contain positive integer shares', 1)
            price = number(holding['price'])
            if price <= 0 or holding['mark_date'] > day:
                raise ValueError('Invalid holding valuation')
            value = actual[sid]*price
            same(value, holding['market_value'], 'Holding value differs', .06)
            assets += value
        if actual != {sid:qty for sid,qty in units.items() if qty}:
            raise ValueError('Daily holdings differ from trades/splits/deliveries')
        same(daily_cost, row['cost'], 'Daily cost differs from trades')
        same(assets, row['market_value'], 'Daily market value differs', .06)
        same(row['nav'], cash+assets+number(row['receivable']), 'Daily balance sheet does not reconcile', .06)
        same(row['nav'], row['opening_nav']+row['market_pnl']+row['dividend_entitlement']+row['execution_basis_pnl']-row['cost'], 'Daily P&L does not reconcile', .06)
        same(row['daily_return'], row['nav']/previous-1, 'Daily return differs', 1e-10)
        same(row['total_return'], row['nav']/initial-1, 'Total return differs', 1e-10)
        peak = max(peak,row['nav'])
        same(row['drawdown'], row['nav']/peak-1, 'Drawdown differs', 1e-10)
        previous = row['nav']
    return dict(cash_ledger_reconciled=True, all_daily_nav_reconciled=True,
        integer_shares=True, no_overdraft=True, capacity_and_order_limits=True,
        fees_recomputed=True, trade_cash_reconciled=True, daily_shares_reconstructed=True,
        opening_nav_continuity=True, trading_days=len(daily), cash_movements=len(account['cash_ledger']))


def input_sources():
    verify_signals()
    return {str(p.relative_to(ROOT)):sha(p) for p in [INPUT/'manifest.json',SIGNALS/'manifest.json',SPEC,OVERRIDES]}


def run_accounts(*, offline):
    signals = json.loads((SIGNALS/'signals.json').read_text())
    entries = signals['entries']
    selected = {'0050'}|{e['members'][0] for e in entries}
    quotes = pd.read_parquet(INPUT/'quotes.parquet')
    quotes = quotes[quotes.stock_id.isin(selected)]
    companies = pd.read_parquet(INPUT/'companies.parquet')
    calendar = pd.read_parquet(INPUT/'calendar.parquet')
    events = pd.read_parquet(INPUT/'events.parquet')
    days = calendar.loc[calendar.is_open,'date']
    token = None if offline else load_config().finmind_token
    feeds = ReplayMarketFeeds(INPUT/'execution-feeds',offline=offline,token=token)
    corp = CorporateActions(events,INPUT/'dividends',token,offline=offline,
                            overrides=json.loads(OVERRIDES.read_text())['overrides'])
    accounts = {}
    for name in ('strategy','benchmark'):
        begin = time.perf_counter()
        class ObservedReplay(Replay):
            last_progress_month = None

            def corporate_day(self, day):
                month = str(day.date())[:7]
                if not offline and month != self.last_progress_month:
                    print(f'{name}: preparing {month}',flush=True)
                    self.last_progress_month = month
                return super().corporate_day(day)
        accounts[name] = ObservedReplay(quotes,companies,days,entries,feeds,corp,benchmark=name=='benchmark').run()
        print(f'{name} {"offline" if offline else "source preparation"} completed in {time.perf_counter()-begin:.2f}s',flush=True)
        audit(accounts[name])
    return accounts, feeds.manifest(), corp.manifest()


def verify_report(output=OUTPUT):
    info = json.loads((output/'manifest.json').read_text())
    expected_sources = {str(p.relative_to(ROOT)) for p in (INPUT/'manifest.json',SIGNALS/'manifest.json',SPEC,OVERRIDES)}
    if (info.get('schema')!=1 or info.get('offline_identical') is not True
            or info.get('live_qualified') is not False
            or set(info.get('source_files_sha256',{}))!=expected_sources
            or set(info.get('files_sha256',{}))!={'report.json','summary.json','prepared-accounts.json'}):
        raise ValueError('Incomplete replay evidence manifest')
    if info['code_sha256'] != {name:sha(ROOT/name) for name in CODE}:
        raise ValueError('Replay code changed')
    if info['runtime_versions'] != dict(python=platform.python_version(),pandas=pd.__version__):
        raise ValueError('Replay runtime changed')
    for name,digest in info['source_files_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Replay source changed: '+name)
    for name,digest in info['files_sha256'].items():
        if sha(output/name)!=digest:
            raise ValueError('Replay output changed: '+name)
    input_sources()
    feeds = ReplayMarketFeeds(INPUT/'execution-feeds',offline=True).manifest()
    if feeds != info['execution_feeds']:
        raise ValueError('Execution source manifest changed')
    for name,digest in info['corporate_sources']['files_sha256'].items():
        if sha(INPUT/'dividends'/name)!=digest:
            raise ValueError('Dividend source changed')
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-and-replay',action='store_true',help='Fetch only required execution evidence, then repeat offline')
    parser.add_argument('--offline-replay',action='store_true',help='Recompute and compare the sealed ledger; no fetch permitted')
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with file_lock(OUTPUT/'.replay.lock',timeout=10):
        if args.verify:
            verify_report()
            print('Replay source, code and output hashes verified')
            return
        if args.offline_replay:
            verify_report()
            before = time.perf_counter()
            accounts,_,_ = run_accounts(offline=True)
            old = json.loads((OUTPUT/'report.json').read_text())
            if any(encoded(accounts[key]) != encoded(old[key]) for key in accounts):
                raise ValueError('Offline ledger differs from sealed output')
            print(f'Identical offline reproduction in {time.perf_counter()-before:.2f}s')
            return
        if not args.prepare_and_replay:
            parser.error('Choose --prepare-and-replay, --offline-replay, or --verify')
        if (OUTPUT/'manifest.json').exists():
            raise ValueError('A sealed replay exists; use --offline-replay or a new explicit output directory')
        sources = input_sources()
        code = {name:sha(ROOT/name) for name in CODE}
        begin = time.perf_counter()
        accounts,feeds,corp = run_accounts(offline=False)
        elapsed = time.perf_counter()-begin
        write(OUTPUT/'prepared-accounts.json',accounts)
        before = time.perf_counter()
        repeated,feeds_after,corp_after = run_accounts(offline=True)
        offline_elapsed = time.perf_counter()-before
        if encoded(accounts)!=encoded(repeated) or feeds!=feeds_after or corp['files_sha256']!=corp_after['files_sha256']:
            raise ValueError('Offline reproduction differs from preparation')
        if sources!=input_sources() or code!={name:sha(ROOT/name) for name in CODE}:
            raise ValueError('Inputs or code changed while replaying')
        summary = {name:summarize(value) for name,value in accounts.items()}
        summary['excess_total_return'] = summary['strategy']['total_return']-summary['benchmark']['total_return']
        report = dict(schema=1,**accounts,summary=summary,
            strategy_name='族群領先股・趨勢通過才進場・3部位／63交易日',
            audit={name:audit(value) for name,value in accounts.items()},
            performance=dict(preparation_and_replay_seconds=elapsed,offline_two_accounts_seconds=offline_elapsed),
            limitations=['條件式日資料成交模擬，普通與零股皆無完整委託簿，不保證可實際成交。',
                '使用舊固定公司名冊，存在存活者偏誤；這段歷史已被多次研究，不是未見資料驗證。',
                '策略已固定；資料延伸與現金股份制使結果不同於舊合成單位回測。',
                '股利毛額，未扣個人所得稅及補充保費；不參與現金增資認購。',
                '成交量為容量檢查，普通盤參考價與零股最後成交价另扣每邊0.45%滑價。',
                '2026-09-09以收盤市值結帳，持倉沒有假設全部當日賣出，未扣未來清倉成本。'],
            source_links=['https://www.twse.com.tw/zh/products/system/trading.html',
                'https://www.tpex.org.tw/zh-tw/mainboard/trading/rules/odd-lot.html',
                'https://finmind.github.io/tutor/TaiwanMarket/Fundamental/'])
        write(OUTPUT/'report.json',report)
        write(OUTPUT/'summary.json',summary)
        manifest = dict(schema=1,created_at=datetime.now(timezone.utc).isoformat(),
            code_sha256=code,source_files_sha256=sources,
            runtime_versions=dict(python=platform.python_version(),pandas=pd.__version__),
            files_sha256={name:sha(OUTPUT/name) for name in ['report.json','summary.json','prepared-accounts.json']},
            execution_feeds=feeds,corporate_sources=corp,offline_identical=True,
            live_qualified=False,performance=report['performance'])
        write(OUTPUT/'manifest.json',manifest)
        verify_report()
        print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
