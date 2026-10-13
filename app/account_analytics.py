"""Supplement sealed accounts with realized, entry-cohort statistics.

This does not replay trades or promote historical research to paper/live trading.
Cash flows already include fees, tax and slippage. Distributions follow their
corporate-action entitlement event, even when paid after the position was sold.
"""
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

SUPPLEMENT = 'artifacts/forward_simulation/account_analytics_20261013.json'
STUDIES = ('legacy_return_comparison_20261012', 'top3_validation_20261012')


def amount(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError('帳戶金額含有無效數字')
    return result


def realized_metrics(account):
    flows, trade_flows, trades = (defaultdict(Decimal) for _ in range(3))
    actions = account['corporate_actions']
    deposits = Decimal(0)
    for row in account['cash_ledger']:
        value = amount(row['cash_change'])
        if row['kind'] == 'initial_deposit':
            deposits += value
            continue
        event = row.get('event_id')
        if not event:
            matches = [a for a in actions
                if a.get('date') == row['date'] and a.get('stock_id') == row['stock_id']
                and ((row['kind'] == 'dividend_payment' and a.get('kind') == 'payment'
                      and a.get('action_id') == row['action_id'] and amount(a['amount']) == value)
                     or (row['kind'] == 'fractional_share_payment' and a.get('kind') == 'share_delivery'))]
            if len(matches) != 1:
                raise ValueError('分配現金無法唯一歸屬原進場批次')
            action = matches[0]
            event = action['event_id']
            # Use the sealed cash amount: corporate terms can require whole-NTD
            # flooring. Recomputing fraction * face here would undo that rule.
        flows[event] += value
        if row['kind'] in ('buy', 'sell'):
            trade_flows[event] += value
    for row in account['trades']:
        trades[row['event_id']] += amount(row['cash_change'])
    if trades != trade_flows:
        raise ValueError('成交與現金日誌不同')

    positions = account['ending_inventory']['positions']
    open_events = {r['event_id'] for r in positions}
    pending = {r['event_id'] for r in account['receivables']}
    final = account['daily'][-1]
    # These studies have cash receivables only. Other claims require an explicit
    # valuation adapter, never an implicit zero or cash fallback.
    if any(r['kind'] != 'cash' for r in account['receivables']):
        raise ValueError('尚未支援此應收權益估值')
    total = (sum(flows.values()) + sum(amount(r['market_value']) for r in positions)
             + sum(amount(r['amount']) for r in account['receivables']))
    if (abs(total - (amount(final['nav']) - deposits)) > Decimal('.011')
            or abs(deposits + sum(flows.values()) - amount(final['cash'])) > Decimal('.011')):
        raise ValueError('批次現金損益與期末帳戶不平')
    cohorts = {r['event_id']: r for r in account['cohorts']}
    benchmark = not cohorts and set(flows) == {'benchmark'}
    if benchmark:
        if any(t['side'] == 'sell' for t in account['trades']):
            raise ValueError('買進持有基準有未支援的賣出')
        closed = set()
    else:
        if (set(flows) != set(cohorts) or len(cohorts) != len(account['cohorts'])
                or {e for e, c in cohorts.items() if not c.get('exit_date')} != open_events):
            raise ValueError('批次與持股結清狀態不同')
        closed = set(cohorts) - open_events - pending
        if any(cohorts[e]['bought_qty'] <= 0 for e in closed):
            raise ValueError('未入金批次不可列入勝率')
    pnl = [flows[e].quantize(Decimal('.01'), rounding=ROUND_HALF_UP) for e in closed]
    wins = sorted((v for v in pnl if v > 0), reverse=True)
    losses = [-v for v in pnl if v < 0]
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    def ratio(a, b):
        return float(a / b) if a is not None and b else None
    return dict(wins=len(wins), losses=len(losses), breakeven=sum(v == 0 for v in pnl),
        settled=len(closed), open=len(open_events), pending_only=len(pending-open_events),
        win_rate=len(wins)/len(pnl) if pnl else None,
        average_win=float(avg_win) if avg_win is not None else None,
        average_loss=float(avg_loss) if avg_loss is not None else None,
        payoff_ratio=ratio(avg_win, avg_loss), profit_factor=ratio(sum(wins), sum(losses)),
        top5_profit_share=ratio(sum(wins[:5]), sum(wins)), realized_profit=float(sum(pnl)),
        mean_realized_pnl=ratio(sum(pnl), len(pnl)),
        not_applicable='買進持有，無已結清批次' if benchmark else None)


def supplement(identifier, publication, publication_sha, root):
    """Read a published supplement; require the exact original account bindings."""
    root = Path(root)
    path = root / SUPPLEMENT
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('績效補充指紋不符')
    data = json.loads(raw)
    if (data['schema'] != 'account_analytics_v1' or data['live_qualified'] is not False
            or data['calculator_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('績效補充計算版本不符')
    study = data['studies'][identifier]
    expected = {k: r['sha256'] for k, r in publication['cases'].items() if r['completed']}
    if (study['publication_sha256'] != publication_sha
            or {k: r['source_sha256'] for k, r in study['cases'].items()} != expected):
        raise ValueError('績效補充與回測版本不同')
    return study['cases']
