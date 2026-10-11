"""Audit non-executed slot rejections without changing financial journals.

The frozen runner independently rebuilds slot membership and cash resources
before invoking this execution auditor. This adapter cross-checks the rejected
order against those journals. All original plan timing, size, legal-limit and
cash-reservation checks still run; only the nonexistent execution children are
excluded from tape matching. It never treats a resource rejection as a halt.
"""
from collections import defaultdict
import inspect

from skills.all_entry_audit_index import private_auditor


POLICY = 'resource_slot_rejection_cross_ledger_v1'


def key(row):
    return row['date'], row['event_id'], row.get('side', 'buy')


def verified_rejections(account):
    plans = {key(p): p for p in account['tick_plans']}
    resources, slots, orders = defaultdict(list), defaultdict(list), defaultdict(list)
    for name, target in [('resource_plans', resources), ('slot_decisions', slots), ('orders', orders)]:
        for row in account.get(name, []):
            target[key(row)].append(row)
    accepted = set()
    for row in account['orders']:
        k = key(row)
        p = plans.get(k)
        if (row.get('failure') != 'resource_slots_locked' or not p
                or not p.get('planned_qty')):
            continue
        if len(resources[k]) != 1 or len(slots[k]) != 1 or len(orders[k]) != 1:
            raise ValueError('Resource rejection lacks unique independent journals')
        r, s = resources[k][0], slots[k][0]
        identity = ('date', 'event_id', 'stock_id', 'signal_date')
        occupied = s.get('occupied_before')
        union = set(s.get('held_before', [])) | set(s.get('opening_members', [])) | set(s.get('attempts_before', []))
        if (any(x.get(f) != p.get(f) for x in (row, r, s) for f in identity)
                or p['side'] != 'buy' or row.get('side') != 'buy' or row.get('channel') != 'event'
                or row.get('filled_qty') != 0 or s.get('filled_qty') != 0 or r.get('spent') != 0
                or r.get('filled_qty', 0) != 0 or s.get('attempted') is not False
                or r.get('failure') != 'resource_slots_locked' or s.get('failure') != 'resource_slots_locked'
                or type(row.get('requested_qty')) is not int
                or not 0 < row['requested_qty'] <= p['planned_qty']
                or r.get('planned_qty') != row['requested_qty']
                or r.get('locked_after') != r.get('locked_unused_before')
                or not isinstance(occupied, list) or occupied != sorted(union)
                or r.get('occupied_before') != occupied
                or not (p['stock_id'] in occupied or len(occupied) >= account['settings']['slots'])
                or not 0 <= r.get('budget', -1) <= p['reserved_cash'] + .011
                or r.get('opening_cash') != p['opening_cash']
                or any(key(t) == k for t in account['trades'])
                or any(t.get('date') == k[0] and t.get('event_id') == k[1]
                       and t.get('kind') == 'buy' for t in account['cash_ledger'])
                or any(c.get('event_id') == k[1] for c in account['cohorts'])):
            raise ValueError('Resource rejection differs from independently audited slot/cash journals')
        accepted.add(k)
    return accepted


def resource_auditor():
    module = private_auditor()
    # inspect reads the hash-verified original source, so restore its same quote
    # lookup optimization before adding the two narrow rejection branches.
    code = inspect.getsource(module.audit_range_children)
    old = "source.loc[source.date.eq(pd.Timestamp(day)) & source.stock_id.eq(sid)]"
    code = code.replace(old, '_indexed_quote_rows(source, day, sid)')
    changes = (
        ("    required = {(key, channel) for key, p in plans.items()\n",
         "    rejected = _verified_resource_rejections(account)\n"
         "    required = {(key, channel) for key, p in plans.items() if key not in rejected\n"),
        ("            if not p['planned_qty']:\n",
         "            if key in rejected:\n                continue\n"
         "            if not p['planned_qty']:\n"),
    )
    for old, new in changes:
        if code.count(old) != 1:
            raise ValueError('Frozen resource rejection audit sites differ')
        code = code.replace(old, new)
    module._verified_resource_rejections = verified_rejections
    exec(compile(code, module.__file__, 'exec'), module.__dict__)
    original = module.audit_range_execution
    def audit(*args, **kwargs):
        result = original(*args, **kwargs)
        rejected = verified_rejections(args[0])
        count = sum(bool(p[c+'_qty']) for p in args[0]['tick_plans'] if key(p) in rejected
                    for c in ('board', 'odd'))
        result['resource_slot_rejections_cross_checked'] = len(rejected)
        result['resource_rejected_children_not_submitted'] = count
        result['original_planned_children'] += count
        result['resource_slot_rejection_policy'] = POLICY
        return result
    return audit


def install_resource_rejection_audit(runner):
    if not hasattr(runner, '_all_entry_indexed_audit'):
        raise ValueError('Install the frozen private audit binding first')
    runner._all_entry_indexed_audit = resource_auditor()
