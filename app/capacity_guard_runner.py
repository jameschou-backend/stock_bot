"""Bounded observer with a per-match guard; sealed legacy code stays unchanged."""
from datetime import time
from pathlib import Path
import time as walltime

from app import capacity_forward as policy, capacity_forward_runner as legacy
from app import capacity_source_guard as guard
from app import forward_journal as j, forward_portfolio as p, forward_market_quotes as market
from app.file_lock import file_lock


def observe(root, clock, sleeper, seconds, fetcher):
    paths = guard.verify(root)
    refresh = legacy.refresh(paths, clock)
    queries, ready, outputs = [], {}, []
    for role, path in paths.items():
        rows = policy.verify(path, role)
        if role != 'benchmark' and policy.risk(rows)['blocked']:
            policy.pause(path, True, '整戶風險條件限制新增；保留既有出場委託', clock)
            rows = policy.verify(path, role)
        audit = legacy.corporate.inspect(path, clock=clock, rows=rows)
        if audit['blocked']:
            outputs.append(dict(role=role, status='blocked', reason='公司行動來源缺漏或待核對事件'))
            continue
        day = str(clock().astimezone(p.TZ).date())
        orders = [o for o in p.state(rows)['orders'].values()
                  if not o['closed'] and o['filled'] < o['qty'] and o['session'] == day]
        lookup = legacy.base.markets(sorted({o['stock_id'] for o in orders}))
        for o in orders:
            key = (lookup[o['stock_id']], o['stock_id'], o['channel'])
            if key not in queries:
                queries.append(key)
        ready[role] = path
        outputs.append(dict(role=role, status='collect', orders=len(orders)))
    if len(queries) > 24:
        raise ValueError('行情種類超過24組，停止非預期抓取')
    for iteration in range(2):
        for key in queries:
            observation = fetcher(*key)
            for role in ready:
                result = guard.match(root, role, observation, clock)
                outputs.append(dict(role=role, status=result['status'], query=list(key),
                                    fills=len(result['fills']), reasons=result['reasons']))
        if iteration == 0 and queries:
            sleeper(seconds)
    return dict(refresh=refresh, observations=outputs, http_queries=len(queries) * 2,
                guard_version=guard.RULES['version'])


def run(root=policy.ROOT, observe_seconds=20, clock=j.now, sleeper=walltime.sleep, fetcher=market.fetch):
    if type(observe_seconds) is not int or not 15 <= observe_seconds <= 30:
        raise ValueError('觀察間隔須15至30秒')
    root = Path(root)
    guard.verify(root)
    now = clock().astimezone(p.TZ)
    if not time(9) <= now.time() < time(13, 30):
        def outside_observation_clock():
            at = clock()
            if time(9) <= at.astimezone(p.TZ).time() < time(13, 30):
                raise ValueError('時段已跨入盤中；本輪停止，等待下一次受保護觀察')
            return at
        return legacy.run(root, observe_seconds, outside_observation_clock, sleeper, fetcher)
    with file_lock(root / '.run.lock', timeout=0):
        if not legacy.base.calendar_day(now.date()):
            return dict(status='closed_market', stages=[])
        stage = legacy.base._stage(root, 'observe', lambda: observe(root, clock, sleeper, observe_seconds, fetcher),
                                   clock, retry_seconds=60, once=False)
        report = legacy.export(root, clock)
        blocked = stage['status'] in ('blocked', 'cooldown') or any(
            o.get('status') == 'blocked' for o in stage.get('detail', {}).get('observations', []))
        blocked = blocked or any(b['source_blocked'] for b in report['books'].values())
        return dict(status='needs_attention' if blocked else 'ok', stages=[stage])
