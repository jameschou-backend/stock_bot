"""Causal surge candidates and full offline accounts on existing audited engines."""
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import math

import numpy as np
import pandas as pd

from skills.surge_anatomy import features
from skills.surge_sector import sector_features
from skills.scenario_exit_replay import ExitSignals
from skills.conservative_diversification import ConservativeDiversification
from skills.board_only_verified_replay import BoardOnlyVerifiedReplay, BoardOnlyVerifiedBenchmark
from skills.execution_resources import ResourceBenchmark, audit_resources
from skills.slot_reuse_replay import audit_slots
from skills.board_only_verified_replay import audit_verified_board_only
from skills.board_only_replay import execution_summary
from skills.backtest_contract import validate_completed_account
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable
from skills.million_replay import UnresolvedAction
from skills.pending_share_entitlements import install_pending_share_rights
from scripts.research_exit_scenarios import summarize, TrackedCorporateActions

ARMS = ('relative_strength', 'strength_with_turnover')
START, END = '2022-01-03', '2026-09-09'


@dataclass(frozen=True)
class SectorAccountInputs:
    quotes: pd.DataFrame
    companies: pd.DataFrame
    days: pd.DatetimeIndex
    entries_by_arm: dict
    events: pd.DataFrame
    features: ExitSignals
    membership_snapshot_date: str
    start: str = START
    end: str = END


def configurations():
    for board in (False, True):
        for stress in ('control', 'combined'):
            for arm in ('benchmark', *ARMS):
                yield f'{arm}_{stress}_{"board_only" if board else "mixed"}', dict(
                    arm=arm, stress=stress, benchmark=arm=='benchmark', board_only=board,
                    position_count=0 if arm=='benchmark' else 5)


def _or(values):
    if values.eq(True).any():
        return True
    return None if values.isna().any() else False


def build_signals(close, raw, volume, companies, members, *, membership_snapshot_date,
                  start=START, end=END):
    """Recompute T-only features; no forward-label table is accepted or read."""
    snapshot = pd.Timestamp(membership_snapshot_date)
    if pd.isna(snapshot) or snapshot.tz is not None or str(snapshot.date()) != membership_snapshot_date:
        raise ValueError('Membership snapshot requires its real ISO retrieval date')
    days = close.index
    if pd.Timestamp(start) not in days or pd.Timestamp(end) not in days:
        raise ValueError('Signal interval endpoints must be supplied market days')
    positions = np.flatnonzero((days>=pd.Timestamp(start)) & (days<=pd.Timestamp(end)))[::21]
    usable = [int(i) for i in positions if i+1<len(days) and days[i+1]<=pd.Timestamp(end)]
    members = members[members.stock_id.isin(companies.stock_id)].copy()
    individual = features(close, raw, volume, companies)
    sectors = sector_features(close, raw, volume, companies, members, days[usable])
    grouped = {key: rows for key, rows in sectors.groupby(['signal_date', 'stock_id'], sort=False)}
    entries = {arm: [] for arm in ARMS}
    coverage, candidates = [], []
    for i in usable:
        day, entry = str(days[i].date()), str(days[i+1].date())
        ids = individual['eligible'].columns[individual['eligible'].iloc[i]]
        # Nullable pandas rows are costly to materialize across ~2,000 stocks.
        # Extract each anchor once instead of rebuilding it for every candidate.
        strength_values = individual['rules']['relative_strength'].iloc[i].to_dict()
        adv_values = individual['adv20'].iloc[i].to_dict()
        relative_values = individual['numeric']['relative20'].iloc[i].to_dict()
        share_window = volume.iloc[i-19:i+1]
        shares_complete = share_window.notna().all().to_dict()
        shares_mean = share_window.mean().to_dict()
        counts = dict(signal_date=day, entry_date=entry, eligible=len(ids), unmapped=0,
                      turnover_unknown=0, common=0, relative_strength=0, strength_with_turnover=0)
        for sid in ids:
            groups = grouped.get((day, sid))
            turnover = None if groups is None else _or(groups.turnover_confirmed)
            common = groups is not None and turnover is not None
            rs = strength_values[sid]
            strength = None if pd.isna(rs) else bool(rs)
            adv = float(adv_values[sid])
            liquidity = dict(as_of=day, complete_20_sessions=bool(shares_complete[sid]),
                observations=len(share_window), adv20_shares=float(shares_mean[sid]), mean_turnover20_twd=adv)
            if not liquidity['complete_20_sessions'] or liquidity['observations']!=20:
                raise ValueError('Eligible signal unexpectedly lacks twenty volume observations')
            row = dict(signal_date=day, entry_date=entry, stock_id=sid,
                adv20=adv, relative_strength=strength, turnover_confirmed=turnover,
                common_peer_observation=common, membership_point_in_time=False,
                membership_snapshot_date=membership_snapshot_date)
            candidates.append(row)
            counts['unmapped'] += int(groups is None)
            counts['turnover_unknown'] += int(groups is not None and turnover is None)
            counts['common'] += int(common)
            if not common or strength is not True:
                continue
            event = dict(event_id=f'sector-{day}-{sid}', members=[sid], stock_id=sid,
                signal_date=day, entry_date=entry, priority=adv,
                feature_cutoff_date=day, group_cutoff_date=day,
                group_cutoff_date_meaning='price_and_turnover_feature_cutoff_only',
                membership_point_in_time=False, membership_snapshot_date=membership_snapshot_date,
                liquidity_at_signal=liquidity, liquidity_before_entry=deepcopy(liquidity),
                relative20=float(relative_values[sid]),
                turnover_confirmed=turnover,
                confirmed_turnover_groups=sorted(groups.loc[groups.turnover_confirmed.eq(True).fillna(False), 'industry']))
            entries['relative_strength'].append(event)
            counts['relative_strength'] += 1
            if turnover:
                entries['strength_with_turnover'].append(deepcopy(event))
                counts['strength_with_turnover'] += 1
        coverage.append(counts)
    for arm in ARMS:
        entries[arm].sort(key=lambda e: (e['signal_date'], -e['priority'], e['stock_id']))
    return dict(entries_by_arm=entries, coverage=coverage, candidates=candidates,
                anchors=[str(days[i].date()) for i in usable], membership_point_in_time=False,
                membership_snapshot_date=membership_snapshot_date, future_labels_used=False)


def validate_entries(entries, days):
    positions = {str(day.date()): i for i, day in enumerate(days)}
    seen = set()
    for e in entries:
        if (e['event_id'] in seen or len(e['members'])!=1 or e['stock_id']!=e['members'][0]
                or positions.get(e['entry_date']) != positions.get(e['signal_date'], -999)+1):
            raise ValueError('Candidate identity or next-session timing is invalid')
        seen.add(e['event_id'])
        if e['membership_point_in_time'] is not False:
            raise ValueError('Current membership must not be promoted to PIT')
        if e['feature_cutoff_date']!=e['signal_date'] or e['group_cutoff_date']!=e['signal_date']:
            raise ValueError('Feature cutoff differs from signal')
        for field in ('liquidity_at_signal', 'liquidity_before_entry'):
            x = e[field]
            if x['as_of']!=e['signal_date'] or x['complete_20_sessions'] is not True or x['observations']!=20:
                raise ValueError('Liquidity metadata is not contemporaneous and complete')
            if any(isinstance(x[k], bool) or not isinstance(x[k], (int,float))
                   or not math.isfinite(x[k]) or x[k]<=0 for k in ('adv20_shares','mean_turnover20_twd')):
                raise ValueError('Invalid signal liquidity')
    return dict(signal_count=len(entries), next_market_day=True, future_labels_used=False,
                membership_point_in_time=False, dated_liquidity_checked=True)


class DatedEvidenceOrders:
    """Missing daily evidence blocks new research instead of looking like a rejection."""
    execution_lot_size = 1

    def _execute_order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        eligible = min(qty, self.holdings.get(sid, {}).get('qty', 0)) if side=='sell' else qty
        if eligible >= self.execution_lot_size:
            values = [self.fields[key].at[day, sid] for key in ('close','high','low','volume')]
            if any(pd.isna(value) or not math.isfinite(float(value)) for value in values):
                raise ReplayDataUnavailable(f'Raw execution quote missing: {sid} {day.date()}')
            if all(self.raw(day, sid, key) for key in ('close','high','low','volume')):
                if not self.feeds.get_limits(sid).get(str(day.date())):
                    raise ReplayDataUnavailable(f'Offline replay is missing price-limit date: {sid} {day.date()}')
            # ResourceDecisions has already approved cash, slots and positive
            # sizing here. Validate corporate evidence before any actual fill.
            self.corporate.prepare_for_execution(sid)
        return super()._execute_order(day, sid, side, qty, reason, event_id, signal_date)


class ExecutionCorporatePreparation:
    """Defer eager candidate fetches until resources authorize an execution.

    Held-asset on_date and explicit execution validation always delegate to the
    original provider. Only Replay's pre-resource candidate hint is deferred.
    """
    def __init__(self, provider):
        self.provider=provider

    def __getattr__(self,name):
        return getattr(self.provider,name)

    def prepare(self,sid):
        return None

    def prepare_for_execution(self,sid):
        return self.provider.prepare(sid)

    def on_date(self,sid,day):
        return self.provider.on_date(sid,day)


class FrozenSignalPriority:
    def corporate_day(self, day):
        income = super().corporate_day(day)
        # The inherited capacity arm ranks execution-day prior liquidity. Keep
        # this experiment's T ranking even when combined delays entry to T+2.
        self.events[day].sort(key=lambda e: (-e['priority'], e['members'][0], e['event_id']))
        return income


class SectorMixedReplay(DatedEvidenceOrders, FrozenSignalPriority, ConservativeDiversification):
    pass


class SectorMixedBenchmark(DatedEvidenceOrders, ResourceBenchmark):
    pass


class SectorBoardReplay(DatedEvidenceOrders, FrozenSignalPriority, BoardOnlyVerifiedReplay):
    execution_lot_size = 1000


class SectorBoardBenchmark(DatedEvidenceOrders, BoardOnlyVerifiedBenchmark):
    execution_lot_size = 1000


def _journals(engine):
    return dict(resource_plans=engine.resource_plans,
        slot_decisions=getattr(engine, 'slot_decisions', []),
        board_decisions=getattr(engine, 'board_decisions', []),
        exit_decisions=getattr(engine, 'exit_decisions', []), exit_states=getattr(engine, 'exit_states', {}))


def run_case(data, config, inputs, overrides, *, feeds=None, corporate=None):
    """Injected in-memory providers support tests; real providers are offline only."""
    if config not in [c for _,c in configurations()]:
        raise ValueError('Use one of the twelve preregistered account configurations')
    entries = [] if config['benchmark'] else data.entries_by_arm[config['arm']]
    contract = validate_entries(entries, data.days)
    schedule, executable_entries = [], []
    positions = {str(day.date()):i for i,day in enumerate(data.days)}
    for entry in entries:
        target = positions[entry['entry_date']] + int(config['stress']=='combined')
        execution = str(data.days[target].date()) if target<len(data.days) else None
        included = execution is not None and execution<=data.end
        schedule.append(dict(event_id=entry['event_id'],stock_id=entry['stock_id'],
            signal_date=entry['signal_date'],control_entry_date=entry['entry_date'],execution_date=execution,
            accepted_for_account=included,reason=None if included else 'scheduled_outside_account_window'))
        if included:
            executable_entries.append(entry)
    metadata = dict(config=config, membership_point_in_time=False,
        membership_snapshot_date=data.membership_snapshot_date, live_qualified=False, unseen_validation=False,
        schedule_decisions=schedule,candidate_count=len(entries))
    inputs = Path(inputs)
    if feeds is None:
        if not (inputs/'execution-feeds/index.json').is_file() or not (inputs/'dividends').is_dir():
            return dict(completed=False, reason='Offline execution/dividend source directory missing', **metadata)
        feeds = ReplayMarketFeeds(inputs/'execution-feeds', offline=True)
    if corporate is None:
        corporate = TrackedCorporateActions(data.events, inputs/'dividends', None, offline=True, overrides=overrides)
    args = (data.quotes, data.companies, data.days, executable_entries, feeds, ExecutionCorporatePreparation(corporate))
    kwargs = dict(start=data.start, end=data.end, initial_cash=1_000_000., stress_mode=config['stress'])
    if config['benchmark']:
        engine = (SectorBoardBenchmark(*args, **kwargs) if config['board_only'] else
                  SectorMixedBenchmark(*args, opening_cash_only=True, lock_unused=True, **kwargs))
    else:
        kwargs.update(exit_signals=data.features, action_dates=list(zip(data.events.stock_id, data.events.event_date)))
        engine = (SectorBoardReplay(*args, **kwargs) if config['board_only'] else
                  SectorMixedReplay(*args, position_count=5, **kwargs))
    try:
        install_pending_share_rights(engine)
        account = engine.run()
        checked = (audit_resources(account, engine.resource_plans, opening_cash_only=True,
                    lock_unused=True, lock_slots=False) if config['benchmark'] else
                   audit_slots(account, engine.resource_plans, engine.slot_decisions,
                    opening_cash_only=True, lock_unused=True, lock_opening_slots=True, lock_failed_slots=True))
        if config['board_only']:
            checked.update(audit_verified_board_only(account, engine.board_decisions, engine.resource_plans))
        checked.update(validate_completed_account(account, [str(d.date()) for d in data.days], data.start, data.end))
    except (ReplayDataUnavailable, UnresolvedAction) as exc:
        account = None; reason = str(exc)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        account = None; reason = str(exc)
    if account is None:
        partial = {key:getattr(engine, attr) for key,attr in (
            ('daily','daily'),('trades','trades'),('orders','orders'),('corporate_actions','actions'),
            ('cash_ledger','cash_ledger'),('holdings','holding_rows'),('cohorts','cohorts'),('receivables','receivables'))}
        return dict(completed=False, reason=reason, partial_account=partial, **metadata, **_journals(engine),
                    partial_scope='Incomplete account; no full-period return is reported', signal_contract=contract)
    return dict(completed=True, account=account, summary=summarize(account), audit=checked,
        signal_contract=contract, execution=execution_summary(account, getattr(engine,'board_decisions',[])),
        **metadata, **_journals(engine))
