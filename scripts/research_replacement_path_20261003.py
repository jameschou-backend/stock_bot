#!/usr/bin/env python3
"""Preregistered one-time replacement of a stagnant independent signal path.

Decisions are completed before original outcomes are loaded. This is a
proportional two-stock path diagnostic, not an executable funded portfolio.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts.export_signal_explorer import verify_evidence
from scripts.research_early_signal_losses import PERIODS, digest, statistics
from scripts.research_signal_rank_20261003 import rank_events
from skills.exit_policy import decide_exit
from skills.independent_three_black import ThreeBlackPath, black_at_close, path_issue
from skills.independent_signals import net_unit_return
from skills.trial_registry import append_trial_registry

EXPECTED_PREREG = "7a8247ef6304e684e085d6360d9e2bf2a4649c1b1e84da48f7afebac0576efa1"
ARMS = ("baseline", "one_replacement_after5_gap10pp")
THRESHOLD = .10
EPS = 1e-12


def positive(*values):
    return all(np.isfinite(v) and v > 0 for v in values)


def adjusted_hl2(path, index):
    if not positive(path.high[index], path.low[index], path.close[index], path.raw_close[index]):
        return None
    return float((path.high[index] + path.low[index]) / 2 * path.close[index] / path.raw_close[index])


def original_exit(path, entry, index):
    """Use the frozen observer's inclusive age and its stop/time/black priority."""
    current, anchor = path.close[index], path.close[entry]
    valid = positive(current, anchor)
    decision = decide_exit(dict(held_sessions=index + 1 - entry, has_signal=valid,
        entry_return=float(current / anchor - 1) if valid else None,
        peak_return=None, peak_drawdown=None, relative20=None, below_ma20_two=False,
        market_off_two=False, strong_trend=False), "loss12")
    return decision["reason"] if decision["exit"] else ("three_black" if black_at_close(path, entry, index) else None)


def relative20_at(path, benchmark, index):
    """Current and prior 20 observations only; ambiguity stays unknown.

    This uses adjusted closes, eligibility and the existing daily/cumulative
    adjustment tolerances, not future returns or raw-price corporate jumps.
    """
    if index < 20:
        return None, "insufficient_relative20_history"
    for label, series in (("incumbent", path), ("benchmark", benchmark)):
        sl = slice(index - 20, index + 1)
        a, b = series.close[sl], series.other[sl]
        if not ((np.isfinite(a) & (a > 0) & np.isfinite(b) & (b > 0)).all() and series.eligible[sl].all()):
            return None, label + "_relative20_price_or_identity"
        ar, br = a[1:] / a[:-1] - 1, b[1:] / b[:-1] - 1
        if ((abs(ar) > .20) | (abs(br) > .20) | (abs(ar - br) > .005)).any() or abs(a[-1] / a[0] - b[-1] / b[0]) > .02:
            return None, label + "_relative20_adjustment_conflict"
    result = path.close[index] / path.close[index - 20] - benchmark.close[index] / benchmark.close[index - 20]
    return float(result), None


def candidate_groups(events):
    """Signal identity and same-close priority only; never inspect outcome keys."""
    groups, seen = defaultdict(list), set()
    for event in events:
        sid, key = event["stock_id"], event["signal_id"]
        score, day = event["priority"], event["signal_date"]
        if key in seen or not isinstance(score, (float, int)) or isinstance(score, bool) or not math.isfinite(score):
            raise ValueError("Unique signal identity and finite T0 score required")
        seen.add(key)
        groups[day].append(dict(signal_id=key, stock_id=sid, signal_date=day,
                                entry_date=event["entry_date"], priority=float(score)))
    return {day: sorted(group, key=lambda e: (-e["priority"], e["signal_id"])) for day, group in groups.items()}


def select_replacement(event, path, benchmark, candidates):
    """Find the first permitted replacement using only information through T.

    The initial day is holding session one. Each close first checks the original
    exit; only a still-held, gross-nonprofitable path after five held sessions
    may switch. Failure to observe a needed decision is not silently skipped.
    """
    path.validate()
    if not benchmark.days.equals(path.days):
        raise ValueError("Benchmark/path calendars differ")
    day = lambda i: str(path.days[i].date())
    result = dict(replacement_action="unchanged", replacement_issue=None,
        replacement_decision_date=None, replacement_execution_date=None,
        replacement_target_signal_id=None, replacement_target_stock_id=None,
        replacement_incumbent_close_return=None, replacement_incumbent_relative20=None,
        replacement_target_relative20=None, replacement_score_gap=None,
        replacement_held_sessions=None, replacement_original_exit_reason=None,
        replacement_original_exit_date=None, replacement_candidate_count=None,
        replacement_available_at=None)
    if event.get("entry_date") is None:
        result["replacement_action"] = "not_entered"
        return result
    entry = int(path.days.get_loc(pd.Timestamp(event["entry_date"])))
    if entry < 1:
        raise ValueError("Original entry needs a previous signal session")
    if day(entry - 1) != event["signal_date"]:
        raise ValueError("Original entry is not next market session")
    initial = adjusted_hl2(path, entry)
    if initial is None or not positive(path.close[entry - 1]):
        result.update(replacement_action="unknown", replacement_issue="missing_entry_or_preentry_price",
                      replacement_decision_date=day(entry))
        return result
    for index in range(entry, min(entry + 62, len(path.days) - 1) + 1):
        issue = path_issue(path, entry, index)
        if issue:
            result.update(replacement_action="unknown", replacement_issue=issue,
                          replacement_decision_date=day(index), replacement_available_at=day(index) + " 收盤資料完成後")
            return result
        reason = original_exit(path, entry, index)
        if reason:
            result.update(replacement_original_exit_reason=reason, replacement_original_exit_date=day(index))
            return result
        held = index - entry + 1
        current_return = float(path.close[index] / initial - 1)
        if held < 5 or current_return > EPS:
            continue
        options = [e for e in candidates.get(day(index), ()) if e["stock_id"] != event["stock_id"]]
        if not options:
            continue
        incumbent, issue = relative20_at(path, benchmark, index)
        if issue:
            result.update(replacement_action="unknown", replacement_issue=issue,
                replacement_decision_date=day(index), replacement_held_sessions=held,
                replacement_incumbent_close_return=current_return,
                replacement_available_at=day(index) + " 收盤資料完成後")
            return result
        target = options[0]  # sorted independently of all subsequent outcomes
        if target["priority"] - incumbent < THRESHOLD - EPS:
            continue
        execution = day(index + 1) if index + 1 < len(path.days) else None
        if target["entry_date"] != execution:
            raise ValueError("Chosen candidate does not execute at T+1")
        result.update(replacement_action="replace" if execution else "pending_replacement",
            replacement_decision_date=day(index), replacement_execution_date=execution,
            replacement_target_signal_id=target["signal_id"], replacement_target_stock_id=target["stock_id"],
            replacement_incumbent_close_return=current_return, replacement_incumbent_relative20=incumbent,
            replacement_target_relative20=target["priority"], replacement_score_gap=target["priority"] - incumbent,
            replacement_held_sessions=held, replacement_candidate_count=len(options),
            replacement_available_at=day(index) + " 收盤資料完成後")
        return result
    return result


def join_outcome(original, decision, path, outcomes):
    """After all decisions are sealed, join the selected target's original path.

    Two complete fee multipliers represent old buy+sell and new buy+sell. A
    missing fill or target observation leaves the chosen target unresolved;
    no replacement is reselected and no failed leg becomes a zero return.
    """
    row = dict(original)
    row.update(decision)
    keys = ("status", "outcome", "net_return", "gross_return", "unrealized_net_return", "holding_days",
            "holding_days_inclusive", "exit_date", "exit_trigger_date", "exit_reason", "observed_end_date", "data_issue")
    variant = {key: original.get(key) for key in keys}
    first_net, first_gross = None, None
    action = decision["replacement_action"]
    if action in ("unknown", "pending_replacement"):
        variant.update(status="unresolved" if action == "unknown" else "pending_replacement",
            outcome="unknown" if action == "unknown" else "unrealized", net_return=None, gross_return=None,
            unrealized_net_return=None, exit_date=None, exit_trigger_date=None, exit_reason=None,
            data_issue=decision["replacement_issue"], observed_end_date=decision["replacement_decision_date"],
            holding_days=None, holding_days_inclusive=None)
    elif action == "replace":
        entry = int(path.days.get_loc(pd.Timestamp(original["entry_date"])))
        execution = int(path.days.get_loc(pd.Timestamp(decision["replacement_execution_date"])))
        issue = path_issue(path, entry, execution)
        if issue is None and not path.volume[execution] > 0:
            issue = "no_volume_on_old_sale"
        target = outcomes.get(decision["replacement_target_signal_id"])
        if target is None or target["stock_id"] != decision["replacement_target_stock_id"] or target["entry_date"] != decision["replacement_execution_date"]:
            raise ValueError("Selected target's sealed original outcome identity differs")
        if issue is None:
            first_gross = adjusted_hl2(path, execution) / adjusted_hl2(path, entry)
            first_net = 1 + net_unit_return(first_gross)
        if issue is not None or target["status"] not in ("closed", "open", "pending_exit"):
            variant.update(status="unresolved", outcome="unknown", net_return=None, gross_return=None,
                unrealized_net_return=None, holding_days=None, holding_days_inclusive=None,
                exit_date=None, exit_trigger_date=None, exit_reason=None,
                observed_end_date=decision["replacement_execution_date"],
                data_issue=issue or "target_" + (target.get("data_issue") or target["status"]))
        else:
            new_gross = target["gross_return"] + 1
            new_net = target["net_return"] if target["status"] == "closed" else target["unrealized_net_return"]
            if not math.isclose(new_net, net_unit_return(new_gross), rel_tol=0, abs_tol=1e-12):
                raise ValueError("Selected target cost formula differs from frozen study")
            combined = first_net * (1 + new_net) - 1
            end = int(path.days.get_loc(pd.Timestamp(target["observed_end_date"])))
            variant.update(status=target["status"], outcome=("profit" if combined > 0 else "loss" if combined < 0 else "flat")
                if target["status"] == "closed" else "unrealized",
                net_return=combined if target["status"] == "closed" else None,
                unrealized_net_return=None if target["status"] == "closed" else combined,
                gross_return=first_gross * new_gross - 1, holding_days=end - entry,
                holding_days_inclusive=end - entry + 1, exit_date=target["exit_date"],
                exit_trigger_date=target["exit_trigger_date"], exit_reason=target["exit_reason"],
                observed_end_date=target["observed_end_date"], data_issue=None)
    row.update({"variant_" + key: value for key, value in variant.items()})
    row.update(replacement_first_leg_net_ratio=first_net, replacement_first_leg_gross_ratio=first_gross)
    return row


def comparison(rows):
    variants = [{**r, **{k.removeprefix("variant_"): v for k, v in r.items() if k.startswith("variant_")}} for r in rows]
    baseline = statistics(rows)
    changed = statistics(variants)
    originals = [r for r in rows if r["status"] == "closed"]
    pairs = [r for r in originals if r["variant_status"] == "closed"]
    original_winners = [r for r in originals if r["net_return"] > 0]
    original_big = [r for r in originals if r["net_return"] >= .30]
    def retained(group, threshold):
        return sum(r["variant_status"] == "closed" and (r["variant_net_return"] > threshold if threshold == 0 else r["variant_net_return"] >= threshold) for r in group)
    def mean(values):
        return float(np.mean(values)) if values else None
    return dict(baseline=baseline, replacement=changed,
        original_closed_opportunities=len(originals), paired_closed_opportunities=len(pairs),
        paired_closed_coverage=len(pairs) / len(originals) if originals else None,
        unfinished_or_unknown_from_original_closed=len(originals) - len(pairs),
        original_closed_variant_status=dict(Counter(r["variant_status"] for r in originals)),
        paired_baseline_mean=mean([r["net_return"] for r in pairs]),
        paired_replacement_mean=mean([r["variant_net_return"] for r in pairs]),
        paired_mean_improvement=mean([r["variant_net_return"] - r["net_return"] for r in pairs]),
        equal_units_return_sum_on_known_originals=float(sum(r["variant_net_return"] for r in pairs)),
        full_original_opportunity_mean=mean([r["variant_net_return"] for r in pairs]) if len(pairs) == len(originals) else None,
        unknown_never_assumed_zero=len(pairs) != len(originals), rejected_original_entries=0,
        not_entered_signal_count=sum(r["status"] == "not_entered" for r in rows),
        original_winner_count=len(original_winners), retained_winner_count=retained(original_winners, 0),
        winner_retention=retained(original_winners, 0) / len(original_winners) if original_winners else None,
        unknown_or_unfinished_original_winners=sum(r["variant_status"] != "closed" for r in original_winners),
        original_return30_count=len(original_big), retained_return30_count=retained(original_big, .30),
        return30_retention=retained(original_big, .30) / len(original_big) if original_big else None,
        unknown_or_unfinished_original_return30=sum(r["variant_status"] != "closed" for r in original_big),
        action_counts=dict(Counter(r["replacement_action"] for r in rows)),
        issue_counts=dict(Counter(r["variant_data_issue"] for r in rows if r["variant_data_issue"])))


def scopes(rows):
    return {"all": rows, **{str(y): [r for r in rows if r["signal_date"].startswith(str(y))] for y in range(2019, 2027)},
            **{name: [r for r in rows if lo <= r["signal_date"] <= hi] for name, (lo, hi) in PERIODS.items()}}


def load_paths(bundle, ids):
    frames = {name: pd.read_parquet(bundle / (name + ".parquet"), columns=["date", *ids]).set_index("date")
              for name in ("close-official", "close-quality", "eligibility")}
    for frame in frames.values():
        frame.index = pd.to_datetime(frame.index)
    days = frames["close-official"].index
    quotes = pd.read_parquet(bundle / "quotes-unmasked.parquet")
    quotes = quotes.loc[quotes.stock_id.isin(ids)].copy()
    quotes.date = pd.to_datetime(quotes.date)
    if quotes.duplicated(["date", "stock_id"]).any():
        raise ValueError("Duplicate raw stock/date quote")
    fields = {name: quotes.pivot(index="date", columns="stock_id", values=name).reindex(index=days, columns=ids)
              for name in ("close", "high", "low", "volume", "open")}
    paths = {sid: ThreeBlackPath(days,
        *[frames[name][sid].to_numpy(bool if name == "eligibility" else float) for name in ("close-official", "close-quality", "eligibility")],
        *[fields[name][sid].to_numpy(float) for name in ("close", "high", "low", "volume", "open")]) for sid in ids}
    return paths, frames["close-official"]


def trial(arm, result, prereg, *, status="completed", record=False):
    row = dict(timestamp=datetime.now(timezone.utc).isoformat(), command=" ".join(sys.argv),
        source="replacement_path_20261003", status=status, sharpe=None,
        params=dict(arm=arm, start="2019-01-01", end="2026-10-02", min_held_sessions=5,
                    score_gap=THRESHOLD, maximum_replacements=1, independent_path_research=True,
                    cash_account=False, unseen_validation=False),
        prereg_sha256=digest(prereg) if prereg.is_file() else None, result=result)
    if record:
        row["registry_row"] = append_trial_registry(row)
    return row


def run(bundle, rank_dir, research, output, prereg, *, record_trials=False):
    bundle, rank_dir, research, output, prereg = [Path(p).resolve() for p in (bundle, rank_dir, research, output, prereg)]
    if output.exists() or not all(p.is_relative_to(ROOT) for p in (bundle, rank_dir, research, output, prereg)):
        raise ValueError("Require repository-local evidence and a new output directory")
    if digest(prereg) != EXPECTED_PREREG:
        raise ValueError("Changed sequential preregistration")
    refs, manifest = verify_evidence(bundle, rank_dir)
    events = json.loads((bundle / "signals.json").read_text())["entries"]
    if len(events) != 30188:
        raise ValueError("Require all original 30188 signal opportunities")
    ids = sorted({e["members"][0] for e in events} | {"0050"})
    paths, adjusted = load_paths(bundle, ids)
    checked_ranks = rank_events(events, adjusted)
    causal_events = [dict(signal_id=e["event_id"], stock_id=e["members"][0], signal_date=e["signal_date"],
                         entry_date=e["entry_date"], priority=e["priority"]) for e in events]
    candidates = candidate_groups(causal_events)
    decisions = {e["signal_id"]: select_replacement(e, paths[e["stock_id"]], paths["0050"], candidates) for e in causal_events}
    # No workbook outcomes are read until every T-only target selection is fixed.
    payload = json.loads((research / "workbook-data.json").read_text())
    originals = payload["rows"]
    if len(originals) != len(events) or {r["signal_id"] for r in originals} != set(decisions):
        raise ValueError("Outcome and decision populations differ")
    if str((research / "workbook-data.json").relative_to(ROOT)) not in refs:
        raise ValueError("Unbound original outcome workbook")
    outcomes = {r["signal_id"]: r for r in originals}
    rows = [join_outcome(r, decisions[r["signal_id"]], paths[r["stock_id"]], outcomes) for r in originals]
    results = {name: comparison(group) for name, group in scopes(rows).items()}
    for path in (Path(__file__), prereg, ROOT / "tests/test_replacement_path_research.py",
                 ROOT / "scripts/export_signal_explorer.py", ROOT / "scripts/research_early_signal_losses.py",
                 ROOT / "skills/independent_three_black.py", ROOT / "skills/independent_signals.py", ROOT / "skills/trial_registry.py"):
        refs[str(path.relative_to(ROOT))] = digest(path)
    records = [trial(arm, {name: r["baseline"] if arm == "baseline" else r for name, r in results.items()},
                     prereg, record=record_trials) for arm in ARMS]
    output.mkdir(parents=True)
    def dump(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pd.DataFrame([dict(signal_id=key, **value) for key, value in decisions.items()]).to_parquet(output / "decisions.parquet", index=False)
    pd.DataFrame(rows).to_parquet(output / "replacement-paths.parquet", index=False)
    dump("comparisons.json", results)
    dump("trials.json", records)
    report = dict(schema="independent_replacement_path_v1", created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=refs, output_sha256={str(p.relative_to(ROOT)): digest(p) for p in output.iterdir()},
        sample_count=len(rows), latest_data=manifest["end"], all_results=results["all"],
        all_three_period_paired_means_improve=all(results[p]["paired_mean_improvement"] is not None and results[p]["paired_mean_improvement"] > 0 for p in PERIODS),
        three_period_paired_coverage={p: results[p]["paired_closed_coverage"] for p in PERIODS},
        source_priority_recomputation_max_error=max(r["rank_priority_error"] for r in checked_ranks.values()),
        registry_recorded=record_trials, live_qualified=False, cash_account=False, unseen_validation=False,
        definitions=dict(trigger_return="gross adjusted T close / initial assumed entry adjusted HL2 - 1 <= 0",
            age="entry session is one; first eligible replacement decision is session5 close",
            priority="original loss12/time63/threeblack before replacement on every close",
            costs="(1+net_unit_return(old_sale/old_entry)) * (1+new_original_net_return) - 1",
            return30_retention="original >=30% opportunities still closed >=30%; unfinished/unknown separately counted",
            promotion="only if each of three fixed periods has positive paired original-opportunity mean improvement; still requires separate account validation"),
        limitations=payload["metadata"]["limitations"] + [
            "Overlapping independent two-stock paths, not portfolio compounding or independent statistical samples.",
            "T+1 HL2 is an after-the-day assumption, not a known limit price or guaranteed fill.",
            "One target selected causally; an unknown or unfilled leg is never replaced by the next candidate.",
            "Sale proceeds are assumed available for the new purchase; actual buying-power sequencing remains unverified.",
            "Common closed comparisons disclose all excluded unknown/unfinished original opportunities; no unknown return is replaced with zero.",
            "Historical reused evidence; no new unseen validation or market-beating account return."])
    dump("report.json", report)
    (output / "report.sha256").write_text(digest(output / "report.json") + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=ROOT / ".cache/all-signals-2019-20261002/inputs")
    p.add_argument("--rank-dir", type=Path, default=ROOT / ".cache/signal-rank-20261003/rank-v2")
    p.add_argument("--research", type=Path, default=ROOT / ".cache/all-signals-2019-20261002/research-v1")
    p.add_argument("--prereg", type=Path, default=ROOT / "docs/research_sequential_prereg_20261003.md")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        report = run(args.inputs, args.rank_dir, args.research, args.output, args.prereg, record_trials=True)
        print(json.dumps({"sample_count": report["sample_count"], "results": report["all_results"],
            "all_three_period_paired_means_improve": report["all_three_period_paired_means_improve"]}, ensure_ascii=False))
    except Exception as exc:
        failures = [trial(arm, {"error": type(exc).__name__ + ": " + str(exc)}, args.prereg, status="failed", record=True) for arm in ARMS]
        if args.output.resolve().is_relative_to(ROOT):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_name(args.output.name + "-failure.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
