"""Compact API/MCP evidence from the exact publications shown in the dashboard."""
from pathlib import Path

from app.research_validation_ui import FAMILIES, load_summary

ROOT=Path(__file__).resolve().parents[1]


def overview(root=ROOT):
    root=Path(root).resolve()
    families={}
    for label,(family,arms) in FAMILIES.items():
        positions=1 if family=='index_exposure' else 5
        version='completed_20260927' if family in ('exit_mechanisms','volatility_budget') else '20260927'
        path=root/'artifacts/forward_simulation'/f'{family}_{version}.json'
        base=dict(label=label,live_qualified=False,unseen_validation=False,position_count=positions,
                  instrument_scope='leveraged_index_ETF' if family=='index_exposure' else 'individual_stocks')
        if not path.exists():
            families[family]=dict(base,available=False,reason='verified_publication_missing')
            continue
        try:
            report=load_summary(path,family,arms,root)
            rows=[]
            for arm,name in arms.items():
                cases=[report['cases'][f'{arm}_{mask}'] for mask in range(8)]
                for mask,case in enumerate(cases):
                    config=case['config']
                    if (config.get('factor_mask')!=mask or config.get('benchmark') is not False
                            or config.get('board_only') is not True or config.get('position_count')!=positions):
                        raise ValueError('Current account execution scope differs')
                def result(mask):
                    case=cases[mask]
                    if not case['completed']:
                        return dict(available=False,reason=case['reason'])
                    return dict(available=True,total_return=case['summary']['total_return'],
                        max_drawdown=case['summary']['max_drawdown'],
                        benchmark_return=case['metrics']['benchmark_return'],
                        excess_return=case['metrics']['excess_return'],
                        account=case['result'])
                rows.append(dict(arm=arm,label=name,normal=result(0),all_stresses=result(7),
                    completed_cases=sum(c['completed'] for c in cases),
                    winning_stresses=sum(c['completed'] and c['metrics']['excess_return']>0 for c in cases),
                    stress_count=8,live_qualified=False))
            families[family]=dict(base,available=True,all_completed=report['all_completed'],
                publication=dict(path=str(path.relative_to(root)),sha256=path.with_suffix('.sha256').read_text().strip()),
                offline_verification=report['offline_verification'],arms=rows,
                data_quality=report.get('data_quality'),limitations=report.get('limitations',[]))
            if family=='index_exposure':
                from app.index_earlier_ui import overview as earlier_overview
                families[family]['earlier_period_replication']=earlier_overview(root)
                from app.index_continuous_ui import overview as continuous_overview
                families[family]['continuous_capital_replication']=continuous_overview(root)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            families[family]=dict(base,available=False,reason='publication_verification_failed',detail=str(exc))
    return dict(schema='current_account_evidence_v1',live_qualified=False,unseen_validation=False,
        start='2022-01-03',end='2026-09-09',initial_cash=1_000_000,position_count=None,
        position_count_scope='See each family; individual stocks use five slots, index exposure uses one ETF',
        execution='board_only',idle_capital='cash',costs_included=True,
        all_stresses='double_slippage_plus_one_extra_entry_day_plus_one_extra_exit_day',
        source_check_scope='published_summaries_and_two_run_manifests_not_all_historical_source_bytes',
        note='以此區比較目前現金帳戶；每列提供封存帳戶。舊研究包含不同資金與成交設定，不能混用報酬。',
        families=families)
