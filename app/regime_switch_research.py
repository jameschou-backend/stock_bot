"""Read sealed market-state research and load one detailed case on demand."""
from functools import lru_cache
from datetime import date
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

from app import diffusion_research

ROOT=Path(__file__).resolve().parents[1]
CACHE='.cache/regime-switch-research'
CODE={'scripts/research_regime_switch.py','skills/regime_state.py','skills/regime_portfolio.py','skills/regime_mix.py'}
RULES={'always','entry_only','idle_cash','exit_cash','mix_always','mix_entry'}
BASES={'official','snapshot'}
SCENARIOS={'base','stress'}
PROBE={'.cache/regime-probe-20260910/'+name for name in
       ('protocol.md','probe.py','states-official.parquet','states-snapshot.parquet','report.json')}


@lru_cache(maxsize=128)
def _digest(path,size,modified):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def sha(path):
    path=Path(path).resolve(strict=True);before=path.stat()
    digest=_digest(str(path),before.st_size,before.st_mtime_ns)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Research artifact changed while reading')
    return digest


def _json(payload):
    def invalid(value):raise ValueError('Non-finite number: '+value)
    return json.loads(payload,parse_constant=invalid)


def read(path):
    return _json(Path(path).read_text())


def _chart_digest(curve):
    points=[{'date':r['date'],'nav':r['nav']} for r in curve]
    return hashlib.sha256(json.dumps(points,sort_keys=True,allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=128)
def _case_header(path,fingerprint):
    # Cold verification reads the sealed case, then retains only small headers.
    # Full transaction records are returned to the UI only by load_case.
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=fingerprint:
        raise ValueError('Case changed during header verification')
    case=_json(payload)
    header={k:case[k] for k in ('summary','valuation_audit','rule','basis','scenario','delay')}
    return header,_chart_digest(case['curve'])


def hashes(mapping,expected,base):
    if not isinstance(mapping,dict) or set(mapping)!=expected:
        raise ValueError('Incomplete research provenance')
    for name,fingerprint in mapping.items():
        if sha(base/name)!=fingerprint:raise ValueError('Changed research file: '+name)


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _summary(row):
    s=row['summary']
    if (s['start']!='2022-01-03' or s['end']!='2026-06-23'
            or not _finite(s['initial_nav']) or s['initial_nav']!=1
            or any(not _finite(s[k]) for k in ('total_return','cagr','max_drawdown','total_cost','turnover',
                'mean_active_weight','mean_cash_weight','final_nav','final_cash'))
            or s['final_nav']<=0 or s['final_cash']<0 or s['total_cost']<0 or s['turnover']<0
            or not -1<=s['max_drawdown']<=0
            or not 0<=s['mean_cash_weight']<=1+1e-10 or not 0<=s['mean_active_weight']<=1+1e-10
            or not math.isclose(s['total_return']+1,s['final_nav'],abs_tol=1e-12)):
        raise ValueError('Invalid strategy numbers')
    years=(date(2026,6,23)-date(2022,1,3)).days/365.25
    if not math.isclose(s['cagr'],s['final_nav']**(1/years)-1,rel_tol=1e-9,abs_tol=1e-12):
        raise ValueError('Annualized return does not match the account period')
    mode={'always':'events','entry_only':'events','idle_cash':'idle_cash','exit_cash':'exit_cash',
          'mix_always':'fixed_mix','mix_entry':'fixed_mix','benchmark':'benchmark'}[row['rule']]
    if (s['mode']!=mode or s['horizon']!=63 or s['slots']!=3
            or s['commission_per_side']!=.001425 or s['stock_sell_tax']!=.003 or s['benchmark_sell_tax']!=.001
            or s['slippage_per_side']!={'base':.003,'stress':.0045}[row['scenario']]):
        raise ValueError('Changed portfolio assumptions')
    annual=s['annual_returns']
    if (not isinstance(annual,dict) or set(annual)!={'2022','2023','2024','2025','2026'}
            or any(not _finite(v) or v<=-1 for v in annual.values())
            or not math.isclose(math.prod(1+v for v in annual.values()),s['final_nav'],rel_tol=1e-9)):
        raise ValueError('Annual compounding mismatch')
    positions=s['unliquidated_positions']
    if (not isinstance(positions,list) or len(positions)!=s['unliquidated_position_count']
            or s['final_liquidation_complete'] is not (not positions) or s['final_nav_is_marked'] is not bool(positions)
            or any(not _finite(p['marked_value']) or p['marked_value']<=0 for p in positions)
            or not math.isclose(s['final_cash']+sum(p['marked_value'] for p in positions),s['final_nav'],rel_tol=1e-9)):
        raise ValueError('Missing liquidation warning or cash reconciliation')
    audit=row['valuation_audit'];findings=audit['findings']
    if (audit['finding_count']!=len(findings)
            or audit['unresolved_valuation_days']!=len({f['date'] for f in findings})):
        raise ValueError('Incomplete valuation audit')


def overview():
    missing={'available':False,'research_only':True,'live_qualified':False,'valid_strategy_evidence':False,
             'note':'尚無完整的情境切換研究；先執行 make prepare-regime-switch，再執行 make research-regime-switch。'}
    folder=ROOT/CACHE
    if not (folder/'report.summary.json').exists():return missing
    try:
        report=read(folder/'report.summary.json')
        if (report['schema']!=1 or report['experiment']!='regime_switch_20260910'
                or report['research_only'] is not True or report['live_qualified'] is not False
                or report['valid_strategy_evidence'] is not False or report['control_reproduction_passed'] is not True
                or report['start']!='2022-01-03' or report['end']!='2026-06-23' or report['signal_end']!='2025-12-31'):
            raise ValueError('Unsupported research')
        hashes(report['code_sha256'],CODE,ROOT)
        if sha(ROOT/'docs/prereg_regime_switch_20260910.md')!=report['protocol_sha256']:
            raise ValueError('Changed protocol')
        if sha(folder/'states.json')!=report['state_manifest_sha256']:
            raise ValueError('Changed state manifest')
        states=read(folder/'states.json')
        if (states!=report['states'] or states['schema']!=1 or states['code_sha256']!=report['code_sha256']
                or states['protocol_sha256']!=report['protocol_sha256'] or states['prefix_invariance_passed'] is not True
                or states['versions']!={name:version(name) for name in ('numpy','pandas','scipy')}):
            raise ValueError('Changed state source or runtime')
        hashes(states['files_sha256'],{'states-official.parquet','states-snapshot.parquet'},folder)
        hashes(states['prior_probe_files_sha256'],PROBE,ROOT)
        parent=diffusion_research.overview()
        if not parent['available'] or states['diffusion_signal_manifest_sha256']!=parent['signal_manifest_sha256']:
            raise ValueError('Parent price and signal evidence changed')
        expected={(r,b,c,0) for r in RULES for b in BASES for c in SCENARIOS}
        expected|={(r,'official','stress',1) for r in RULES}
        key=lambda r:(r['rule'],r['basis'],r['scenario'],r['delay'])
        rows=report['results'];bench=report['baselines']
        if len(rows)!=30 or {key(r) for r in rows}!=expected:
            raise ValueError('Incomplete thirty contrasts')
        if len(bench)!=4 or {key(r) for r in bench}!={('benchmark',b,c,0) for b in BASES for c in SCENARIOS}:
            raise ValueError('Incomplete benchmark controls')
        cases={f'case-{rule}-{b}-{c}-{d}.json' for rule,b,c,d in expected}
        cases|={f'case-benchmark-{b}-{c}-0.json' for b in BASES for c in SCENARIOS}
        hashes(report['case_files_sha256'],cases,folder)
        for row in rows+bench:
            _summary(row)
            if row['case_file']!=f"case-{row['rule']}-{row['basis']}-{row['scenario']}-{row['delay']}.json":
                raise ValueError('Incorrect case link')
            header,chart_hash=_case_header(str(folder/row['case_file']),report['case_files_sha256'][row['case_file']])
            if any(row[k]!=v for k,v in header.items()):
                raise ValueError('Displayed summary differs from the sealed account')
            if (row['basis'],row['scenario'],row['delay'])==('official','stress',0):
                if _chart_digest(report['charts'][row['rule']])!=chart_hash:
                    raise ValueError('Displayed chart differs from the sealed account')
        indexed={key(r):r for r in rows+bench}
        for row in rows:
            for compare in ('0050','entry_only','mix_entry'):
                other=indexed[('benchmark',row['basis'],row['scenario'],0) if compare=='0050'
                              else (compare,row['basis'],row['scenario'],row['delay'])]
                if not math.isclose(row['excess_vs_'+compare],row['summary']['total_return']-other['summary']['total_return'],abs_tol=1e-12):
                    raise ValueError('Inconsistent comparison')
        if (set(report['charts'])!=RULES|{'benchmark'} or not report['limitations']
                or any(r['diagnostic_only'] is not True for r in report['leave_one_year_diagnostic'])):
            raise ValueError('Missing charts or limitations')
        diagnostics=report['leave_one_year_diagnostic']
        if len(diagnostics)!=4 or {r['omitted_year'] for r in diagnostics}!={'2022','2023','2024','2025'}:
            raise ValueError('Incomplete year concentration diagnostic')
        for diagnostic in diagnostics:
            values=diagnostic['compounded_remaining_years']
            if not isinstance(values,dict) or set(values)!=RULES|{'benchmark'}:
                raise ValueError('Incomplete diagnostic comparisons')
            for rule,value in values.items():
                annual=indexed[(rule,'official','stress',0)]['summary']['annual_returns']
                expected=math.prod(1+v for year,v in annual.items() if year!=diagnostic['omitted_year'])-1
                if not _finite(value) or not math.isclose(value,expected,abs_tol=1e-12):
                    raise ValueError('Incorrect year concentration diagnostic')
        return {**report,'available':True}
    except (OSError,ValueError,KeyError,TypeError,ImportError,OverflowError):
        return {**missing,'note':'情境切換研究不完整，或來源／程式已變更；請重新準備狀態與研究結果。'}


def load_case(report,row):
    """Only accept a case named by the verified report; never use an arbitrary path."""
    try:
        if not report.get('available') or row not in report['results']+report['baselines']:
            raise ValueError('Case is not in the verified report')
        name=row['case_file']
        if Path(name).name!=name or sha(ROOT/CACHE/name)!=report['case_files_sha256'][name]:
            raise ValueError('Case changed')
        case=read(ROOT/CACHE/name)
        if (case['summary']!=row['summary'] or case['valuation_audit']!=row['valuation_audit']
                or any(case[k]!=row[k] for k in ('rule','basis','scenario','delay'))):
            raise ValueError('Case summary differs from displayed result')
        return {**case,'available':True}
    except (OSError,ValueError,KeyError,TypeError):
        return {'available':False,'note':'這份成交紀錄已變更或不完整，請重新研究後再查看。'}
