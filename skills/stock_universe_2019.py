"""Two fixed universe hypotheses; retain causal technical and data quality checks."""
from copy import deepcopy
import pandas as pd
from skills.candidate_quality import candidate_features

ARMS = ('original', 'ungrouped300', 'liquid_universe', 'benchmark')


def generate(frames, companies, signals, cutoff):
    f = candidate_features(frames, companies)
    c, days = f['close'], f['close'].index
    volume = frames['raw-volume'].where(f['close'].notna() & frames['raw-volume'].gt(0))
    previous_high = c.shift(1).rolling(60, min_periods=60).max()
    previous_volume = volume.shift(1).rolling(20, min_periods=20).mean()
    own = c/c.shift(20)-1
    technical = f['quality'] & c.gt(previous_high) & own.gt(0) & f['relative20'].gt(0)
    technical &= volume.ge(previous_volume*1.5)
    technical['0050'] = False
    groups = {g['month']: g for g in signals['diffusion']['groups']}
    out = {'original': [deepcopy(e) for e in signals['entries'] if e['signal_date'] <= cutoff],
           'ungrouped300': [], 'liquid_universe': []}
    for i, day in enumerate(days[:-1]):
        if day < pd.Timestamp('2019-01-01') or day > pd.Timestamp(cutoff) or f['trend'].at[day] != 'ON':
            continue
        month = groups[str(day.to_period('M'))]
        if pd.Timestamp(month['cutoff_date']) >= day:
            raise ValueError('Monthly universe is not known before signal')
        pool = set(month['selected_ids']) - set(month['exclusions'].get('zero_residual_variance', []))
        for sid in technical.columns[technical.loc[day]]:
            if len(sid) != 4 or not sid.isdigit() or sid.startswith('0'):
                raise ValueError('Only individual stocks are eligible')
            for arm in ('ungrouped300', 'liquid_universe'):
                if arm == 'ungrouped300' and sid not in pool:
                    continue
                date = str(day.date())
                out[arm].append(dict(event_id=f'{arm}-{date}-{sid}', signal_date=date,
                    entry_date=str(days[i+1].date()), members=[sid], priority=float(f['relative20'].at[day, sid]),
                    group_id=arm+'-'+str(day.to_period('M')), group_cutoff_date=month['cutoff_date'],
                    group_members=[sid], selection_reason='causal breakout and volume expansion; '+arm,
                    leader_evidence=dict(leader_return20=float(own.at[day, sid]),
                        benchmark_return20=float(own.at[day, '0050']),
                        leader_volume_ratio=float(volume.at[day, sid]/previous_volume.at[day, sid]))))
    for arm in ('ungrouped300', 'liquid_universe'):
        out[arm].sort(key=lambda e: (e['entry_date'], -e['priority'], e['event_id']))
    return out
