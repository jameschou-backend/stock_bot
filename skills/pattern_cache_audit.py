"""Do not count verified reused accounts as newly executed experiments."""


def audit_cache_reuse(report, parent, trials, arms, load_case):
    expected = {f'{a}_{m}' for a in arms for m in range(8)}
    if set(report['cases']) != expected:
        raise ValueError('Pattern comparison requires every registered arm and stress')
    executed, reused = set(), 0
    for arm,(mode,enabled) in arms.items():
        for mask in range(8):
            name=f'{arm}_{mask}';row=report['cases'][name]
            case=load_case(row['result'])
            if (row['cache_hit'] is not (not enabled) or case['config']['factor_mask']!=mask
                    or case['config']['technical_mode']!=mode or row['config']!=case['config']
                    or row['completed']!=case['completed']):
                raise ValueError('Cache status or arm configuration does not match the account')
            if enabled:
                if row['reused_case'] is not None or case['config'].get('pattern_filter') is not True:
                    raise ValueError('A new pattern account cannot claim a reused result')
                executed.add(name)
            else:
                original=parent['cases'][f'{mode}_{mask}']['result']
                if (row['reused_case']!=original or case!=load_case(original)
                        or case['completed'] is not True):
                    raise ValueError('Reused account is not the exact previously verified control')
                reused+=1
    if (len(trials)!=len(executed) or {r['case'] for r in trials}!=executed
            or report['trial_count']!=len(executed) or report['cache_hits']!=reused):
        raise ValueError('New execution count and reused account count must remain distinct')
    for trial in trials:
        case=report['cases'][trial['case']]
        expected_status='completed' if case['completed'] else 'blocked'
        if trial['status']!=expected_status or trial['result_path']!=case['result']['path']:
            raise ValueError('Execution registry differs from the resulting account')
    return dict(exact_parent_accounts_reused=True,newly_executed_cases=len(executed),cache_hits=reused)
