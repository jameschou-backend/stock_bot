from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import residual_slots_ui as ui
from scripts.research_residual_slots import comparison
from scripts.research_exit_scenarios import summarize, write, sha


def publication(root):
    dates = [str(d.date()) for d in pd.bdate_range('2022-01-03', periods=252)] + ['2026-09-09']
    account = dict(settings={'initial_cash': 1_000_000},
        daily=[dict(date=d, opening_nav=1_000_000., nav=1_000_000., total_return=0.,
            cash=1_000_000., market_value=0., receivable=0., drawdown=0., stale_holdings=0, holdings=0) for d in dates],
        trades=[], orders=[], holdings=[], receivables=[], cohorts=[])
    cases, rows, refs, old_rows = {}, {}, {}, {}
    for name in ['release_'+str(i) for i in range(8)] + ['keep_0','keep_7','benchmark_control','benchmark_combined']:
        release = name.startswith('release_')
        result = dict(completed=True, account=deepcopy(account), summary=summarize(account), stock_pnl={},
            config=dict(residual_policy='release' if release else 'keep', factor_mask=int(name.split('_')[1]) if name[:1] in 'rk' else None),
            audit={k:True for k in ('residual_classification_rebuilt','residual_assets_retained','residual_risk_budget_rebuilt','active_slots_rebuilt')},
            residual_days=[dict(residual_value=0.,opening_nav=1_000_000.,block_new_buys=False,opening_active=[]) for _ in dates])
        path = root / (name+'.json');write(path,result);refs[path.name]=sha(path)
        cases[name]=result
        rows[name]=dict(completed=True,summary=result['summary'],result=dict(path=path.name,sha256=sha(path)))
        if not name.startswith('keep_'):
            old_rows['factor_'+name[8:] if release else name]=deepcopy(rows[name])
    original = dict(cases=old_rows)
    old = root/'original.json';write(old,original);refs[old.name]=sha(old)
    manifest = root/'manifest.json';write(manifest,dict(files_sha256=refs));refs[manifest.name]=sha(manifest)
    proof = root/'proof.json';write(proof,dict(schema='residual_slots_offline_v1',passed=True,all_completed=True,
        compared_cases=12,source_sha256=refs))
    return dict(schema='residual_slots_publication_v1',candidate_count=454,all_completed=True,
        live_qualified=False,strict_data_ready=False,unseen_validation=False,cases=rows,
        comparison=comparison(cases,original),original_publication=dict(path=old.name,sha256=sha(old)),
        offline_verification=dict(path=proof.name,sha256=sha(proof)),run_manifest=dict(path=manifest.name,sha256=sha(manifest)))


def test_annual_rolling_concentration_and_qualification_are_verified(tmp_path):
    value=publication(tmp_path)
    assert ui.validate(value,tmp_path) is value
    value['comparison']['release_0']['rolling252_win_rate']=1.
    with pytest.raises(ValueError,match='滾動'):
        ui.validate(value,tmp_path)
    value=publication(tmp_path);value['live_qualified']=True
    with pytest.raises(ValueError,match='資格'):
        ui.validate(value,tmp_path)


def test_missing_case_or_proof_mismatch_hides_result(tmp_path):
    value=publication(tmp_path);del value['cases']['release_7']
    with pytest.raises(ValueError,match='完整因素'):
        ui.validate(value,tmp_path)
    value=publication(tmp_path)
    value['cases']['release_7']['result']=value['cases']['release_0']['result']
    with pytest.raises(ValueError,match='標籤'):
        ui.validate(value,tmp_path)


def test_source_change_invalidates_cache(monkeypatch,tmp_path):
    leaf=tmp_path/'source.txt';leaf.write_text('source')
    report=tmp_path/'report.json';write(report,dict(source_sha256={leaf.name:sha(leaf)}))
    report.with_suffix('.sha256').write_text(sha(report))
    monkeypatch.setattr(ui,'validate',lambda value,root:value)
    ui.load(report,tmp_path);leaf.write_text('changed')
    with pytest.raises(ValueError,match='版本已改變'):
        ui.load(report,tmp_path)


def test_display_all_scenarios_details_and_existing_broker_records(monkeypatch,tmp_path):
    value=publication(tmp_path)
    monkeypatch.setattr(ui,'ROOT',tmp_path);monkeypatch.setattr(ui,'REPORT',Path(__file__))
    monkeypatch.setattr(ui,'load',lambda:deepcopy(value))
    app=AppTest.from_string('from app.residual_slots_ui import render\nrender()').run()
    assert not app.exception
    assert app.dataframe[0].value['情境'].tolist()==list(ui.LABELS)
    assert any('尚未取得實戰資格' in row.value for row in app.warning)
    app.selectbox[0].set_value(7).run();app.checkbox[0].check().run()
    assert not app.exception and len(app.get('download_button'))==4
    assert any('大戶投既有紀錄核對' in row.value for row in app.subheader)


def test_missing_evidence_never_displays_returns(monkeypatch,tmp_path):
    monkeypatch.setattr(ui,'REPORT',tmp_path/'missing.json')
    app=AppTest.from_string('from app.residual_slots_ui import render\nrender()').run()
    assert not app.exception and not app.dataframe
