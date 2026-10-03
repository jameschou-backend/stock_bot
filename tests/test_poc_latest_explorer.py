from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from scripts.export_poc_latest_explorer import (ARMS, build_payload, load_latest,
    pending_decision, report_entries, terminal_signals)
from scripts.export_signal_explorer import digest
from test_poc_signal_explorer_data import fixture_data, case
from test_poc_latest_account import event, write_partitioned_bundle


def latest_fixture():
    entries,a,q,eligible,names=fixture_data()
    pending=deepcopy(entries[0])
    pending.update(event_id='p2222',signal_date='2026-01-06',entry_date=None)
    entries.append(pending)
    cases={'original':case(original=True),'poc_base':case(),'poc_red':case(red=True),'benchmark':case(original=True)}
    return entries,cases,{arm:[] for arm in ARMS},a,a.copy(),q,eligible,names


def test_exporter_reads_the_builders_separate_pending_file(tmp_path):
    rows=[event(),event('pending','2026-10-02',None)]
    active,pending=write_partitioned_bundle(tmp_path,rows)
    report=dict(pending_terminal_signals=pending,source_signal_count=len(rows))
    assert report_entries(tmp_path,report)==active+pending==rows
    for bad in ({**report,'pending_terminal_signals':[]},{**report,'source_signal_count':len(active)}):
        with pytest.raises(ValueError,match='sealed terminal'):
            report_entries(tmp_path,bad)


def test_terminal_signal_is_kept_with_known_candle_and_unknown_future_entry():
    payload=build_payload(*latest_fixture(),end='2026-01-06')
    final=payload['days'][-1]
    assert final['signal_count']==1
    assert final['signal_status']=='pending_next_session'
    s=payload['signals'][-1]
    assert s['signal_id']=='p2222' and s['entry_date'] is None and s['candle']=='black'
    assert s['daily_rank']==1 and s['candidate_count']==1
    assert payload['metadata']['pending_signal_count']==1
    assert payload['metadata']['date_end']=='2026-01-06'
    assert payload['metadata']['zero_signal_days']==1
    assert any('21筆TWSE全日成交量' in note and '原因未釐清' in note
               for note in payload['metadata']['limitations'])
    for arm,strategy in payload['strategies'].items():
        d=strategy['decisions']['p2222']
        assert d['poc_status']=='not_evaluated' and d['poc_reason']=='pending_next_session'
        assert d['pending_entry'] and d['entry_date'] is None
        assert d['simulated_buy_qty']==0 and not d['selected_for_planning']
        assert d['red_gate'] is (False if arm=='poc_red' else None)
    stock=payload['stocks']['2222']
    assert stock['signal_ids'][-1]=='p2222' and stock['prices'][-1][7]==s['signal_close_adjusted']


def test_terminal_red_gate_is_evaluated_without_calling_profile_or_guessing_date():
    s={'signal_id':'p','candle':'red'}
    d=pending_decision(s,True)
    assert d['red_gate'] is True and d['red_gate_status']=='red'
    assert d['selection_status']=='pending_next_session'
    assert d['selection_available_at'] is None
    assert not {'poc_before','poc_after','poc_window_end'}.intersection(d)


def test_pending_never_accepts_a_modeled_trade_or_future_profile_query():
    data=list(latest_fixture())
    data[1]['poc_red']['account']['trades']=[dict(date='2026-01-06',stock_id='2222',
        event_id='p2222',signal_date='2026-01-06',side='buy',qty=5,reference_price=125)]
    with pytest.raises(ValueError,match='already has a modeled fill'):
        build_payload(*data,end='2026-01-06')
    data=list(latest_fixture())
    data[1]['poc_red']['profile_queries']=[dict(event_id='p2222',stock_id='2222',signal_date='2026-01-06')]
    with pytest.raises(ValueError,match='future POC planning query'):
        build_payload(*data,end='2026-01-06')


def test_old_trades_and_opening_inventory_keep_their_actual_date_scope():
    data=list(latest_fixture())
    for c in data[1].values():
        c['account']['trades']=[dict(date=day,stock_id='3333',event_id='old2025',
            signal_date='2025-12-29',side='sell',qty=1,reference_price=140)
            for day in ['2025-12-31','2026-01-02','2026-01-05']]
    p=build_payload(*data,end='2026-01-06')
    for s in p['strategies'].values():
        assert [t['date'] for t in s['trades']]==['2026-01-02','2026-01-05']
        assert [t['date'] for t in s['trades'] if t['date']<='2026-01-02']==['2026-01-02']
        assert s['opening_inventory'][0]['event_id']=='old2025'
        assert s['opening_account_day']['nav']==2e6


def test_nonterminal_pending_cannot_silently_become_a_zero_signal_day():
    entries,a,q,_,names=fixture_data()
    bad=deepcopy(entries[0]);bad['entry_date']=None
    with pytest.raises(ValueError,match='Unresolved non-terminal'):
        terminal_signals([bad],a,q,names,end='2026-01-06')


def test_complete_four_arm_parent_and_verified_prefix_are_required(tmp_path):
    folder=tmp_path/'.cache/poc-latest-20261003/test'
    folder.mkdir(parents=True)
    p=folder/'report.json'
    report=dict(all_completed=False,cases={},start='2024-01-02',end='2026-10-02',
        input_bundle='.cache/poc-latest-20261003/inputs-v1',live_qualified=False)
    p.write_text(json.dumps(report));p.with_suffix('.sha256').write_text(digest(p))
    with pytest.raises(ValueError,match='all four complete'):
        load_latest(p,tmp_path)
    report.update(all_completed=True,cases={a:{'completed':True,'prefix_parity':{'all_exact':False}} for a in ARMS},
        source_sha256={},profile_data={'schema':'poc_latest_profiles_v1'})
    p.write_text(json.dumps(report));p.with_suffix('.sha256').write_text(digest(p))
    with pytest.raises(ValueError,match='prefix equality'):
        load_latest(p,tmp_path)


def test_latest_template_has_pending_state_and_dynamic_period():
    template=(Path(__file__).resolve().parents[1]/'ui/poc_latest_signal_explorer.html').read_text()
    assert template.count('<!-- SIGNAL_DATA -->')==1
    assert "d.pending_entry?'待下一交易日'" in template
    assert 's.entry_date?esc(s.entry_date)' in template
    assert "meta.account_start+'—'+dateEnd" in template
    assert '資料只到 9/9' not in template and '2024/1/2—2026/9/9' not in template
    assert "'archive-link'" in template
    assert '尚未生成新' not in template
    assert '本圖含延伸行情；原始價格已比對，還原價與成交量仍有認證限制。' in template
    assert '尚未完成官方交叉核對' not in template


def test_actual_ui_pending_render_and_asof_price_poc_trade_boundaries():
    """Execute the actual template JS with a minimal DOM, without browser/network."""
    node=shutil.which('node')
    assert node, 'Install Node.js to verify the standalone explorer JavaScript'
    template=(Path(__file__).resolve().parents[1]/'ui/poc_latest_signal_explorer.html').read_text()
    script=re.findall(r'<script>([\s\S]*?)</script>',template)[0]
    script=script.replace('})();','globalThis.testUI={data,state,selectDay,renderAll,chartBars};})();')
    harness=r"""
const assert=require('node:assert/strict'),vm=require('node:vm');
const script=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const days=['2026-09-10','2026-09-28','2026-10-02'];
const signal=(id,sid,date,entry)=>({signal_id:id,stock_id:sid,name:'測試股',signal_date:date,
 entry_date:entry,candle:'red',daily_rank:1,candidate_count:21,priority:.2,volume_ratio:2});
const signals=[signal('early','2330',days[0],'2026-09-11'),
 ...Array.from({length:21},(_,i)=>({...signal('pending'+i,String(2330+i),days[2],null),daily_rank:i+1}))];
const stocks=Object.fromEntries(signals.map(s=>[s.stock_id,{name:s.name,
 prices:['2026-09-09','2026-09-10','2026-09-11','2026-09-28','2026-10-02','2026-10-05'].map(date=>
 ({date,raw_open:10,raw_high:12,raw_low:9,raw_close:11,adjusted_close:11,adjustment_factor:1,volume:1000}))}]));
const decisions=Object.fromEntries(signals.map(s=>[s.signal_id,{pending_entry:!s.entry_date,
 poc_status:s.entry_date?'up':'not_evaluated',poc_reason:s.entry_date?'known_true':'pending_next_session',
 poc_before:s.entry_date?900:null,poc_after:s.entry_date?999:null,
 selection_status:s.entry_date?'reserved':'pending_next_session',simulated_buy_qty:0}]));
const arm={label:'POC紅K',decisions,summary:{},account_days:Object.fromEntries(days.map(date=>[date,{holdings:[]}])) ,
 trades:[{stock_id:'2330',date:'2026-09-11',side:'buy',qty:1,reference_price:11}]};
const data={metadata:{date_end:'2026-10-02',data_as_of:'2026-10-02',historical_end:'2026-09-09',account_start:'2024-01-02'},
 default_strategy:'poc_red',days:days.map(date=>({date,signal_status:date===days[2]?'pending_next_session':'complete'})),
 signals,stocks,strategies:{poc_red:arm}};
const elements=new Map();
const element=key=>{if(!elements.has(key))elements.set(key,{textContent:'',innerHTML:'',hidden:false,value:'',
 classList:{toggle(){}},setAttribute(){},addEventListener(){},querySelector(){return null;}});return elements.get(key);};
element('signal-data').textContent=JSON.stringify(data);
const context={document:{getElementById:element,querySelector:element,querySelectorAll:()=>[],addEventListener(){}},
 requestAnimationFrame(){},ResizeObserver:class{observe(){}},window:{devicePixelRatio:1},console};
vm.runInNewContext(script,context);
const ui=context.testUI;
ui.selectDay('2026-10-02');
assert.match(element('date-subtitle').textContent,/本日已產生 21 筆/);
assert.match(element('date-subtitle').textContent,/待下一交易日/);
assert.equal(element('stat-day').textContent,'21');
assert.equal((element('candidate-body').innerHTML.match(/待下一交易日/g)||[]).length,21);
assert(!element('chart-foot').innerHTML.includes('POC：'));
assert.match(element('chart-foot').innerHTML,/日期與成交尚未知/);
assert.equal(ui.chartBars().at(-1).date,'2026-10-02');
ui.selectDay('2026-09-10');
assert.equal(ui.chartBars().at(-1).date,'2026-09-10');
assert(!element('stock-trades').innerHTML.includes('2026-09-11'));
const before=element('candidate-body').innerHTML;
assert(!before.includes('badge up')&&!before.includes('badge warn'));
assert(!element('chart-foot').innerHTML.includes('900'));
// Changing a future planning result must not alter historical rendering.
const actualDecision=ui.data.strategies.poc_red.decisions.early;
actualDecision.poc_status='unknown';actualDecision.poc_before=123;actualDecision.poc_after=456;
ui.renderAll();
assert.equal(element('candidate-body').innerHTML,before);
assert(!element('chart-foot').innerHTML.includes('123'));
actualDecision.poc_status='up';actualDecision.poc_before=900;actualDecision.poc_after=999;
ui.state.asof=false;ui.state.range='all';ui.renderAll();
assert.equal(ui.chartBars().at(-1).date,'2026-10-02');
assert(element('candidate-body').innerHTML.includes('badge up'));
assert(element('chart-foot').innerHTML.includes('900.00 → 999.00'));
assert(element('stock-trades').innerHTML.includes('2026-09-11'));
ui.selectDay('2026-10-02');
assert.match(element('chart-foot').innerHTML,/尚無下一交易日可供模擬成交/);
assert(!element('date-subtitle').textContent.includes('未生成'));
console.log('actual-ui: pending21, asof prices/fills/POC and final cutoff PASS');
"""
    result=subprocess.run([node,'-e',harness],input=json.dumps(script),text=True,capture_output=True,check=True)
    assert 'actual-ui:' in result.stdout
