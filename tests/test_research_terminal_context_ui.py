"""Execute the research terminal's actual JS for contextual source selection."""
import json
from pathlib import Path
import shutil
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / 'ui/research_terminal/app.js'
HARNESS = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element {
 constructor(tag='div'){this.tagName=tag;this.children=[];this._text='';this.listeners={};this.attrs={};this.value='';this.hidden=false;this.disabled=false;this.className='';this.dataset={};this.offsetWidth=0;this.classList={toggle(){}};this.parentElement={querySelector:()=>new Element()};}
 get textContent(){return this._text+this.children.map(x=>x.textContent).join('');}
 set textContent(v){this._text=String(v);this.children=[];}
 get innerHTML(){throw Error('No HTML injection');} set innerHTML(v){throw Error('No HTML injection');}
 get childNodes(){return this.children;}
 append(...rows){this.children.push(...rows);} prepend(...rows){this.children.unshift(...rows);}
 replaceChildren(...rows){this._text='';this.children=[...rows];}
 setAttribute(k,v){this.attrs[k]=v;}
 addEventListener(k,fn){this.listeners[k]=fn;}
 fire(k){return this.listeners[k]?.({target:this});}
 querySelector(){return new Element();}
}
function mount(url='http://localhost/terminal/#signals') {
 const map=new Map(),element=id=>{if(!map.has(id))map.set(id,new Element());return map.get(id);},requests=[];
 const location=new URL(url),context={document:{getElementById:element,createElement:tag=>new Element(tag),createTextNode:text=>{const e=new Element('#text');e.textContent=text;return e;},querySelector:element},location,history:{replaceState(a,b,to){const next=new URL(to,location);for(const k of['href'])location[k]=next[k];}},URL,URLSearchParams,Headers,console,requestAnimationFrame(){},fetch:async(path)=>{requests.push(path);return {ok:true,json:async()=>path.includes('/stocks/')?{stock_id:'6187',name:'萬潤',date:'2026-10-06',candles:[],results:[{strategy_id:'entry_contraction_narrow',status:'not_matched',reasons:['市場廣度未低於 50%'],metrics:{market_breadth_value:.558495}}],assessment:{label:'未符合本組條件'},coverage:{daily_snapshot_available:true}}:{rows:[],total:0,context_summary:input.summary}}}};
 vm.runInNewContext(input.script,context);
 const ui=context.terminalUI;
 ui.state.catalog=[{id:'legacy_course_breakout',name:'400 日價量高點',kind:'entry',status:'active'},{id:'poc_up_red',name:'POC + 紅 K',kind:'entry',status:'active'},{id:'entry_contraction_narrow',name:'前期價格收斂＋市場廣度偏弱',kind:'entry',status:'active',signal_only:true},{id:'entry_peer_narrow',name:'價格同儕強勢＋市場廣度偏弱',kind:'entry',status:'active',signal_only:true}];
 ui.state.overview={dates:['2026-09-29','2026-10-05'],default_date:'2026-10-05',source_end:'2026-10-05',entry_context:{strategy_ids:['entry_contraction_narrow','entry_peer_narrow'],dates:['2026-09-29','2026-10-05','2026-10-06'],start:'2026-09-01',end:'2026-10-06',default_strategy_id:'entry_contraction_narrow'}};
 return {ui,element,context,requests};
}
'''


def run_js(body):
    node = shutil.which('node')
    assert node, 'Install Node.js to validate the research terminal interactions'
    script = SCRIPT.read_text().replace('bind();init();', '''globalThis.terminalUI={state,selectInitialStrategy,populateCatalog,populateSignalScope,renderContextSummary,renderStock,loadStock,loadSignals,backtestEntries,renderRows,renderSignalLifecycle,drawSignalMarker};''')
    summary = dict(market_breadth_value=.5584950029, market_breadth_threshold=.5,
                   market_breadth_coverage=.878162106, parent_candidates=3, matched=0,
                   unknown=0, not_matched=3, explanation='市場廣度 55.85%，未低於 50%，本組無新訊號。',
                   checks=[dict(stock_id='6187', name='萬潤', status='not_matched',
                                reasons=['市場廣度未低於 50%'])])
    result = subprocess.run([node, '-e', HARNESS + '\n' + body],
                            input=json.dumps(dict(script=script, summary=summary)),
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_context_default_and_strategy_dates_stay_separate_from_ordinary_sources():
    run_js(r'''
const {ui,element,context}=mount();
ui.selectInitialStrategy();ui.populateCatalog();ui.populateSignalScope();
assert.equal(ui.state.strategy,'entry_contraction_narrow');assert.equal(ui.state.date,'2026-10-06');
assert.match(element('source-asof').textContent,/情境研究.*2026-10-06/);
assert.equal(element('first-only').disabled,true);
assert.equal(new URLSearchParams(context.location.search).get('date'),'2026-10-06');
assert.deepEqual(Array.from(element('backtest-strategy').children,x=>x.value),['legacy_course_breakout','poc_up_red']);
ui.state.strategy='all';ui.populateSignalScope();
assert.equal(ui.state.date,'2026-10-05');
assert.match(element('source-asof').textContent,/一般策略.*2026-10-05/);
assert.match(element('signal-scope').textContent,/不含.*前期價格收斂/);
assert.equal(element('first-only').disabled,false);
''')


def test_deep_links_select_requested_rule_and_date_and_reject_unknown_rule():
    run_js(r'''
const {ui,element}=mount('http://localhost/terminal/?strategy=entry_peer_narrow&date=2026-09-29#signals');
ui.selectInitialStrategy();ui.populateCatalog();ui.populateSignalScope();
assert.equal(ui.state.strategy,'entry_peer_narrow');assert.equal(ui.state.date,'2026-09-29');
assert.equal(element('strategy-filter').value,'entry_peer_narrow');
const all=mount('http://localhost/terminal/?strategy=all#signals');all.ui.selectInitialStrategy();all.ui.populateSignalScope();
assert.equal(all.ui.state.strategy,'all');assert.equal(all.ui.state.date,'2026-10-05');
assert.throws(()=>mount('http://localhost/terminal/?strategy=missing#signals').ui.selectInitialStrategy(),/策略不存在/);
''')


def test_zero_match_loads_failed_parent_chart_and_preserves_failure_reasons():
    run_js(r'''
(async()=>{
 const {ui,element,requests}=mount();ui.selectInitialStrategy();ui.populateSignalScope();await ui.loadSignals();
 assert.match(requests[0],/strategy_id=entry_contraction_narrow/);
 assert.match(requests[1],/stocks\/6187\?date=2026-10-06.*strategy_id=entry_contraction_narrow/);
 assert.equal(ui.state.rows.length,0);assert.equal(element('stat-matched').textContent,'0檔');
 assert.equal(ui.state.selected,'6187');assert.equal(element('stock-detail').hidden,false);
 assert.match(element('stock-signals').textContent,/未符合/);assert.match(element('stock-signals').textContent,/市場廣度未低於 50%/);
 assert.equal(element('entry-context').hidden,false);assert.match(element('entry-context').textContent,/55.85%/);
 assert.match(element('entry-context').textContent,/87.82%/);
 assert.match(element('signal-empty').textContent,/上方市場廣度/);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_unknown_context_status_and_untrusted_names_remain_text_only():
    run_js(r'''
const {ui,element}=mount();ui.selectInitialStrategy();
ui.renderContextSummary({...input.summary,unknown:1,not_matched:2,checks:[{stock_id:'6187',name:'<img src=x onerror=alert(1)>',status:'unknown',reasons:['資料不足']}]});
assert.match(element('entry-context').textContent,/資料待核 1 檔/);
assert.match(element('entry-context').textContent,/<img src=x onerror=alert\(1\)>/);
assert.match(element('entry-context').textContent,/資料不足/);
ui.state.strategy='all';ui.renderContextSummary(null);assert.equal(element('entry-context').hidden,true);
''')


def test_context_no_parent_event_is_known_absence_not_missing_daily_snapshot():
    run_js(r'''
const {ui,element}=mount();ui.selectInitialStrategy();
ui.state.chartStrategy='entry_peer_narrow';
const data={stock_id:'6505',name:'台塑化',date:'2026-10-06',candles:[],results:[],assessment:{status:'no_parent_first_event',label:'當日沒有 400 日價量高點的已知首日事件'},coverage:{daily_snapshot_available:false}};
ui.renderStock(data);
assert.match(element('stock-signals').textContent,/當日沒有 400 日價量高點的首次訊號/);
assert.match(element('stock-signals').textContent,/不列入本組訊號/);
assert.doesNotMatch(element('stock-signals').textContent,/當日完整掃描未封存/);
ui.state.chartStrategy='all';ui.renderStock({...data,assessment:{status:'data_incomplete'}});
assert.match(element('stock-signals').textContent,/當日完整掃描未封存/);
ui.populateSignalScope();
assert.match(element('signal-scope').textContent,/400 日價量高點訊號/);
assert.doesNotMatch(element('signal-scope').textContent,/原始價量/);
''')


def test_lifecycle_card_and_rows_identify_retrigger_without_claiming_trade_return():
    run_js(r'''
const {ui,element}=mount();ui.state.chartStrategy='poc_up_red';
const lifecycle={state:'retrigger',label:'再次觸發',explanation:'前一交易日未符合，本日再次符合',previous_trigger_date:'2026-10-02',sessions_since_previous_trigger:5,previous_trigger_close:48.15,current_close:57.9,price_change_since_previous_trigger:.2024922,window_start:'2026-04-16',window_end:'2026-10-12',observed_sessions:120,prior_known_triggers:4,coverage_note:'未記錄不代表未發生'};
const result={strategy_id:'poc_up_red',name:'POC 上移＋紅 K',status:'matched',first_signal:true,lifecycle,metrics:{poc_before:43.228125,poc_after:52.974375}};
ui.state.rows=[{stock_id:'4927',name:'泰鼎-KY',matches:[result],first_count:1,matched_count:1}];ui.renderRows();
assert.match(element('signal-rows').textContent,/再次觸發/);assert.match(element('signal-rows').textContent,/前次 10\/02/);assert.doesNotMatch(element('signal-rows').textContent,/新訊號/);
ui.renderSignalLifecycle({date:'2026-10-12',results:[result],candles:[{date:'2026-10-12',open:54.5,close:57.9}]});
const text=element('signal-lifecycle').textContent;
assert.match(text,/相隔 5 個交易日/);assert.match(text,/20.25%/);assert.match(text,/不是策略進場報酬/);assert.match(text,/43.2281.*52.9744/);assert.match(text,/120 個交易日/);
ui.state.chartStrategy='all';ui.renderSignalLifecycle({results:[result]});assert.equal(element('signal-lifecycle').hidden,true);
''')


def test_selected_rule_reasons_do_not_mix_other_rule_matches():
    run_js(r'''
const {ui,element}=mount();ui.state.chartStrategy='poc_up_red';
ui.renderStock({stock_id:'4927',date:'2026-10-12',candles:[],results:[{strategy_id:'poc_up_red',name:'POC',status:'matched',reasons:['所選條件']},{strategy_id:'momentum',name:'動能',status:'matched',reasons:['另一套條件']}],coverage:{}});
assert.match(element('stock-signals').textContent,/所選條件/);assert.doesNotMatch(element('stock-signals').textContent,/另一套條件/);
''')


def test_chart_lifecycle_markers_use_distinct_shapes_and_unknown_is_hollow():
    run_js(r'''
const {ui}=mount();function trace(type){const calls=[];const ctx=new Proxy({},{set(t,k,v){t[k]=v;return true;},get(t,k){return t[k]||((...args)=>calls.push([k,...args]));}});ui.drawSignalMarker(ctx,type,10,20);return calls;}
const first=trace('first_observed'),repeat=trace('retrigger'),ongoing=trace('continuing'),unknown=trace('unknown');
assert.equal(first.filter(x=>x[0]==='lineTo').length,2);assert.equal(repeat.filter(x=>x[0]==='lineTo').length,3);
assert(ongoing.some(x=>x[0]==='arc'));assert(ongoing.some(x=>x[0]==='fill'));assert(unknown.some(x=>x[0]==='stroke'));assert(!unknown.some(x=>x[0]==='fill'));
''')
