"""Execute the independent daily-POC UI; no network or account recalculation."""
import json
from pathlib import Path
import re
import shutil
import subprocess

TEMPLATE=Path(__file__).resolve().parents[1]/'ui/poc_daily_signal_explorer.html'
HARNESS=r"""
const assert=require('node:assert/strict'),vm=require('node:vm');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const dates=['2026-09-10','2026-09-11','2026-10-02'];
function prior20(day){const out=[],d=new Date(day+'T00:00:00Z');while(out.length<20){d.setUTCDate(d.getUTCDate()-1);if(d.getUTCDay()!==0&&d.getUTCDay()!==6)out.unshift(d.toISOString().slice(0,10));}return out;}
const signals=['up','down','unknown','pending_data'].map((status,i)=>({signal_id:'s'+i,stock_id:String(2330+i),name:'股票'+i,
 signal_date:dates[0],entry_date:dates[1],daily_rank:i+1,candidate_count:4,candle:i===1?'black':'red',priority:.3-i*.02,volume_ratio:2}));
signals.push({...signals[0],signal_id:'terminal',signal_date:dates[2],entry_date:null,daily_rank:1,candidate_count:1});
const statuses=['up','down','unknown','pending_data','up'];
const opportunities=Object.fromEntries(signals.map((s,i)=>{const ds=prior20(s.signal_date),known=i<2||i===4;return [s.signal_id,{
 signal_id:s.signal_id,stock_id:s.stock_id,signal_date:s.signal_date,status:statuses[i],available:known,
 reason:i===2?'ordinary_tape_conflict':i===3?'raw_tape_unavailable':null,prior_dates:ds,
 window_start:ds[0],window_end:ds[19],source_date_end:ds[19],poc_before:known?100:null,poc_after:known?(i===1?95:105):null,
 account_independent:true,reconstructed:true,computed_at:'2026-10-04T01:00:00+00:00'}];}));
const stocks=Object.fromEntries(signals.map(s=>[s.stock_id,{name:s.name,prices:['2026-09-09',...dates,'2026-10-05'].map(date=>
 ({date,raw_open:10,raw_high:12,raw_low:9,raw_close:11,adjusted_close:11,adjustment_factor:1,volume:1000}))}]));
const decisions=Object.fromEntries(signals.map(s=>[s.signal_id,{pending_entry:!s.entry_date,poc_status:'not_evaluated',
 poc_reason:s.entry_date?'not_needed_resource_exhausted':'pending_next_session',selection_status:s.entry_date?'not_needed_resource_exhausted':'pending_next_session',
 selection_reason:s.entry_date?'opening_slots_locked':'no_observed_next_market_session',simulated_buy_qty:0,simulated_buy_dates:[]}]));
const arm={label:'POC＋紅K',decisions,summary:{total_return:1,max_drawdown:-.2},account_days:Object.fromEntries(dates.map(date=>[date,{holdings:[],nav:1000000}])) ,
 trades:[{stock_id:'2330',date:'2026-09-11',side:'buy',qty:1,reference_price:11}]};
const fixture={metadata:{date_end:'2026-10-02',data_as_of:'2026-10-02',historical_end:'2026-09-09',account_start:'2024-01-02'},
 opportunity_metadata:{computed_at:'2026-10-04T01:00:00+00:00'},opportunities,default_strategy:'poc_red',
 days:dates.map(date=>({date,signal_status:date===dates[2]?'pending_next_session':'complete'})),signals,stocks,
 strategies:Object.fromEntries(['poc_red','poc_base','original'].map(id=>[id,JSON.parse(JSON.stringify({...arm,label:id}))]))};
function mount(data=input.payload||fixture){
 const elements=new Map(),element=key=>{if(!elements.has(key))elements.set(key,{textContent:'',innerHTML:'',hidden:false,value:'',
 classList:{toggle(){}},setAttribute(){},addEventListener(){},querySelector(){return null;}});return elements.get(key);};
 element('signal-data').textContent=JSON.stringify(data);
 const context={document:{getElementById:element,querySelector:element,querySelectorAll:()=>[],addEventListener(){}},
 requestAnimationFrame(){},ResizeObserver:class{observe(){}},window:{devicePixelRatio:1},console};
 vm.runInNewContext(input.script,context);return {ui:context.testUI,element};
}
"""


def run_js(body,payload=None):
    node=shutil.which('node')
    assert node,'Install Node.js to verify the standalone daily POC explorer'
    script=re.findall(r'<script>([\s\S]*?)</script>',TEMPLATE.read_text())[0]
    script=script.replace('})();','globalThis.testUI={data,state,selectDay,selectSignal,renderAll,chartBars,opportunity,setRankFilter};})();')
    result=subprocess.run([node,'-e',HARNESS+'\n'+body],input=json.dumps({'script':script,'payload':payload}),text=True,capture_output=True)
    assert result.returncode==0,result.stderr


def test_daily_poc_is_visible_despite_full_account_and_historical_mode():
    run_js(r"""
const {ui,element}=mount();ui.selectDay(dates[0]);
assert.equal(ui.state.asof,true);
assert.equal(element('stat-day').textContent,'4');assert.equal(element('stat-signals').textContent,'1');assert.equal(element('stat-stocks').textContent,'2');
assert.match(element('candidate-body').innerHTML,/badge up/);
assert.match(element('chart-foot').innerHTML,/100.00 → 105.00/);
assert.match(element('chart-foot').innerHTML,/依訊號前 20 日資料重建/);
assert(!element('chart-foot').innerHTML.includes('未評估'));
assert(!element('account-decision').innerHTML.includes('持倉已占滿'));
assert.match(element('account-decision').innerHTML,/隱藏後續/);
assert(!element('stock-trades').innerHTML.includes('2026-09-11'));
assert.equal(ui.chartBars().at(-1).date,dates[0]);
""")


def test_poc_remains_identical_across_strategy_candle_and_account_mutations():
    run_js(r"""
const {ui,element}=mount();const values=JSON.stringify(ui.data.opportunities),before=element('chart-foot').innerHTML;
const saved=JSON.stringify(ui.data.strategies);
ui.renderAll();assert.equal(JSON.stringify(ui.data.strategies),saved);
ui.data.strategies.poc_red.decisions.s0.poc_status='down';ui.data.strategies.poc_red.decisions.s0.poc_before=888;ui.data.strategies.poc_red.decisions.s0.poc_after=1;
ui.data.strategies.poc_red.decisions.s0.simulated_buy_qty=999;ui.renderAll();
assert.equal(element('chart-foot').innerHTML,before);assert(!element('candidate-body').innerHTML.includes('888'));
ui.state.strategy='original';ui.renderAll();assert.equal(element('chart-foot').innerHTML,before);
assert.equal(element('stat-signals').textContent,'1');assert.equal(JSON.stringify(ui.data.opportunities),values);
ui.state.strategy='poc_red';ui.selectSignal(ui.data.signals.find(s=>s.signal_id==='s1'));
assert.match(element('chart-foot').innerHTML,/100.00 → 95.00/);assert.match(element('signal-label').innerHTML,/非紅 K/);
ui.setRankFilter(true);assert(!element('candidate-body').innerHTML.includes('股票1'));
assert.equal(element('stat-day').textContent,'4');assert.equal(element('stat-signals').textContent,'1');
""")


def test_terminal_poc_known_while_next_entry_unknown_and_account_review_separate():
    run_js(r"""
const {ui,element}=mount();ui.selectDay(dates[2]);
assert.match(element('candidate-body').innerHTML,/badge up/);assert(!element('candidate-body').innerHTML.includes('待下一交易日'));
assert.match(element('chart-foot').innerHTML,/100.00 → 105.00/);assert.match(element('chart-foot').innerHTML,/日期與成交尚未知/);
assert.equal(ui.chartBars().at(-1).date,dates[2]);
ui.state.asof=false;ui.renderAll();assert.match(element('account-decision').innerHTML,/尚無下一交易日可供模擬成交/);
assert.equal(ui.chartBars().at(-1).date,dates[2]);assert.match(element('chart-foot').innerHTML,/100.00 → 105.00/);
ui.selectDay(dates[0]);assert.match(element('account-decision').innerHTML,/既有持倉已占滿名額/);
assert(!element('chart-foot').innerHTML.includes('占滿名額'));
""")


def test_quality_unknown_and_pending_are_distinct_never_false_poc():
    run_js(r"""
const {ui,element}=mount();
assert.match(element('candidate-body').innerHTML,/資料品質不符/);assert.match(element('candidate-body').innerHTML,/待補資料/);
assert.match(element('daily-poc-note').textContent,/未上移 1 · 品質不符 1 · 待補 1/);
ui.selectSignal(ui.data.signals.find(s=>s.signal_id==='s2'));assert.match(element('chart-foot').innerHTML,/官方普通盤行情不一致/);
assert(!element('chart-foot').innerHTML.includes(' → '));
ui.selectSignal(ui.data.signals.find(s=>s.signal_id==='s3'));assert.match(element('chart-foot').innerHTML,/逐筆成交資料尚未齊全/);
assert(!element('chart-foot').innerHTML.includes('未上移'));
""")


def test_missing_opportunity_record_fails_closed_without_account_substitution():
    run_js(r"""
const data=JSON.parse(JSON.stringify(fixture));delete data.opportunities.s0;data.strategies.poc_red.decisions.s0={poc_status:'up',poc_before:99,poc_after:100};
const {ui,element}=mount(data);assert.equal(element('stat-signals').textContent,'0');assert.equal(element('stat-stocks').textContent,'3');
assert.match(element('chart-foot').innerHTML,/缺少每日 POC 紀錄/);assert(!element('chart-foot').innerHTML.includes('99.00'));
const broken=JSON.parse(JSON.stringify(fixture));delete broken.opportunities;const x=mount(broken);
assert.equal(x.ui,undefined);assert.match(x.element('workspace').innerHTML,/資料格式不完整/);
""")


def test_future_window_wrong_identity_and_direction_cannot_count_as_up():
    run_js(r"""
for(const mutate of [o=>{o.prior_dates[19]=dates[0];o.window_end=dates[0];o.source_date_end=dates[0];},
 o=>{o.stock_id='9999';},o=>{o.poc_after=90;},o=>{o.account_independent=false;}]){
 const data=JSON.parse(JSON.stringify(fixture));mutate(data.opportunities.s0);const {ui,element}=mount(data);
 assert.equal(element('stat-signals').textContent,'0');assert.match(element('chart-foot').innerHTML,/紀錄不完整/);
 assert(!element('chart-foot').innerHTML.includes('100.00 → 105.00'));
}
""")


def test_zero_signal_day_clears_detail_and_opportunity_counts():
    run_js(r"""
const {ui,element}=mount();ui.selectDay(dates[1]);
assert.equal(element('stat-day').textContent,'0');assert.equal(element('stat-signals').textContent,'0');assert.equal(element('stat-stocks').textContent,'0');
assert.equal(element('candidate-body').innerHTML,'');assert.equal(element('detail-content').hidden,true);assert.equal(ui.state.signal,null);
""")


def test_new_template_has_one_marker_and_honest_historical_reconstruction_copy():
    text=TEMPLATE.read_text()
    assert text.count('<!-- SIGNAL_DATA -->')==1
    assert '每日 POC 為依訊號前 20 日資料事後重建，並非當時即時發布或可用性的認證' in text
    assert '僅帳戶有空位時按需要評估' not in text
    assert '回測帳戶執行' in text and 'id="account-decision"' in text
    assert '原始價格已比對，還原價與成交量仍有認證限制' in text


def test_real_exporter_payload_integrates_calendar_and_account_archive_contract():
    from test_poc_daily_explorer import fixture
    from scripts.export_poc_daily_explorer import build_payload
    base,profiles=fixture()
    for i,signal in enumerate(base['signals']):
        signal.update(name='股票'+str(i),daily_rank=i+1,priority=.2,volume_ratio=2)
    for strategy in base['strategies'].values():strategy.update(account_days={},label='紅K帳戶')
    base['metadata'].update(account_start='2024-01-02',historical_end='2026-09-09',
                            limitations=['POC只顯示本版本真實查詢；未查不等於未上移。'])
    payload=build_payload(base,profiles)
    run_js(r"""
const {ui,element}=mount();
assert.equal(ui.opportunity(ui.data.signals[0]).status,'up');
assert.equal(element('stat-signals').textContent,'1');
assert.match(element('chart-foot').innerHTML,/100.00 → 110.00/);
assert.equal(element('archive-link').href,'signal_explorer_account_20261002.html');
assert.match(element('archive-link').textContent,/原三檔帳戶/);
assert.match(element('method-coverage').textContent,/不是全市場股票每日掃描/);
assert(!element('method-data').textContent.includes('POC只顯示本版本真實查詢'));
assert.match(element('method-data').textContent,/與三檔帳戶名額無關/);
ui.selectDay('2026-10-02');
assert.equal(element('stat-signals').textContent,'1');assert.equal(element('stat-stocks').textContent,'1');
assert.match(element('daily-poc-note').textContent,/POC 上移不是完整買進條件/);
""",payload)


def test_pending_reason_codes_have_readable_chinese_explanations():
    run_js(r"""
for(const reason of ['missing_tick_data','request_budget_paused','quota_paused','provider_error','empty','orphaned_started_attempt']){
 const data=JSON.parse(JSON.stringify(fixture));data.opportunities.s3.reason=reason;
 const {ui,element}=mount(data);ui.selectSignal(ui.data.signals.find(s=>s.signal_id==='s3'));
 assert.match(element('chart-foot').innerHTML,/待補資料/);assert(!element('chart-foot').innerHTML.includes(reason));
}
""")
