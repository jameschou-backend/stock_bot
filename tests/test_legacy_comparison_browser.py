"""Exercise actual renderer mode changes, archived references and stale requests."""
import json
from pathlib import Path
import shutil
import subprocess


def test_modes_references_errors_and_stale_detail_responses():
    script=(Path(__file__).resolve().parents[1]/'ui/research_terminal/legacy-comparison.js').read_text()
    node=shutil.which('node');assert node,'Install Node.js for renderer verification'
    harness=r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
class E {
 constructor(tag,text='',cls=''){this.tagName=tag;this._text=text||'';this.children=[];this.attrs={};this.listeners={};this.value='';}
 append(...cs){this.children.push(...cs);if(this.tagName==='select'&&!this.value)this.value=this.children[0].value;}
 replaceChildren(...cs){this.children=cs;this._text='';}
 get textContent(){return this._text+this.children.map(e=>e.textContent).join('');}
 set textContent(v){this._text=v;}
 set innerHTML(v){throw Error('unsafe HTML');}
 setAttribute(k,v){this.attrs[k]=v;}
 addEventListener(k,v){this.listeners[k]=v;}
 scrollIntoView(){}
}
const context={window:{}};vm.runInNewContext(JSON.parse(require('node:fs').readFileSync(0,'utf8')),context);
const el=(...a)=>new E(...a),host=el('section'),walk=e=>[e,...e.children.flatMap(walk)];
const modes={baseline:{commission:.001425,slippage:.0045},fee2:{commission:.00285,slippage:.0045},slip2:{commission:.001425,slippage:.009},combined:{commission:.00285,slippage:.009}};
const cases={},hard={};
for(const mode of Object.keys(modes))for(const arm of ['benchmark','liquid_universe','poc_priority','poc_up_red']){
 const row={name:arm,completed:true,total_return:.3,excess_return:.1,max_drawdown:-.2,annual_excess:[{excess:0},{excess:.1},{excess:.2}],metrics:{costs:{total_cost:100},funded_events:42},data_gap_exclusions:2,selection_fallback_days:3};
 if(arm==='poc_up_red')hard[mode]=row;else cases[`${arm}:${mode}`]=row;
}
cases['poc_priority:combined']={name:'POC',completed:false,reason:'missing data'};
const report={start:'2024-01-02',end:'2026-10-02',initial_cash:1000000,completed_cases:11,cases,modes,hard_filter_reference:hard,diagnostics:{},old_versions:[],differences:['<script>literal</script>'],candidate_differences:{old_count:1,current_count:2,added:['x'],removed:[],changed:[]},candidate_count:2,trial_accounting:{total_recorded_attempts:12,incomplete_attempts:1},limitations:['already seen']};
report.path_differences={liquid_universe:{old_funded_events:2,new_funded_events:1,common_funded_events:0,old_opening_nav_2024:1902115.9,old_carry_positions:[{stock_id:'1234'}],new_opening_nav_2024:1000000,focus_5386:{old:{trades:[]},new:{trades:[]}},missed_5386_legacy_entries:[{date:'2026-02-11',new_orders:[{failure:'slots_full'}],opening_holdings:[{stock_id:'6443'},{stock_id:'2367'},{stock_id:'3138'}],opening_cash:42}]}};
cases['liquid_universe:combined'].settlement_evidence_notes=['restricted certificate'];
const requests=[];let pending;
const helpers={el,tag:t=>el('span',t),number:String,pct:n=>String(n*100)+'%',api:path=>{requests.push(path);return new Promise(resolve=>pending=resolve);},renderAccountDetail:(d,h)=>h.append(el('p',d.name))};
(async()=>{
 context.window.renderLegacyComparison({id:'legacy_return_comparison_20261012',status:'verified_publication',legacy_comparison:report},host,helpers);
 assert.match(host.textContent,/11 \/ 12/);assert.match(host.textContent,/POC硬篩選（上一輪同口徑）/);
 assert.match(host.textContent,/持股名額已滿/);assert.match(host.textContent,/6443、2367、3138/);
 const select=walk(host).find(e=>e.tagName==='select');
 const detail=walk(host).filter(e=>e.tagName==='button')[3].listeners.click();
 assert.equal(requests[0],'/archive/top3_validation_20261012/cases/poc_up_red_baseline');
 select.value='combined';select.listeners.change();pending({name:'stale curve'});await detail;
 assert.doesNotMatch(host.textContent,/stale curve/);assert.match(host.textContent,/missing data/);assert.match(host.textContent,/0.8999999999999999%|0.9%/);
 const next=walk(host).filter(e=>e.tagName==='button')[1].listeners.click();
 assert.equal(requests[1],'/archive/legacy_return_comparison_20261012/cases/liquid_universe_combined');
 pending({name:'verified curve'});await next;assert.match(host.textContent,/verified curve/);
 assert.match(host.textContent,/restricted certificate/);
 assert.match(host.textContent,/<script>literal<\/script>/);
 context.window.renderLegacyComparison(null,host,helpers);assert.match(host.textContent,/尚未發布/);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    result=subprocess.run([node,'-e',harness],input=json.dumps(script),text=True,capture_output=True)
    assert result.returncode==0,result.stderr
