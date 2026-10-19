"""Run the renderer in a small DOM test double; no browser/network automation."""
import json
from pathlib import Path
import shutil
import subprocess


def test_cap40_costs_and_curve_requests_use_matching_controls():
    node = shutil.which('node')
    assert node, 'Install Node.js to validate the research terminal renderer'
    script = (Path(__file__).resolve().parents[1]/'ui/research_terminal/poc-cap40.js').read_text()
    harness = r'''
const assert=require('node:assert/strict'), vm=require('node:vm');
const script=JSON.parse(require('node:fs').readFileSync(0,'utf8')).script;
class Element {
  constructor(type,text){this.type=type;this.text=text||'';this.children=[];this.dataset={};this.attrs={};this.listeners={};this.value='';}
  append(...nodes){this.children.push(...nodes);if(this.type==='select'&&!this.value&&nodes.length)this.value=nodes[0].value;}
  replaceChildren(...nodes){this.children=[];this.append(...nodes);}
  setAttribute(k,v){this.attrs[k]=v;}
  addEventListener(k,f){this.listeners[k]=f;}
}
const context={window:{}};vm.runInNewContext(script,context);
const nodes=x=>[x,...x.children.flatMap(nodes)];
const host=new Element('main'),requests=[],drawn=[];
const helpers={el:(t,s)=>new Element(t,s),number:(n,d=0)=>Number(n).toFixed(d),pct:n=>(n*100).toFixed(2)+'%',
  api:async path=>{requests.push(path);return {initial_cash:1000000,id:path};},
  renderAccountDetail:(a,h,b)=>{if(b)assert.equal(a.initial_cash,b.initial_cash);drawn.push([a,b]);}};
context.window.renderPocCap40(null,host,helpers);
assert.match(nodes(host).map(n=>n.text).join(' '),/全部核對完成/);
const report={start:'2024-01-02',end:'2026-10-02',cases:{},comparisons:{}};
const modes=['baseline','fee2','slip2','combined'];
for(const [index,mode] of modes.entries()){
  report.comparisons[mode]={};
  for(const [a,v] of [['poc_up_red','original'],['benchmark','original'],['poc_up_red','cap40']]){
    report.cases[`${a}:${v}:${mode}`]={metrics:{total_return:.8-index*.1,max_drawdown:-.3,final_nav:1800000,costs:{total_cost:100},annual:[{year:'2026',partial_year:true,total_return:.4,max_drawdown:-.3}]},cap40_audit:{triggers:7},data_gap_exclusions:1,profile_data_exclusions:2};
  }
  for(const a of ['original','benchmark'])report.comparisons[mode][a]={drawdown_improvement:.1,net_profit_retained:.8,return_change:-.2,profit_change:-200000};
}
context.window.renderPocCap40({id:'cap40',status:'verified_publication',cap40:report,download_url:'/archive/cap40'},host,helpers);
const select=nodes(host).find(n=>n.type==='select');
(async()=>{
for(const mode of modes){
  select.value=mode;select.listeners.change();
  assert.match(nodes(host).map(n=>n.text).join(' '),/未建立委託或成交/);
  const buttons=nodes(host).filter(n=>n.type==='button');assert.equal(buttons.length,3);
  for(const button of buttons){
    requests.length=0;await button.onclick();assert.ok(requests.every(p=>p.endsWith('__'+mode)));
    assert.equal(requests.length,button.text.includes('0050')?1:2);
    assert.ok(requests[0].includes(button.text.includes('單股40%減碼')?'poc_up_red__cap40':button.text.includes('0050')?'benchmark__original':'poc_up_red__original'));
  }
}
assert.equal(drawn.length,12);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    result=subprocess.run([node,'-e',harness],input=json.dumps(dict(script=script)),text=True,capture_output=True)
    assert result.returncode==0,result.stderr
