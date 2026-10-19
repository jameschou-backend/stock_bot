"""Exercise all five drilldowns with the actual, hash-sealed publication."""
import json
from pathlib import Path
import shutil
import subprocess


def test_all_five_drawdown_details_show_separate_costs_and_readonly_scope():
    root=Path(__file__).resolve().parents[1]
    node=shutil.which('node')
    assert node, 'Install Node.js for terminal renderer verification'
    harness=r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const {script,report}=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element {
 constructor(type,text){this.type=type;this.text=text||'';this.children=[];this.dataset={};this.value='';this.listeners={};}
 append(...n){this.children.push(...n);if(this.type==='select'&&!this.value&&n.length)this.value=n[0].value;}
 replaceChildren(...n){this.children=[];this.append(...n);}
 setAttribute(){} addEventListener(k,f){this.listeners[k]=f;}
}
const c={window:{}};vm.runInNewContext(script,c);
const host=new Element('main'),walk=x=>[x,...x.children.flatMap(walk)];
const helpers={el:(t,s)=>new Element(t,s),number:(n,d=0)=>Number(n).toFixed(d),pct:n=>(100*n).toFixed(2)+'%'};
c.window.renderPocDrawdown(null,host,helpers);assert.match(walk(host).map(x=>x.text).join(' '),/尚未完成核對/);
c.window.renderPocDrawdown({status:'verified_publication',drawdown_attribution:report,download_url:'/result'},host,helpers);
const select=walk(host).find(x=>x.type==='select');assert.equal(select.children.length,5);
for(const e of report.episodes){select.value=String(e.rank);select.listeners.change();const s=walk(host).map(x=>x.text).join(' ');
 assert.ok(s.includes(e.peak_date+' → '+e.trough_date));assert.match(s,/回測重跑0次/);
 assert.match(s,/手續費/);assert.match(s,/交易稅/);assert.match(s,/滑價/);assert.match(s,/原版策略維持不變/);
 for(const row of e.cohorts)assert.ok(s.includes(row.stock_id+' '+row.name+' / '+row.entry_date));
 assert.equal(walk(host).filter(x=>x.type==='button').length,0);
}
'''
    result=subprocess.run([node,'-e',harness],input=json.dumps(dict(
        script=(root/'ui/research_terminal/poc-drawdown.js').read_text(),
        report=json.loads((root/'artifacts/forward_simulation/poc_drawdown_attribution_20261019.json').read_text()))),
        text=True,capture_output=True)
    assert result.returncode==0,result.stderr
