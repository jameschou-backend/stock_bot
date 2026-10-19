"""Exercise all five episodes and all three fixed diagnostic rules."""
import json
from pathlib import Path
import shutil
import subprocess


def test_warning_renderer_all_fifteen_views_and_no_trading_actions():
    root=Path(__file__).resolve().parents[1]
    node=shutil.which('node')
    assert node, 'Install Node.js for terminal renderer verification'
    harness=r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const {script,report}=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element {
 constructor(type,text){this.type=type;this.text=text||'';this.children=[];this.dataset={};this.value='';this.listeners={};this.attrs={};}
 append(...n){this.children.push(...n);if(this.type==='select'&&!this.value&&n.length)this.value=n[0].value;}
 replaceChildren(...n){this.children=[];this.append(...n);}
 setAttribute(k,v){this.attrs[k]=v;} addEventListener(k,f){this.listeners[k]=f;}
}
const c={window:{}};vm.runInNewContext(script,c);
const host=new Element('main'),walk=x=>[x,...x.children.flatMap(walk)];
const h={el:(t,s)=>new Element(t,s),number:(n,d=0)=>Number(n).toFixed(d),pct:n=>(100*n).toFixed(2)+'%'};
c.window.renderPocWeakness(null,host,h);assert.match(walk(host).map(x=>x.text).join(' '),/尚未完成核對/);
c.window.renderPocWeakness({status:'verified_publication',weakness:report,download_url:'/result'},host,h);
const [episode,rule]=walk(host).filter(x=>x.type==='select');
assert.equal(episode.children.length,5);assert.equal(rule.children.length,3);
for(const e of report.episodes)for(const key of Object.keys(report.rules)){
 episode.value=String(e.rank);rule.value=key;rule.listeners.change();
 const s=walk(host).map(x=>x.text).join(' ');
 assert.ok(s.includes(e.peak_date+' → '+e.trough_date));assert.match(s,/回測重跑0次/);
 assert.match(s,/原版保留/);assert.match(s,/不是交易勝率/);assert.match(s,/沒有減碼後的新帳戶/);
 for(const row of e.rows)assert.ok(s.includes(row.stock_id+' '+row.name+' / '+row.entry_date));
 if(e.rank===5)assert.match(s,/沒有賣出/);
 assert.equal(walk(host).filter(x=>x.type==='button').length,0);
}
'''
    result=subprocess.run([node,'-e',harness],input=json.dumps(dict(
        script=(root/'ui/research_terminal/poc-weakness.js').read_text(),
        report=json.loads((root/'artifacts/forward_simulation/poc_weakness_20261019.json').read_text()))),
        text=True,capture_output=True)
    assert result.returncode==0,result.stderr
