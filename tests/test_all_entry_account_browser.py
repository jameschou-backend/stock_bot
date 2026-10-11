"""Execute the actual comparison renderer, including its asynchronous details."""
import json
from pathlib import Path
import shutil
import subprocess


def test_comparison_renders_partial_results_and_uses_verified_detail_transport():
    script=(Path(__file__).resolve().parents[1]/'ui/research_terminal/account-comparison.js').read_text()
    node=shutil.which('node')
    assert node, 'Install Node.js to validate account comparison interactions'
    harness=r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
class E {
 constructor(tag,text='',cls=''){this.tagName=tag;this._text=text||'';this.children=[];this.attrs={};this.listeners={};this.value='';this.className=cls;}
 append(...children){this.children.push(...children);if(this.tagName==='select'&&!this.value)this.value=this.children[0].value;}
 replaceChildren(...children){this.children=children;this._text='';}
 get textContent(){return this._text+this.children.map(x=>x.textContent).join('');}
 set innerHTML(v){throw Error('Untrusted HTML injection');}
 setAttribute(k,v){this.attrs[k]=v;}
 addEventListener(k,f){this.listeners[k]=f;}
 scrollIntoView(){}
}
const context={window:{}};vm.runInNewContext(JSON.parse(require('node:fs').readFileSync(0,'utf8')),context);
const el=(...args)=>new E(...args),host=el('section'),walk=e=>[e,...e.children.flatMap(walk)],requests=[];
const metric={total_return:.25,max_drawdown:-.3,cagr:.08,costs:{total_cost:123},mean_daily_cash_weight:.2,maximum_held_symbols:4,days_above_active_slot_count:5};
const completed={completed:true,completed_sessions:666,metrics:metric,annual:[{year:'2024',total_return:.12}],cohorts:{settled:10,win_rate:.6},candidate_coverage:{executable:100}};
const report={start:'2024-01-02',end:'2026-10-02',initial_cash:1000000,completed_strategies:1,best_is_global:false,best_completed_strategy:'one',cases:{
 benchmark:{...completed,id:'benchmark',name:'0050',rank:null},
 one:{...completed,id:'one',name:'<img src=x onerror=bad()>',rank:1},
 two:{id:'two',name:'Incomplete rule',completed:false,metrics:null,completed_sessions:20,rank:null,reason:'Necessary source missing'},
}};
const helpers={el,tag:(text,cls)=>el('span',text,cls),number:n=>n==null?'—':String(n),pct:n=>n==null?'—':`${n*100}%`,api:async path=>{requests.push(path);return {name:'detail'};},renderAccountDetail:(result,dest)=>dest.append(el('p','verified curve'))};
(async()=>{
 context.window.renderAccountComparison({id:'study',status:'verified_publication',comparison:report},host,helpers);
 assert.match(host.textContent,/已完成組別中最高/);assert.match(host.textContent,/尚無策略取得實戰資格/);
 assert.match(host.textContent,/20 日／缺件/);assert.match(host.textContent,/Necessary source missing/);
 assert.match(host.textContent,/<img src=x onerror=bad\(\)>/);
 const rows=walk(host).filter(e=>e.tagName==='tbody')[0].children;
 assert.equal(rows.length,3);assert.match(rows[0].textContent,/0050/);
 const buttons=walk(host).filter(e=>e.tagName==='button');assert.equal(buttons.length,2);
 await buttons[1].listeners.click();assert.deepEqual(requests,['/archive/study/cases/one']);
 assert.match(host.textContent,/verified curve/);
 assert.match(host.textContent,/歷史最多同持 4 檔/);
 const search=walk(host).find(e=>e.tagName==='input');search.value='Incomplete';search.listeners.input();
 assert.equal(walk(host).find(e=>e.tagName==='tbody').children.length,1);
 context.window.renderAccountComparison(null,host,helpers);assert.match(host.textContent,/尚未發布/);assert.doesNotMatch(host.textContent,/25%/);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    result=subprocess.run([node,'-e',harness],input=json.dumps(script),text=True,capture_output=True)
    assert result.returncode==0,result.stderr
