"""Execute the actual comparison renderer, including its asynchronous details."""
import json
from pathlib import Path
import shutil
import subprocess


def test_stress_selector_and_detail_requests_never_imply_unseen_or_paper_success():
    script=(Path(__file__).resolve().parents[1]/'ui/research_terminal/top3-validation.js').read_text()
    node=shutil.which('node')
    assert node, 'Install Node.js for browser renderer checks'
    harness=r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
class E {
 constructor(tag,text='',cls=''){this.tagName=tag;this._text=text||'';this.children=[];this.attrs={};this.listeners={};this.value='';}
 append(...children){this.children.push(...children);if(this.tagName==='select'&&!this.value)this.value=this.children[0].value;}
 replaceChildren(...children){this.children=children;this._text='';}
 get textContent(){return this._text+this.children.map(x=>x.textContent).join('');}
 set textContent(v){this._text=v;}
 set innerHTML(v){throw Error('Untrusted HTML injection');}
 setAttribute(k,v){this.attrs[k]=v;}
 addEventListener(k,f){this.listeners[k]=f;}
 scrollIntoView(){this.scrolled=true;}
}
const context={window:{}};vm.runInNewContext(JSON.parse(require('node:fs').readFileSync(0,'utf8')),context);
const el=(...args)=>new E(...args),host=el('section'),walk=e=>[e,...e.children.flatMap(walk)],requests=[];
const modes={baseline:{commission:.001425,slippage:.0045},fee2:{commission:.00285,slippage:.0045},slip2:{commission:.001425,slippage:.009},combined:{commission:.00285,slippage:.009}};
const cases={};for(const mode of Object.keys(modes))for(const arm of ['benchmark','obv20_breakout','poc_up_red','entry_peer_narrow']) cases[`${arm}:${mode}`]={name:arm,completed:true,total_return:mode==='baseline'?.3:.1,excess_return:-.1,max_drawdown:-.4,annual_excess:[{excess:0},{excess:.1},{excess:-.2}],metrics:{costs:{total_cost:123}},rolling252:{win_fraction:.5}};
cases['poc_up_red:combined']={name:'POC',completed:false,reason:'required source missing'};
const report={start:'2024-01-02',end:'2026-10-02',initial_cash:1000000,completed_cases:15,cases,modes,out_of_sample:{sessions:0,minimum_sessions:126,minimum_settled_cohorts:30,explanation:'already seen history'},paper:{enabled_strategies:[],explanation:'no hypothetical fills'},qualification:{a:{checks:{historical_stress_pass:false}}},limitations:[],poc_note:'not same version',poc_reconciliation:[{name:'<img src=x>',return_pct:949.76,start:'2019-01-02',end:'2026-09-09',difference:'not POC'}]};
const helpers={el,tag:t=>el('span',t),number:n=>n==null?'—':String(n),pct:n=>n==null?'—':`${Number((n*100).toFixed(4))}%`,api:async path=>{requests.push(path);return {name:'verified'};},renderAccountDetail:(r,h)=>h.append(el('p','verified curve'))};
(async()=>{
 context.window.renderTop3Validation({id:'top3_validation_20261012',status:'verified_publication',validation:report},host,helpers);
 assert.match(host.textContent,/真正未見樣本0 日/);assert.match(host.textContent,/已啟用紙上交易0 套/);
 assert.match(host.textContent,/\(最低＋最高\) ÷ 2/);
 const select=walk(host).find(e=>e.tagName==='select');select.value='combined';select.listeners.change();
 assert.match(host.textContent,/required source missing/);assert.match(host.textContent,/0.9%/);
 const buttons=walk(host).filter(e=>e.tagName==='button');assert.equal(buttons.length,3);
 await buttons[1].listeners.click();assert.equal(requests[0],'/archive/top3_validation_20261012/cases/obv20_breakout_combined');
 assert.match(host.textContent,/verified curve/);assert.match(host.textContent,/<img src=x>/);
 select.value='baseline';select.listeners.change();assert.doesNotMatch(host.textContent,/verified curve/);
 context.window.renderTop3Validation(null,host,helpers);assert.match(host.textContent,/尚未發布/);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    result=subprocess.run([node,'-e',harness],input=json.dumps(script),text=True,capture_output=True)
    assert result.returncode==0,result.stderr
