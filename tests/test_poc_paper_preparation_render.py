import json
from pathlib import Path
import shutil
import subprocess


def test_preparation_shows_blockers_without_orders_or_backtest_action():
    root=Path(__file__).resolve().parents[1]
    node=shutil.which('node')
    assert node,'Install Node.js to validate the research terminal renderer'
    payload=dict(script=(root/'ui/research_terminal/poc-paper-preparation.js').read_text(),
        report=json.loads((root/'artifacts/forward_simulation/poc_original_paper_preparation_20261015.json').read_text()))
    harness=r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element {
 constructor(type,text){this.type=type;this.text=text||'';this.children=[];}
 append(...x){this.children.push(...x);}
 replaceChildren(...x){this.children=x;}
}
const context={window:{}};vm.runInNewContext(input.script,context);
const nodes=x=>[x,...x.children.flatMap(nodes)],host=new Element('main');
const h={el:(t,s)=>new Element(t,s),number:n=>String(n),pct:n=>(n*100).toFixed(2)+'%'};
context.window.renderPocPaperPreparation(null,host,h);
assert.match(nodes(host).map(n=>n.text).join(' '),/尚未通過核對/);
context.window.renderPocPaperPreparation({status:'verified_publication',paper_preparation:input.report,download_url:'/report'},host,h);
const text=nodes(host).map(n=>n.text).join(' ');
assert.match(text,/未回填委託/);assert.match(text,/委託 0 筆、成交 0 筆/);
assert.match(text,/不需先累積126天/);assert.match(text,/269.75%/);assert.match(text,/254.25%/);
assert.match(text,/持續符合/);assert.match(text,/你回報買進日期/);assert.match(text,/若日後需要自動模擬成交/);
assert.equal(nodes(host).filter(n=>n.type==='button').length,0);
'''
    result=subprocess.run([node,'-e',harness],input=json.dumps(payload),text=True,capture_output=True)
    assert result.returncode==0,result.stderr
