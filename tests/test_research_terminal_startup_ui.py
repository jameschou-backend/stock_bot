"""Actual startup must explain cold loads and recover from stalled requests."""
import json
from pathlib import Path
import shutil
import subprocess

from tests.test_research_terminal_context_ui import HARNESS


def run_js(body):
    node = shutil.which('node')
    assert node, 'Install Node.js to validate startup interactions'
    script = (Path(__file__).resolve().parents[1] / 'ui/research_terminal/app.js').read_text()
    script = script.replace('bind();init();', 'globalThis.terminalUI={state,init};')
    setup = r'''
const {ui,element,context}=mount();
const overview=ui.state.overview,catalog=ui.state.catalog;ui.state.overview=null;
context.document.querySelectorAll=()=>[];context.window={scrollTo(){}};
const timers=new Map();let counter=0;
context.setTimeout=(fn,ms)=>{timers.set(++counter,{fn,ms});return counter;};context.clearTimeout=id=>timers.delete(id);
const payload=path=>path.endsWith('/overview')?overview:path.endsWith('/strategies')?{catalog}:{rows:[],total:0};
const ready=path=>({ok:true,json:async()=>payload(path)});
'''
    result = subprocess.run([node, '-e', HARNESS + '\n(async()=>{\n' + setup + body
                             + '\n})().catch(e=>{console.error(e);process.exitCode=1;});'],
                            input=json.dumps(dict(script=script)), text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_cold_load_shows_progress_and_disables_empty_filters_until_dates_ready():
    run_js(r'''
const pending=[];context.fetch=(path,options)=>new Promise(resolve=>pending.push({path,resolve}));
const loading=ui.init();
assert.equal(element('startup-status').hidden,false);assert.match(element('startup-status').textContent,/日期.*策略/);
assert.equal(element('scan-date').disabled,true);assert.equal(element('strategy-filter').disabled,true);
assert.equal(element('refresh').disabled,true);
context.fetch=async path=>ready(path);pending.forEach(r=>r.resolve(ready(r.path)));await loading;
assert.ok(element('scan-date').children.length>0);assert.equal(element('scan-date').disabled,false);
assert.equal(element('strategy-filter').disabled,false);assert.equal(element('first-only').disabled,true);
assert.equal(element('startup-status').hidden,true);assert.equal(element('refresh').disabled,false);
assert.equal(element('global-error').hidden,true);assert.equal(timers.size,0);
''')


def test_timeout_leaves_explicit_error_and_refresh_recovers_same_strategy_date():
    run_js(r'''
context.fetch=(path,options)=>path.endsWith('/overview')?new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')),{once:true})):Promise.resolve(ready(path));
const loading=ui.init();await Promise.resolve();await Promise.resolve();
const timeout=[...timers.values()].find(t=>t.ms===30000);assert.ok(timeout,'startup has a deadline');timeout.fn();await loading;
assert.equal(element('startup-status').hidden,true);assert.equal(element('global-error').hidden,false);
assert.match(element('global-error').textContent,/逾時.*重新載入/);assert.equal(element('scan-date').disabled,true);
assert.equal(element('refresh').disabled,false);assert.equal(timers.size,0);
context.fetch=async path=>ready(path);await ui.init();
assert.equal(element('global-error').hidden,true);assert.equal(element('scan-date').disabled,false);
assert.equal(ui.state.strategy,'entry_contraction_narrow');assert.equal(element('scan-date').value,'2026-10-06');
assert.equal(timers.size,0);
''')
