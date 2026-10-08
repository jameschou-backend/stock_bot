"""Exercise the actual browser JSON transport across a public ngrok share."""

import json
from pathlib import Path
import shutil
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / 'ui/research_terminal/app.js'
HARNESS = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const iphone='Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1';
function mount(url='https://example.ngrok-free.dev/terminal/', responder) {
 const location=new URL(url),requests=[];
 const context={location,navigator:{userAgent:iphone},URL,URLSearchParams,Headers,console,
  fetch:async(path,options={})=>{
   const request={url:new URL(path,location),...options,headers:new Headers(options.headers)};
   requests.push(request);
   if(responder)return responder(request);
   // The public edge can answer browser API requests with an HTML notice, HTTP 200.
   if(request.url.hostname.endsWith('.ngrok-free.dev')&&!request.headers.get('ngrok-skip-browser-warning'))
    return new Response('<!doctype html><title>ngrok</title>ERR_NGROK_6024',{status:200,headers:{'Content-Type':'text/html'}});
   return Response.json({source_end:'2026-10-07',total:12});
  }};
 vm.runInNewContext(input.script,context);
 return {api:context.terminalTransport.api,requests};
}
'''


def run_js(body):
    node = shutil.which('node')
    assert node, 'Install Node.js to validate the research terminal JSON transport'
    script = SCRIPT.read_text()
    assert script.count('bind();init();') == 1, 'Update the actual app.js test entry point'
    script = script.replace('bind();init();', 'globalThis.terminalTransport={api};')
    result = subprocess.run(
        [node, '-e', HARNESS + '\n(async()=>{\n' + body
         + '\n})().catch(e=>{console.error(e);process.exitCode=1;});'],
        input=json.dumps(dict(script=script)), text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_mobile_public_share_reads_json_instead_of_ngrok_html_notice():
    run_js(r'''
const {api,requests}=mount();
const data=await api('/overview');
assert.equal(data.source_end,'2026-10-07');
assert.equal(requests[0].url.pathname,'/research-terminal/api/overview');
assert.equal(requests[0].url.origin,'https://example.ngrok-free.dev');
assert.ok(requests[0].headers.get('ngrok-skip-browser-warning'));
assert.equal(requests[0].headers.get('content-type'),'application/json');
''')


def test_caller_headers_merge_without_losing_transport_headers_method_or_body():
    run_js(r'''
for(const headers of [
 {'X-Request-ID':'transport-test'},
 new Headers({'X-Request-ID':'transport-test'}),
 [['X-Request-ID','transport-test']],
]){
 const originalHeaders=Array.from(new Headers(headers).entries());
 const {api,requests}=mount();
 const body=JSON.stringify({mode:'entry_only',strategy_id:'poc_up_red'});
 await api('/backtests',{method:'POST',headers,body,credentials:'same-origin'});
 const request=requests[0];
 assert.equal(request.method,'POST');
 assert.equal(request.body,body);
 assert.equal(request.credentials,'same-origin');
 assert.equal(request.headers.get('x-request-id'),'transport-test');
 assert.equal(request.headers.get('content-type'),'application/json');
 assert.ok(request.headers.get('ngrok-skip-browser-warning'));
 assert.deepEqual(Array.from(new Headers(headers).entries()),originalHeaders);
}
''')


def test_local_json_transport_keeps_paths_queries_and_response_data():
    run_js(r'''
const {api,requests}=mount('http://127.0.0.1:8000/terminal/');
const data=await api('/signals?date=2026-10-07&strategy_id=poc_up_red');
assert.equal(data.total,12);
assert.equal(requests[0].url.origin,'http://127.0.0.1:8000');
assert.equal(requests[0].url.pathname,'/research-terminal/api/signals');
assert.equal(requests[0].url.searchParams.get('date'),'2026-10-07');
assert.equal(requests[0].headers.get('content-type'),'application/json');
''')


def test_json_http_failures_preserve_server_detail():
    run_js(r'''
const denied=mount(undefined,()=>Response.json({detail:'Read-only share'},{status:403}));
await assert.rejects(()=>denied.api('/backtests'),error=>error.message==='Read-only share');
const validation=mount(undefined,()=>Response.json({detail:{field:'date',error:'invalid'}},{status:422}));
await assert.rejects(()=>validation.api('/signals'),error=>error.message==='{"field":"date","error":"invalid"}');
const unavailable=mount(undefined,()=>Response.json({},{status:503}));
await assert.rejects(()=>unavailable.api('/overview'),error=>error.message==='HTTP 503');
''')


def test_explicit_caller_content_type_is_preserved():
    run_js(r'''
const {api,requests}=mount();
await api('/backtests',{method:'POST',headers:{'content-type':'application/json; charset=utf-8'},body:'{}'});
assert.equal(requests[0].headers.get('content-type'),'application/json; charset=utf-8');
assert.ok(requests[0].headers.get('ngrok-skip-browser-warning'));
''')


def test_non_json_response_is_reported_without_displaying_html():
    run_js(r'''
for(const status of [200,502]){
 const {api}=mount(undefined,()=>new Response('<script>alert("private upstream text")</script>',{
  status,headers:{'Content-Type':'text/html'},
 }));
 await assert.rejects(()=>api('/overview'),error=>{
  assert.match(error.message,/有效資料/);
  assert.ok(error.message.includes(`HTTP ${status}`));
  assert.doesNotMatch(error.message,/<script>|private upstream text|請確認研究 API 已啟動/);
  return true;
 });
}
''')
