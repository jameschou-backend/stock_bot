"""Bounded worker launched by the workbench job service (no arbitrary commands)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from app.file_lock import file_lock
from app.workbench_jobs import ROOT,JOBS_DIR,WorkRequest,read_job,write_job,command_for


def main(job_id):
    # Submit holds this lock while saving the process id; avoid a startup write race.
    with file_lock(JOBS_DIR/'submit.lock',timeout=10):
        job=read_job(job_id)
        message={'update_data':'更新資料','backtest':'載入資料與驗證策略',
                 'verified_backtest':'核對封存資料並執行所選歷史研究',
                 'sector_backtest':'核對新族群來源並執行固定帳戶研究',
                 'index_backtest':'核對封存來源與連續ETF帳戶',
                 'news_scan':'整理新聞題材','news_review':'重建歷史新聞時間線',
                 'chain_flow':'整理族群成交分布與法人方向'}[job['request']['kind']]
        job.update(status='running',message=message,pid=os.getpid())
        write_job(job)
    started=time.perf_counter()
    try:
        request=WorkRequest(**job['request'])
        output=JOBS_DIR/f'{job_id}.result.json'
        env=dict(os.environ,AI_ASSIST_ENABLED='0',PYTHONUNBUFFERED='1')
        # Explicit Sponsor data route for this workbench job; never edits the user's .env.
        if request.kind=='update_data':
            env['INGEST_PRICES_SOURCE']='finmind'
        # CLI entry points share the same lock with make pipeline / manual backtests.
        result=subprocess.run(command_for(request,output),cwd=ROOT,env=env,timeout=1800)
        if result.returncode:
            job.update(status='failed',message='工作未完成，請查看資料狀態及本機工作日誌',exit_code=result.returncode)
            if result.returncode==75 and request.kind in ('news_scan','news_review','chain_flow') and output.exists():
                value=json.loads(output.read_text())
                retry=int(value.get('retry_after_seconds',0))+1
                job.update(message=f'FinMind 達額度暫停，約 {retry} 秒後重試；已完成日期會重用',retry_after_seconds=retry)
        else:
            job.update(status='completed',message='已完成；研究結果仍需資料與策略驗證')
            if request.kind=='index_backtest':
                import hashlib
                from app.index_case_backtest_ui import load_result
                value, _=load_result(output,root=ROOT,request=job['request'])
                reused=value['metrics']['reused_cases']==1
                job.update(message=('來源核對完成，重用一致帳戶' if reused else '重新計算完成，完整帳戶與封存結果一致')+'；僅供歷史研究',
                           report_status='exploratory',result_path=str(output),
                           result_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                           research_only=True,live_qualified=False,unseen_validation=False)
            elif request.kind=='sector_backtest':
                import hashlib
                from app.backtest_completion_ui import load_sector_job
                value, report=load_sector_job(output,root=ROOT,request=job['request'])
                messages={
                    'preflight_complete':'族群資料預檢完成；尚未計算報酬，請查看來源缺件',
                    'blocked':'族群研究檢查完成；被阻擋案例不提供中途收益',
                    'exploratory':'族群日資料帳戶完成；已核對案例與明細指紋，僅供歷史研究',
                }
                job.update(message=messages[value['status']],report_status=value['status'],
                           result_path=str(output),result_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                           research_only=True,live_qualified=False,unseen_validation=False)
            elif request.kind=='verified_backtest':
                import hashlib
                from app.backtest_tool_ui import load_report
                value=load_report(output,root=ROOT)
                messages={
                    'preflight_ready':'資料預檢完成；尚未計算報酬，可開始背景驗證回測',
                    'blocked':'研究檢查完成；部分或全部案例被阻擋，僅完成案例可查歷史日資料估算',
                    'exploratory':'日資料估算完成；僅供歷史研究，尚未取得逐筆驗證或實盤資格',
                }
                job.update(message=messages[value['status']],report_status=value['status'],
                           result_path=str(output),result_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                           research_only=True,live_qualified=False,unseen_validation=False)
            elif output.exists():
                value=json.loads(output.read_text())
                if value.get('error'):
                    raise ValueError(value['error'])
                job['summary']=value.get('summary',{})
                job['result_path']=str(output)
                job['equity_curve']=value.get('equity_curve',[])
                job['research_only']=True
    except subprocess.TimeoutExpired:
        job.update(status='failed',message='超過 30 分鐘上限，工作已停止；請縮短區間或查看效能紀錄')
    except Exception as exc:
        job.update(status='failed',message=f'工作失敗 ({type(exc).__name__})，請查看本機日誌')
        print(f'worker failed: {type(exc).__name__}',file=sys.stderr)
    job['elapsed_seconds']=round(time.perf_counter()-started,3)
    job['ended_at']=time.time()
    # Scientific reports can contain NaN/Infinity; convert them to explicit missing values.
    def clean(value):
        import math
        if isinstance(value,float) and not math.isfinite(value): return None
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list): return [clean(v) for v in value]
        return value
    write_job(clean(job))


if __name__=='__main__':
    main(sys.argv[1])
