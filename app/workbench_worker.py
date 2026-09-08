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
        job.update(status='running',message='更新資料' if job['request']['kind']=='update_data' else '載入資料與驗證策略',pid=os.getpid())
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
        else:
            job.update(status='completed',message='已完成；研究結果仍需資料與策略驗證')
            if output.exists():
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
