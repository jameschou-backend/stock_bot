"""One bounded background task at a time. Results survive API/UI restarts."""
from __future__ import annotations
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from uuid import uuid4
from pydantic import BaseModel, Field
from typing import Literal
from app.file_lock import file_lock

ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / '.cache/workbench/jobs'


class WorkRequest(BaseModel):
    kind: Literal['update_data','backtest']
    months: int = Field(default=12,ge=3,le=120)
    topn: int = Field(default=10,ge=1,le=50)
    quick: bool = False
    cost: float = Field(default=.00585,ge=.00585,le=.05,allow_inf_nan=False)
    stoploss: float = Field(default=-.12,ge=-.5,le=-.01,allow_inf_nan=False)


def job_path(job_id):
    if not re.fullmatch(r'[0-9a-f]{32}',job_id):
        raise ValueError('工作識別碼錯誤')
    return JOBS_DIR / f'{job_id}.json'


def write_job(job):
    JOBS_DIR.mkdir(parents=True,exist_ok=True)
    path=job_path(job['job_id'])
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(job,ensure_ascii=False,allow_nan=False))
    os.replace(tmp,path)


def read_job(job_id):
    path=job_path(job_id)
    if not path.exists():
        raise ValueError('找不到工作')
    return json.loads(path.read_text())


def recent_jobs(limit=10):
    paths = sorted((p for p in JOBS_DIR.glob('*.json') if re.fullmatch(r'[0-9a-f]{32}',p.stem)),
                   key=lambda p:p.stat().st_mtime,reverse=True)[:limit]
    result=[]
    for path in paths:
        job=json.loads(path.read_text())
        if job['status'] in ('queued','running'):
            job['elapsed_seconds']=round(time.time()-job['created_at'],1)
            if job.get('pid'):
                try:
                    os.kill(job['pid'],0)
                except ProcessLookupError:
                    job.update(status='failed',message='工作程序已中斷，可以重新執行')
            log=path.with_suffix('.log')
            if log.exists():
                with log.open('rb') as f:
                    f.seek(max(0,log.stat().st_size-16384))
                    tail=f.read().decode('utf-8',errors='replace')
                matches=re.findall(r'\[TIMER\] ([\w]+) (start|done)',tail)
                if matches and job['status']=='running':
                    names={'load_prices':'載入股價','load_features':'載入特徵','load_labels':'載入標籤',
                           'precompute':'預先計算指標','prepare':'準備回測資料','backtest':'執行回測'}
                    name,phase=matches[-1]
                    job['message']=names.get(name,'計算策略')+('中' if phase=='start' else '完成，進入下一階段')
        result.append(job)
    return result


def submit(request: WorkRequest):
    JOBS_DIR.mkdir(parents=True,exist_ok=True)
    with file_lock(JOBS_DIR/'submit.lock',timeout=0):
        for job in recent_jobs(100):
            if job['status'] in ('queued','running'):
                pid=job.get('pid')
                alive=False
                if pid:
                    try:
                        os.kill(pid,0)
                        alive=True
                    except ProcessLookupError:
                        pass
                if alive or time.time()-job['created_at']<10:
                    if job['request']==request.model_dump():
                        return job
                    raise ValueError('已有工作執行中；完成後再開始下一個，避免資源競爭')
                job.update(status='failed',message='前次工作已中斷，可重新執行')
                write_job(job)
        job={'job_id':uuid4().hex,'request':request.model_dump(),'status':'queued',
             'created_at':time.time(),'message':'準備執行','pid':None}
        write_job(job)
        env=dict(os.environ,AI_ASSIST_ENABLED='0',PYTHONUNBUFFERED='1')
        try:
            with (JOBS_DIR/f"{job['job_id']}.log").open('w') as log:
                proc=subprocess.Popen([sys.executable,'-m','app.workbench_worker',job['job_id']],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except OSError:
            job.update(status='failed',message='無法啟動背景工作，請確認 Python 執行環境')
            write_job(job)
            raise ValueError(job['message']) from None
        job['pid']=proc.pid
        write_job(job)
        return job


def command_for(request, output):
    if request.kind=='update_data':
        return [sys.executable,'scripts/run_daily.py']
    args=[sys.executable,'scripts/run_backtest.py','--months',str(request.months),
          '--topn',str(request.topn),'--entry-delay','1','--cost',str(request.cost),
          '--stoploss',str(request.stoploss),'--train-lookback','730','--pruned-features','--slippage',
          '--eval-end',datetime.now().date().isoformat(),'--output',str(output)]
    if request.quick:
        args.append('--fast')
    return args
