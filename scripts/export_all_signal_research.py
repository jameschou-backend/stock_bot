#!/usr/bin/env python3
"""Reproduce all median50m signals as separate units with three-black exits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from skills.independent_three_black import ThreeBlackPath, observe, summarize
from skills.ordinary_volume_bundle import bind, digest


ISSUES = {
    'missing_or_invalid_price':'持有路徑缺少有效行情或還原價',
    'historical_identity_or_eligibility':'持有路徑市場身分或資格待核對',
    'raw_ohlc_conflict':'開高低收資料矛盾',
    'missing_or_invalid_volume':'持有路徑成交量缺漏',
    'no_volume_on_assumed_entry':'預定買進日沒有成交量',
    'no_volume_on_assumed_exit':'預定賣出日沒有成交量',
    'daily_adjustment_conflict':'單日異動或還原來源差異待核對',
    'cumulative_adjustment_conflict':'累計還原來源差異待核對',
    'missing_pre_entry_close':'缺少買進前一日還原收盤',
}

ENTRY_RULES = [
    '四碼個股，排除ETF與正二；使用本次固定歷史股票池及當時上市、可交易身分。',
    '至少126個交易日的上市與品質觀察，至少100個有效基準共同報酬日。',
    '最近20交易日估算成交值（原始收盤價×成交股數）的平均、中位數均至少5,000萬元，含訊號日。',
    '當日還原收盤嚴格突破前60交易日的最高還原收盤，不含當日。',
    '個股20日漲幅大於0，且高於0050同期漲幅。',
    '當日成交股數至少為前20交易日平均的1.5倍，均量不含當日。',
    '0050當日還原收盤高於最近120個有效觀察收盤的平均。',
    '收盤資料完成後確認訊號；下一市場交易日以日高低中點假設買入。',
    '本版本未直接使用新聞題材、基本面、法人連買或籌碼集中條件。',
]
EXIT_RULES = [
    '第一順位：還原收盤相對買進當日的還原收盤下跌至少12%，隔交易日賣出。停損錨點不是高低中點買價。',
    '第二順位：買進索引+62日收盤確認到期，買進索引+63日賣出。經過63交易日，含買賣日共64日。',
    '第三順位：連續三根原始收盤低於開盤的黑K，且各日還原收盤都低於前日；三根K均須在持有期間，下一交易日賣出。',
    '以上最先成立的原因鎖定；不使用未收盤資料，不以最後一天強制賣出未到期訊號。',
]


def run(bundle, output):
    bundle, output = bundle.resolve(), output.resolve()
    if not bundle.is_relative_to(ROOT) or not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Use repository inputs and a new repository output directory')
    refs = {}
    def source(path, expected=None):
        path=Path(path)
        return bind(ROOT,refs,str(path.relative_to(ROOT)),expected or digest(path))
    manifest_path=source(bundle/'manifest.json')
    if digest(manifest_path)!=(bundle/'manifest.sha256').read_text().strip():
        raise ValueError('Input manifest differs from its sealed hash')
    source(bundle/'manifest.sha256')
    manifest=json.loads(manifest_path.read_text())
    for name,h in manifest['files_sha256'].items():source(bundle/name,h)
    for name,h in manifest.get('source_sha256',{}).items():bind(ROOT,refs,name,h)
    signals=json.loads((bundle/'signals.json').read_text())
    if signals['schema']!='all_independent_median50m_signals_v1' or not signals['prefix_exact']:
        raise ValueError('Require unchanged current-strategy history plus explicit extension')
    entries=sorted(signals['entries'], key=lambda e:(e['signal_date'],e['event_id']))
    if not entries or len({e['event_id'] for e in entries}) != len(entries):
        raise ValueError('Require all unique current-strategy signal events')
    ids=sorted({e['members'][0] for e in entries})
    if any(len(s)!=4 or not s.isdigit() or s.startswith('0') for s in ids):
        raise ValueError('Only individual stock signals belong in this study')
    frames={n:pd.read_parquet(bundle/(n+'.parquet'),columns=['date',*ids]).set_index('date') for n in
        ('close-official','close-quality','eligibility','raw-close','raw-volume')}
    for frame in frames.values():frame.index=pd.to_datetime(frame.index)
    c,other,eligible=[frames[n] for n in ('close-official','close-quality','eligibility')]
    days=pd.DatetimeIndex(c.index)
    if any(not f.index.equals(days) or list(f.columns)!=ids for f in frames.values()):
        raise ValueError('Input arrays must share the same ordered axes')
    if eligible.isna().any().any() or any(t!=np.dtype(bool) for t in eligible.dtypes):
        raise ValueError('No unknown or coerced eligibility booleans')
    raw=pd.read_parquet(bundle/'quotes-unmasked.parquet')
    raw=raw[raw.stock_id.isin(ids)].copy();raw['date']=pd.to_datetime(raw.date)
    if raw.duplicated(['date','stock_id']).any():raise ValueError('Duplicate raw OHLC')
    raw_fields={n:raw.pivot(index='date',columns='stock_id',values=n).reindex(index=days,columns=ids)
                for n in ('close','open','high','low','volume')}
    companies=pd.read_parquet(bundle/'companies.parquet').set_index('stock_id')
    paths={sid:ThreeBlackPath(days,c[sid].to_numpy(float),other[sid].to_numpy(float),
            eligible[sid].to_numpy(bool),*[raw_fields[k][sid].to_numpy(float)
                for k in ('close','high','low','volume','open')]) for sid in ids}
    amount=frames['raw-close']*frames['raw-volume']
    median=amount.rolling(20,min_periods=20).median()
    mean=amount.rolling(20,min_periods=20).mean()
    rows=[]
    for e in entries:
        sid=e['members'][0]; signal=pd.Timestamp(e['signal_date'])
        index=int(days.get_loc(signal)); entry=e.get('entry_date')
        if index+1<len(days):
            if entry!=str(days[index+1].date()):raise ValueError('Signal entry is not T+1')
            result=observe(paths[sid],index+1)
        else:
            if entry is not None:raise ValueError('Last-day signal must not invent an unobserved fill')
            result=dict(status='not_entered',outcome='unrealized',exit_reason=None,data_issue=None,
                        observed_end_date=str(days[-1].date()))
        evidence=e.get('leader_evidence',{})
        item=dict(signal_id=e['event_id'],stock_id=sid,name=str(companies.at[sid,'name']),
            market=str(companies.at[sid,'market']) if 'market' in companies.columns else '',
            signal_date=e['signal_date'],signal_stage='收盤後確認',entry_date=entry,
            entry_reason='60日收盤突破、放量、相對強勢、0050多頭及20日成交值平均與中位數達門檻',
            return20=evidence.get('leader_return20'),relative20=e.get('priority'),
            volume_ratio=evidence.get('leader_volume_ratio'),
            turnover_median20=float(median.at[signal,sid]),turnover_mean20=float(mean.at[signal,sid]),
            **result)
        item['data_issue_code']=item.get('data_issue')
        item['data_issue']=ISSUES.get(item.get('data_issue'),item.get('data_issue'))
        item['data_source_scope']=('含9/10後FinMind延伸，未完成官方交叉核對'
            if item['observed_end_date']>'2026-09-09' else '修復後封存歷史資料，仍非全市場與成交認證')
        rows.append(item)
    summary=summarize(rows)
    annual={str(year):summarize([r for r in rows if r['signal_date'].startswith(str(year))])
            for year in range(2019,days[-1].year+1)}
    for p in (Path(__file__),ROOT/'skills/independent_three_black.py',ROOT/'skills/independent_signals.py',
              ROOT/'skills/exit_policy.py',ROOT/'skills/three_black_exit.py',ROOT/'skills/million_replay.py'):
        source(p)
    metadata=dict(strategy_name='60日突破＋出量＋成交值中位數5,000萬＋三黑K出場',
        requested_start='2019-01-01',requested_end='2026-10-02',data_through=str(days[-1].date()),
        first_signal=entries[0]['signal_date'],last_signal=entries[-1]['signal_date'],
        entry_price_method='下一交易日日高低中點（HL2）假設成交',
        exit_price_method='出場訊號隔交易日日高低中點（HL2）假設成交',
        entry_rule=ENTRY_RULES,exit_rule=EXIT_RULES,
        fee_rule='每邊佣金0.1425%、滑價0.45%，賣出稅0.3%。比例單位試算，未計最低手續費及整股／零股捨入。',
        win_rate_definition='扣費後獲利筆數／資料可判定且已出場筆數；未出場、待賣、未買、資料不足不列分母。',
        mfe_definition='日線最高漲幅上界包含買賣日，無法判斷當日高點在成交前或後，不代表可實得報酬。最高收盤排除賣出日；完整持有日最高價另排除買进日。',
        limitations=[
            '同一股票不同訊號日各自獨立買入，允許重疊。取消資金、名額、重複持股限制，並未驗證每筆理想成交。',
            '這是選股訊號研究，單筆平均報酬不可加總或複利成帳戶績效。',
            '原始價用於呈現買賣點；報酬以還原價比例處理除權息與分割，沒有重建股利付款、配股交付或資金可用日。',
            '歷史股票池不是完整歷史全市場；歷史來源也不是當年首次發布版本，不能宣稱無存活者偏誤或未見期間驗證。',
            '9/10之後使用FinMind還原價銜接封存尺度，尚未完成同期間官方公司行動與行情交叉核對。',
            '日線中點可成交、漲跌停排隊、普通盤／零股容量均未逐筆驗證。資料不足保留明細並不計勝率。',
        ],source_bundle=str(bundle.relative_to(ROOT)),
        fee_commission=.001425,fee_slippage=.0045,fee_sell_tax=.003,
        no_capital_limit=True,no_position_limit=True,live_qualified=False,actual_fill_verified=False)
    output.mkdir(parents=True)
    payload=dict(metadata=metadata,summary=summary,annual=annual,rows=rows)
    p=output/'workbook-data.json'
    p.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(rows).to_parquet(output/'signals.parquet',index=False)
    report=dict(schema='all_independent_three_black_v1',created_at=datetime.now(timezone.utc).isoformat(),
        metadata=metadata,summary=summary,annual=annual,source_sha256=refs,
        output_sha256={str(p.relative_to(ROOT)):digest(p) for p in output.iterdir()},
        live_qualified=False,actual_fill_verified=False,cash_account=False)
    p=output/'report.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    p.with_suffix('.sha256').write_text(digest(p)+'\n')
    print(json.dumps(dict(summary=summary,annual=annual,rows=len(rows),data_through=metadata['data_through']),
                     ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.inputs,args.output)
