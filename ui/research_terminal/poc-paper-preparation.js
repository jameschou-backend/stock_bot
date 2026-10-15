'use strict';
window.renderPocPaperPreparation = function(item, host, h) {
  const {el,number,pct}=h;
  host.replaceChildren(el('div','SIGNAL PREPARATION','eyebrow'),el('h2','POC硬篩＋紅K原版：訊號準備'));
  if(item?.status!=='verified_publication'||!item.paper_preparation){
    host.append(el('p','準備紀錄尚未通過核對。紙上交易未啟用。','alert'));return;
  }
  const r=item.paper_preparation,s=r.latest_inputs,counts=r.observation_summary;
  host.append(el('p',`檢查時間 ${r.checked_at} · 訊號截至 ${s.signal_date} · 本頁為封存狀態，不是即時成交帳戶`,'caption'),
    el('p','目前只準備訊號。你回報買進日期、股票、股數與成交價後，再依回報記錄；現在沒有委託或成交。','result-note'),
    el('p','準備本金100萬元、3個活躍持股名額，閒置資金保留現金。採全部每日符合訊號，按訊號日20日平均成交值排序；T+1進場，保留12%收盤停損、63交易日期限及進場後三黑出場。','result-note'));
  host.append(el('h3','已完成並沿用'));
  const done=el('ul');r.completed.forEach(x=>done.append(el('li',x)));host.append(done);
  host.append(el('p',`本次新增歷史帳戶重跑 ${r.historical_backtests_replayed} 次、準備流程新增FinMind請求 ${r.finmind_requests} 次。126個交易日與30批結清是之後的評估門檻。`,'caption'));
  host.append(el('h3','訊號資料核對'));
  for(const gap of r.remaining.filter(x=>['timely_signal','daily_pit'].includes(x.id))){const row=el('article',undefined,'reason-item');row.append(el('strong',gap.id==='timely_signal'?'訊號日期與封存時間不同':gap.title),el('p',gap.id==='timely_signal'?`本次沿用 ${s.signal_date} 已完成掃描，只作準備核對；不會補記為當天事前已知的訊號。`:gap.detail));host.append(row);}
  const execution=el('details',undefined,'comparison-method');execution.append(el('summary','若日後需要自動模擬成交，仍須補的驗證'));
  for(const gap of r.remaining.filter(x=>['fill_source','causal_execution','poc_account_integration'].includes(x.id))){execution.append(el('strong',gap.title),el('p',gap.detail));}host.append(execution);
  const future=r.remaining.find(x=>x.id==='forward_evidence');
  if(future)host.append(el('h3','後續績效評估'),el('p',future.detail+' 這不影響先準備訊號；不需先累積126天才能開始觀察。','result-note'));
  host.append(el('h3','本次封存'),el('p',`${s.signal_date}完整股票池：${s.matched_count}檔符合、${s.unknown_count}檔資料未知；符合者中${s.rank_unknown_count}檔排序資料不足。合格事前訊號日 ${counts.prospective_signal_days} 天；委託 ${counts.orders} 筆、成交 ${counts.fills} 筆。`));
  if(!r.observation.body.eligible_observation)host.append(el('p','本次只作準備核對，未計入鎖版後事前觀察，也未回填委託。','result-note'));
  const details=el('details',undefined,'comparison-method');details.append(el('summary','查看封存候選與資料未知原因'));
  const list=el('ol');s.candidates.forEach(c=>list.append(el('li',`${c.members[0]} ${c.name} · 20日平均成交值 ${number(c.priority,0)} · ${c.first_signal?'首日符合':'持續符合'}`)));details.append(list);
  s.unknown.forEach(c=>details.append(el('p',`${c.stock_id} ${c.name}：${c.reasons.join('；')}`,'caption')));
  host.append(details,el('h3','原版與同成本0050：沿用歷史結果'));
  const table=el('table'),head=el('thead'),tr=el('tr'),body=el('tbody'),wrap=el('div',undefined,'compact-comparison');
  ['成本','原版報酬／回撤','0050報酬／回撤'].forEach(x=>tr.append(el('th',x)));head.append(tr);
  for(const [mode,label] of Object.entries({baseline:'原成本',fee2:'佣金加倍',slip2:'滑價加倍',combined:'佣金與滑價加倍'})){
    const row=el('tr');row.append(el('td',label));
    for(const arm of ['poc_up_red','benchmark']){const c=r.completed_controls[`${arm}:original:${mode}`];row.append(el('td',`${pct(c.total_return)}／${pct(c.max_drawdown)}`));}body.append(row);
  }
  table.append(head,body);wrap.append(table);host.append(wrap,el('p',`${r.start} → ${r.end}，本金100萬。這些是已見歷史的成交價估算，尚未完成可執行成交及鎖版後未見驗證。`,'caption'));
  const link=el('a','查看完整準備紀錄 ↗');link.href=item.download_url;link.target='_blank';link.rel='noopener';host.append(link);
};
