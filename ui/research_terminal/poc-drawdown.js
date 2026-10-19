'use strict';
window.renderPocDrawdown = function(item, host, h) {
  const {el, number, pct}=h;
  host.replaceChildren(el('div','POC ORIGINAL · LEDGER ATTRIBUTION','eyebrow'),el('h2','原版五次主要回撤：損失從哪裡來'));
  if(item?.status!=='verified_publication'||!item.drawdown_attribution){
    host.append(el('p','回撤拆解尚未完成核對，暫不顯示結論。','alert'));return;
  }
  const r=item.drawdown_attribution;
  const labels={winner_decline:'舊贏家跌價',other_existing_decline:'其他舊持股跌價',new_entry_decline:'新買虧損',gain_offset:'期間獲利抵銷',cost:'費稅與滑價'};
  const groups={existing_winner:'高點時獲利的舊批次',existing_other:'其他舊批次',new_entry:'高點後新買'};
  const money=x=>number((x||0)/10000,2)+' 萬';
  function table(parent,heads,rows){
    const wrap=el('div',undefined,'compact-comparison'),t=el('table'),head=el('thead'),tr=el('tr'),body=el('tbody');
    heads.forEach(x=>tr.append(el('th',x)));head.append(tr);
    rows.forEach(cells=>{const row=el('tr');cells.forEach((x,i)=>{const td=el('td',String(x));td.dataset.label=heads[i];row.append(td);});body.append(row);});
    t.append(head,body);wrap.append(t);parent.append(wrap);
  }
  host.append(el('p',`${r.start} → ${r.end} · 正常成本 · 本金100萬元 · 重用原版與0050兩份帳本 · 回測重跑0次`,'caption'),
    el('p','原版策略維持不變。以下拆解整個帳戶從歷史高點到低點的損失，並非單筆停損，也不是從本金計算的虧損。','result-note'));
  table(host,['高點 → 低點','帳戶回撤','帳戶減少','舊贏家跌價','新買虧損','費稅滑價','獲利抵銷','同期0050報酬'],r.episodes.map(e=>[
    e.peak_date+' → '+e.trough_date,pct(e.drawdown),money(-e.net_change),money(-e.components.winner_decline),
    money(-e.components.new_entry_decline),money(-e.components.cost),money(e.components.gain_offset),pct(e.benchmark_same_window_return)]));
  host.append(el('p','表內損失以正金額列示；舊贏家跌價＋其他舊持股跌價＋新買虧損＋成本－獲利抵銷＝帳戶減少。這五段其他舊持股跌價均為0。0050欄是相同日期報酬，並非其最大回撤。','caption'));
  const findings=el('section');findings.append(el('h3','最值得改善的地方'));
  r.findings.forEach(x=>findings.append(el('p',x)));host.append(findings);
  const label=el('label',undefined,'study-version'),select=el('select');select.setAttribute('aria-label','回撤段落');
  r.episodes.forEach(e=>{const o=el('option',`第${e.rank}深 · ${e.peak_date} → ${e.trough_date} · ${pct(e.drawdown)}`);o.value=String(e.rank);select.append(o);});
  label.append(el('span','查看個股與費用明細'),select);const detail=el('section');host.append(label,detail);
  function render(){
    const e=r.episodes.find(x=>String(x.rank)===select.value);detail.replaceChildren(el('h3',`${e.peak_date} → ${e.trough_date}`));
    detail.append(el('p',`淨值 ${money(e.peak_nav)} → ${money(e.trough_nav)}；下跌歷時 ${e.decline_sessions} 個交易日。${e.recovery_date?'於 '+e.recovery_date+' 回復高點，距高點 '+e.recovery_sessions+' 個交易日。':'截至 '+r.end+' 尚未回復高點。'} 高點後新買 ${e.new_cohorts} 個批次；區間內 ${e.child_trade_count} 筆整股／零股成交紀錄。`));
    table(detail,['損益來源','對帳戶損益貢獻','占淨損失比例'],Object.entries(labels).map(([key,name])=>[name,money(e.components[key]),pct(e.component_loss_shares[key])]));
    detail.append(el('p','負金額造成損失，正金額抵銷損失；獲利抵銷的損失占比為負，因此損失項占比合計可以超過100%。所有分類均已拆出成本。','caption'));
    table(detail,['手續費','交易稅','滑價'],[[money(e.costs.commission),money(e.costs.tax),money(e.costs.slippage)]]);
    detail.append(el('p',`舊贏家跌價中，${money(e.winner_giveback)} 屬高點時已有累計淨獲利的回吐，另 ${money(e.winner_beyond_peak_profit)} 跌穿該獲利；這是批次累計損益拆解，包含先前已實現損益及應收權益。`,'caption'));
    table(detail,['股票／進場日','分類','高點持股比重','高點累計淨損益','區間價差與收益','區間成本','區間淨損益','區間賣出'],e.cohorts.map(c=>[
      `${c.stock_id} ${c.name} / ${c.entry_date}`,groups[c.group],pct(c.peak_weight),money(c.peak_net_pnl),money(c.gross_pnl),money(c.cost),money(c.net_pnl),
      c.sales.map(s=>`${s.date} ${number(s.qty)}股 ${({three_black:'三黑K',loss12:'12%停損',time63:'63日期限',scheduled_exit:'期限出場'})[s.reason]||s.reason}`).join('；')||'本區間未賣出']));
  }
  select.addEventListener('change',render);render();
  const method=el('details',undefined,'comparison-method');method.append(el('summary','計算方法與限制'));
  method.append(el('p','依歷史高點劃分尚未回到前高的獨立期間，取最深五段；只計高點次日至低點日。逐批次核對：持股市值變化＋現金流＋應收權益變化＝淨損益。配股及股息沿用原帳本，配息收款不重複算獲利。'));
  r.limitations.forEach(x=>method.append(el('p',x)));host.append(method);
  const link=el('a','下載完整拆解與來源指紋 ↗');link.href=item.download_url;link.target='_blank';link.rel='noopener';host.append(link);
};
