'use strict';
window.renderPocWeakness = function(item, host, h) {
  const {el, number, pct}=h;
  host.replaceChildren(el('div','POC ORIGINAL · EARLY WEAKNESS','eyebrow'),el('h2','三黑K之前，有沒有可辨識的轉弱？'));
  if(item?.status!=='verified_publication'||!item.weakness){host.append(el('p','轉弱診斷尚未完成核對，暫不顯示結論。','alert'));return;}
  const r=item.weakness,order=['two_black_below_ma20','two_black','below_ma20_two'];
  const reasons={three_black:'三黑K',time63:'63日期限',loss12:'12%停損'};
  const locations={before_peak:'回撤高點之前',during_decline:'本段下跌期間',after_trough:'低點之後，無法提前處理本段'};
  function table(parent,heads,rows){const wrap=el('div',undefined,'compact-comparison'),t=el('table'),head=el('thead'),tr=el('tr'),body=el('tbody');heads.forEach(x=>tr.append(el('th',x)));head.append(tr);rows.forEach(cells=>{const row=el('tr');cells.forEach((x,i)=>{const td=el('td',String(x));td.dataset.label=heads[i];row.append(td);});body.append(row);});t.append(head,body);wrap.append(t);parent.append(wrap);}
  host.append(el('p',`${r.start} → ${r.end} · 正常成本 · 檢查全部 ${r.total_cohorts} 個持股批次與五次主要回撤 · 回測重跑0次`,'caption'),
    el('p','原版保留。這次只診斷原持股路徑，尚未計算減碼後的報酬、回撤或新增成本。','result-note'));
  host.append(el('h3','優先值得測：兩黑K＋跌破20日均線，只減三分之一'));
  const best=r.rule_summaries.two_black_below_ma20;
  host.append(el('p',`加上「還原收盤仍高於進場日」篩選後，${best.triggered} 個批次有提前警訊，其中 ${best.original_decision_lower} 個到原版出場決策日收盤更低，${best.original_decision_higher} 個反而更高。提前時間中位數只有 ${number(best.lead_median,1)} 個交易日，保護空間有限。`),
    el('p','最深回撤中，大立光6/29、國巨7/7、群創7/14收盤已符合；最早分別在6/30、7/8、7/15提出減碼，原三黑K出場指令則在7/8、7/9、7/21。這些是可事前計算的日期，並非已證明可成交或省下損失。'),
    el('p','反例也要保留：鴻海2024/4/16已出現同一警訊，之後到原出場決策前最高收盤仍再漲31.56%。只減三分之一是待測折衷，不是已證明不會少賺。','result-note'));
  table(host,['候選條件（均需仍有價格獲利）','提前警訊批次／全部','其中三黑K出場','提前交易日中位數','到原出場決策日較低／較高','警訊後曾再創持有期新高'],order.map(key=>{const s=r.rule_summaries[key];return [r.rules[key],`${s.triggered} / ${s.cohorts}`,`${s.exit_reasons.three_black||0} / ${r.original_exit_counts.three_black}`,number(s.lead_median,1),`${s.original_decision_lower} / ${s.original_decision_higher}`,`${s.subsequent_new_high} / ${s.resolved}`];}));
  host.append(el('p','較低／較高以同一還原收盤基準比較，不是交易勝率；相等不計入兩者。新高相對警訊當時已知的持有期最高收盤。警訊統計包含回撤之外的全部批次；未出場右設限批次不計入這些後續結果。','caption'));
  host.append(el('h3','後續各自獨立測，不疊加改動'));
  for(const key of order){const c=r.candidate_rules.find(x=>x.id===key);host.append(el('p',`${key===order[0]?'優先：':key==='two_black'?'對照：':'暫緩：'}${c.name}。${c.action}。`));}
  host.append(el('p','首次符合後隔日賣 floor(當時可交易股數/3)，同批只減一次；未成交保留固定目標續掛。12%停損、三黑K、63日期限皆優先完整出場。剩餘部位仍占名額、不補買，沿用原成本與整零股限制。單純兩黑K常過早；單純連兩日破MA20也仍有大量反彈，不宜直接採用。'));
  const controls=el('div',undefined,'research-controls'),episode=el('select'),rule=el('select');
  episode.setAttribute('aria-label','轉弱診斷回撤段落');rule.setAttribute('aria-label','轉弱診斷條件');
  r.episodes.forEach(e=>{const o=el('option',`第${e.rank}深 · ${e.peak_date} → ${e.trough_date} · ${pct(e.drawdown)}`);o.value=String(e.rank);episode.append(o);});
  order.forEach(key=>{const o=el('option',r.rules[key]);o.value=key;rule.append(o);});controls.append(episode,rule);const detail=el('section');host.append(controls,detail);
  function render(){const e=r.episodes.find(x=>String(x.rank)===episode.value),key=rule.value;detail.replaceChildren(el('h3',`${e.peak_date} → ${e.trough_date} · ${pct(e.drawdown)}`));
    if(e.rank===5)detail.append(el('p','這段是短期急跌，原帳本在高點至低點之間沒有賣出；該候選沒有在此段提供合格提前警訊。低點之後的訊號不能算作避開這次回撤。','result-note'));
    const old=e.rows.filter(c=>c.group!=='new_entry');
    table(detail,['舊持股／進場日','原版出場決策 → 最早指令日','首次警訊 → 最早減碼日','警訊相對回撤位置','提前交易日','警訊時價格獲利','之後至原決策最高收盤相對警訊'],old.map(c=>{const w=c.warnings[key];return [`${c.stock_id} ${c.name} / ${c.entry_date}`,c.original_decision?`${c.original_decision} → ${c.original_target}（${reasons[c.original_reason]}${c.exit_during_decline?'':'；本段低點後'}）`:'期末仍持有',w?`${w.date} → ${w.next_session}`:'無合格提前警訊',w?locations[w.location]:'—',w?w.lead_sessions??'右設限':'—',w?pct(w.close_return_from_entry):'—',w?pct(w.subsequent_max_return):'—'];}));
    const newer=e.rows.filter(c=>c.group==='new_entry'),covered=newer.filter(c=>c.warnings[key]&&c.warnings[key].date<=e.trough_date);
    detail.append(el('p',`本段另有 ${newer.length} 個新買批次出現價格與收益損失；只有 ${covered.length} 個在低點前出現這項合格警訊。保護既有贏家的規則不能解決全部新買虧損。`));
    if(newer.length){const more=el('details');more.append(el('summary','新買虧損批次與警訊'));table(more,['新買股票／進場日','原出場原因','首次警訊','位置'],newer.map(c=>{const w=c.warnings[key];return [`${c.stock_id} ${c.name} / ${c.entry_date}`,reasons[c.original_reason]||'期末仍持有',w?.date||'無合格提前警訊',w?locations[w.location]:'—'];}));detail.append(more);}
    detail.append(el('p','僅列原拆解中價格與收益為負的批次。首次警訊從進場起算，不能在事後高點重新起算挑最好日期；提前天數對照原出場決策，不把原出場未成交等待期當成提前。','caption'));
  }
  episode.addEventListener('change',render);rule.addEventListener('change',render);render();
  const method=el('details');method.append(el('summary','定義、原帳本核對與限制'),el('p','黑K須原始收盤低於開盤且還原收盤低於前日；兩天皆須在買入後。MA20用當天與前19個市場交易日。價格獲利不等於扣費損益。'),el('p',`已核對原 ${r.original_black_audit.decisions_checked} 筆三黑K判斷及 ${r.original_black_audit.exits} 次出場觸發；未重播現金帳戶。`));r.limitations.forEach(x=>method.append(el('p',x)));host.append(method);
  const a=el('a','下載73個批次、五段回撤與來源指紋 ↗');a.href=item.download_url;a.target='_blank';a.rel='noopener';host.append(a);
};
