'use strict';

window.renderPocOptimization = function(item, host, helpers) {
  const {el,tag,number,pct,api,renderAccountDetail}=helpers;
  host.replaceChildren(el('div','ONE CHANGE · FULL ACCOUNT REPLAY','eyebrow'),el('h2','POC 優化：冷卻訊號，是否值得？'));
  if(item?.status!=='verified_publication'||!item.optimization){host.append(el('p','本批尚無可核對的完整發布結果，暫不顯示績效。原有研究仍可從上方選單查看。','alert'));return;}
  const report=item.optimization,modes={baseline:'原成本',fee2:'佣金加倍',slip2:'滑價加倍',combined:'佣金與滑價同時加倍'};
  if(item.download_url?.startsWith('/research-terminal/api/archive/')){const link=el('a','查看本批結果、試次與成交缺件清單 ↗');link.href=item.download_url;link.target='_blank';link.rel='noopener';host.append(link);}
  host.append(el('p',`${report.start} → ${report.end} · 本金 ${number(report.initial_cash)} 元 · 個股3名額 · ${report.completed_new_accounts}/${report.expected_new_accounts} 新帳戶完成`,'caption'),
    el('p','原版與0050沿用已核對的封存帳戶；本批新跑兩套冷卻版各四種成本。每組成本皆為完整帳戶結果。','caption'),
    el('p','每次保留訊號後，同檔接下來5個市場交易日的候選暫不採用，再按原規則排序與分配資金。不是出場後冷卻；停損與原出場規則維持原設定。','scope-line'),
    el('p','冷卻先作用於原始進場候選，之後才套用 POC 硬篩或優先排序。保留候選不代表已符合 POC 或已成交；這批未測「實際出場後5日內不再買」。','caption'),
    el('p','買價仍為訊號隔日 (H+L)/2，賣價為 L+30%×(H−L)。兩者皆是事後區間價估算；本頁未啟用紙上交易。','result-note'));
  const controls=el('div',undefined,'study-controls'),strategy=el('select'),cost=el('select');
  strategy.setAttribute('aria-label','POC優化策略');cost.setAttribute('aria-label','POC優化成本');
  for(const [id,name] of [['poc_up_red','POC 硬篩＋紅 K'],['poc_priority','POC 優先＋紅 K']]){const o=el('option',name);o.value=id;strategy.append(o);}
  for(const [id,name] of Object.entries(modes)){const o=el('option',name);o.value=id;cost.append(o);}
  controls.append(strategy,cost);host.append(controls);
  const decision=el('section',undefined,'qualification-card'),matrix=el('div'),actions=el('div',undefined,'study-actions'),
    stress=el('details',undefined,'comparison-method'),robust=el('details',undefined,'comparison-method'),
    evidence=el('section',undefined,'qualification-card'),detail=el('div',undefined,'comparison-detail');
  decision.setAttribute('aria-live','polite');detail.setAttribute('aria-live','polite');host.append(decision,matrix,actions,stress,robust,evidence,detail);
  let generation=0;
  const pp=v=>v==null?'—':`${v>0?'+':''}${number(v*100,2)}`;
  const win=r=>r.analytics?.win_rate==null?'不適用':`${pct(r.analytics.win_rate,2)} (${r.analytics.wins}/${r.analytics.settled})`;
  function table(parent,labels,rows){const wrap=el('div',undefined,'compact-comparison'),t=el('table'),head=el('thead'),h=el('tr'),body=el('tbody');labels.forEach(x=>h.append(el('th',x)));head.append(h);for(const values of rows){const r=el('tr');values.forEach((v,i)=>{const c=el('td',String(v));c.dataset.label=labels[i];r.append(c);});body.append(r);}t.append(head,body);wrap.append(t);parent.append(wrap);}
  function render(){
    ++generation;[decision,matrix,actions,stress,robust,evidence,detail].forEach(x=>x.replaceChildren());
    const arm=strategy.value,mode=cost.value,get=(v,m=mode)=>report.cases[`${arm}:${v}:${m}`],
      original=get('original'),changed=get('cooldown5'),benchmark=report.cases[`benchmark:original:${mode}`],verdict=report.decisions[arm];
    decision.append(el('h3',verdict.improvement_pass?'四組成本皆改善，仍需前瞻驗證':'冷卻版未通過改善門檻，保留原版作研究基準'));
    const badges=el('div',undefined,'qualification-badges');
    for(const [name,ok] of [['四組都提高報酬且回撤不惡化',verdict.improvement_pass],['通過歷史壓力門檻',verdict.historical_stress_pass],['真正樣本外通過',false],['紙上交易啟用',false]])badges.append(tag(`${ok?'✓':'未'} ${name}`,ok?'green':'neutral'));
    decision.append(badges,el('p','改善門檻與原版比較；歷史壓力門檻另要求四組成本都獲利、贏0050且回撤不差於0050。勝率提高不代表帳戶報酬提高。','caption'));
    table(matrix,['本成本情境','淨報酬','相對0050（百分點）','最大回撤','已結清勝率','總成本','成交筆數'],
      [[original,'原版'],[changed,'冷卻5日'],[benchmark,'0050']].map(([r,name])=>r.completed?
        [name,pct(r.metrics.total_return),pp(r.excess_return),pct(r.metrics.max_drawdown),win(r),number(r.metrics.costs.total_cost),number(r.metrics.trade_count)]:[name,'未完成','—','—','—','—','—']));
    if(!changed.completed)matrix.append(el('p',changed.reason||'資料或核帳缺件','alert error'));
    if(changed.admission)matrix.append(el('p',`候選 ${number(changed.admission.input_candidates)} → ${number(changed.admission.retained_candidates)}；抑制 ${number(changed.admission.suppressed_candidates)} 個重複候選。候選數不是成交筆數；成交筆數包含整股、零股及分批成交。`,'caption'));
    if(changed.funded_path_changes){const p=changed.funded_path_changes;matrix.append(el('p',`入金批次：共同 ${p.common} 批、原版獨有 ${p.original_only} 批、冷卻版新增 ${p.cooldown_only} 批。篩選會改變資金和名額，並非只從原版刪掉幾筆成交。`,'caption'));}
    for(const [variant,label] of [['original','原版'],['cooldown5','冷卻5日']]){
      if(!get(variant).completed)continue;
      const button=el('button',`${label}曲線與0050`,'primary-button');
      button.addEventListener('click',async()=>{const request=++generation;button.disabled=true;detail.replaceChildren(el('p','正在核對帳戶…'));try{
        const prefix=`/archive/${item.id}/cases/`,[account,base]=await Promise.all([api(prefix+`${arm}__${variant}__${mode}`),api(prefix+`benchmark__original__${mode}`)]);
        if(request!==generation)return;detail.replaceChildren();renderAccountDetail(account,detail,base);
      }catch(error){if(request===generation)detail.replaceChildren(el('p',error.message,'alert error'));}finally{button.disabled=false;}});actions.append(button);
    }
    stress.append(el('summary','四種成本：原版與冷卻版完整對照'));stress.open=true;
    table(stress,['成本情境','原版淨報酬','冷卻淨報酬','報酬變化（百分點）','冷卻最大回撤','冷卻勝率'],Object.entries(modes).map(([m,name])=>{const a=get('original',m),b=get('cooldown5',m);return [name,pct(a.metrics.total_return),b.completed?pct(b.metrics.total_return):'未完成',pp(b.change_from_original?.total_return),b.completed?pct(b.metrics.max_drawdown):'—',b.completed?win(b):'—'];}));
    robust.append(el('summary','分年、獲利集中度與成本設定'));
    table(robust,['版本','2024超額（百分點）','2025超額（百分點）','2026至10/2超額（百分點）','前5筆占正損益','獲利因子'],[[original,'原版'],[changed,'冷卻5日']].map(([r,name])=>r.completed?[name,...r.annual_excess.map(y=>pp(y.excess)),pct(r.analytics.top5_profit_share),number(r.analytics.profit_factor,2)]:[name,'—','—','—','—','—']));
    robust.append(el('p',`資料缺件排除：原版 ${original.data_gap_exclusions??'—'} 筆、冷卻版 ${changed.data_gap_exclusions??'—'} 筆。缺件排除可能改變交易路徑，仍需補齊後驗證。`,'caption'));
    robust.append(el('p',`POC 資料品質排除：原版 ${original.profile_data_exclusions??'—'} 筆、冷卻版 ${changed.profile_data_exclusions??'—'} 筆；POC 未知時恢復原排序：原版 ${original.selection_fallback_days??'—'} 日、冷卻版 ${changed.selection_fallback_days??'—'} 日。硬篩版不採用未知 POC。`,'caption'));
    robust.append(el('p','前5筆占比的分母是所有獲利批次的正損益合計，並非帳戶淨利。分年與滾動窗口皆為已見歷史；不能當作未見樣本外。勝率已扣費稅滑價，未平倉與應收未結批次不列入。','caption'));
    const cfg=report.modes[mode];robust.append(el('p',`每邊佣金 ${pct(cfg.commission,4)}、滑價 ${pct(cfg.slippage,2)}；最低每通路20元。賣出稅個股0.3%、0050為0.1%。`,'caption'));
    const coverage=original.execution_coverage;
    evidence.append(el('h3','離可執行測試還缺什麼'),el('p',`原版封存買單：普通盤 ${coverage.board_buy_orders} 筆，完整逐筆序列驗證 ${coverage.board_complete_sequence} 筆；零股 ${coverage.odd_buy_orders} 筆，逐筆時序驗證 ${coverage.odd_chronological_verified} 筆。這是買單證據盤點，不代表供應商全部資料覆蓋率。`,'caption'),
      el('p','下一階段先補齊普通盤與零股成交時序，建立可事前決定的委託規則，再重播開盤／限價帳戶。日高低均價不能提供成交先後、排隊或部分成交證明。'));
    const steps=el('ol');for(const label of ['成交時序與可執行委託驗證','非停損出場多一日確認，停損維持原規則','POC 強度排序','依歷史價格控制持股相關性'])steps.append(el('li',`${label}：待測`));evidence.append(steps,el('p','本批只改候選冷卻；以上項目尚未完成，沒有自動切換策略。','caption'));
  }
  strategy.addEventListener('change',render);cost.addEventListener('change',render);render();
};
