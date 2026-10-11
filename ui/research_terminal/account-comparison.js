'use strict';

// Kept separate from signal studies: every row here is a funded cash account.
window.renderAccountComparison = function (item, host, helpers) {
  const {el, tag, number, pct, api, renderAccountDetail} = helpers;
  host.replaceChildren();
  host.append(el('div', 'COMMON ACCOUNT COMPARISON', 'eyebrow'), el('h2', '31 組進場條件，同一套資金規則'));
  if (!item?.comparison || item.status !== 'verified_publication') {
    host.append(el('p', '31 組帳戶比較尚未發布完整紀錄；完成與缺件狀態將在這裡顯示。', 'result-note'));
    return;
  }
  const report = item.comparison, cases = Object.values(report.cases), best = report.cases[report.best_completed_strategy];
  host.append(el('p', `${report.start} → ${report.end} · 本金 ${number(report.initial_cash)} 元 · 3 個活躍名額 · 含費稅與滑價`, 'caption'));
  const stats = el('div', undefined, 'research-stats');
  for (const [label, val] of [
    ['完整期間已完成', `${report.completed_strategies} / 31 組`],
    [report.best_is_global ? '歷史淨報酬最高' : '已完成組別中最高', best ? `${best.name} ${pct(best.metrics.total_return)}` : '尚無完整結果'],
    ['0050 同期含息帳戶', pct(report.cases.benchmark.metrics?.total_return)],
  ]) {const box = el('div', undefined, 'research-stat');box.append(el('span', label), el('strong', val));stats.append(box);}
  host.append(stats, tag('尚無策略取得實戰資格', 'neutral'));
  const method = el('details', undefined, 'comparison-method');
  method.append(el('summary', '共同規格與實戰缺件'));
  method.append(el('p', 'T 收盤訊號、T+1 執行；同日候選按前 20 日均成交值排序。共同出場：12% 收盤停損、63 個交易日期限、進場後三黑 K。0050 為買進持有基準。'));
  method.append(el('p', '3 個名額限制新的活躍部位；餘股或延後配股交付可能短暫增加實際持股檔數，帳戶明細另列歷史最多同持檔數。'));
  method.append(el('p', '配股權益在普通股可交易前保留為應收；已核對的權利證書期間採普通股收盤價估值，未模擬證書買賣。各帳戶明細列出相關股票與日期；未核實的畸零款不作為可用現金。'));
  method.append(el('p', '買 L＋70%×(H−L)、賣 L＋30%×(H−L)，為事後日區間代理；佣金每邊 0.1425%、每通路至少 20 元、摩擦每邊 0.45%，另含賣出稅。一般規則每日可成立，2 組情境取母訊號首日。'));
  const gaps = el('ul');for (const text of report.qualification_gaps || []) gaps.append(el('li', text));
  method.append(gaps, el('p', (report.limitations || []).join('；'), 'caption'));host.append(method);
  const controls = el('div', undefined, 'research-controls'), sorting = el('select'), filter = el('input');
  sorting.setAttribute('aria-label', '帳戶比較排序');filter.setAttribute('aria-label', '搜尋帳戶策略');filter.placeholder = '搜尋策略名稱';
  for (const [key, name] of [['total_return', '累積淨報酬'], ['max_drawdown', '回撤較小'], ['2024', '2024 年報酬'], ['2025', '2025 年報酬'], ['2026', '2026 年截至 10/2']]) {
    const option = el('option', name);option.value = key;sorting.append(option);
  }
  controls.append(sorting, filter);host.append(controls);
  host.append(el('p', '表格可左右滑動查看分年、成本與現金；最右側「曲線與明細」可展開帳本。', 'caption'));
  const wrap = el('div', undefined, 'research-table comparison-table'), table = el('table'), head = el('thead'), tr = el('tr');
  for (const label of ['名次／策略', '狀態', '累積淨報酬', '最大回撤', 'CAGR', '2024', '2025', '2026 至 10/2', '已結清批次勝率', '總成本', '平均現金', '明細']) tr.append(el('th', label));
  head.append(tr);const body = el('tbody');table.append(head, body);wrap.append(table);host.append(wrap);
  const detail = el('div', undefined, 'comparison-detail');detail.setAttribute('aria-live', 'polite');host.append(detail);
  let generation = 0;
  function score(row) {const key = sorting.value;return row.completed ? (/^20/.test(key) ? row.annual.find(a => a.year === key)?.total_return : row.metrics[key]) ?? -Infinity : -Infinity;}
  async function open(row, button) {
    const request = ++generation;button.disabled = true;detail.replaceChildren(el('p', `讀取 ${row.name}…`, 'caption'));
    try {
      const result = await api(`/archive/${encodeURIComponent(item.id)}/cases/${encodeURIComponent(row.id)}`);
      if (request !== generation) return;
      detail.replaceChildren();
      const coverage = row.candidate_coverage || {}, metrics = row.metrics;
      detail.append(el('p', `候選 ${number(coverage.executable)}；期末待執行 ${number(coverage.pending)}。已進場 ${number(row.cohorts.funded)} 批、已結清 ${number(row.cohorts.settled)} 批、未結清 ${number(row.cohorts.unresolved)} 批。期末現金 ${number(metrics.cash)}、持股市值 ${number(metrics.market_value)}、應收權益 ${number(metrics.receivable)} 元。`, 'result-note'));
      detail.append(el('p', `歷史最多同持 ${number(metrics.maximum_held_symbols)} 檔；持股超過活躍名額 ${number(metrics.days_above_active_slot_count)} 個交易日（包含餘股與延後配股）。`, 'caption'));
      const pending = row.pending_cash_valuation;
      if (pending?.gross_unavailable_cash > 0) {
        detail.append(el('p', `應收款估值待核：${number(pending.gross_unavailable_cash)} 元尚無確認支付日，未用於買進。若該款項淨額為零，期末資產為 ${number(pending.nav_if_these_claims_pay_zero)} 元；按帳載毛額為 ${number(pending.nav_using_recorded_gross_claims)} 元。這只反映該款項的期末估值差異。`, 'result-note'));
      }
      for (const right of result.certificate_restrictions || []) {
        detail.append(el('p', `${right.stock_id} 配股限制：除權日 ${right.ex_date}，${number(right.whole_new_shares)} 股新權益；證書 ${right.certificate_delivery_date} 交付，普通股 ${right.ordinary_share_available_date} 才可交易。期間按普通股收盤價估值，未模擬證書成交或把它當作可賣普通股。`, 'result-note'));
      }
      for (const note of result.settlement_evidence_notes || []) detail.append(el('p', note, 'result-note'));
      detail.append(el('p', `未知股票日 ${number(coverage.unknown_stock_days)}；情境未知事件 ${number(coverage.unknown_context)}；POC 品質排除 ${number(row.profile_quality_exclusions)}；單邊換手 ${number(metrics.one_way_turnover_multiple, 2)} 倍。未成交或資料排除紀錄：${Object.entries(row.execution_exclusion_reasons || {}).map(([k,v]) => `${k} ${number(v)}`).join('；') || '無'}。`, 'caption'));
      renderAccountDetail(result, detail);detail.scrollIntoView({behavior: 'smooth', block: 'start'});
    } catch (error) {if (request === generation) detail.replaceChildren(el('p', error.message, 'alert error'));}
    finally {button.disabled = false;}
  }
  function render() {
    body.replaceChildren();
    const query = filter.value.trim().toLowerCase();
    const rows = cases.filter(r => `${r.name} ${r.id}`.toLowerCase().includes(query)).sort((a,b) =>
      a.id === 'benchmark' ? -1 : b.id === 'benchmark' ? 1 : Number(b.completed)-Number(a.completed) || score(b)-score(a) || a.id.localeCompare(b.id));
    for (const row of rows) {
      const tr = el('tr');if (row.id === 'benchmark') tr.className = 'comparison-benchmark';
      const name = el('td');name.append(el('strong', `${row.id === 'benchmark' ? '基準' : row.rank ? '#'+row.rank : '—'} · ${row.name}`));tr.append(name);
      tr.append(el('td', row.completed ? '666 日完成' : row.replay_completed ? '回放完成／驗證缺件' : `${row.completed_sessions || 0} 日／缺件`));
      const m = row.metrics, years = Object.fromEntries((row.annual || []).map(a => [a.year, a.total_return]));
      for (const val of [pct(m?.total_return), pct(m?.max_drawdown), pct(m?.cagr), pct(years['2024']), pct(years['2025']), pct(years['2026']), row.completed && row.id !== 'benchmark' ? `${pct(row.cohorts.win_rate,1)} / ${number(row.cohorts.settled)} 批` : '—', number(m?.costs.total_cost), pct(m?.mean_daily_cash_weight,1)]) tr.append(el('td', val));
      const action = el('td');
      if (row.completed) {const button = el('button', '曲線與明細', 'mini-action');button.setAttribute('aria-label', `${row.name} 曲線與明細`);button.addEventListener('click', () => open(row, button));action.append(button);}
      else {const d = el('details');d.append(el('summary', '缺件原因'), el('p', row.reason || '來源尚未齊備'));action.append(d);}
      tr.append(action);body.append(tr);
    }
  }
  sorting.addEventListener('change', render);filter.addEventListener('input', render);render();
  host.append(el('p', '名次依完整期間淨報酬固定；改變排序不改名次。勝率只計已結清訊號批次，未結清持股和應收仍列入 NAV。', 'caption'));
};
