#!/usr/bin/env node
/** Read a sealed replay report and author one Traditional Chinese Excel ledger.
 * Author with the bundled Node/runtime only; no API calls or strategy reruns.
 * --synthetic --preview-only is an explicit layout test and never writes XLSX.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_REPORT = path.join(ROOT, '.cache/million-replay/report.json');
const DEFAULT_OUTPUT = path.join(ROOT, 'outputs/million_replay_20260910');
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(name);
  if (i < 0) return fallback;
  if (!args[i + 1] || args[i + 1].startsWith('--')) throw new Error(`Missing ${name} value`);
  return args[i + 1];
};
const synthetic = args.includes('--synthetic');
const previewOnly = args.includes('--preview-only');
if (synthetic && !previewOnly) throw new Error('Synthetic data may only produce test previews, never the deliverable XLSX.');
const reportPath = path.resolve(option('--input', DEFAULT_REPORT));
const outputDir = path.resolve(option('--output-dir', synthetic ? path.join(ROOT, '.cache/million-workbook-preview') : DEFAULT_OUTPUT));
const modules = option('--runtime-modules', path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'));
const runtime = path.join(ROOT, '.cache/million-workbook-runtime');
await fs.mkdir(runtime, { recursive: true });
const moduleLink = path.join(runtime, 'node_modules');
try { await fs.symlink(modules, moduleLink, 'dir'); }
catch (error) { if (error.code !== 'EEXIST') throw error; }
if (await fs.realpath(moduleLink) !== await fs.realpath(modules)) throw new Error('The task runtime points to different dependencies.');
const require = createRequire(path.join(runtime, 'package.json'));
const { Workbook, SpreadsheetFile } = await import(require.resolve('@oai/artifact-tool'));

const COLORS = { ink: '#263548', navy: '#17365D', blue: '#3266A0', pale: '#EAF0F6',
  gray: '#66788A', line: '#CBD5DF', green: '#008000', red: '#A32129', amber: '#FFF0D1' };
const FONT = 'Arial';
const MONEY = '#,##0;(#,##0);"-"';
const PRICE = '0.00;(0.00);"-"';
const PERCENT = '0.0%;(0.0%);"-"';
const DATE = 'yyyy/mm/dd';
const SHEET_NAMES = ['摘要', '策略全部買賣', '每日資產', '每日持倉', '公司行動', '未成交', '現金流水', '0050對照買賣'];
const REASONS = {
  initial_allocation: '初始配置0050', idle_cash: '閒置資金配置0050', fund_stock: '賣0050準備買股',
  leader_entry: '族群領先訊號進場', scheduled_exit: '63交易日到期退出',
  missing_or_zero_quote_volume: '缺當日價量或成交量為零', single_price_session: '當日只有單一成交價',
  missing_price_limits: '缺漲跌停價格', at_upper_limit: '買進觸及漲停', at_lower_limit: '賣出觸及跌停',
  missing_adv20: '缺完整20日均量', no_odd_lot_trade: '當日無可用零股成交',
  invalid_odd_lot_quote: '零股買賣報價無效', no_odd_lot_opposing_quote_quantity: '零股對手報價量為零',
  odd_lot_at_price_limit: '零股觸及價格限制', proceeds_below_costs: '賣出所得不足交易成本',
  partial_capacity_or_cash: '容量或資金限制，部分成交', capacity_or_cash_zero: '容量或資金不足，未成交',
  overlapping_member: '該股票已有持倉', slots_full: '3個持倉名額已滿',
  prior_liquidity_below_50m_or_missing: '前20日均額不足5,000萬或缺資料', no_prior_price: '缺前日有效價格',
  unsupported_no_price_limit_session: '當日無漲跌幅限制，暫不模擬成交',
  initial_deposit: '投入本金', buy: '買入', sell: '賣出', dividend_payment: '現金股利入帳',
  fractional_share_payment: '畸零股份現金入帳', cash_dividend: '現金股利除息', split: '股份分割',
  stock_dividend: '股票股利除權', payment: '股利付款', share_delivery: '新股交付',
  waive_subscription: '不參與現金增資', cash: '現金應收', shares: '股份應收', fraction: '畸零股份應收',
};
const text = (value) => value == null ? null : String(value).startsWith('=') ? `'${value}` : String(value);
const translate = (value) => value == null ? null : REASONS[value] ?? text(value);
const side = (value) => ({ buy: '買入', sell: '賣出' }[value] ?? translate(value));
const channel = (value) => ({ board: '整張', odd: '零股', event: '事件篩選' }[value] ?? text(value));
const number = (value) => value == null ? null : Number.isFinite(value) ? value : (() => { throw new Error('Non-finite or nonnumeric report value'); })();
const date = (value) => {
  if (value == null || value === '') return null;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error(`Invalid date ${value}`);
  return new Date(`${value}T00:00:00Z`);
};
const col = (i) => { let result = ''; for (i++; i; i = Math.floor((i - 1) / 26)) result = String.fromCharCode(65 + ((i - 1) % 26)) + result; return result; };
const equal = (a, b, label, tolerance = .061) => { if (!Number.isFinite(a) || !Number.isFinite(b) || Math.abs(a - b) > tolerance) throw new Error(`${label}: ${a} != ${b}`); };
const digest = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const accountName = (key) => key === 'strategy' ? '策略' : '0050對照';

function fixture() {
  const prices = [100, 104, 101, 110, 112, 109, 118, 115];
  const fixtureDates = ['2022-01-03','2022-04-01','2022-07-01','2022-10-03','2023-01-03','2023-04-03','2023-07-03','2023-10-02'];
  function makeAccount(key) {
    const qty = key === 'strategy' ? 4000 : 6000;
    const sid = key === 'strategy' ? '1101' : '0050';
    const notional = qty * 100, commission = Math.round(notional * .001425), slippage = Math.ceil(notional * .0045);
    const cost = commission + slippage, cash = 1e6 - notional - cost;
    let previous = 1e6, peak = 1e6;
    const daily = prices.map((price, i) => {
      const day = fixtureDates[i];
      const nav = cash + qty * price;
      peak = Math.max(peak, nav);
      const row = { date: day, opening_nav: previous, market_pnl: i ? qty * (price - prices[i - 1]) : 0,
        dividend_entitlement: 0, execution_basis_pnl: 0, cost: i ? 0 : cost, cash, market_value: qty * price,
        receivable: 0, nav, daily_return: nav / previous - 1, total_return: nav / 1e6 - 1,
        drawdown: nav / peak - 1, holdings: key === 'strategy' ? 1 : 0, stale_holdings: 0 };
      previous = nav; return row;
    });
    const trade = { date: daily[0].date, stock_id: sid, name: key === 'strategy' ? '合成測試股票' : '合成0050',
      side: 'buy', channel: 'board', qty, reference_price: 100, gross: notional, commission, tax: 0, slippage,
      total_cost: cost, cash_change: -notional - cost, cash_after: cash, remaining_shares: qty,
      reason: key === 'strategy' ? 'leader_entry' : 'initial_allocation', signal_date: '2022-01-02',
      requested_qty: qty, capacity_qty: 10000, day_volume: 1000000, prior_avg_volume20: 1200000,
      prior_avg_amount20: 120e6, day_participation: qty / 1e6, event_id: 'synthetic-event', sequence: 1 };
    const holdings = daily.map(row => ({ date: row.date, stock_id: sid, name: trade.name, qty,
      price: row.market_value / qty, market_value: row.market_value, mark_date: row.date,
      event_id: trade.event_id, stale: false }));
    return { daily, trades: [trade], holdings, orders: [{ ...trade, filled_qty: qty, failure: null },
      { ...trade, date: daily[2].date, requested_qty: 800, filled_qty: 0, channel: 'odd', failure: 'no_odd_lot_trade' }],
      corporate_actions: [], receivables: [], cohorts: [],
      cash_ledger: [{ date: daily[0].date, kind: 'initial_deposit', cash_change: 1e6, cash_after: 1e6 },
        { date: daily[0].date, kind: 'buy', stock_id: sid, event_id: trade.event_id, channel: 'board', cash_change: trade.cash_change, cash_after: cash }],
      settings: { initial_cash: 1e6, slots: 3, horizon: 63 } };
  }
  const report = { schema: 1, strategy: makeAccount('strategy'), benchmark: makeAccount('benchmark'),
    summary: {}, strategy_name: '合成資料排版測試，不是研究結果', limitations: ['合成資料僅供版面測試，不代表實際歷史收益。'],
    source_links: ['https://www.twse.com.tw/zh/products/system/trading.html'] };
  for (const key of ['strategy', 'benchmark']) {
    const a = report[key], last = a.daily.at(-1);
    report.summary[key] = { initial_cash: 1e6, final_nav: last.nav, profit: last.nav - 1e6,
      total_return: last.total_return, cagr: null, max_drawdown: Math.min(...a.daily.map(x => x.drawdown)),
      cash: last.cash, market_value: last.market_value, receivable: 0, trade_count: a.trades.length,
      costs: Object.fromEntries(['commission', 'tax', 'slippage', 'total_cost'].map(k => [k, a.trades.reduce((s, t) => s + t[k], 0)])),
      annual: [{ year: '2022', start_nav: 1e6, end_nav: last.nav, total_return: last.total_return, max_drawdown: -.1, partial_year: true }],
      final_holdings: a.holdings.filter(x => x.date === last.date), start: a.daily[0].date, end: last.date };
  }
  return report;
}

const reportBytes = synthetic ? Buffer.from(JSON.stringify(fixture())) : await fs.readFile(reportPath);
const report = JSON.parse(reportBytes);
const reportSha = digest(reportBytes);
if (!synthetic) {
  const manifest = JSON.parse(await fs.readFile(path.join(path.dirname(reportPath), 'manifest.json'), 'utf8'));
  if (manifest.files_sha256?.[path.basename(reportPath)] !== reportSha || manifest.offline_identical !== true)
    throw new Error('The replay report must be sealed and identically reproduced offline before export.');
}
if (report.schema !== 1 || !report.summary?.strategy || !report.summary?.benchmark) throw new Error('Unsupported replay report schema');
for (const key of ['strategy', 'benchmark']) {
  const a = report[key], s = report.summary[key];
  for (const field of ['daily', 'trades', 'orders', 'corporate_actions', 'cash_ledger', 'holdings', 'cohorts', 'receivables'])
    if (!Array.isArray(a?.[field])) throw new Error(`${key}.${field} is required`);
  if (!a.daily.length || a.settings?.initial_cash !== 1e6) throw new Error('A funded NT$1,000,000 account is required');
  equal(a.daily.at(-1).nav, s.final_nav, `${key} final NAV`);
  equal(s.profit, s.final_nav - s.initial_cash, `${key} profit`);
  for (const row of a.daily) {
    equal(row.nav, row.cash + row.market_value + row.receivable, `${key} daily balance`);
    equal(row.nav, row.opening_nav + row.market_pnl + row.dividend_entitlement + row.execution_basis_pnl - row.cost, `${key} daily P&L`);
  }
  for (const trade of a.trades) {
    if (!Number.isInteger(trade.qty) || trade.qty <= 0) throw new Error('Trade quantities must be positive integer shares');
    equal(trade.total_cost, trade.commission + trade.tax + trade.slippage, 'Trade cost', .011);
    equal(trade.gross, Math.round(trade.qty * trade.reference_price * 100) / 100, 'Trade notional', .011);
    equal(trade.cash_change, (trade.side === 'buy' ? -trade.gross : trade.gross) - trade.total_cost, 'Trade cash change', .011);
  }
  equal(a.trades.reduce((sum, trade) => sum + trade.total_cost, 0), s.costs.total_cost, `${key} total costs`);
}
const days = report.strategy.daily.map(r => r.date);
if (JSON.stringify(days) !== JSON.stringify(report.benchmark.daily.map(r => r.date)) || new Set(days).size !== days.length
  || JSON.stringify([...days].sort()) !== JSON.stringify(days)) throw new Error('Strategy and benchmark need the same ordered unique market dates');
const dailyRows = new Map(days.map((day, i) => [day, i + 7]));
const lastRow = days.length + 6;
const wb = Workbook.create();
const sheets = Object.fromEntries(SHEET_NAMES.map(name => [name, wb.worksheets.add(name)]));
for (const sheet of Object.values(sheets)) { sheet.showGridLines = false; }
const artifacts = [], checks = [];

function baseSheet(name, title, width, subtitle = null) {
  const sheet = sheets[name];
  sheet.getRange(`A1:${col(width - 1)}${Math.max(10, 7)}`).format.font = { name: FONT, size: 10, color: COLORS.ink };
  sheet.getRange('A2').values = [[title]];
  sheet.getRange('A2').format.font = { name: FONT, size: 16, bold: true, color: COLORS.navy };
  sheet.getRange(`A3:${col(width - 1)}3`).format.borders = { bottom: { style: 'thin', color: COLORS.line } };
  sheet.getRange('A3').format.rowHeight = 9;
  if (subtitle) { sheet.getRange('A4').values = [[subtitle]]; sheet.getRange('A4').format.font = { name: FONT, size: 9, italic: true, color: COLORS.gray }; }
  return sheet;
}

function table(name, title, columns, rows, id, { source = true } = {}) {
  const sourceText = source ? `來源：本次封存重播帳本，報告SHA256 ${reportSha.slice(0, 16)}。金額NT$；股數為整數；空白為不適用或來源未提供。` : null;
  const sheet = baseSheet(name, title, columns.length, sourceText);
  const end = Math.max(7, rows.length + 6);
  sheet.getRange(`A6:${col(columns.length - 1)}6`).values = [columns.map(c => c.label)];
  const values = rows.map((row, i) => columns.map(c => {
    const value=c.get ? c.get(row, i + 7) : row[c.key] ?? null;
    return c.format==='@' && value!=null ? String(value) : value;
  }));
  if (values.length) sheet.getRange(`A7:${col(columns.length - 1)}${end}`).values = values;
  const area = sheet.getRange(`A6:${col(columns.length - 1)}${end}`);
  area.format.font = { name: FONT, size: 10, color: COLORS.ink };
  area.format.rowHeight = 22;
  area.format.verticalAlignment = 'center';
  columns.forEach((c, i) => {
    const range = sheet.getRange(`${col(i)}6:${col(i)}${end}`);
    range.format.columnWidth = c.width ?? 16;
    // The renderer formats numeric-looking strings even with '@'. Keep the
    // underlying identifier a string and display its four digits explicitly.
    if (c.format) sheet.getRange(`${col(i)}7:${col(i)}${end}`).setNumberFormat(c.format==='@'?'0000':c.format);
    range.format.horizontalAlignment = c.format==='@' || c.format===DATE ? 'center' : c.format ? 'right' : 'left';
  });
  const t = sheet.tables.add(`A6:${col(columns.length - 1)}${end}`, true, id);
  t.style = 'TableStyleLight9'; t.showFilterButton = true;
  const header = sheet.getRange(`A6:${col(columns.length - 1)}6`);
  header.format = { fill: COLORS.navy, font: { name: FONT, size: 10, bold: true, color: '#FFFFFF' },
    rowHeight: 32, horizontalAlignment: 'center', verticalAlignment: 'center', wrapText: true,
    borders: { insideVertical: { style: 'thin', color: '#FFFFFF' } } };
  sheet.freezePanes.freezeRows(6); sheet.freezePanes.freezeColumns(2);
  if (!rows.length) sheet.getRange('A8').values = [['本期間無紀錄']];
  checks.push({ sheet: name, sourceRows: rows.length, firstDataRow: 7, lastDataRow: rows.length + 6, columns: columns.length });
  return { sheet, end, values };
}
const c = (key, label, format = null, width = 16, get = null) => ({ key, label, format, width, get });

// Daily cash, market values and receivables are source inputs. NAV/returns and
// drawdown are visible spreadsheet formulas, independently checked against JSON.
const dailyColumns = [c('date', '日期', DATE, 13, r => date(r.date)),
  ...[['opening_nav','期初資產'],['market_pnl','市價損益'],['dividend_entitlement','股利權利'],
      ['execution_basis_pnl','成交價差損益'],['cost','交易成本'],['cash','現金'],['market_value','持倉市值'],['receivable','應收權利估值']]
    .map(([key,label]) => c(key,label,MONEY,16,r => number(r[key]))),
  c('nav','策略淨資產',MONEY), c('pnl','單日損益',MONEY), c('total_return','累積報酬',PERCENT),
  c('peak','歷史最高資產',MONEY), c('drawdown','距高點回撤',PERCENT),
  c('bench_cash','0050現金',MONEY),c('bench_assets','0050市值',MONEY),c('bench_receivable','0050應收權利',MONEY),
  c('bench_nav','0050淨資產',MONEY), c('bench_return','0050累積報酬',PERCENT),
  c('bench_peak','0050歷史最高',MONEY),c('bench_drawdown','0050回撤',PERCENT),
  c('holdings','事件股票檔數', '#,##0', 14), c('stale_holdings','前值估價檔數', '#,##0',14)];
const dailyData = report.strategy.daily.map((row, i) => ({ ...row, bench_cash: report.benchmark.daily[i].cash,
  bench_assets: report.benchmark.daily[i].market_value, bench_receivable: report.benchmark.daily[i].receivable }));
table('每日資產','每日資產變化',dailyColumns,dailyData,'DailyAssets');
sheets['每日資產'].getRange(`J7:N${lastRow}`).formulas=days.map((_,i)=>{
  const row=i+7;return [`=SUM(G${row}:I${row})`,`=J${row}-B${row}`,`=J${row}/$B$7-1`,
    i ? `=MAX(M${row-1},J${row})` : `=MAX($B$7,J${row})`, `=J${row}/M${row}-1`];
});
sheets['每日資產'].getRange(`R7:U${lastRow}`).formulas=days.map((_,i)=>{
  const row=i+7;return [`=SUM(O${row}:Q${row})`,`=R${row}/$B$7-1`,
    i ? `=MAX(T${row-1},R${row})` : `=MAX($B$7,R${row})`, `=R${row}/T${row}-1`];
});
sheets['每日資產'].getRange(`N7:N${lastRow}`).conditionalFormats.add('cellIs', { operator: 'lessThan', formula: -.3, format: { fill: '#FCE8E8', font: { color: COLORS.red } } });

function tradeColumns(key) {
  return [c('date','成交日',DATE,13,r=>date(r.date)),c('side','買賣',null,8,r=>side(r.side)),
    c('stock_id','代號','@',8,r=>text(r.stock_id)),c('name','名稱',null,16,r=>text(r.name)),
    c('channel','交易別',null,10,r=>channel(r.channel)),c('qty','成交股數','#,##0',13),
    c('reference_price','成交參考價',PRICE,16),c('gross','成交金額',MONEY),c('commission','佣金',MONEY,13),
    c('tax','證交稅',MONEY,13),c('slippage','滑價現金成本',MONEY),c('total_cost','合計成本',MONEY),
    c('cash_change','現金收付',MONEY),c('cash_after','成交後現金',MONEY),
    c('end_nav','當日期末資產',MONEY,18),c('remaining_shares','交易後持股','#,##0',15),
    c('reason','交易原因',null,27,r=>translate(r.reason)),c('signal_date','訊號日',DATE,13,r=>date(r.signal_date)),
    c('requested_qty','委託上限股數','#,##0'),c('capacity_qty','當日容量股數','#,##0'),
    c('day_volume','普通盤當日量','#,##0',18),c('prior_avg_volume20','前20日均量','#,##0',18),
    c('prior_avg_amount20','前20日均額',MONEY,18),c('day_participation','本筆量參與率',PERCENT,16),
    c('odd_volume','零股當日量','#,##0',16),c('odd_bid','零股買價',PRICE,13),c('odd_ask','零股賣價',PRICE,13),
    c('price_limit_source','價格限制來源',null,30,r=>text(r.price_limit_source)),
    c('event_id','事件編號',null,48,r=>text(r.event_id)),
    c('odd_bid_qty','零股最後買量','#,##0',18),c('odd_ask_qty','零股最後賣量','#,##0',18),
    c('above_last_opposite','高於最後對手量',null,30,r=>r.channel!=='odd'?null:
      (r.odd_bid_qty==null||r.odd_ask_qty==null?'無最後揭示量資料':aboveLastOpposite(r)?'是：僅日量假設':'否'))];
}
function aboveLastOpposite(row) {
  const opposite=row.side==='buy'?row.odd_ask_qty:row.odd_bid_qty;
  return row.channel==='odd'&&Number.isFinite(opposite)&&row.qty>opposite;
}
const oddAudit=Object.fromEntries(['strategy','benchmark'].map(key=>{
  const rows=report[key].trades.filter(row=>row.channel==='odd');
  return [key,{oddTrades:rows.length,aboveLastOpposite:rows.filter(aboveLastOpposite).length}];
}));
for (const [key,name] of [['strategy','策略全部買賣'],['benchmark','0050對照買賣']]) {
  const rows=report[key].trades;
  table(name,key==='strategy'?'策略全部買賣':'0050對照全部買賣',tradeColumns(key),rows,key==='strategy'?'StrategyTrades':'BenchmarkTrades');
  const sheet=sheets[name];
  sheet.freezePanes.freezeColumns(4);
  sheet.getRange('A5').values=[['整張參考普通收盤；零股參考零股最後價；滑價另扣現金。「當日期末資產」不是交易瞬間資產。']];
  sheet.getRange('A5').format.font={name:FONT,size:9,color:COLORS.gray};
  if(rows.length) {
    const end=rows.length+6;
    sheet.getRange(`H7:H${end}`).formulas=rows.map((_,i)=>[`=ROUND(F${i+7}*G${i+7},2)`]);
    sheet.getRange(`L7:M${end}`).formulas=rows.map((_,i)=>[`=SUM(I${i+7}:K${i+7})`,`=IF(B${i+7}="買入",-H${i+7},H${i+7})-L${i+7}`]);
    sheet.getRange(`O7:O${end}`).formulas=rows.map(row=>{
      const daily=dailyRows.get(row.date);if(!daily) throw new Error('Trade date outside daily NAV ledger');
      return [`='每日資產'!${key==='strategy'?'J':'R'}${daily}`];
    });
    sheet.getRange(`O7:O${end}`).format.font.color=COLORS.green;
    sheet.getRange(`AF7:AF${end}`).conditionalFormats.add('containsText',{text:'是：',format:{fill:COLORS.amber}});
  }
}

const accounts=['strategy','benchmark'];
const holdings=accounts.flatMap(key=>report[key].holdings.map(row=>({...row,account:accountName(key)})))
  .sort((a,b)=>a.date.localeCompare(b.date)||a.account.localeCompare(b.account)||a.stock_id.localeCompare(b.stock_id));
table('每日持倉','每日持倉與估價日',[c('account','帳戶',null,12),c('date','日期',DATE,13,r=>date(r.date)),
  c('stock_id','代號','@',8),c('name','名稱',null,17),c('qty','持有股數','#,##0',15),c('price','估價單價',PRICE,15),
  c('market_value','持倉市值',MONEY,19),c('mark_date','報價所屬日',DATE,14,r=>date(r.mark_date)),
  c('stale','估價狀態',null,16,r=>r.stale?'沿用之前報價':'當日報價'),c('event_id','事件編號',null,48)],holdings,'DailyHoldings');
if(holdings.length) sheets['每日持倉'].getRange(`G7:G${holdings.length+6}`).formulas=holdings.map((_,i)=>[`=E${i+7}*F${i+7}`]);
if(holdings.length) sheets['每日持倉'].getRange(`I7:I${holdings.length+6}`).conditionalFormats.add('containsText',{text:'沿用',format:{fill:COLORS.amber}});

const actions=accounts.flatMap(key=>[
  ...report[key].corporate_actions.map(row=>({...row,account:accountName(key),display_kind:translate(row.kind)})),
  ...report[key].receivables.map(row=>({...row,account:accountName(key),date:days.at(-1),display_kind:`期末未收取：${translate(row.kind)}`,
    note:row.kind==='shares'?'股利新股尚未交付，交付前不可交易；已計入期末權利估值。':row.kind==='cash'?'尚未收取的現金股利，尚不可作為買股資金。':row.note}))
]).sort((a,b)=>(a.date??'').localeCompare(b.date??''));
table('公司行動','公司行動與期末應收',[c('account','帳戶',null,12),c('date','記帳日',DATE,13,r=>date(r.date)),
  c('stock_id','代號','@',8),c('display_kind','公司行動',null,25),c('action_id','公告識別碼',null,34),
  c('entitled_qty','享有權利股數','#,##0',18),c('cash_per_share','每股現金股利',PRICE,18),
  c('entitlement_value','新認列現金權利',MONEY,18),c('amount','現金應收或付款',MONEY,18),
  c('qty_after','分割後股數','#,##0',17),c('whole_new_shares','應交付整數股','#,##0',18),
  c('fractional_right','畸零股份權利','0.0000',18,r=>r.fractional_right??r.fraction??null),c('qty','待交付股數','#,##0',16),
  c('ex_date','原除權息日',DATE,14,r=>date(r.ex_date)),c('pay_date','付款或交付日',DATE,15,r=>date(r.pay_date)),
  c('event_id','事件編號',null,48),c('multiplier','分割換股倍數','0.0000',18),
  c('shares_per_share','每股配發股數','0.0000',18),c('fractional_cash_per_share','畸零股現金單價',PRICE,19),
  c('note','說明',null,65),c('source','公告來源',null,80,r=>text(r.source))],actions,'CorporateActions');

const failures=accounts.flatMap(key=>report[key].orders.filter(row=>row.failure).map(row=>({...row,account:accountName(key)})))
  .sort((a,b)=>a.date.localeCompare(b.date));
table('未成交','未成交與部分成交',[c('account','帳戶',null,12),c('date','委託日',DATE,13,r=>date(r.date)),
  c('stock_id','代號','@',8),c('name','名稱',null,17),c('side','買賣',null,8,r=>side(r.side)),
  c('channel','交易別',null,12,r=>channel(r.channel)),c('requested_qty','委託股數','#,##0',15),
  c('filled_qty','已成交股數','#,##0',15),c('unfilled','未成交股數','#,##0',15),
  c('failure','未完成原因',null,36,r=>translate(r.failure)),c('reference_price','成交參考價',PRICE,16),
  c('prior_avg_volume20','前20日均量','#,##0',18),c('prior_avg_amount20','前20日均額',MONEY,18),
  c('day_volume','普通盤當日量','#,##0',18),c('odd_volume','零股當日量','#,##0',18),
  c('reason','委託目的',null,28,r=>translate(r.reason)),c('signal_date','訊號日',DATE,13,r=>date(r.signal_date)),
  c('event_id','事件編號',null,48)],failures,'UnfilledOrders');
if(failures.length) sheets['未成交'].getRange(`I7:I${failures.length+6}`).formulas=failures.map((_,i)=>[`=G${i+7}-H${i+7}`]);

const cash=accounts.flatMap(key=>report[key].cash_ledger.map((row,i)=>({...row,account:accountName(key),sequence:i+1})))
  .sort((a,b)=>a.date.localeCompare(b.date)||a.account.localeCompare(b.account)||a.sequence-b.sequence);
table('現金流水','現金收付流水',[c('account','帳戶',null,12),c('date','日期',DATE,13,r=>date(r.date)),
  c('sequence','帳戶序號','#,##0',13),c('kind','收付原因',null,25,r=>translate(r.kind)),c('stock_id','代號','@',8),
  c('channel','交易別',null,12,r=>channel(r.channel)),c('cash_change','現金收付',MONEY,18),c('cash_after','收付後現金',MONEY,19),
  c('event_id','事件編號',null,48),c('action_id','公司行動編號',null,40)],cash,'CashLedger');
const sourceSheet=sheets['現金流水'];
sourceSheet.getRange('A2:J2').merge();
sourceSheet.getRange('A4:J4').merge();
sourceSheet.getRange('L2').values=[['資料與制度來源']];
sourceSheet.getRange('L2').format.font={name:FONT,size:10,bold:true,color:COLORS.navy};
const sources=[['本次封存逐筆帳本',`報告SHA256：${reportSha}`],
  ...((report.source_links??[]).map((url,i)=>[`制度來源${i+1}`,url]))];
sourceSheet.getRange(`L3:M${sources.length+2}`).values=sources;
sourceSheet.getRange(`L3:L${sources.length+2}`).format.columnWidth=23;
sourceSheet.getRange(`M3:M${sources.length+2}`).format.columnWidth=95;
sourceSheet.getRange(`L3:M${sources.length+2}`).format.font={name:FONT,size:9,color:COLORS.gray};

const summary=baseSheet('摘要',synthetic?'合成資料版面測試':'一百萬歷史交易帳本',13);
summary.tabColor=COLORS.navy;
summary.getRange('A4').values=[[`${days[0].replaceAll('-','/')} 至 ${days.at(-1).replaceAll('-','/')}；期末持倉以收盤市值列帳，未假設全部賣出。`]];
summary.getRange('A4').format.font={name:FONT,size:10,color:COLORS.gray};
summary.getRange('A5:B5').values=[['投入本金（NT$）',1e6]];
summary.getRange('B5').setNumberFormat(MONEY);
summary.getRange('A7:C7').values=[['項目','策略','0050對照']];
const labels=['期末淨資產','累積盈虧','累積報酬','最大回撤','現金','持倉市值','應收股利及新股','佣金、稅及滑價','已成交筆數'];
summary.getRange('A8:A16').values=labels.map(v=>[v]);
const strategyEnd=report.strategy.trades.length+6,benchmarkEnd=report.benchmark.trades.length+6;
const totals=key=>report[key].trades.length?`=SUM('${key==='strategy'?'策略全部買賣':'0050對照買賣'}'!L7:L${key==='strategy'?strategyEnd:benchmarkEnd})`:'=0';
summary.getRange('B8:C16').formulas=[
  [`='每日資產'!J${lastRow}`,`='每日資產'!R${lastRow}`],['=B8-$B$5','=C8-$B$5'],
  [`='每日資產'!L${lastRow}`,`='每日資產'!S${lastRow}`],
  [`=MIN('每日資產'!N7:N${lastRow})`,`=MIN('每日資產'!U7:U${lastRow})`],
  [`='每日資產'!G${lastRow}`,`='每日資產'!O${lastRow}`],
  [`='每日資產'!H${lastRow}`,`='每日資產'!P${lastRow}`],
  [`='每日資產'!I${lastRow}`,`='每日資產'!Q${lastRow}`],[totals('strategy'),totals('benchmark')],
  [report.strategy.trades.length?`=COUNTA('策略全部買賣'!A7:A${strategyEnd})`:'=0',report.benchmark.trades.length?`=COUNTA('0050對照買賣'!A7:A${benchmarkEnd})`:'=0']];
summary.getRange('A7:C16').format.rowHeight=27;
summary.getRange('A7:C7').format={fill:COLORS.navy,font:{name:FONT,size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:30};
summary.getRange('A8:C8').format={fill:COLORS.pale,font:{name:FONT,size:10,bold:true,color:COLORS.navy}};
summary.getRange('B8:C15').setNumberFormat(MONEY);summary.getRange('B10:C11').setNumberFormat(PERCENT);
summary.getRange('B16:C16').setNumberFormat('#,##0');
summary.getRange('A8:C16').format.borders={bottom:{style:'thin',color:COLORS.line}};
summary.getRange('A18:C18').values=[['相對0050累積報酬差',null,'百分點']];
summary.getRange('B18').formulas=[['=(B10-C10)*100']];summary.getRange('B18').setNumberFormat('0.0;(0.0);"-"');
summary.getRange('A20').values=[['採用原3部位規則，63個市場交易日後開始退出。成交不足保留未成交明細。']];
summary.getRange('A20').format.font={name:FONT,size:9,color:COLORS.gray};
const pendingStocks=report.strategy.receivables.filter(row=>row.kind==='shares'&&row.qty>0);
if(pendingStocks.length) {
  summary.getRange('A21').values=[[`期末${[...new Set(pendingStocks.map(row=>row.stock_id))].join('、')}股票股利新股尚未交付；已計入權利估值，交付前不能交易。交付日期見「公司行動」。`]];
  summary.getRange('A21').format.font={name:FONT,size:9,color:COLORS.red};
}

// Native, range-backed chart uses quarter-end NAV links. Daily detail remains
// complete; plotting quarterly snapshots is explicitly labelled.
const quarterly=[];
for(let i=0;i<days.length;i++) {
  const label=`${days[i].slice(0,4)} Q${Math.ceil(Number(days[i].slice(5,7))/3)}`;
  const next=i+1<days.length?`${days[i+1].slice(0,4)} Q${Math.ceil(Number(days[i+1].slice(5,7))/3)}`:null;
  if(label!==next) quarterly.push({label:i===days.length-1?`${days[i].slice(0,7)}`:label,row:i+7});
}
summary.getRange('A23:D23').values=[['期間末','策略淨資產','0050淨資產','金額差']];
summary.getRange('A23:D23').format={fill:COLORS.pale,font:{name:FONT,size:10,bold:true,color:COLORS.navy},rowHeight:28};
for(let i=0;i<quarterly.length;i++) {
  const row=i+24,q=quarterly[i];
  summary.getRange(`A${row}`).values=[[q.label]];
  summary.getRange(`B${row}:D${row}`).formulas=[[`='每日資產'!J${q.row}`,`='每日資產'!R${q.row}`,`=B${row}-C${row}`]];
}
summary.getRange(`B24:D${quarterly.length+23}`).setNumberFormat(MONEY);
summary.getRange(`A23:D${quarterly.length+23}`).format.rowHeight=23;
summary.getRange('F23:J23').values=[['年度','策略報酬','0050報酬','差距(百分點)','策略期末資產']];
summary.getRange('F23:J23').format={fill:COLORS.pale,font:{name:FONT,size:10,bold:true,color:COLORS.navy},rowHeight:28,wrapText:true};
const annualYears=[...new Set(days.map(day=>day.slice(0,4)))];
for(let i=0;i<annualYears.length;i++) {
  const year=annualYears[i],yearDates=days.filter(day=>day.startsWith(year));
  const first=dailyRows.get(yearDates[0]),last=dailyRows.get(yearDates.at(-1)),row=24+i;
  summary.getRange(`F${row}`).values=[[year===days.at(-1).slice(0,4)&&!days.at(-1).endsWith('12-31')?`${year}截至${days.at(-1).slice(5).replace('-','/')}`:year]];
  summary.getRange(`G${row}:J${row}`).formulas=[[`='每日資產'!J${last}/'每日資產'!B${first}-1`,
    `='每日資產'!R${last}/'每日資產'!${first===7?'B7':`R${first-1}`}-1`,`=(G${row}-H${row})*100`,`='每日資產'!J${last}`]];
}
summary.getRange(`G24:H${annualYears.length+23}`).setNumberFormat(PERCENT);
summary.getRange(`I24:I${annualYears.length+23}`).setNumberFormat('0.0;(0.0);"-"');
summary.getRange(`J24:J${annualYears.length+23}`).setNumberFormat(MONEY);
// All sessions are retained; the text helper only makes sparse date ticks
// legible and never aggregates or removes daily NAV observations.
const dailySheet=sheets['每日資產'];
dailySheet.getRange('Y6').values=[['走勢圖日期']];
dailySheet.getRange(`Y7:Y${lastRow}`).formulas=days.map((_,i)=>[`=TEXT(A${i+7},"yyyy/mm/dd")`]);
dailySheet.getRange(`Y6:Y${lastRow}`).format.columnWidth=15;
const chart=summary.charts.add('line',[dailySheet.getRange(`Y6:Y${lastRow}`),dailySheet.getRange(`J6:J${lastRow}`),dailySheet.getRange(`R6:R${lastRow}`)]);
chart.setPosition('E7','M19');chart.title=`每日資產走勢（${days.length}日，NT$）`;
chart.titleTextStyle.typeface=FONT;chart.titleTextStyle.fontSize=14;
chart.legend={position:'top',textStyle:{typeface:FONT,fontSize:11}};
chart.xAxis={axisType:'textAxis',tickLabelInterval:Math.max(1,Math.ceil(days.length/6)),textStyle:{typeface:FONT,fontSize:10}};
chart.yAxis={numberFormatCode:'0.00,,"百萬"',numberFormatSourceLinked:false,textStyle:{typeface:FONT,fontSize:11}};
chart.series.items[0].line={fill:COLORS.blue,style:'solid',width:2};
chart.series.items[1].line={fill:'#808891',style:'dashed',width:2};
const noteStart=Math.max(47,quarterly.length+26);
summary.getRange(`A${noteStart}`).values=[['本次模擬的範圍']];
summary.getRange(`A${noteStart}`).format.font={name:FONT,size:10,bold:true,color:COLORS.navy};
const workbookLimitations=[...(report.limitations??[]),
  `策略零股 ${oddAudit.strategy.aboveLastOpposite}/${oddAudit.strategy.oddTrades} 筆高於最後揭示對手量；0050 對照 ${oddAudit.benchmark.aboveLastOpposite}/${oddAudit.benchmark.oddTrades} 筆。僅符合全日日量假設，無法證明最後揭示時點全成交。`];
workbookLimitations.forEach((value,i)=>{
  summary.getRange(`A${noteStart+i+1}`).values=[[text(value)]];
  summary.getRange(`A${noteStart+i+1}`).format.font={name:FONT,size:9,color:COLORS.gray};
});
summary.getRange('A1:A60').format.columnWidth=28;
summary.getRange('B1:D60').format.columnWidth=19;
summary.getRange('E1:M60').format.columnWidth=11;
summary.getRange('F1:F60').format.columnWidth=20;
summary.getRange('J1:J60').format.columnWidth=18;
summary.getRange(`A1:M${noteStart+workbookLimitations.length+2}`).format.verticalAlignment='center';

// One final recalculation, bounded content/error checks, then previews for all
// eight sheets. JSON control values are independent of workbook formulas.
wb.recalculate();
const summaryValues=summary.getRange('B8:C16').values;
for(const [key,index] of [['strategy',0],['benchmark',1]]) {
  equal(summaryValues[0][index],report.summary[key].final_nav,`${key} Excel final NAV`);
  equal(summaryValues[1][index],report.summary[key].profit,`${key} Excel profit`);
  equal(summaryValues[2][index],report.summary[key].total_return,`${key} Excel return`,1e-9);
  equal(summaryValues[3][index],report.summary[key].max_drawdown,`${key} Excel drawdown`,1e-9);
  equal(summaryValues[7][index],report.summary[key].costs.total_cost,`${key} Excel total cost`);
  if(summaryValues[8][index]!==report[key].trades.length) throw new Error('Excel trade count does not match the report');
  const tradeSheet=sheets[key==='strategy'?'策略全部買賣':'0050對照買賣'];
  if(report[key].trades.length) {
    const ids=tradeSheet.getRange(`C7:C${report[key].trades.length+6}`).values.flat();
    if(ids.some((value,i)=>value!==report[key].trades[i].stock_id)) throw new Error('Excel stock identifiers lost their original text/leading zeroes');
  }
}
const inspect=await wb.inspect({kind:'table',range:'摘要!A7:C18',include:'values,formulas',tableMaxRows:12,tableMaxCols:3,maxChars:3500});
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:3500});
if(/#REF!|#DIV\/0!|#VALUE!|#NAME\?|#N\/A|#NUM!|#NULL!|#SPILL!|#CALC!/.test(errors.ndjson??'')) throw new Error('Unexpected workbook formula errors: '+errors.ndjson);
await fs.mkdir(outputDir,{recursive:true});
await fs.writeFile(path.join(outputDir,'workbook-checks.json'),JSON.stringify({synthetic,report_sha256:reportSha,rows:checks,
  summary_values:summaryValues,odd_lot_audit:oddAudit,chart_point_count:days.length,inspect:inspect.ndjson,formula_errors:errors.ndjson,source_dates:[days[0],days.at(-1)]},null,2));
const previewRanges=SHEET_NAMES.map(name=>({name,range:name==='摘要'?'A1:M21':name==='每日資產'?'A1:J13':name==='公司行動'?'A1:I13':
  ['每日持倉','現金流水'].includes(name)?'A1:J13':'A1:O13',suffix:''}));
previewRanges.push({name:'摘要',range:`A22:M${noteStart+workbookLimitations.length+1}`,suffix:'-年度與說明'},
  {name:'每日資產',range:'K5:W13',suffix:'-報酬與0050'},
  {name:'策略全部買賣',range:'P5:AC13',suffix:'-容量與原因'},
  {name:'公司行動',range:'J5:U13',suffix:'-股份與來源'},
  {name:'策略全部買賣',range:'AD5:AF16',suffix:'-零股最後對手量'},
  {name:'0050對照買賣',range:'AD5:AF16',suffix:'-零股最後對手量'},
  {name:'公司行動',range:`A${Math.max(7,actions.length)}:O${actions.length+6}`,suffix:'-期末應收'},
  {name:'公司行動',range:`M${Math.max(7,actions.length)}:U${actions.length+6}`,suffix:'-期末交付說明'},
  {name:'每日資產',range:`A${Math.max(7,lastRow-4)}:J${lastRow}`,suffix:'-期末資產'},
  {name:'現金流水',range:`L1:M${sources.length+3}`,suffix:'-來源'});
for(const {name,range,suffix} of previewRanges) {
  const blob=await wb.render({sheetName:name,range,scale:1.3,format:'png'});
  const filename=`preview-${name}${suffix}.png`;
  await fs.writeFile(path.join(outputDir,filename),new Uint8Array(await blob.arrayBuffer()));artifacts.push(filename);
}
if(!previewOnly) {
  const xlsx=await SpreadsheetFile.exportXlsx(wb);
  const output=path.join(outputDir,'一百萬歷史交易帳本.xlsx');
  await xlsx.save(output);
  console.log(JSON.stringify({output,report_sha256:reportSha,sheets:SHEET_NAMES,previews:artifacts},null,2));
} else console.log(JSON.stringify({synthetic,preview_only:true,output_directory:outputDir,previews:artifacts},null,2));
