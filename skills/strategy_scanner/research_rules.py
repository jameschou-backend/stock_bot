"""Past-only daily adapters for existing price/volume research hypotheses.

These are new scanner versions, not the old cohorts, accounts or return series.
All rolling windows retain missing market sessions; no prices, RSI changes or
unobserved events are filled with zero. The caller supplies date-aligned, quality
and dated-eligibility-masked matrices and owns the final three-state evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


VERSION = 'scanner_research_v2_20261005'
RESEARCH_IDS = (
    'legacy_momentum_trend', 'legacy_mean_reversion', 'legacy_course_breakout',
    'first_volume_bar_price', 'early_rotation', 'launch_breakout_strength',
    'launch_turnover_heat', 'entry_not_extended', 'entry_strong_close',
    'liquidity_median50m', 'liquidity_prior50m', 'liquidity_persistent50m',
)
_SELF = 'skills/strategy_scanner/research_rules.py'
_FACTORY = ('skills/strategy_factory/strategies.py', 'skills/strategy_factory/data.py')
_GAPS = (
    '這是新的每日研究適配器；不繼承舊母體、持倉配置、成交假設或報酬。',
    '成交值為原始收盤乘成交股數的估算，不等於實際成交金額或資金淨流入。',
    '日期身分及行情品質沿用掃描輸入；不是完整歷史市場或實際可成交認證。',
)


def _item(identifier, name, family, kind, description, sources, *, regimes=(),
          required=(), differences=(), parameters=None):
    return dict(id=identifier, name=name, family=family, kind=kind, status='active',
        description=description, version=VERSION, preferred_regimes=list(regimes),
        required_data=['raw_ohlcv', 'adjusted_close', 'raw_close_times_volume',
                       'market_calendar', 'historical_eligibility', *required],
        source_paths=[_SELF, *sources], source_urls=[], variants=[],
        data_gaps=[*_GAPS, *differences], reusable_interfaces=['add_research_rules'],
        parameters=dict(parameters or {}), signal_timing='T_close_confirmed',
        live_qualified=False, returns_inherited=False)


RESEARCH_CATALOG = [
    _item('legacy_momentum_trend', '舊工廠：動能趨勢（日掃描版）', 'legacy_factory', 'entry',
        '還原收盤高於MA60、20日漲幅>10%、含今日20日均量高於60日均量；20日均估計成交值≥5,000萬元。',
        _FACTORY, regimes=('trend_up',),
        differences=('保留舊主要公式；改用還原收盤、完整市場日窗口及缺值未知，並非原工廠回測。',)),
    _item('legacy_mean_reversion', '舊工廠：均值回歸（日掃描版）', 'legacy_factory', 'entry',
        '14日簡單平均漲跌幅RSI<30且還原收盤低於20日均線減兩倍樣本標準差；均估計成交值≥5,000萬元。',
        _FACTORY, regimes=('range',),
        differences=('RSI採14個完整日變動，非Wilder平滑；缺價不補零，全無漲跌時RSI未知。布林標準差ddof=1。',)),
    _item('legacy_course_breakout', '舊工廠：400日價量高點（日掃描版）', 'legacy_factory', 'entry',
        '收盤≥含今日400日最高還原收盤的99.8%，今日量≥含今日10日最大量；20日均估計成交值≥5億元。',
        _FACTORY, regimes=('trend_up',),
        differences=('保留含今日窗口及5億元門檻；完整400日不足為未知，沒有接入舊出場與資金配置。',)),
    _item('first_volume_bar_price', '盤整後第一根放量紅K（純價量子版）', 'early_launch', 'entry',
        '120日行情完整，前20日收盤區間≤15%，今日量≥前20日均量2倍、漲幅≥2%、嚴格紅K且收在區間上方35%；'
        '前10日沒有同脈衝，且前20日沒有任何合格價量第一根候選；均估計成交值≥5,000萬元。',
        ('skills/first_bar.py', 'docs/prereg_first_bar_20260927.md'), regimes=('range', 'trend_up'),
        differences=('僅價量階段，未使用集保集中、突破等待、指定贏家或未來報酬；不重用旧固定名冊及15%跳價剔除。',
            '冷卻依任何原始價量候選，包括被冷卻擋下的候選；距離至少21市場日。這比舊已接受事件冷卻更嚴，兩者不可視為同策略。',
            '冷卻不讀持倉，缺歷史候選狀態為未知；含120日暖機及20日冷卻觀測，至少140日才可完整判斷。'),
        parameters=dict(price_history_sessions=120, prior_burst_sessions=10,
                        prior_candidate_cooldown_sessions=20, minimum_event_spacing=21)),
    _item('early_rotation', '成交升溫與早期相對強勢（日掃描版）', 'early_launch', 'entry',
        '120日行情完整，近5日均估計成交值／此前不重疊20日均值≥1.5、20日報酬高於0050、'
        '收盤距含今日60日最高收盤≤5%、60日漲幅≤30%；均估計成交值≥5,000萬元。',
        ('skills/stock_launch.py', 'skills/surge_anatomy.py', 'docs/prereg_stock_launch_20260927.md'),
        regimes=('range', 'trend_up'), required=('benchmark_prices',),
        differences=('保留研究價量公式；不用事後急漲標籤、指定案例選樣、旧15%跳價排除或現今產業來產生訊號。',)),
    _item('launch_breakout_strength', '60日收盤突破＋領先0050', 'early_launch', 'entry',
        '120日行情完整，今日收盤嚴格突破此前60日最高收盤，20日報酬領先0050至少10百分點；均估計成交值≥5,000萬元。',
        ('skills/stock_launch.py', 'skills/surge_anatomy.py'), regimes=('trend_up',),
        required=('benchmark_prices',)),
    _item('launch_turnover_heat', '近五日成交值升溫', 'early_launch', 'entry',
        '120日行情完整，近5日均估計成交值至少為此前不重疊20日均值1.5倍；含今日20日均估計成交值≥5,000萬元。',
        ('skills/stock_launch.py',), regimes=('range', 'trend_up'),
        differences=('只代表價量活動升溫，不等於法人淨買超、題材已核實或股價必然上漲。',)),
    _item('entry_not_extended', '進場品質：不過度乖離', 'entry_quality', 'filter',
        '收盤相對MA20乖離≤15%且近5日漲幅≤15%，均估計成交值≥5,000萬元；只是一項獨立篩選，不自行產生進場依據。',
        ('skills/entry_filters.py', 'skills/candidate_quality.py'),
        differences=('未要求乖離下限，深跌亦可能通過；需與另一進場條件配對，沒有套用原三檔候選或市場廣度。',)),
    _item('entry_strong_close', '進場品質：紅K收在高位', 'entry_quality', 'filter',
        '原始收盤嚴格高於開盤，且位於全日高低區間上方30%；均估計成交值≥5,000萬元。全日同價的收盤位置未知。',
        ('skills/entry_filters.py',),
        differences=('此子版嚴格排除十字K；舊green>=0容許同價K，不能繼承其結果。此條件須與進場訊號搭配。',)),
    _item('liquidity_median50m', '流動性：20日成交值中位數', 'liquidity', 'filter',
        '含今日20日估計成交值的均值與中位數都≥5,000萬元；全窗口必須可觀測，僅作流動性篩選。',
        ('skills/liquidity_candidates.py', 'skills/liquidity_diagnostics.py'),
        differences=('目前逐股輸出篩選狀態，未與原候選名单求交；不代表指定價位成交或防操縱保證。',)),
    _item('liquidity_prior50m', '流動性：訊號日前20日均值', 'liquidity', 'filter',
        '含今日20日均估計成交值≥5,000萬元，且不含今日的前20日均值也≥5,000萬元；避免只靠今日放量達標。',
        ('skills/liquidity_candidates.py', 'skills/liquidity_diagnostics.py')),
    _item('liquidity_persistent50m', '流動性：中位數與前期均值', 'liquidity', 'filter',
        '含今日20日成交值均值、中位數，以及不含今日的前20日均值三者都≥5,000萬元；僅為篩選。',
        ('skills/liquidity_candidates.py', 'skills/liquidity_diagnostics.py')),
]


def _finite(*values):
    good = pd.DataFrame(True, index=values[0].index, columns=values[0].columns)
    for frame in values:
        good &= np.isfinite(frame)
    return good.fillna(False)


def _complete(frame, sessions):
    return _finite(frame).rolling(sessions, min_periods=sessions).sum().eq(sessions)


def add_research_rules(f, z, add):
    """Register twelve date-aligned independent rules through engine ``add``.

    ``f`` and ``z`` are shared, wide market-calendar matrices. Only new prefixed
    features are added to ``z``. Rules consume data at or before their own row;
    they do not accept portfolio state, return labels, I/O or execution prices.
    """
    c, h, l, v, amount = (f[k] for k in ('c', 'h', 'l', 'v', 'a'))
    sources = [h, l, v, amount, f['close'], f['open'], f['valid'], f['eligible'],
               *(z[k] for k in ('ma20', 'ma60', 'amount20', 'ret20', 'volume_ratio'))]
    if (not c.index.is_unique or not c.index.is_monotonic_increasing or not c.columns.is_unique
            or any(not q.index.equals(c.index) or not q.columns.equals(c.columns) for q in sources)):
        raise ValueError('Research rule matrices require identical ordered market axes')
    p = 'research_'
    def feature(name, frame):
        key = p+name
        if key in z:
            raise ValueError('Duplicate research feature registration: '+key)
        z[key] = frame
        return frame
    volume20 = feature('volume20_inclusive', v.rolling(20, min_periods=20).mean())
    volume60 = feature('volume60_inclusive', v.rolling(60, min_periods=60).mean())
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rsi = feature('rsi14_simple', (100*gain/(gain+loss)).where((gain+loss).gt(0)))
    lower = feature('bb_lower_sample', z['ma20']-2*c.rolling(20, min_periods=20).std(ddof=1))
    high400 = feature('highest_close400', c.rolling(400, min_periods=400).max())
    max_volume10 = feature('max_volume10', v.rolling(10, min_periods=10).max())
    location = feature('close_location', ((c-l)/(h-l)).where(h.gt(l)))
    return1 = feature('return1', c/c.shift(1)-1)
    return5 = feature('return5', c/c.shift(5)-1)
    return60 = feature('return60', c/c.shift(60)-1)
    distance20 = feature('distance20', c/z['ma20']-1)
    range20 = feature('prior_range20', c.shift(1).rolling(20, min_periods=20).max()
                      /c.shift(1).rolling(20, min_periods=20).min()-1)
    prior_high60 = feature('prior_high_close60', c.shift(1).rolling(60, min_periods=60).max())
    distance_high60 = feature('distance_high60', c/c.rolling(60, min_periods=60).max()-1)
    turnover5 = amount.rolling(5, min_periods=5).mean()
    turnover_prior20 = amount.shift(5).rolling(20, min_periods=20).mean()
    heat = feature('turnover_heat', turnover5/turnover_prior20.where(turnover_prior20.gt(0)))
    median20 = feature('amount20_median', amount.rolling(20, min_periods=20).median())
    prior_amount20 = feature('amount20_prior', amount.shift(1).rolling(20, min_periods=20).mean())
    # Benchmark absence/gaps remain unknown; never substitute a market average.
    relative = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    if '0050' in c:
        own_complete = _complete(c, 21)
        relative = z['ret20'].sub(z['ret20']['0050'], axis=0)
        relative = relative.where(own_complete.mul(own_complete['0050'], axis=0))
    relative = feature('relative20', relative)
    full120 = _complete(c, 120)
    raw_red = f['close'].gt(f['open'])
    liquid = z['amount20'].ge(50_000_000)
    specs = {row['id']: row for row in RESEARCH_CATALOG}
    def register(identifier, match, fields, *, known=None):
        add(identifier, match & liquid, [*fields, 'amount20'], specs[identifier]['description'], known=known)

    register('legacy_momentum_trend', c.gt(z['ma60']) & z['ret20'].gt(.10) & volume20.gt(volume60),
             ['ma60', 'ret20', p+'volume20_inclusive', p+'volume60_inclusive'])
    register('legacy_mean_reversion', rsi.lt(30) & c.lt(lower), [p+'rsi14_simple', p+'bb_lower_sample'])
    register('legacy_course_breakout', c.ge(high400*.998) & v.ge(max_volume10) & z['amount20'].ge(500_000_000),
             [p+'highest_close400', p+'max_volume10'])

    burst_known = _finite(z['volume_ratio'], return1, location) & f['valid'].fillna(False)
    burst = z['volume_ratio'].ge(2) & return1.ge(.02) & raw_red & location.ge(.65)
    prior_bursts = feature('prior_burst_count10', burst.astype(float).where(burst_known)
                          .shift(1).rolling(10, min_periods=10).sum())
    setup = burst & prior_bursts.eq(0) & range20.le(.15) & liquid
    setup_known = full120 & burst_known & _finite(prior_bursts, range20, z['amount20'])
    prior_setups = feature('prior_first_candidate_count20', setup.astype(float).where(setup_known)
                          .shift(1).rolling(20, min_periods=20).sum())
    register('first_volume_bar_price', setup & prior_setups.eq(0),
        ['volume_ratio', p+'return1', p+'close_location', p+'prior_range20',
         p+'prior_burst_count10', p+'prior_first_candidate_count20'], known=setup_known)
    register('early_rotation', heat.ge(1.5) & relative.gt(0) & distance_high60.ge(-.05) & return60.le(.30),
        [p+'turnover_heat', p+'relative20', p+'distance_high60', p+'return60'], known=full120)
    register('launch_breakout_strength', c.gt(prior_high60) & relative.ge(.10),
        [p+'prior_high_close60', p+'relative20'], known=full120)
    register('launch_turnover_heat', heat.ge(1.5), [p+'turnover_heat'], known=full120)
    register('entry_not_extended', distance20.le(.15) & return5.le(.15), [p+'distance20', p+'return5'])
    register('entry_strong_close', raw_red & location.ge(.70), [p+'close_location'])
    register('liquidity_median50m', median20.ge(50_000_000), [p+'amount20_median'])
    register('liquidity_prior50m', prior_amount20.ge(50_000_000), [p+'amount20_prior'])
    register('liquidity_persistent50m', median20.ge(50_000_000) & prior_amount20.ge(50_000_000),
        [p+'amount20_median', p+'amount20_prior'])
