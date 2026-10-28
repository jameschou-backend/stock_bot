import inspect
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import backtest_rotation as rotation
from skills import backtest, train_ranker


@pytest.mark.parametrize('module,method', [(rotation, '_train_model'), (backtest, '_train_model'), (train_ranker, '_build_model')])
def test_missing_lightgbm_is_an_explicit_error(monkeypatch, module, method):
    monkeypatch.setattr(module, '_HAS_LGBM', False)
    with pytest.raises(RuntimeError, match='LightGBM'):
        getattr(module, method)(np.ones((3, 2)), np.zeros(3))


def test_research_defaults_delay_and_charge_slippage():
    cfg = backtest.WalkForwardConfig()
    assert cfg.entry_delay_days >= 1
    assert cfg.enable_slippage
    sig = inspect.signature(rotation.run_rotation).parameters
    assert sig['signal_lag'].default >= 1
    assert sig['slippage_pct'].default > 0


def test_training_identity_changes_when_historical_data_is_corrected():
    frame = pd.DataFrame({'ret_20':[.1,.2], 'amt_20':[1.,2.]})
    labels = pd.Series([.1,.2])
    original = train_ranker.training_data_fingerprint(frame, labels, None, True)
    assert train_ranker.training_data_fingerprint(frame.copy(), labels.copy(), None, True) == original
    changed = frame.copy(); changed.loc[0,'ret_20'] = .15
    assert train_ranker.training_data_fingerprint(changed, labels, None, True) != original
    assert train_ranker.training_data_fingerprint(frame, labels + .01, None, True) != original
    assert train_ranker.training_data_fingerprint(frame, labels, None, False) != original


def test_risk_module_failure_cannot_silently_disable_cash_control(monkeypatch):
    from skills import vol_targeting
    monkeypatch.setattr(vol_targeting, 'compute_vol_scaler_for_picks', lambda *a, **k: (_ for _ in ()).throw(ValueError('bad covariance')))
    pipe = backtest.BacktestPipeline(SimpleNamespace(), None, backtest.WalkForwardConfig(vol_target_pct=.2))
    pipe.price_df = pd.DataFrame()
    with pytest.raises(RuntimeError, match='volatility'):
        pipe._compute_vol_target_cash_share(pd.DataFrame({'stock_id':['2330','6488'], 'score':[2,1]}), date(2026,10,27),2)


def test_rotation_account_marks_open_positions_and_cannot_spend_cash_twice():
    from skills.rotation_account import RotationAccount
    book = RotationAccount(10000)
    book.buy('2330', 100, 6000, .001425)
    book.buy('6488', 100, 6000, .001425)
    assert book.cash >= -1e-9
    assert book.nav({'2330':50, '6488':100}) == pytest.approx(7000 / 1.001425)
    with pytest.raises(ValueError): book.buy('2317',100,1000,.001425)


def test_realized_loss_is_not_clipped_and_terminal_nav_reconciles():
    from skills.rotation_account import RotationAccount
    book = RotationAccount(10000)
    book.buy('2330',100,10000,0)
    assert book.nav({'2330':20}) == 2000
    result = book.sell('2330',20,0)
    assert result['realized_pnl_pct'] == pytest.approx(-.8)
    assert book.nav({}) == book.cash == 2000


def test_rotation_benchmark_uses_actual_price_returns():
    frame = pd.DataFrame({'stock_id':['2330']*3, 'trading_date':[date(2026,10,d) for d in (20,21,22)], 'adj_close':[100,90,110]})
    curve = rotation.compute_benchmark_curve(frame, [date(2026,10,21),date(2026,10,22)],10000)
    assert curve[-1]['equity'] == pytest.approx(11000)
    assert curve[0]['equity'] == pytest.approx(9000)


def test_no_history_cannot_use_a_calendar_day_cutoff():
    days = [date(2026,10,20), date(2026,10,21)]
    assert rotation.compute_trading_day_cutoff(days, days[1],20) == days[0]


def test_rotation_end_to_end_marks_drawdown_before_sale(monkeypatch):
    days = pd.bdate_range('2025-01-01', periods=80).date.tolist()
    ids = [str(2300 + i) for i in range(20)]
    prices = pd.DataFrame([dict(stock_id=sid, trading_date=d, close=20. if sid==ids[0] and d==days[66] else 100., volume=1000000)
                           for d in days for sid in ids])
    features = pd.DataFrame([dict(stock_id=sid, trading_date=d, signal=20-i)
                             for d in days for i,sid in enumerate(ids)])
    labels = features[['stock_id','trading_date']].assign(future_ret_h=0.01)
    values = iter([days[-1], days[0], days[-1]])
    session = SimpleNamespace(query=lambda *a: SimpleNamespace(scalar=lambda:next(values)))
    monkeypatch.setattr(rotation.data_store,'get_prices',lambda *a: prices.copy())
    monkeypatch.setattr(rotation.data_store,'get_features',lambda *a: features.copy())
    monkeypatch.setattr(rotation.data_store,'get_labels',lambda *a: labels.copy())
    factors = prices[['stock_id','trading_date']].assign(adj_factor=1.)
    monkeypatch.setattr(rotation,'_load_adj_factor_df',lambda *a:factors.copy())
    monkeypatch.setattr(rotation,'_train_model',lambda *a,**k:SimpleNamespace(predict=lambda x:x[:,0]))
    result = rotation.run_rotation(SimpleNamespace(),session,eval_period_start=days[65],eval_period_end=days[69],
        fixed_train_end=days[60],max_positions=1,top_entry_n=1,slippage_pct=0,transaction_cost_pct=0)
    nav = {r['date']:r['equity'] for r in result['equity_curve']}
    assert nav[str(days[66])] == pytest.approx(2000)
    assert result['summary']['max_drawdown'] == pytest.approx(-.8)
    assert result['equity_curve'][-1]['equity'] == pytest.approx(10000*(1+result['summary']['total_return']))
    assert result['summary']['total_trades'] == 1
    assert result['summary']['benchmark_total_return'] != 0


@pytest.mark.parametrize('module_name', ['scripts.backtest_rotation','scripts.run_backtest'])
def test_cli_help_formats_without_data_access(monkeypatch, module_name):
    import importlib
    import sys
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys,'argv',[module_name,'--help'])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0


@pytest.mark.parametrize('module_name', ['scripts.backtest_rotation','scripts.run_backtest'])
def test_cli_refuses_to_overwrite_existing_evidence_before_loading_data(monkeypatch, tmp_path, module_name):
    import importlib
    import sys
    module = importlib.import_module(module_name)
    output = tmp_path/'sealed.json'; output.write_text('original')
    monkeypatch.setattr(sys,'argv',[module_name,'--output',str(output)])
    monkeypatch.setattr(module,'load_config',lambda:(_ for _ in ()).throw(AssertionError('must not access data')))
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert output.read_text() == 'original'
