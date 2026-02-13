from skills.strategy_factory.portfolio import Position, Portfolio, execute_order, risk_position_size, apply_pyramiding


def test_risk_position_size():
    qty = risk_position_size(100000, 0.01, entry_price=100, stop_price=95)
    assert round(qty, 2) == 200.0


def test_execute_order_buy_sell():
    pf = Portfolio(initial_capital=100000, cash=100000)
    execute_order(pf, "2330", "BUY", 10, 100, 1)
    assert pf.cash == 100000 - 1000 - 1
    assert "2330" in pf.positions
    execute_order(pf, "2330", "SELL", 10, 110, 1)
    assert "2330" not in pf.positions
    assert pf.cash > 100000


def test_apply_pyramiding():
    pos = Position(stock_id="2330", qty=10, avg_cost=100, last_price=105)
    add_ratio = apply_pyramiding(pos, last_high=104, current_price=105)
    assert add_ratio == 0.30
