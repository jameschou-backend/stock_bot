import numpy as np
import pytest
import lightgbm as lgb
from skills.training_cache import fit_cached, training_key


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('BACKTEST_TRAIN_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('BACKTEST_TRAIN_CACHE', 'on')


def estimator(n=10):
    return lgb.LGBMRegressor(n_estimators=n, n_jobs=1, verbosity=-1, random_state=42)


def test_reuse_has_exact_predictions_and_no_refit(monkeypatch):
    rng = np.random.default_rng(42)
    X, y = rng.normal(size=(150, 5)), rng.normal(size=150)
    cold = fit_cached(estimator(), X, y)
    assert not cold.stock_bot_cache_hit_
    monkeypatch.setattr(lgb.LGBMRegressor, 'fit', lambda *a, **k: pytest.fail('cache hit fitted again'))
    hot = fit_cached(estimator(), X.copy(), y.copy())
    assert hot.stock_bot_cache_hit_
    np.testing.assert_array_equal(cold.predict(X), hot.predict(X))


def test_content_order_params_weights_groups_invalidate():
    X = np.arange(100, dtype=float).reshape(20, 5)
    y = np.arange(20, dtype=float)
    base = training_key(estimator(), X, y)
    changed = X.copy()
    changed[0, 0] += .01
    variants = [
        training_key(estimator(), changed, y),
        training_key(estimator(), X[::-1], y[::-1]),
        training_key(estimator(11), X, y),
        training_key(estimator(), X, y + .1),
        training_key(estimator(), X, y, np.ones(20)),
        training_key(estimator(), X, y, group=np.array([10, 10])),
    ]
    assert base not in variants
    assert len(set(variants)) == len(variants)


def test_corrupt_cache_fails_explicitly(tmp_path):
    X, y = np.ones((20, 2)), np.ones(20)
    key = training_key(estimator(), X, y)
    (tmp_path / f'{key}.joblib').write_bytes(b'corrupt')
    with pytest.raises(Exception):
        fit_cached(estimator(), X, y)
