import pytest
from scripts.research_legacy_return_comparison_v4 import stock_only
from skills.legacy_merger_settlement import MergerSettlement


def test_merger_does_not_add_a_stock_identity_contract_to_benchmark():
    class Stock:
        def identity(self,*args):return 'stock'
    class Benchmark:
        def _costs(self,*args):return 'original_cost_function'
    wrapped_stock=type('StockWithMerger',(MergerSettlement,Stock),{})
    wrapped_benchmark=type('BenchmarkWithMerger',(MergerSettlement,Benchmark),{})
    factory=stock_only(lambda:(wrapped_stock,wrapped_benchmark))
    actual_stock,actual_benchmark=factory()
    assert actual_stock is wrapped_stock
    assert actual_benchmark is Benchmark
    assert not hasattr(actual_benchmark,'identity')
    assert actual_benchmark()._costs()=='original_cost_function'
    assert factory()[1] is Benchmark


def test_changed_merger_factory_cannot_silently_strip_another_adapter():
    with pytest.raises(ValueError):stock_only(lambda:(object,object))()
