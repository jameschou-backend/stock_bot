from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import Strategy
from . import rules_entry as re
from . import rules_filter as rf
from . import rules_exit as rx


@dataclass
class MomentumTrend(Strategy):
    name: str = "MomentumTrend"
    entry_rules: List = None
    filter_rules: List = None
    exit_rules: List = None

    def __post_init__(self):
        self.entry_rules = [
            re.entry_close_above_ma("ma_60"),
            re.entry_ret_above("ret_20", 0.10),
            re.entry_volume_above("volume_20", "volume_60_mean"),
        ]
        self.filter_rules = [
            rf.filter_min_turnover("avg_turnover_20", 0.5),
        ]
        self.exit_rules = [
            rx.exit_stop_loss_fixed(-0.07),
            rx.exit_trailing_stop(0.10),
            rx.exit_below_ma("ma_20"),
        ]


@dataclass
class MeanReversion(Strategy):
    name: str = "MeanReversion"
    entry_rules: List = None
    filter_rules: List = None
    exit_rules: List = None

    def __post_init__(self):
        self.entry_rules = [
            re.entry_rsi_below("rsi_14", 30),
            re.entry_below_bollinger("bb_lower"),
        ]
        self.filter_rules = [
            rf.filter_min_turnover("avg_turnover_20", 0.5),
        ]
        self.exit_rules = [
            rx.exit_take_profit_fixed(0.08),
            rx.exit_time_stop(10),
            rx.exit_stop_loss_fixed(-0.05),
        ]


@dataclass
class DefensiveLowVol(Strategy):
    name: str = "DefensiveLowVol"
    entry_rules: List = None
    filter_rules: List = None
    exit_rules: List = None

    def __post_init__(self):
        self.entry_rules = [
            rf.filter_low_vol_percentile("vol_pct_60", 0.3),
            re.entry_close_above_ma("ma_20"),
        ]
        self.filter_rules = [
            rf.filter_min_turnover("avg_turnover_20", 0.5),
        ]
        self.exit_rules = [
            rx.exit_below_ma("ma_20"),
            rx.exit_stop_loss_fixed(-0.06),
        ]
