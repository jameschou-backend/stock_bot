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
    rank_col: str = "ret_20"
    rank_ascending: bool = False
    stoploss_fixed_pct: float = 0.07
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
    rank_col: str = "rsi_14"
    rank_ascending: bool = True
    stoploss_fixed_pct: float = 0.05
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
    rank_col: str = "vol_pct_60"
    rank_ascending: bool = True
    stoploss_fixed_pct: float = 0.06
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


@dataclass
class CourseVolumeMomentum(Strategy):
    """課程版：量大動能（多方）"""

    name: str = "CourseVolumeMomentum"
    rank_col: str = "ret_20"
    rank_ascending: bool = False
    stoploss_fixed_pct: float = 0.07
    entry_rules: List = None
    filter_rules: List = None
    exit_rules: List = None

    def __post_init__(self):
        self.entry_rules = [
            re.entry_close_above_ma("ma_20"),
            re.entry_ret_above("ret_20", 0.10),
            re.entry_volume_above("volume_20", "volume_60_mean"),
            re.entry_col_gte("foreign_net_3", 1000),
        ]
        self.filter_rules = [
            rf.filter_min_turnover("avg_turnover_20", 5.0),
        ]
        self.exit_rules = [
            rx.exit_stop_loss_fixed(-0.07),
            rx.exit_trailing_stop(0.12),
            rx.exit_below_ma("ma_5"),
            rx.exit_below_ma("ma_10"),
            rx.exit_below_ma("ma_20"),
            rx.exit_time_stop(20),
        ]


@dataclass
class CourseBreakout(Strategy):
    """課程版：價量創新高 / 突破壓力"""

    name: str = "CourseBreakout"
    rank_col: str = "ret_5"
    rank_ascending: bool = False
    stoploss_fixed_pct: float = 0.06
    entry_rules: List = None
    filter_rules: List = None
    exit_rules: List = None

    def __post_init__(self):
        self.entry_rules = [
            re.entry_close_at_high("high_400", tolerance=0.002),
            re.entry_volume_new_high("volume", "volume_max_10"),
        ]
        self.filter_rules = [
            rf.filter_min_turnover("avg_turnover_20", 5.0),
        ]
        self.exit_rules = [
            rx.exit_stop_loss_fixed(-0.06),
            rx.exit_trailing_stop(0.10),
            rx.exit_below_ma("ma_5"),
            rx.exit_below_ma("ma_10"),
            rx.exit_below_ma("ma_20"),
            rx.exit_time_stop(12),
        ]


@dataclass
class CoursePullback(Strategy):
    """課程版：回檔低成本進場（多方）"""

    name: str = "CoursePullback"
    rank_col: str = "rsi_14"
    rank_ascending: bool = True
    stoploss_fixed_pct: float = 0.05
    entry_rules: List = None
    filter_rules: List = None
    exit_rules: List = None

    def __post_init__(self):
        self.entry_rules = [
            re.entry_close_above_ma("ma_60"),
            re.entry_close_near_ma("ma_10", tolerance=0.03),
            re.entry_rsi_below("rsi_14", 45),
            re.entry_col_gte("foreign_trust_same_side", 1),
        ]
        self.filter_rules = [
            rf.filter_min_turnover("avg_turnover_20", 0.8),
        ]
        self.exit_rules = [
            rx.exit_take_profit_fixed(0.12),
            rx.exit_stop_loss_fixed(-0.05),
            rx.exit_below_ma("ma_10"),
            rx.exit_below_ma("ma_20"),
            rx.exit_time_stop(20),
        ]
