"""Numba-accelerated backtesting engine for trading strategies."""

__version__ = "0.1.0"

from backtest_engine._types import (
    LONG, SHORT,
    EXIT_SL, EXIT_TP, EXIT_TIME, EXIT_BE, EXIT_CUSTOM, EXIT_TRAIL, EXIT_NO_FILL,
    TRADE_RESULT_DTYPE,
)
from backtest_engine.core import simulate_trades
from backtest_engine.indicators import (
    sma, true_range, atr, bollinger_bands, rci, expanding_quantile, map_higher_tf,
    parabolic_sar,
)
from backtest_engine.costs import BrokerCost
from backtest_engine.montecarlo import MonteCarloDD
from backtest_engine.validation import WalkForward, CSCV
from backtest_engine.utils import load_ohlcv, find_signal_bar, resample_ohlcv
from backtest_engine.bug_guard import (
    run_all_checks as bug_guard,
    check_look_ahead, check_cost_registry, check_bfill_in_source,
    check_resolution, check_data_period, check_same_bar_reentry,
    check_entry_price_type, check_min_trades,
    FUNDORA_COST_PIPS, cost_pips_to_price,
)
from backtest_engine.gatekeeper import GateKeeper

__all__ = [
    "simulate_trades",
    "LONG", "SHORT",
    "EXIT_SL", "EXIT_TP", "EXIT_TIME", "EXIT_BE", "EXIT_CUSTOM", "EXIT_TRAIL", "EXIT_NO_FILL",
    "TRADE_RESULT_DTYPE",
    "sma", "true_range", "atr", "bollinger_bands", "rci", "expanding_quantile", "map_higher_tf",
    "parabolic_sar",
    "BrokerCost",
    "MonteCarloDD",
    "WalkForward", "CSCV",
    "load_ohlcv", "find_signal_bar", "resample_ohlcv",
    "bug_guard", "GateKeeper",
    "check_look_ahead", "check_cost_registry", "check_bfill_in_source",
    "check_resolution", "check_data_period", "check_same_bar_reentry",
    "check_entry_price_type", "check_min_trades",
    "FUNDORA_COST_PIPS", "cost_pips_to_price",
]
