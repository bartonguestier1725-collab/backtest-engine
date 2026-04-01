"""Numba-accelerated backtesting engine for trading strategies."""

__version__ = "0.5.1"

from backtest_engine._types import (
    LONG, SHORT,
    EXIT_SL, EXIT_TP, EXIT_TIME, EXIT_BE, EXIT_CUSTOM, EXIT_TRAIL, EXIT_NO_FILL,
    TRADE_RESULT_DTYPE,
)
from backtest_engine.core import simulate_trades
from backtest_engine.preflight import BacktestQualityWarning, PreflightReport
from backtest_engine._results import TradeResults
from backtest_engine.indicators import (
    sma, true_range, atr, rsi, bollinger_bands, rci, expanding_quantile, map_higher_tf,
    parabolic_sar,
)
from backtest_engine.costs import BrokerCost
from backtest_engine.montecarlo import MonteCarloDD, StressTest
from backtest_engine.validation import WalkForward, CSCV
from backtest_engine.utils import load_ohlcv, find_signal_bar, resample_ohlcv
from backtest_engine.data import fetch_aggvault
from backtest_engine.bug_guard import (
    run_all_checks as bug_guard,
    check_look_ahead, check_cost_registry, check_bfill_in_source,
    check_resolution, check_data_period, check_same_bar_reentry,
    check_entry_price_type, check_min_trades,
    check_open_prices_provided, check_fixed_cost_usage, check_spread_filter,
    check_effective_no_sl,
)
from backtest_engine.gatekeeper import GateKeeper

__all__ = [
    "simulate_trades",
    "LONG", "SHORT",
    "EXIT_SL", "EXIT_TP", "EXIT_TIME", "EXIT_BE", "EXIT_CUSTOM", "EXIT_TRAIL", "EXIT_NO_FILL",
    "TRADE_RESULT_DTYPE",
    "BacktestQualityWarning", "PreflightReport", "TradeResults",
    "sma", "true_range", "atr", "rsi", "bollinger_bands", "rci", "expanding_quantile", "map_higher_tf",
    "parabolic_sar",
    "BrokerCost",
    "MonteCarloDD", "StressTest",
    "WalkForward", "CSCV",
    "load_ohlcv", "find_signal_bar", "resample_ohlcv",
    "fetch_aggvault",
    "bug_guard", "GateKeeper",
    "check_look_ahead", "check_cost_registry", "check_bfill_in_source",
    "check_resolution", "check_data_period", "check_same_bar_reentry",
    "check_entry_price_type", "check_min_trades",
    "check_open_prices_provided", "check_fixed_cost_usage", "check_spread_filter",
    "check_effective_no_sl",
]
