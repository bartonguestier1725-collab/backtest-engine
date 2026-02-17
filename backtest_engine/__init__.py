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
)
from backtest_engine.costs import BrokerCost
from backtest_engine.montecarlo import MonteCarloDD
from backtest_engine.validation import WalkForward, CSCV
from backtest_engine.utils import load_ohlcv, find_signal_bar, resample_ohlcv

__all__ = [
    "simulate_trades",
    "LONG", "SHORT",
    "EXIT_SL", "EXIT_TP", "EXIT_TIME", "EXIT_BE", "EXIT_CUSTOM", "EXIT_TRAIL", "EXIT_NO_FILL",
    "TRADE_RESULT_DTYPE",
    "sma", "true_range", "atr", "bollinger_bands", "rci", "expanding_quantile", "map_higher_tf",
    "BrokerCost",
    "MonteCarloDD",
    "WalkForward", "CSCV",
    "load_ohlcv", "find_signal_bar", "resample_ohlcv",
]
