"""Shared fixtures for backtest_engine tests."""

import numpy as np
import pytest


@pytest.fixture
def simple_uptrend():
    """100-bar uptrend: price rises from 100 to 199."""
    n = 100
    close = np.arange(100.0, 100.0 + n, dtype=np.float64)
    high = close + 0.5
    low = close - 0.5
    return high, low, close


@pytest.fixture
def simple_downtrend():
    """100-bar downtrend: price falls from 200 to 101."""
    n = 100
    close = np.arange(200.0, 200.0 - n, -1.0, dtype=np.float64)
    high = close + 0.5
    low = close - 0.5
    return high, low, close


@pytest.fixture
def flat_market():
    """100-bar flat market at 100.0."""
    n = 100
    close = np.full(n, 100.0, dtype=np.float64)
    high = np.full(n, 100.5, dtype=np.float64)
    low = np.full(n, 99.5, dtype=np.float64)
    return high, low, close
