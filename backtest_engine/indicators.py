"""Numba-accelerated technical indicators."""

from __future__ import annotations

import numpy as np
import numba


@numba.njit(cache=True)
def sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average. First period-1 values are NaN."""
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[:period - 1] = np.nan

    # Initial window sum
    window_sum = 0.0
    for i in range(period):
        window_sum += data[i]
    out[period - 1] = window_sum / period

    # Sliding window
    for i in range(period, n):
        window_sum += data[i] - data[i - period]
        out[i] = window_sum / period

    return out


@numba.njit(cache=True)
def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range. First value is high[0] - low[0]."""
    n = len(high)
    out = np.empty(n, dtype=np.float64)
    out[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        out[i] = max(hl, max(hc, lc))

    return out


@numba.njit(cache=True)
def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range (SMA method, not Wilder)."""
    tr = true_range(high, low, close)
    return sma(tr, period)


@numba.njit(cache=True)
def bollinger_bands(
    close: np.ndarray, period: int, num_std: float = 2.0,
) -> tuple:
    """Bollinger Bands. Returns (upper, mid, lower)."""
    n = len(close)
    upper = np.empty(n, dtype=np.float64)
    mid = np.empty(n, dtype=np.float64)
    lower = np.empty(n, dtype=np.float64)

    upper[:period - 1] = np.nan
    mid[:period - 1] = np.nan
    lower[:period - 1] = np.nan

    for i in range(period - 1, n):
        start = i - period + 1
        mean = 0.0
        for j in range(start, i + 1):
            mean += close[j]
        mean /= period

        var = 0.0
        for j in range(start, i + 1):
            diff = close[j] - mean
            var += diff * diff
        var /= period  # population std (same as pandas default ddof=0 for rolling)
        std = var ** 0.5

        mid[i] = mean
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std

    return upper, mid, lower


@numba.njit(cache=True)
def rci(close: np.ndarray, period: int) -> np.ndarray:
    """RCI (Rank Correlation Index) = Spearman correlation × 100.

    Uses insertion sort for ranking within each window.
    """
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    out[:period - 1] = np.nan

    # Pre-allocate work arrays
    values = np.empty(period, dtype=np.float64)
    indices = np.empty(period, dtype=np.int64)

    for i in range(period - 1, n):
        start = i - period + 1

        # Copy window values
        for j in range(period):
            values[j] = close[start + j]
            indices[j] = j

        # Insertion sort by value (ascending)
        for j in range(1, period):
            key_val = values[j]
            key_idx = indices[j]
            k = j - 1
            while k >= 0 and values[k] > key_val:
                values[k + 1] = values[k]
                indices[k + 1] = indices[k]
                k -= 1
            values[k + 1] = key_val
            indices[k + 1] = key_idx

        # Compute sum of squared rank differences
        d2_sum = 0.0
        for j in range(period):
            # Price rank (1-based, ascending) = j + 1
            price_rank = j + 1
            # Time rank (1-based): original position + 1
            time_rank = indices[j] + 1
            d = price_rank - time_rank
            d2_sum += d * d

        # Spearman: 1 - 6*sum(d^2) / (n*(n^2-1))
        rci_val = (1.0 - 6.0 * d2_sum / (period * (period * period - 1))) * 100.0
        out[i] = rci_val

    return out


@numba.njit(cache=True)
def parabolic_sar(
    high: np.ndarray,
    low: np.ndarray,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> tuple:
    """Parabolic SAR (Wilder).

    Returns (sar, trend, sar_stop) where:
      sar      : float64 array of SAR values (post-reversal, for charting)
      trend    : int8 array (1=uptrend, -1=downtrend)
      sar_stop : float64 array of pre-reversal SAR (for trailing stop use)
                 On reversal bars, sar_stop has the computed SAR level BEFORE
                 it jumps to the extreme point. Use this for exit price.
    """
    n = len(high)
    sar = np.empty(n, dtype=np.float64)
    trend = np.empty(n, dtype=np.int8)
    sar_stop = np.empty(n, dtype=np.float64)

    # Initialise: assume uptrend starting at first bar
    sar[0] = low[0]
    sar_stop[0] = low[0]
    trend[0] = 1
    ep = high[0]   # extreme point
    af = af_start

    for i in range(1, n):
        prev_sar = sar[i - 1]

        if trend[i - 1] == 1:
            # --- Uptrend ---
            new_sar = prev_sar + af * (ep - prev_sar)
            # Clamp: SAR must not be above prior two lows
            new_sar = min(new_sar, low[i - 1])
            if i >= 2:
                new_sar = min(new_sar, low[i - 2])

            # Store pre-reversal SAR for trailing stop
            sar_stop[i] = new_sar

            if low[i] < new_sar:
                # Reversal to downtrend
                trend[i] = -1
                sar[i] = ep          # SAR flips to extreme point
                ep = low[i]
                af = af_start
            else:
                trend[i] = 1
                sar[i] = new_sar
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            # --- Downtrend ---
            new_sar = prev_sar + af * (ep - prev_sar)
            # Clamp: SAR must not be below prior two highs
            new_sar = max(new_sar, high[i - 1])
            if i >= 2:
                new_sar = max(new_sar, high[i - 2])

            # Store pre-reversal SAR for trailing stop
            sar_stop[i] = new_sar

            if high[i] > new_sar:
                # Reversal to uptrend
                trend[i] = 1
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                trend[i] = -1
                sar[i] = new_sar
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    return sar, trend, sar_stop


@numba.njit(cache=True)
def expanding_quantile(data: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Expanding quantile: percentile rank of value[i] within data[0:i+1].

    Returns values in [0, 1].
    """
    n = len(data)
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        count_below = 0
        for j in range(i + 1):
            if data[j] <= value[i]:
                count_below += 1
        out[i] = count_below / (i + 1)

    return out


@numba.njit(cache=True)
def map_higher_tf(
    lower_ts: np.ndarray,
    higher_ts: np.ndarray,
    higher_vals: np.ndarray,
) -> np.ndarray:
    """Map higher-timeframe values to lower-timeframe bars.

    Uses the most recent *completed* higher-TF bar (side='right' - 2).
    First bars before any higher-TF bar completes are NaN.

    Parameters
    ----------
    lower_ts : int64 array of lower-timeframe timestamps.
    higher_ts : int64 array of higher-timeframe bar *open* timestamps (sorted).
    higher_vals : float64 array of higher-timeframe values.
    """
    n_lower = len(lower_ts)
    n_higher = len(higher_ts)
    out = np.empty(n_lower, dtype=np.float64)

    h_idx = 0
    for i in range(n_lower):
        # Advance h_idx to the last higher_ts <= lower_ts[i]
        while h_idx < n_higher and higher_ts[h_idx] <= lower_ts[i]:
            h_idx += 1
        # h_idx points to first higher_ts > lower_ts[i]
        # The completed bar is h_idx - 2 (current open bar is h_idx - 1)
        completed = h_idx - 2
        if completed >= 0:
            out[i] = higher_vals[completed]
        else:
            out[i] = np.nan

    return out
