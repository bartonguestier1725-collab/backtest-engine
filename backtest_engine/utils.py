"""Utility functions: CSV loading, timestamp operations, resampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_ohlcv(
    filepath: str | Path,
    time_col: str = "time",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load OHLCV data from CSV into numpy arrays.

    Returns
    -------
    (timestamps, open, high, low, close, volume) — all numpy float64/int64 arrays.
    timestamps are Unix epoch in seconds (int64).
    """
    df = pd.read_csv(filepath)
    ts = pd.to_datetime(df[time_col])
    timestamps = (ts.astype(np.int64) // 10**9).values  # ns → seconds

    return (
        timestamps,
        df[open_col].values.astype(np.float64),
        df[high_col].values.astype(np.float64),
        df[low_col].values.astype(np.float64),
        df[close_col].values.astype(np.float64),
        df[volume_col].values.astype(np.float64) if volume_col in df.columns else np.zeros(len(df), dtype=np.float64),
    )


def find_signal_bar(timestamps: np.ndarray, signal_time: int) -> int:
    """Find the bar index for a given signal timestamp using binary search.

    Returns the index of the last bar with timestamp <= signal_time.
    """
    idx = int(np.searchsorted(timestamps, signal_time, side="right")) - 1
    return max(0, idx)


def resample_ohlcv(
    timestamps: np.ndarray,
    open_arr: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    rule: str = "1h",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample OHLCV data to a higher timeframe.

    Parameters
    ----------
    timestamps : int64 array of Unix timestamps (seconds).
    open_arr, high, low, close, volume : float64 arrays.
    rule : pandas resample rule (e.g. '1h', '4h', '1D').

    Returns
    -------
    (timestamps, open, high, low, close, volume) for the higher timeframe.
    """
    index = pd.to_datetime(timestamps, unit="s")
    df = pd.DataFrame({
        "open": open_arr,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=index)

    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    ts_out = (resampled.index.astype(np.int64) // 10**9).values
    return (
        ts_out,
        resampled["open"].values.astype(np.float64),
        resampled["high"].values.astype(np.float64),
        resampled["low"].values.astype(np.float64),
        resampled["close"].values.astype(np.float64),
        resampled["volume"].values.astype(np.float64),
    )
