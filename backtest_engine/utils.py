"""Utility functions: CSV loading, timestamp operations, resampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _to_epoch_seconds(values) -> np.ndarray:
    """Convert datetimes (strings, datetime64 of any resolution, tz-aware or
    naive=UTC) or numeric epochs to int64 Unix seconds.

    ``astype(np.int64) // 10**9`` is wrong on pandas >= 3, where datetimes are
    no longer always nanoseconds (GitHub issue #1).
    """
    if isinstance(values, pd.DatetimeIndex):
        idx = values
    else:
        ser = pd.Series(values)
        if pd.api.types.is_numeric_dtype(ser) and not pd.api.types.is_bool_dtype(ser):
            v = ser.to_numpy(dtype=np.int64)
            # Infer the unit from the magnitude (epoch seconds stay < 1e11
            # until the year 5138): s < 1e11 <= ms < 1e14 <= us < 1e17 <= ns.
            mag = int(np.max(np.abs(v))) if len(v) else 0
            for limit, div in ((10**11, 1), (10**14, 10**3), (10**17, 10**6)):
                if mag < limit:
                    return v // div
            return v // 10**9
        idx = pd.DatetimeIndex(pd.to_datetime(ser, utc=True))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return np.asarray((idx - _EPOCH) // pd.Timedelta(seconds=1), dtype=np.int64)


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

    The time column may hold date strings (naive strings are read as UTC;
    strings with an offset are converted to UTC) or numeric Unix epochs in
    seconds or milliseconds.
    """
    df = pd.read_csv(filepath)
    timestamps = _to_epoch_seconds(df[time_col])

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

    ts_out = _to_epoch_seconds(resampled.index)
    return (
        ts_out,
        resampled["open"].values.astype(np.float64),
        resampled["high"].values.astype(np.float64),
        resampled["low"].values.astype(np.float64),
        resampled["close"].values.astype(np.float64),
        resampled["volume"].values.astype(np.float64),
    )
