"""Data fetching utilities for backtest_engine.

Provides helpers to fetch OHLC data from supported sources.
Uses only stdlib (urllib) — no extra dependencies required.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional

import numpy as np


__all__ = ["fetch_aggvault"]

_USER_AGENT = "backtest-engine/0.5.1"


def _iso_to_epoch(date_str: str, label: str) -> int:
    """Convert ISO date string (YYYY-MM-DD) to Unix epoch seconds."""
    try:
        return int(
            datetime.strptime(date_str, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    except ValueError:
        raise ValueError(
            f"Invalid {label} date: {date_str!r}. Expected YYYY-MM-DD format."
        )


def fetch_aggvault(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    api_key: Optional[str] = None,
    base_url: str = "https://tick.hugen.tokyo",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fetch OHLC data from AggVault API.

    Returns the same 6-tuple format as ``load_ohlcv`` for easy swapping
    between CSV and API data sources::

        timestamps, opens, highs, lows, closes, volume = fetch_aggvault(
            "EURUSD", "1h", "2021-04-01", "2026-03-31",
        )

    Parameters
    ----------
    symbol : str
        Currency pair (e.g. ``"EURUSD"``, ``"XAUUSD"``).
        Case-insensitive, slashes optional (``"EUR/USD"`` also works).
    timeframe : str
        Bar size. One of ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``.
    start : str
        Start date inclusive, ISO format ``"YYYY-MM-DD"``.
    end : str
        End date inclusive, ISO format ``"YYYY-MM-DD"``.
    api_key : str, optional
        AggVault API key (``tk_...``).
        Falls back to ``AGGVAULT_KEY`` environment variable if not provided.
    base_url : str
        API base URL. Defaults to ``https://tick.hugen.tokyo``.

    Returns
    -------
    (timestamps, opens, highs, lows, closes, volume)
        Same format as :func:`load_ohlcv`.
        ``timestamps`` is int64 (Unix epoch seconds), others are float64.
        ``volume`` is always zeros (AggVault does not provide volume).

    Raises
    ------
    ValueError
        Missing API key, invalid date format, or invalid parameters.
    RuntimeError
        API returned an error (401/403/404/429/5xx).
    """
    key = api_key or os.environ.get("AGGVAULT_KEY", "")
    if not key:
        raise ValueError(
            "AggVault API key required.\n"
            "  Option 1: pass api_key='tk_...'\n"
            "  Option 2: export AGGVAULT_KEY=tk_your_key"
        )

    valid_tf = {"1m", "5m", "15m", "1h"}
    if timeframe not in valid_tf:
        raise ValueError(
            f"Invalid timeframe: {timeframe!r}. Must be one of {sorted(valid_tf)}."
        )

    from_epoch = _iso_to_epoch(start, "start")
    to_epoch = _iso_to_epoch(end, "end")

    if from_epoch >= to_epoch:
        raise ValueError(
            f"start ({start}) must be before end ({end})."
        )

    url = (
        f"{base_url}/api/v1/historical/{symbol}"
        f"?tf={timeframe}&from={from_epoch}&to={to_epoch}"
    )

    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}",
        "User-Agent": _USER_AGENT,
    })

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:300]
        except Exception:
            pass
        messages = {
            401: "Invalid API key. Check your AGGVAULT_KEY.",
            403: "Access denied. Historical data requires a 'pro' or 'enterprise' plan.",
            404: f"No data found for {symbol}/{timeframe}.",
            429: "Rate limited (max 10 requests/minute). Wait and retry.",
        }
        msg = messages.get(e.code, f"API error {e.code}")
        if body:
            msg += f" Server response: {body}"
        raise RuntimeError(msg) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach AggVault API at {base_url}: {e.reason}"
        ) from e

    data = json.loads(raw)

    if not data:
        raise RuntimeError(
            f"No bars returned for {symbol}/{timeframe} ({start} → {end}). "
            f"Check symbol name and date range."
        )

    times = np.array([bar["time"] for bar in data], dtype=np.int64)
    opens = np.array([bar["open"] for bar in data], dtype=np.float64)
    highs = np.array([bar["high"] for bar in data], dtype=np.float64)
    lows = np.array([bar["low"] for bar in data], dtype=np.float64)
    closes = np.array([bar["close"] for bar in data], dtype=np.float64)
    volume = np.zeros(len(data), dtype=np.float64)

    return times, opens, highs, lows, closes, volume
