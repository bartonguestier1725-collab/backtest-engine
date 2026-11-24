"""Data fetching utilities for backtest_engine.

Provides helpers to fetch OHLC data from supported sources.
Uses only stdlib (urllib) — no extra dependencies required.
"""

from __future__ import annotations

import json
import os
import time
import warnings
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from backtest_engine.core import DataQualityWarning


__all__ = ["fetch_aggvault"]

_USER_AGENT = "backtest-engine/0.6.1"


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


# Max bars per /api/v1/historical reply (server default LIMIT, oldest first).
_SERVER_LIMIT = 500_000
# Request window per timeframe, sized to stay far below _SERVER_LIMIT
# (1m: 180 days <= 259,200 bars). Coarser timeframes fit in one request.
_CHUNK_DAYS = {"1m": 180, "5m": 900, "15m": 2700}
# Server rate limit is 10 requests/minute.
_MIN_INTERVAL_SEC = 6.5
_RETRY_WAIT_SEC = 61.0
# Inside the requested range, a reply ending this long before its window end
# is treated as cut short (FX weekends + holidays are shorter).
_HOLE_SEC = 5 * 86400
_sleep = time.sleep
# Time of the last request, shared by all calls so a 1h fetch followed by a
# 1m fetch is paced too.
_LAST_REQUEST = [0.0]


def _ymd(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


class _NotFound(Exception):
    pass


def _get_bars(url, key, symbol, timeframe, base_url, max_retries, last_request):
    """GET one window. Retries HTTP 429 after waiting; paces repeated requests."""
    attempt = 0
    while True:
        if last_request[0]:
            wait = _MIN_INTERVAL_SEC - (time.monotonic() - last_request[0])
            if wait > 0:
                _sleep(wait)
        last_request[0] = time.monotonic()
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {key}",
            "User-Agent": _USER_AGENT,
        })
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                try:
                    raw = resp.read().decode("utf-8")
                except UnicodeDecodeError as ue:
                    raise RuntimeError(
                        f"Invalid text encoding in AggVault API response: {ue}"
                    ) from ue
                except (OSError, ConnectionError) as re_:
                    raise RuntimeError(
                        f"Connection lost while reading AggVault API response: {re_}"
                    ) from re_
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < max_retries:
                attempt += 1
                wait = _RETRY_WAIT_SEC
                try:
                    wait = min(max(float(e.headers.get("Retry-After", wait)), 1.0), 120.0)
                except (TypeError, ValueError, AttributeError):
                    pass
                e.close()  # else a ResourceWarning names the unread 429 reply
                _sleep(wait)
                last_request[0] = 0.0  # the wait above already paces the retry
                continue
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
            if e.code == 404:
                raise _NotFound(msg) from e
            raise RuntimeError(msg) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach AggVault API at {base_url}: {e.reason}"
            ) from e

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON response from AggVault API: {e}. "
                f"Response preview: {raw[:200]!r}"
            ) from e
        if not isinstance(data, list):
            raise RuntimeError(
                f"Unexpected AggVault response type: {type(data).__name__}. "
                f"Response preview: {raw[:200]!r}"
            )
        return data


_REQUIRED_KEYS = {"time", "open", "high", "low", "close"}


def _to_arrays(data: list, offset: int = 0) -> tuple[np.ndarray, ...]:
    """Validate one reply and convert it to (time, open, high, low, close)."""
    for idx, bar in enumerate(data):
        if not isinstance(bar, dict) or not _REQUIRED_KEYS.issubset(bar):
            missing = _REQUIRED_KEYS - set(bar) if isinstance(bar, dict) else _REQUIRED_KEYS
            raise RuntimeError(
                f"Invalid bar format at index {offset + idx}. Missing keys: {missing}. "
                f"Bar: {bar!r}"
            )
    try:
        return (
            np.array([bar["time"] for bar in data], dtype=np.int64),
            np.array([bar["open"] for bar in data], dtype=np.float64),
            np.array([bar["high"] for bar in data], dtype=np.float64),
            np.array([bar["low"] for bar in data], dtype=np.float64),
            np.array([bar["close"] for bar in data], dtype=np.float64),
        )
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"Failed to convert bar data to numpy arrays: {e}"
        ) from e


def fetch_aggvault(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    api_key: Optional[str] = None,
    base_url: str = "https://tick.hugen.tokyo",
    max_retries: int = 2,
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
        Bar size. One of ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"4h"``, ``"1d"``.
    start : str
        Start date inclusive, ISO format ``"YYYY-MM-DD"``.
    end : str
        End date inclusive, ISO format ``"YYYY-MM-DD"``.
    api_key : str, optional
        AggVault API key (``tk_...``).
        Falls back to ``AGGVAULT_KEY`` environment variable if not provided.
    base_url : str
        API base URL. Defaults to ``https://tick.hugen.tokyo``.
    max_retries : int
        How many times to wait and retry after HTTP 429 or 503 (~1 minute,
        or the server's Retry-After).

    Notes
    -----
    The server returns at most 500,000 bars per request. Long 1m/5m/15m
    ranges are fetched in several requests (1m: 180-day windows, paced to the
    10 requests/minute limit). 5 years of 1m data takes 2-3 minutes, mostly
    download time (the pacing alone is about a minute).
    A window with no bars inside the range emits a DataQualityWarning.

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

    valid_tf = {"1m", "5m", "15m", "1h", "4h", "1d"}
    if timeframe not in valid_tf:
        raise ValueError(
            f"Invalid timeframe: {timeframe!r}. Must be one of {sorted(valid_tf)}."
        )

    from_epoch = _iso_to_epoch(start, "start")
    to_epoch = _iso_to_epoch(end, "end") + 86400 - 1  # end日の23:59:59まで含む

    if from_epoch > to_epoch:
        raise ValueError(
            f"start ({start}) must be on or before end ({end})."
        )

    import re
    symbol_norm = symbol.upper().replace("/", "").strip()
    if not re.fullmatch(r"[A-Z0-9]+", symbol_norm):
        raise ValueError(
            f"Invalid symbol: {symbol!r}. Expected letters/digits only (e.g. 'EURUSD')."
        )

    # The server returns at most _SERVER_LIMIT bars per request, oldest first,
    # with HTTP 200 and no truncation flag. EURUSD 1m for 5 years is ~1.8M
    # bars, so a single request silently lost everything after ~1.3 years
    # (before v0.6.0). Split the range into windows that stay well under the
    # limit, and keep requesting within a window if a reply is still full.
    chunk_days = _CHUNK_DAYS.get(timeframe)
    if chunk_days is None:
        windows = [(from_epoch, to_epoch)]
    else:
        step = chunk_days * 86400
        windows = [(a, min(a + step - 1, to_epoch))
                   for a in range(from_epoch, to_epoch + 1, step)]

    parts: list[tuple[np.ndarray, ...]] = []
    n_seen = 0
    not_found: Optional[RuntimeError] = None
    empty: list[tuple[int, int]] = []
    for wi, (w_from, w_to) in enumerate(windows):
        interior = wi < len(windows) - 1
        cur = w_from
        got_any = False
        while cur <= w_to:
            # limit is sent explicitly: if the server ever lowers its maximum,
            # the request fails (HTTP 422) instead of being cut silently.
            url = (
                f"{base_url}/api/v1/historical/{symbol_norm}"
                f"?tf={timeframe}&from={cur}&to={w_to}&limit={_SERVER_LIMIT}"
            )
            try:
                chunk = _get_bars(url, key, symbol, timeframe, base_url,
                                  max_retries, _LAST_REQUEST)
            except _NotFound as e:
                if len(windows) == 1:
                    raise RuntimeError(str(e)) from None
                not_found = not_found or RuntimeError(str(e))
                chunk = []
            if not chunk:
                break
            # Convert per window so a long 1m fetch never holds millions
            # of dicts at once.
            parts.append(_to_arrays(chunk, n_seen))
            n_seen += len(chunk)
            got_any = True
            last_t = int(parts[-1][0][-1])
            if last_t + 1 <= cur:
                break
            # Keep going while the reply was full, or (inside the range) while
            # it stopped well before the window end: a lower server limit or a
            # hole. The last window may end early because the data does.
            if len(chunk) < _SERVER_LIMIT and not (interior and last_t < w_to - _HOLE_SEC):
                break
            cur = last_t + 1
        if not got_any:
            empty.append((w_from, w_to))

    if parts and empty:
        warnings.warn(
            f"fetch_aggvault: no {timeframe} bars for {len(empty)} of {len(windows)} "
            f"request windows ({', '.join(f'{_ymd(a)}..{_ymd(b)}' for a, b in empty[:3])}"
            f"{', ...' if len(empty) > 3 else ''}). The returned data has a hole there "
            f"(or starts later than requested).",
            DataQualityWarning,
            stacklevel=2,
        )

    if not parts:
        if not_found is not None:
            raise not_found
        raise RuntimeError(
            f"No bars returned for {symbol}/{timeframe} ({start} → {end}). "
            f"Check symbol name and date range."
        )

    times, opens, highs, lows, closes = (
        np.concatenate([p[i] for p in parts]) for i in range(5))

    # Sort and drop duplicate timestamps (window edges), keeping the first.
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        order = np.argsort(times, kind="stable")
        times, opens, highs, lows, closes = (
            a[order] for a in (times, opens, highs, lows, closes))
        keep = np.r_[True, np.diff(times) > 0]
        times, opens, highs, lows, closes = (
            a[keep] for a in (times, opens, highs, lows, closes))
    volume = np.zeros(len(times), dtype=np.float64)

    return times, opens, highs, lows, closes, volume
