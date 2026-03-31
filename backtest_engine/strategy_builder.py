"""Strategy Builder — Convert time-based strategy specs to bar indices.

Bridges the gap between human-readable strategy definitions (timezone,
clock times, day-of-week) and backtest_engine's bar-index arrays.

Solves:
  - Timezone conversion (user thinks in JST, data is UTC)
  - Market hours awareness (no Sunday 17:00 UTC bars)
  - Entry/exit bar matching with full accountability
  - Silent trade loss prevention (reports every unmatched entry)

Incident: 2026-04-01 Gold Rollover — 25% of entries silently dropped
with no warning because exit bars didn't exist (market closed on
weekends, Sunday market open gap). Users would have trusted the
incomplete results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MatchReport:
    """Report of entry/exit matching results.

    Every entry candidate is accounted for — nothing is silently dropped.
    """

    matched: int = 0
    unmatched_no_exit_bar: int = 0
    unmatched_market_closed: int = 0
    unmatched_other: int = 0
    details: list[str] = field(default_factory=list)

    @property
    def total_candidates(self) -> int:
        return self.matched + self.unmatched_no_exit_bar + self.unmatched_market_closed + self.unmatched_other

    @property
    def drop_rate(self) -> float:
        if self.total_candidates == 0:
            return 0.0
        return 1.0 - self.matched / self.total_candidates

    def summary(self) -> str:
        lines = [
            f"Entry/Exit Match Report",
            f"  Candidates:  {self.total_candidates}",
            f"  Matched:     {self.matched}",
        ]
        if self.unmatched_no_exit_bar > 0:
            lines.append(f"  No exit bar: {self.unmatched_no_exit_bar} (market closed at exit time)")
        if self.unmatched_market_closed > 0:
            lines.append(f"  Market closed at entry: {self.unmatched_market_closed}")
        if self.unmatched_other > 0:
            lines.append(f"  Other:       {self.unmatched_other}")
        if self.total_candidates > 0:
            lines.append(f"  Drop rate:   {self.drop_rate:.1%}")
        if self.drop_rate > 0.1:
            lines.append(
                f"  ⚠ WARNING: {self.drop_rate:.0%} of entries dropped. "
                f"Check market hours and exit timing."
            )
        return "\n".join(lines)

    def print_report(self):
        print(self.summary())


def _to_utc_hour(hour: int, tz_offset_hours: int) -> int:
    """Convert a local hour to UTC hour.

    Parameters
    ----------
    hour : Local hour (0-23).
    tz_offset_hours : Timezone offset from UTC (e.g. 9 for JST).

    Returns
    -------
    UTC hour (0-23) and day offset (-1, 0, or +1).
    """
    utc_hour = hour - tz_offset_hours
    day_offset = 0
    if utc_hour < 0:
        utc_hour += 24
        day_offset = -1
    elif utc_hour >= 24:
        utc_hour -= 24
        day_offset = 1
    return utc_hour, day_offset


def build_time_based_signals(
    timestamps: np.ndarray,
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    entry_hour_local: int,
    exit_hour_local: int,
    tz_offset_hours: int = 9,
    direction: int = 1,
    skip_weekdays_local: Optional[list[int]] = None,
    entry_price_mode: str = "open",
) -> tuple[pd.DataFrame, MatchReport]:
    """Build entry/exit pairs for a time-based strategy.

    Parameters
    ----------
    timestamps : Unix epoch timestamps (seconds) for each bar.
    open_prices : Open prices for each bar.
    close_prices : Close prices for each bar.
    entry_hour_local : Entry hour in local timezone (0-23).
    exit_hour_local : Exit hour in local timezone (0-23).
    tz_offset_hours : Timezone offset from UTC. Default 9 (JST).
    direction : 1 for LONG, -1 for SHORT.
    skip_weekdays_local : List of weekday numbers (0=Mon) to skip in local
                          timezone. E.g. [3] to skip Thursday in JST.
    entry_price_mode : "open" for bar open price, "close" for bar close price.

    Returns
    -------
    (trades_df, match_report) where trades_df has columns:
        entry_bar_idx, exit_bar_idx, entry_price, exit_price,
        entry_time, exit_time, direction
    """
    if skip_weekdays_local is None:
        skip_weekdays_local = []

    # Convert to pandas DatetimeIndex (UTC)
    dt_index = pd.to_datetime(timestamps, unit="s", utc=True)

    # Convert local entry/exit hours to UTC
    entry_hour_utc, entry_day_offset = _to_utc_hour(entry_hour_local, tz_offset_hours)
    exit_hour_utc, exit_day_offset = _to_utc_hour(exit_hour_local, tz_offset_hours)

    # Convert local skip weekdays to UTC weekdays
    # When entry is at 02:00 JST (= 17:00 UTC prev day), the UTC weekday
    # is one day earlier than the local weekday
    skip_weekdays_utc = set()
    for local_dow in skip_weekdays_local:
        utc_dow = (local_dow + entry_day_offset) % 7
        skip_weekdays_utc.add(utc_dow)

    # Find entry bar candidates
    entry_mask = (dt_index.hour == entry_hour_utc) & (~dt_index.dayofweek.isin(skip_weekdays_utc))
    entry_indices = np.where(entry_mask)[0]

    # Build exit lookup: hour == exit_hour_utc
    exit_mask = dt_index.hour == exit_hour_utc
    exit_times = dt_index[exit_mask]
    exit_indices = np.where(exit_mask)[0]

    # Calculate hold duration in hours
    if exit_hour_local > entry_hour_local:
        hold_hours = exit_hour_local - entry_hour_local
    else:
        hold_hours = 24 - entry_hour_local + exit_hour_local

    report = MatchReport()
    trades = []

    for entry_idx in entry_indices:
        entry_time = dt_index[entry_idx]

        # Expected exit time: entry date + hold duration
        expected_exit = entry_time + pd.Timedelta(hours=hold_hours)

        # Find exit bar within ±30 min of expected
        exit_window = exit_times[
            (exit_times >= expected_exit - pd.Timedelta(minutes=30))
            & (exit_times <= expected_exit + pd.Timedelta(minutes=30))
        ]

        if len(exit_window) == 0:
            # Determine reason
            local_exit_time = expected_exit + pd.Timedelta(hours=tz_offset_hours)
            if local_exit_time.dayofweek >= 5:  # Saturday or Sunday
                report.unmatched_no_exit_bar += 1
                report.details.append(
                    f"  {entry_time} → exit would be {expected_exit} (weekend)"
                )
            else:
                report.unmatched_other += 1
                report.details.append(
                    f"  {entry_time} → exit bar not found near {expected_exit}"
                )
            continue

        exit_time = exit_window[0]
        exit_idx = exit_indices[exit_times == exit_time][0]

        if entry_price_mode == "open":
            entry_price = float(open_prices[entry_idx])
        else:
            entry_price = float(close_prices[entry_idx])

        exit_price = float(open_prices[exit_idx])

        trades.append({
            "entry_bar_idx": int(entry_idx),
            "exit_bar_idx": int(exit_idx),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": direction,
        })
        report.matched += 1

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_bar_idx", "exit_bar_idx", "entry_price", "exit_price",
                 "entry_time", "exit_time", "direction"]
    )

    return trades_df, report
