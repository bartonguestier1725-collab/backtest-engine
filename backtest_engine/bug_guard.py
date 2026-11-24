"""BugGuard — Automated backtest bug detection.

Every bug that has EVER occurred in our backtesting is encoded here as an
automated check. This module MUST be used before reporting any backtest result.

Incident log (why each check exists):
  BG-01: Look-ahead bias — signal from bar[N], entry at bar[N] (2026-02-28 v1)
  BG-02: Cost underestimation — JPY commission missing from cost model (2026-02-28)
  BG-03: Commission one-way — $6 RT treated as $3 one-way (2026-02-17)
  BG-04: bfill data leak — future values used to fill NaN (2026-02-28 v2)
  BG-05: 1h OHLC phantom wins — SL/TP hit order unknown in bar (2026-02-28 v5)
  BG-06: Short period overfitting — 48-day "champion" dead on 13.5mo (2026-02-28)
  BG-07: ATR percentile leak — full-period quantile used (2026-02 earlier)
  BG-08: Same-bar exit+reentry — enter at Open after seeing High/Low for exit (Codex)
  BG-09: Entry at Close — should be Open of next bar (2026-02-28 v1 pre-fix)
  BG-10: Incomplete bars — last bar still forming; unsorted/duplicate timestamps
  BG-11: Spread/cost mismatch — published spread used instead of measured (2026-02-28)
  BG-12: Fixed cost with ATR-varying SL distances
  BG-13: Spread filter used live but not in the backtest
  BG-14: Effectively no SL (2026-04-01 Gold Rollover)

Checks that need trade data (BG-01/06b/08/09b/12/14) only run when the trade
arrays are passed. ``GateKeeper.gate0_validate(trades=...)`` does this.
"""

from __future__ import annotations

import inspect
import re
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── Known-good cost registry (measured 2026-02) ────────────────────────────

# Total round-trip cost in PIPS (spread + commission).
# Source: actual trading, not published typical spreads.
_FUNDORA_COST_PIPS = {
    # Measured directly
    "EURUSD": 0.75,   # spread 0.15 + commission 0.60
    "USDJPY": 1.70,   # spread 0.80 + commission 0.90
    "USDCHF": 1.24,   # spread 0.70 + commission 0.54
    "XAUUSD": 46.0,   # spread 40.0 + commission 6.0
    # Estimated: original spread + commission
    # Commission: $6/lot RT → non-JPY = 0.6 pip, JPY = 0.9 pip
    "GBPUSD": 1.40,   # spread ~0.8 + commission 0.60 (conservative)
    "AUDUSD": 1.20,   # spread ~0.6 + commission 0.60
    "NZDUSD": 1.40,   # spread ~0.8 + commission 0.60
    "USDCAD": 1.40,   # spread ~0.8 + commission 0.60
    "EURGBP": 1.40,   # spread ~0.8 + commission 0.60
    "EURJPY": 2.50,   # spread ~1.6 + commission 0.90
    "GBPJPY": 3.10,   # spread ~2.2 + commission 0.90
    "AUDJPY": 2.50,   # spread ~1.6 + commission 0.90
    "EURAUD": 1.80,   # spread ~1.2 + commission 0.60
    "EURNZD": 2.40,   # spread ~1.8 + commission 0.60
    "GBPAUD": 2.20,   # spread ~1.6 + commission 0.60
    "GBPNZD": 2.80,   # spread ~2.2 + commission 0.60
    "AUDNZD": 1.60,   # spread ~1.0 + commission 0.60
    "AUDCAD": 1.60,   # spread ~1.0 + commission 0.60
    "NZDCAD": 1.80,   # spread ~1.2 + commission 0.60
    "EURCAD": 1.60,   # spread ~1.0 + commission 0.60
    "GBPCAD": 1.80,   # spread ~1.2 + commission 0.60
    "CADCHF": 1.60,   # spread ~1.0 + commission 0.60
    "CADJPY": 2.50,   # spread ~1.6 + commission 0.90
    "CHFJPY": 2.50,   # spread ~1.6 + commission 0.90
    "NZDJPY": 2.50,   # spread ~1.6 + commission 0.90
    "GBPCHF": 1.80,   # spread ~1.2 + commission 0.60
    "AUDCHF": 1.60,   # spread ~1.0 + commission 0.60
    "NZDCHF": 1.80,   # spread ~1.2 + commission 0.60
    "EURCHF": 1.40,   # spread ~0.8 + commission 0.60
}

_PIP_SIZES = {}
for _p in _FUNDORA_COST_PIPS:
    if _p == "XAUUSD":
        _PIP_SIZES[_p] = 0.01
    elif _p.endswith("JPY"):
        _PIP_SIZES[_p] = 0.01
    else:
        _PIP_SIZES[_p] = 0.0001


def _cost_pips_to_price(pair: str) -> float:
    """Convert known-good cost from pips to price units."""
    pips = _FUNDORA_COST_PIPS.get(pair)
    pip_size = _PIP_SIZES.get(pair, 0.0001)
    if pips is None:
        return 0.0
    return pips * pip_size


def _fundora_expected_costs() -> dict[str, float]:
    """Build expected_costs dict for Fundora.

    Delegates to ``BrokerCost.fundora().cost_prices()`` so that only one
    cost registry exists.  The legacy ``_FUNDORA_COST_PIPS`` dict is kept
    for reference but is no longer used in validation.
    """
    from backtest_engine.costs import BrokerCost

    return BrokerCost.fundora().cost_prices()


# ── Check results ──────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    """Result of a single bug guard check."""
    check_id: str
    passed: bool
    message: str
    severity: str = "ERROR"  # ERROR (blocks), WARN (flags)


@dataclass
class GuardReport:
    """Full report from all checks."""
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "ERROR")

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "ERROR")

    @property
    def n_warnings(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "WARN")

    def print_report(self):
        print("=" * 70)
        print("BugGuard Report")
        print("=" * 70)
        for r in self.results:
            status = "PASS" if r.passed else f"FAIL({r.severity})"
            print(f"  [{status:>11}] {r.check_id}: {r.message}")
        print("-" * 70)
        if self.passed:
            print(f"  RESULT: ALL CLEAR ({self.n_warnings} warnings)")
        else:
            print(f"  RESULT: BLOCKED — {self.n_errors} errors, {self.n_warnings} warnings")
        print("=" * 70)

    def assert_passed(self):
        """Raise RuntimeError if any ERROR-level check failed."""
        if not self.passed:
            failures = [r for r in self.results if not r.passed and r.severity == "ERROR"]
            msg = "BugGuard BLOCKED execution:\n"
            for f in failures:
                msg += f"  [{f.check_id}] {f.message}\n"
            raise RuntimeError(msg)


# ── Individual checks ──────────────────────────────────────────────────────

def check_look_ahead(signal_bars: np.ndarray, entry_bars: np.ndarray) -> CheckResult:
    """BG-01: Verify signal bar < entry bar (no look-ahead).

    Every trade's signal must come from a bar BEFORE the entry bar.
    signal_bars[i] must be < entry_bars[i] for all i.
    """
    if len(signal_bars) == 0:
        return CheckResult("BG-01", True, "No trades to check")
    violations = np.sum(signal_bars >= entry_bars)
    if violations > 0:
        pct = violations / len(signal_bars) * 100
        return CheckResult(
            "BG-01", False,
            f"LOOK-AHEAD BIAS: {violations}/{len(signal_bars)} trades "
            f"({pct:.1f}%) have signal_bar >= entry_bar",
        )
    return CheckResult("BG-01", True, "No look-ahead bias detected")


def check_cost_registry(
    spreads_used: dict[str, float],
    expected_costs: dict[str, float] | None = None,
    tolerance: float = 0.20,
) -> CheckResult:
    """BG-02/03/11: Verify costs match expected values.

    Catches: JPY commission missing, one-way commission, published spread used.

    Parameters
    ----------
    spreads_used : {pair: cost_in_price_units} actually used in the backtest.
    expected_costs : {pair: cost_in_price_units} expected (e.g. from
                     BrokerCost.cost_prices()). If None, the check is skipped.
    tolerance : max fractional deviation allowed (0.20 = 20%).
    """
    if expected_costs is None:
        return CheckResult(
            "BG-02", False,
            "No expected costs provided. Pass expected_costs=BrokerCost.cost_prices() "
            "to validate cost accuracy.",
            severity="WARN",
        )

    errors = []
    for pair, used_price in spreads_used.items():
        expected_price = expected_costs.get(pair)
        if expected_price is None or expected_price == 0:
            continue
        if used_price == 0:
            errors.append(f"{pair}: cost=0 (expected {expected_price:.6f})")
            continue
        ratio = used_price / expected_price
        if ratio < (1 - tolerance):
            errors.append(
                f"{pair}: {used_price:.6f} used vs {expected_price:.6f} expected "
                f"({ratio:.0%} — UNDER by {(1-ratio)*100:.0f}%)"
            )
    if errors:
        return CheckResult(
            "BG-02", False,
            f"COST MISMATCH: {len(errors)} pairs underestimated:\n    "
            + "\n    ".join(errors),
        )
    return CheckResult("BG-02", True, f"All {len(spreads_used)} pair costs within tolerance")


def check_bfill_in_source(source_path: str | Path) -> CheckResult:
    """BG-04: Check source code for bfill() usage (future data leak)."""
    path = Path(source_path)
    if not path.exists():
        return CheckResult("BG-04", True, f"File not found: {path}", severity="WARN")
    text = path.read_text()
    bfill_matches = re.findall(r'\.(?:bfill|backfill)\(', text)
    fillna_bfill = re.findall(
        r"fillna\([^)]*method\s*=\s*['\"](?:bfill|backfill)['\"]", text)
    interp_back = re.findall(
        r"interpolate\([^)]*limit_direction\s*=\s*['\"](?:backward|both)['\"]", text)
    all_matches = bfill_matches + fillna_bfill + interp_back
    if all_matches:
        return CheckResult(
            "BG-04", False,
            f"BFILL LEAK: {len(all_matches)} backward-fill calls (bfill/backfill/"
            f"interpolate backward) found in {path.name}. Use ffill() instead.",
        )
    return CheckResult("BG-04", True, f"No bfill() in {path.name}")


def check_resolution(
    resolution_minutes: int,
    sl_atr_mult: float | None = None,
    avg_atr_pips: float | None = None,
) -> CheckResult:
    """BG-05: Warn if using 1h+ bars for SL/TP simulation.

    1h bars cannot determine SL/TP hit order. 1m or finer is required
    for accurate fill simulation. Pass ``sl_atr_mult`` and ``avg_atr_pips``
    to have the warning quote your SL size.
    """
    if resolution_minutes >= 60:
        sl_note = ""
        if sl_atr_mult is not None and avg_atr_pips is not None:
            sl_note = f"With SL={sl_atr_mult}×ATR≈{avg_atr_pips*sl_atr_mult:.0f}pip, "
        return CheckResult(
            "BG-05", False,
            f"COARSE BARS: Using {resolution_minutes}min bars for SL/TP detection. "
            f"{sl_note}{'i' if sl_note else 'I'}ntra-bar SL/TP hit order is unknowable "
            f"(a bar touching both counts as SL). Use ≤5min bars for accurate simulation.",
            severity="WARN",
        )
    if resolution_minutes >= 15:
        return CheckResult(
            "BG-05", True,
            f"Using {resolution_minutes}min bars — acceptable but 1m is ideal",
            severity="WARN",
        )
    return CheckResult("BG-05", True, f"Using {resolution_minutes}min bars for execution")


def _period_months(n_bars: int, bar_minutes: int, start_ts=None, end_ts=None) -> tuple[float, str]:
    """Months of data and how they were measured (see check_data_period)."""
    if start_ts is not None and end_ts is not None:
        import pandas as pd

        def _ts(x):
            if isinstance(x, (int, float, np.integer, np.floating)):
                # Numbers are Unix seconds (pd.Timestamp(int) reads ns).
                # 1e11 s is the year 5138; larger values are ms or ns.
                if np.isnan(x):
                    raise ValueError("check_data_period: start_ts/end_ts is NaN")
                if not abs(float(x)) < 1e11:
                    raise ValueError(
                        f"check_data_period: {x!r} is not Unix seconds "
                        "(milliseconds or nanoseconds?). Pass seconds or a Timestamp.")
                t = pd.Timestamp(float(x), unit="s")
            else:
                t = pd.Timestamp(x)
            if t is pd.NaT:
                raise ValueError("check_data_period: start_ts/end_ts is NaT")
            return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

        # From the first bar's open to the last bar's close.
        delta = _ts(end_ts) - _ts(start_ts) + pd.Timedelta(minutes=bar_minutes)
        return delta.total_seconds() / 86400 / 30.44, "from the first and last bar times"
    # FX: 22 trading days/month × 24 hours/day = 528 1h-bars/month
    bars_per_month = (22 * 24 * 60) / bar_minutes
    months = n_bars / bars_per_month if bars_per_month > 0 else 0
    return months, f"estimated from {n_bars} bars × {bar_minutes}min at FX hours"


# About a week. Even a gap-free non-leap year is 365 / 30.44 = 11.99 months,
# and a calendar year of FX data starts after the New Year close and ends at
# the last close of the year (the Friday when Dec 31 falls on a weekend), so
# it measures 11.86-11.99 months (EURUSD and XAUUSD 1h, 2022-2024). The
# slack lets a full year pass.
_PERIOD_SLACK_MONTHS = 0.25


def _period_result(total_months: float, basis: str, min_months: int,
                   estimated: bool = False) -> CheckResult:
    floor = min_months - _PERIOD_SLACK_MONTHS
    if total_months < floor:
        hint = (" The period was estimated at FX hours; pass the bar times "
                "(timestamps= to bug_guard()/gate0_validate(), or start_ts/end_ts "
                "to check_data_period()) to measure it exactly." if estimated else "")
        return CheckResult(
            "BG-06", False,
            f"SHORT PERIOD: {total_months:.2f} months ({basis}). "
            f"Minimum {min_months} months less 1 week ({floor:.2f}) required. "
            f"Results will be overfitted.{hint}",
        )
    return CheckResult("BG-06", True,
                       f"Data period: {total_months:.2f} months (>={floor:.2f}, {basis})")


def check_data_period(
    n_bars: int,
    bar_minutes: int = 60,
    min_months: int = 12,
    start_ts=None,
    end_ts=None,
) -> CheckResult:
    """BG-06: Verify sufficient data period to avoid short-period overfitting.

    Passes at ``min_months`` less 1 week (11.75 for 12), so that a calendar
    year of FX data, which starts and ends at a weekend or holiday close,
    passes.

    If start_ts/end_ts are provided (the first and last bar open times: Unix
    seconds like the rest of the library, pd.Timestamp, datetime64 or
    datetime), uses the calendar period up to the last bar's close.
    Otherwise estimates from bar count assuming FX market hours (24h/day,
    22 trading days/month = 528 bars/month for 1h). FX trades about 21.7
    days a month, so the estimate errs short (5 years of AggVault EURUSD 1h
    reads as 59 months). Markets with fewer hours read much shorter, and a
    market open every day (crypto, 30.4 days a month) reads about 1.4× too
    long, so pass the dates for those.
    """
    months, basis = _period_months(n_bars, bar_minutes, start_ts, end_ts)
    return _period_result(months, basis, min_months,
                          estimated=start_ts is None or end_ts is None)


def check_expanding_quantile(source_path: str | Path) -> CheckResult:
    """BG-07: Check for full-period quantile/percentile usage (data leak)."""
    path = Path(source_path)
    if not path.exists():
        return CheckResult("BG-07", True, f"File not found: {path}", severity="WARN")
    text = path.read_text()
    # Look for np.percentile/quantile on full arrays without expanding window
    suspicious = []
    for pattern in [
        r'np\.(?:nan)?percentile\([^,]+,',
        r'np\.(?:nan)?quantile\([^,]+,',
        r'\.quantile\(',
        r'\.percentile\(',
    ]:
        for m in re.finditer(pattern, text):
            line_num = text[:m.start()].count('\n') + 1
            # Check if it's inside an expanding/rolling context
            context_start = max(0, m.start() - 200)
            context = text[context_start:m.start()]
            if 'expanding' not in context and 'rolling' not in context and 'for bar' not in context:
                suspicious.append(f"line {line_num}: {m.group()}")
    if suspicious:
        return CheckResult(
            "BG-07", False,
            f"PERCENTILE LEAK: {len(suspicious)} potential full-period quantile uses:\n    "
            + "\n    ".join(suspicious[:5]),
            severity="WARN",  # WARN because it may be intentional (param ranges, etc.)
        )
    return CheckResult("BG-07", True, "No obvious percentile leak detected")


def check_same_bar_reentry(
    entry_bars: np.ndarray,
    exit_bars: np.ndarray,
    resolution_minutes: int = 60,
) -> CheckResult:
    """BG-08: Check for same-bar exit → reentry without sub-bar resolution.

    If trade[i] exits at bar N and trade[i+1] enters at bar N, this is a
    look-ahead leak unless sub-bar (1m) resolution is used.
    """
    if len(entry_bars) < 2:
        return CheckResult("BG-08", True, "Too few trades to check")
    same_bar = 0
    for i in range(1, len(entry_bars)):
        if entry_bars[i] == exit_bars[i - 1]:
            same_bar += 1
    if same_bar > 0 and resolution_minutes >= 60:
        pct = same_bar / (len(entry_bars) - 1) * 100
        return CheckResult(
            "BG-08", False,
            f"SAME-BAR REENTRY: {same_bar} trades ({pct:.1f}%) enter on the same "
            f"{resolution_minutes}min bar as previous exit. Use 1m resolution to validate.",
        )
    if same_bar > 0:
        pct = same_bar / (len(entry_bars) - 1) * 100
        return CheckResult(
            "BG-08", True,
            f"{same_bar} same-bar reentries ({pct:.1f}%) — OK with {resolution_minutes}min resolution",
        )
    return CheckResult("BG-08", True, "No same-bar reentries")


def check_entry_price_type(source_path: str | Path) -> CheckResult:
    """BG-09: Check that entry uses Open, not Close.

    Entry at Close[bar] means you're using information that wasn't available
    at the time the signal was generated.
    """
    path = Path(source_path)
    if not path.exists():
        return CheckResult("BG-09", True, f"File not found: {path}", severity="WARN")
    text = path.read_text()
    # Look for entry_price = ...Close...values[ (the original incident: ERROR)
    close_entry = re.findall(
        r'entry_price\s*=.*\bClose\b.*values\s*\[',
        text,
    )
    if close_entry:
        return CheckResult(
            "BG-09", False,
            f"ENTRY AT CLOSE: Found {len(close_entry)} instance(s) of "
            f"entry_price using Close. Use Open of next bar instead.",
        )
    # Other spellings (close[i], closes[i], df["close"].iloc[i]) — WARN,
    # since the pattern is looser. "+ 1]" (next bar) is not flagged.
    loose = [
        m.group(0) for m in re.finditer(
            r'entry_price\s*=\s*[^\n#]*\b[Cc]loses?\b[^\n#]*\[[^\n#]*', text)
        if not re.search(r'\+\s*1\s*\]', m.group(0))
    ]
    if loose:
        return CheckResult(
            "BG-09", False,
            f"POSSIBLE ENTRY AT CLOSE: {len(loose)} line(s) set entry_price from a "
            f"close price (e.g. {loose[0].strip()[:60]!r}). If the signal uses "
            f"that close, enter at the next bar's open instead.",
            severity="WARN",
        )
    return CheckResult("BG-09", True, "Entry price does not use Close")


def check_open_prices_provided(open_prices_provided: bool) -> CheckResult:
    """BG-09b: Verify that open_prices are used for entry.

    When open_prices is not provided to simulate_trades(), entry occurs at
    close[signal_bar] which uses information not available at signal time.
    This is the API-level complement to BG-09's source code scan.

    Parameters
    ----------
    open_prices_provided : True if open_prices was passed to simulate_trades().
    """
    if not open_prices_provided:
        return CheckResult(
            "BG-09b", False,
            "ENTRY AT CLOSE: open_prices not provided to simulate_trades(). "
            "Trades enter at close[signal_bar], which is a look-ahead bias. "
            "Pass open_prices= to use next-bar-open entry.",
            severity="WARN",
        )
    return CheckResult("BG-09b", True, "open_prices provided for next-bar-open entry")


def check_fixed_cost_usage(
    sl_distances: np.ndarray,
    cost_is_fixed: bool = False,
) -> CheckResult:
    """BG-12: Warn if fixed cost is used with varying SL distances.

    When SL distance varies across trades (ATR-based), using a fixed cost_r
    (scalar) is inaccurate. Use BrokerCost.per_trade_cost() instead.

    Parameters
    ----------
    sl_distances : SL distances for all trades.
    cost_is_fixed : True if caller used scalar as_r() instead of per-trade cost.
                    GateKeeper's Gate 0 infers it from ``cost_r``.
    """
    if not cost_is_fixed:
        return CheckResult("BG-12", True,
                           "No fixed cost reported (per-trade costs, or none applied)")

    if len(sl_distances) < 2:
        return CheckResult("BG-12", True, "Too few trades to check cost variation")

    cv = np.std(sl_distances) / np.mean(sl_distances) if np.mean(sl_distances) > 0 else 0.0
    if cv > 0.10:
        return CheckResult(
            "BG-12", False,
            f"FIXED COST WITH VARYING SL: SL distance CV={cv:.1%}. "
            f"Use BrokerCost.per_trade_cost() for accurate per-trade costs.",
            severity="WARN",
        )
    return CheckResult("BG-12", True, f"SL distance CV={cv:.1%} — fixed cost acceptable")


def check_spread_filter(
    signal_spreads: np.ndarray | None = None,
    max_spread: float = 0.0,
) -> CheckResult:
    """BG-13: Verify backtest signals respect max spread constraint.

    If a max_spread filter is used in live trading, the backtest should
    also filter out signals where spread exceeded the threshold.

    Parameters
    ----------
    signal_spreads : Spread at signal time for each trade. None if not tracked.
    max_spread : Maximum spread allowed (in price units). 0 = no constraint.
    """
    if max_spread <= 0.0:
        return CheckResult("BG-13", True, "No spread filter constraint to check")
    if signal_spreads is None:
        return CheckResult(
            "BG-13", False,
            f"max_spread={max_spread:.6f} specified but signal_spreads not provided. "
            "Cannot verify backtest/live spread parity.",
            severity="WARN",
        )

    violations = np.sum(signal_spreads > max_spread)
    if violations > 0:
        pct = violations / len(signal_spreads) * 100
        return CheckResult(
            "BG-13", False,
            f"SPREAD FILTER LEAK: {violations}/{len(signal_spreads)} signals "
            f"({pct:.1f}%) had spread > {max_spread:.6f}. "
            f"Apply spread filter in backtest to match live behavior.",
            severity="WARN",
        )
    return CheckResult("BG-13", True, f"All {len(signal_spreads)} signals within max_spread")


def check_incomplete_bars(
    timestamps: np.ndarray,
    expected_interval_minutes: int = 60,
    now: Optional[float] = None,
) -> CheckResult:
    """BG-10: Unsorted/duplicate timestamps, or a last bar still forming.

    Parameters
    ----------
    timestamps : Bar OPEN times (Unix seconds).
    expected_interval_minutes : Bar size.
    now : Current Unix time (default: time.time()). For tests.
    """
    try:
        ts = np.asarray(timestamps, dtype=np.int64)
    except (TypeError, ValueError):
        return CheckResult("BG-10", False,
                           "BAD TIMESTAMPS: bar times must be Unix seconds (integers).")
    if len(ts) < 2:
        return CheckResult("BG-10", True, "Too few bars to check")
    n_bad = int(np.sum(np.diff(ts) <= 0))
    if n_bad:
        return CheckResult(
            "BG-10", False,
            f"UNSORTED TIMESTAMPS: {n_bad} bar(s) are not later than the previous "
            f"bar (duplicates or out of order). Sort and de-duplicate first.",
        )
    if now is None:
        import time as _time
        now = _time.time()
    last_close = int(ts[-1]) + expected_interval_minutes * 60
    if last_close > now:
        return CheckResult(
            "BG-10", False,
            f"INCOMPLETE LAST BAR: the last {expected_interval_minutes}min bar closes "
            f"{(last_close - now) / 60:.0f} min from now, so its OHLC is still "
            f"changing. Drop it before computing signals.",
            severity="WARN",
        )
    return CheckResult("BG-10", True, "Timestamps sorted, last bar complete")


def check_min_trades(
    n_trades: int,
    min_required: int = 100,
) -> CheckResult:
    """BG-06b: Minimum trade count for statistical significance."""
    if n_trades < min_required:
        return CheckResult(
            "BG-06b", False,
            f"LOW TRADE COUNT: {n_trades} trades (minimum {min_required}). "
            f"Results are statistically unreliable.",
            severity="WARN",
        )
    return CheckResult("BG-06b", True, f"Trade count: {n_trades} (>={min_required})")


def check_effective_no_sl(
    sl_distances: np.ndarray,
    avg_price: float,
    threshold: float = 0.05,
    resolution_minutes: int = 60,
) -> CheckResult:
    """BG-14: Warn when SL distances are effectively disabled.

    If median SL distance exceeds ``threshold`` fraction of the average price,
    the strategy has no meaningful downside protection.  Combined with coarse
    bars (≥60min), intra-bar MAE is severely underestimated.

    Incident: 2026-04-01 Gold Rollover — SL=None strategy on 1h bars hid
    true max adverse excursion.  Backtest showed -1,014 pips max DD, but
    real intra-bar drawdown was unknowable.

    Parameters
    ----------
    sl_distances : SL distances in price units for all trades.
    avg_price : Representative price level (e.g. median of close array).
    threshold : Fraction of price above which SL is considered "no SL".
    resolution_minutes : Bar size, used to escalate severity.
    """
    if len(sl_distances) == 0:
        return CheckResult("BG-14", True, "No trades to check")

    if avg_price <= 0:
        return CheckResult("BG-14", True, "avg_price not provided — skipped", severity="WARN")

    median_sl = float(np.median(sl_distances))
    sl_pct = median_sl / avg_price

    if sl_pct > threshold:
        severity = "WARN"
        extra = ""
        if resolution_minutes >= 60:
            extra = (
                f" Combined with {resolution_minutes}min bars, intra-bar MAE "
                f"(max adverse excursion) is severely underestimated. "
                f"Use 1min bars to measure true intra-trade risk."
            )
        return CheckResult(
            "BG-14", False,
            f"EFFECTIVELY NO SL: Median SL = {median_sl:.2f} "
            f"({sl_pct:.1%} of price level {avg_price:.2f}). "
            f"Trades have no meaningful downside protection.{extra}",
            severity=severity,
        )
    return CheckResult(
        "BG-14", True,
        f"SL distance = {sl_pct:.1%} of price (within {threshold:.0%} threshold)",
    )


# ── Aggregate runner ───────────────────────────────────────────────────────

def run_all_checks(
    *,
    source_path: Optional[str | Path] = None,
    signal_bars: Optional[np.ndarray] = None,
    entry_bars: Optional[np.ndarray] = None,
    exit_bars: Optional[np.ndarray] = None,
    spreads_used: Optional[dict[str, float]] = None,
    expected_costs: Optional[dict[str, float]] = None,
    resolution_minutes: int = 60,
    n_bars: int = 0,
    bar_minutes: int = 60,
    n_trades: int = 0,
    min_months: int = 12,
    min_trades: int = 100,
    open_prices_provided: Optional[bool] = None,
    sl_distances: Optional[np.ndarray] = None,
    cost_is_fixed: bool = False,
    signal_spreads: Optional[np.ndarray] = None,
    max_spread: float = 0.0,
    avg_price: float = 0.0,
    timestamps: Optional[np.ndarray] = None,
    strict: bool = True,
    _suppress_print: bool = False,
    **kwargs,
) -> GuardReport:
    """Run all applicable checks and return a report.

    Parameters
    ----------
    source_path : Path to the backtest script (for source code checks).
    signal_bars : Array of bar indices where signals were generated.
    entry_bars : Array of bar indices where trades were entered.
    exit_bars : Array of bar indices where trades were exited.
    spreads_used : Dict of {pair: cost_in_price_units} used in the backtest.
    expected_costs : Dict of {pair: cost_in_price_units} for cost validation.
                     Use BrokerCost.cost_prices() to generate this.
                     If None, cost check is skipped.
    resolution_minutes : Bar size used for SL/TP simulation.
    n_bars : Number of bars in the dataset (BG-06 estimates the period from
             it when ``timestamps`` is not given; with fewer ``n_bars`` than
             timestamps, the timestamp span is scaled by n_bars/len(timestamps)).
    bar_minutes : Size of each bar in minutes.
    n_trades : Total number of trades.
    min_months : Minimum data period in months.
    min_trades : Minimum trade count.
    open_prices_provided : True if open_prices was passed to simulate_trades().
    sl_distances : SL distances for all trades (for BG-12 fixed-cost check).
    cost_is_fixed : True if scalar cost was used instead of per-trade cost.
    signal_spreads : Spread at signal time for each trade (for BG-13).
    max_spread : Maximum spread allowed in price units (for BG-13). 0 = no check.
    avg_price : Representative price level for BG-14 no-SL detection
                (e.g. median of close). 0 = skip check.
    timestamps : Bar open times of the data, Unix seconds (for BG-10, and
                 BG-06 measures the period from the first to the last bar).
                 None = skip BG-10.
    strict : If True, assert_passed() is called (raises on ERROR).
    """
    # Backward compat: accept deprecated broker kwarg
    if "broker" in kwargs:
        warnings.warn(
            "broker parameter is deprecated. "
            "Use expected_costs=BrokerCost.fundora().cost_prices() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        broker = kwargs.pop("broker")
        if expected_costs is None and broker == "fundora":
            expected_costs = _fundora_expected_costs()
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")

    report = GuardReport()

    # Source code checks
    if source_path is not None:
        report.results.append(check_bfill_in_source(source_path))
        report.results.append(check_expanding_quantile(source_path))
        report.results.append(check_entry_price_type(source_path))

    # Signal/trade checks
    if signal_bars is not None and entry_bars is not None:
        report.results.append(check_look_ahead(signal_bars, entry_bars))
    if entry_bars is not None and exit_bars is not None:
        report.results.append(check_same_bar_reentry(entry_bars, exit_bars, resolution_minutes))

    # Cost checks
    if spreads_used is not None:
        report.results.append(check_cost_registry(spreads_used, expected_costs))

    # Data quality checks
    report.results.append(check_resolution(resolution_minutes))
    ts_span = timestamps is not None and len(timestamps) >= 2
    if ts_span:
        # The bar times give the calendar period; the bar count only estimates it.
        t = np.asarray(timestamps)
        try:
            months, basis = _period_months(0, bar_minutes, t.min(), t.max())
        except (ValueError, TypeError) as e:  # ms/ns, NaN/NaT, naive mixed with aware
            r = CheckResult("BG-06", False, f"BAD TIMESTAMPS: {e}")
        else:
            share = n_bars / len(t) if n_bars and n_bars < len(t) else 1.0
            if share < 1.0:
                # Fewer bars than bar times, e.g. close sliced to a short
                # window but the full ts passed: count only the bars' share.
                months *= share
                basis += f", × n_bars/timestamps = {n_bars}/{len(t)}"
            r = _period_result(months, basis, min_months)
            if not r.passed and share < 0.5:
                # More likely two timeframes than a sliced close (1m times
                # with the 1h n_bars): say that before the period.
                r = CheckResult("BG-06", False,
                                f"n_bars ({n_bars}) and timestamps ({len(t)}) count different "
                                f"bars? Pass the times of the signal-timeframe bars that "
                                f"n_bars counts (the 1h times also with hires). " + r.message)
        report.results.append(r)
    elif n_bars > 0:
        report.results.append(check_data_period(n_bars, bar_minutes, min_months))
    if n_trades > 0:
        report.results.append(check_min_trades(n_trades, min_trades))
    if timestamps is not None:
        report.results.append(check_incomplete_bars(timestamps, bar_minutes))

    # API-level entry check
    if open_prices_provided is not None:
        report.results.append(check_open_prices_provided(open_prices_provided))

    # BG-12: Fixed cost with varying SL
    if sl_distances is not None:
        report.results.append(check_fixed_cost_usage(sl_distances, cost_is_fixed))

    # BG-13: Spread filter violations
    if signal_spreads is not None or max_spread > 0.0:
        report.results.append(check_spread_filter(signal_spreads, max_spread))

    # BG-14: Effectively no SL
    if sl_distances is not None and avg_price > 0:
        report.results.append(
            check_effective_no_sl(sl_distances, avg_price, resolution_minutes=resolution_minutes)
        )

    if not _suppress_print:
        report.print_report()

    if strict:
        report.assert_passed()

    return report
