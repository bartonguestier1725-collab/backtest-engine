"""Strategy audit: dynamic checks that BugGuard's pattern checks cannot do.

Two tests, both independent of what the strategy code looks like:

1. **Look-ahead test** (``check_lookahead``) — rewrite every bar after a cut
   point and re-run the signal function. A causal signal at a bar at or before
   the cut cannot change when only later bars change. Four rewrites are used
   (mirror around the cut close, 2x deviations, +25% jump, -20% jump), because
   some leaks survive one of them (a full-period volatility quantile does not
   change when the future is mirrored). Half of the cuts are placed on, or just
   before, bars where the strategy has a signal, so rare signals are tested.
   The test is probabilistic: a leak that rarely flips a signal can pass.

2. **Null-data test** (``check_null_edge``) — run the whole backtest on
   driftless random walks. Each bar's high and low are the extremes of a
   continuous (Brownian bridge) path between its open and close, so a stop or
   target inside [low, high] was really touched and fills at its level fairly.
   The expected PnL before costs is then zero, minus a small pessimistic bias
   from the SL-first rule when a bar touches both SL and TP. The same entries
   are also run with coin-flip directions (inside simulate_trades, with the
   SL/TP distances unchanged), which keeps that bias and removes any
   information in the directions.

A strategy file for the CLI defines ``signals(data)`` and/or
``backtest(data)``; ``data`` is a dict of numpy arrays with keys
``time, open, high, low, close`` (and ``volume`` if available)::

    python -m backtest_engine.audit my_strategy.py            # human-readable
    python -m backtest_engine.audit my_strategy.py --json     # for AI agents

Passing both tests does not prove a strategy is free of bias (a leak that
never changes a signal, survivorship in the instrument list, or bad data are
invisible to them). Failing either one is strong evidence of a problem.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from backtest_engine._types import EXIT_NO_FILL

__all__ = [
    "AuditResult", "check_lookahead", "check_null_edge", "random_walk_bars",
    "audit_strategy", "main",
]

_REWRITES = ("mirror", "scale2", "jump_up", "jump_down")
_OHLC = ("open", "high", "low", "close")
_MIN_SIGNALS = 5
_MIN_PATHS = 10


@dataclass
class AuditResult:
    """Outcome of one audit test.

    verdict : "pass", "fail", "warn" or "error".
    """
    check: str
    verdict: str
    summary: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Data helpers ────────────────────────────────────────────────────────────

def _as_data(data) -> dict:
    """Accept a dict with open/high/low/close (time, volume optional) or the
    6-tuple returned by fetch_aggvault / load_ohlcv."""
    from collections.abc import Mapping
    if isinstance(data, Mapping):
        d = {k: np.asarray(data[k]) for k in data.keys()}
    else:
        ts, o, h, l, c = data[:5]
        d = {"time": np.asarray(ts), "open": np.asarray(o), "high": np.asarray(h),
             "low": np.asarray(l), "close": np.asarray(c)}
        if len(data) > 5:
            d["volume"] = np.asarray(data[5], dtype=np.float64)
    for k in _OHLC:
        if k not in d:
            raise ValueError(f"data needs '{k}'")
        d[k] = np.asarray(d[k], dtype=np.float64)
    n = len(d["close"])
    if "time" not in d:
        d["time"] = np.arange(n, dtype=np.int64) * 60
    if any(len(v) != n for v in d.values()):
        raise ValueError("all data columns must have the same length")
    return d


def random_walk_bars(
    n_bars: int = 5000,
    bar_minutes: int = 60,
    vol: float = 0.002,
    start_price: float = 100.0,
    seed=0,
) -> dict:
    """Driftless random-walk OHLC bars (plus a random volume column).

    ``seed`` is an int or a numpy SeedSequence.

    The close follows an arithmetic random walk with per-bar standard
    deviation ``vol * start_price``. Each bar opens at the previous close, and
    its high and low are drawn from the distribution of the maximum and the
    minimum of a Brownian bridge from the open to the close, so every level
    inside [low, high] was touched by a continuous path: stops and targets
    fill at their level fairly. Prices are shifted up if the walk would come
    near zero.
    """
    rng = np.random.default_rng(seed)
    sig = vol * start_price
    c = start_price + np.cumsum(rng.normal(0.0, sig, n_bars))
    o = np.r_[start_price, c[:-1]]
    dlt = c - o
    h = (o + c + np.sqrt(dlt * dlt - 2.0 * sig * sig * np.log(rng.random(n_bars)))) / 2.0
    l = (o + c - np.sqrt(dlt * dlt - 2.0 * sig * sig * np.log(rng.random(n_bars)))) / 2.0
    floor = 0.2 * start_price
    if l.min() < floor:
        k = floor - l.min()
        o, h, l, c = o + k, h + k, l + k, c + k
    t = np.arange(n_bars, dtype=np.int64) * bar_minutes * 60 + 1_704_067_200
    vol_col = np.round(rng.lognormal(6.0, 0.5, n_bars))
    return {"time": t, "open": o, "high": h, "low": l, "close": c, "volume": vol_col}


def _rewrite_future(d: dict, cut: int, mode: str) -> dict:
    """Copy of d with every bar after ``cut`` rewritten. OHLC are mirrored,
    scaled or jumped around close[cut]; other numeric columns (e.g. volume)
    are scaled in the non-mirror modes. ``time`` is unchanged."""
    out = {k: v.copy() for k, v in d.items()}
    p = d["close"][cut]
    f = slice(cut + 1, None)
    o, h, l, c = (d[k][f] for k in _OHLC)
    if mode == "mirror":
        o2, h2, l2, c2 = 2 * p - o, 2 * p - l, 2 * p - h, 2 * p - c
        k_extra = 1.0
    elif mode == "scale2":
        o2, h2, l2, c2 = (p + 2 * (x - p) for x in (o, h, l, c))
        k_extra = 2.0
    elif mode in ("jump_up", "jump_down"):
        k_extra = 1.25 if mode == "jump_up" else 0.8
        o2, h2, l2, c2 = o * k_extra, h * k_extra, l * k_extra, c * k_extra
    else:
        raise ValueError(mode)
    # Mirroring/scaling can push far-away prices to <= 0; keep them positive.
    floor = 1e-6 * abs(p)
    out["open"][f], out["high"][f], out["low"][f], out["close"][f] = (
        np.maximum(x, floor) for x in (o2, h2, l2, c2))
    for k, v in d.items():
        if k in _OHLC or k == "time" or k_extra == 1.0:
            continue
        if np.issubdtype(v.dtype, np.number) and not np.issubdtype(v.dtype, np.bool_):
            out[k] = out[k].astype(np.float64)
            out[k][f] = v[f] * k_extra
    return out


def _per_bar(sig, n: int) -> np.ndarray:
    """Signal output -> float array per bar. Accepts an array of length n, or
    a tuple whose first two items are (signal_bars, directions); for a tuple
    only those two are compared."""
    if isinstance(sig, tuple):
        bars = np.asarray(sig[0], dtype=np.int64)
        vals = np.asarray(sig[1], dtype=np.float64) if len(sig) > 1 else np.ones(len(bars))
        out = np.zeros(n, dtype=np.float64)
        if len(bars) and (bars.min() < 0 or bars.max() >= n):
            raise ValueError("signal_bars out of range")
        out[bars] = vals
        return out
    arr = np.asarray(sig, dtype=np.float64)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(
            f"signal function must return an array with one value per bar "
            f"(got shape {arr.shape}, expected ({n},)) or (signal_bars, directions)")
    return arr


def _same(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> np.ndarray:
    eq = (a == b) | (np.isnan(a) & np.isnan(b))
    if rtol > 0 or atol > 0:
        eq |= np.isclose(a, b, rtol=rtol, atol=atol)
    return eq


def _diagnose(hits: list[dict]) -> str:
    """Plain-language guess at the kind of leak from which bars changed."""
    if all(h["latest_changed"] < 5 for h in hits):
        return ("only the first bars of the data changed: probably a wrap-around "
                "(np.roll) that puts the last bars at the start")
    near = min(h["cut"] - h["latest_changed"] + 1 for h in hits)
    far = max(h["reach_bars"] for h in hits)
    if far <= 2:
        return f"the signal uses the next {far} bar(s)"
    if near <= 2 and far <= 60:
        return f"a window looking up to {far} bars ahead (e.g. a centered window or a pivot)"
    return (f"signals up to {far} bars before the cut changed: a statistic, filter or model "
            f"over the whole sample")


# ── Test 1: look-ahead ──────────────────────────────────────────────────────

def check_lookahead(
    signal_fn: Callable[[dict], Any],
    data=None,
    n_cuts: int = 32,
    warmup: Optional[int] = None,
    seed: int = 0,
    rtol: float = 1e-9,
) -> AuditResult:
    """Look-ahead test: do signals at or before a cut change when only bars
    after the cut change?

    Parameters
    ----------
    signal_fn : ``signal_fn(data) -> array (one value per bar) or
                (signal_bars, directions)``. ``data`` is a dict of numpy arrays
                (time/open/high/low/close, and volume if present). Must not
                modify ``data``.
    data : Bars to test on (dict or 6-tuple). Default: 5,000 random-walk bars.
           Real data is better when the signal depends on real market
           structure (sessions, volatility regimes).
    n_cuts : Number of cut points. Half are random bars after ``warmup``, half
             are on or 1-3 bars before bars with a non-zero signal. Each cut
             costs 4 signal runs. More cuts catch rarer leaks.
    warmup : First bar a cut may be placed at (default: 10% of the bars).
    rtol : Relative tolerance for float signals (default 1e-9, so numerical
           noise from e.g. FFT convolution is not reported as look-ahead).
    """
    if n_cuts < 1:
        return AuditResult("lookahead", "error", "n_cuts must be >= 1")
    d = _as_data(data if data is not None else random_walk_bars())
    n = len(d["close"])
    warmup = max(1, n // 10) if warmup is None else int(warmup)
    if n - 2 <= warmup:
        return AuditResult("lookahead", "error", f"Too few bars ({n}) for warmup={warmup}")
    t0 = time.time()
    try:
        base = _per_bar(signal_fn({k: v.copy() for k, v in d.items()}), n)
        again = _per_bar(signal_fn({k: v.copy() for k, v in d.items()}), n)
    except Exception as e:  # surfaced to the user/agent as an error verdict
        return AuditResult("lookahead", "error", f"signal function raised {type(e).__name__}: {e}")
    finite = np.abs(base[np.isfinite(base)])
    atol = 1e-12 * float(finite.max()) if len(finite) else 0.0
    if not np.all(_same(base, again, rtol, atol)):
        return AuditResult(
            "lookahead", "error",
            "Signal function is not deterministic (two runs on the same data differ). "
            "Fix the random seed before testing.")
    active = np.flatnonzero(np.nan_to_num(base) != 0)
    active = active[(active >= warmup) & (active < n - 1)]
    if len(active) < _MIN_SIGNALS:
        return AuditResult(
            "lookahead", "error",
            f"Only {len(active)} bars with a signal in the test data ({n} bars); too few "
            f"to test. Use more or real data (--csv, or NULL_BARS in the strategy file).",
            {"n_bars": n, "signal_bars": int(len(active))})
    rng = np.random.default_rng(seed)
    n_rand = n_cuts - n_cuts // 2
    pool = np.arange(warmup, n - 1)
    cuts = set(rng.choice(pool, size=min(n_rand, len(pool)), replace=False).tolist())
    near = np.clip(rng.choice(active, size=n_cuts // 2) - rng.integers(0, 4, n_cuts // 2),
                   warmup, n - 2)
    cuts |= set(near.tolist())
    cuts = np.array(sorted(cuts))
    hits = []
    for cut in cuts:
        for mode in _REWRITES:
            try:
                pert = _per_bar(signal_fn(_rewrite_future(d, int(cut), mode)), n)
            except Exception as e:
                return AuditResult("lookahead", "error",
                                   f"signal function raised {type(e).__name__} on rewritten data: {e}")
            diff = ~_same(base[: cut + 1], pert[: cut + 1], rtol, atol)
            if diff.any():
                idx = np.flatnonzero(diff)
                hits.append({
                    "cut": int(cut), "rewrite": mode, "bars_changed": int(len(idx)),
                    "earliest_changed": int(idx.min()), "latest_changed": int(idx.max()),
                    # how many bars ahead the earliest changed signal looked
                    "reach_bars": int(cut - idx.min() + 1),
                })
    cut_hit = sorted({h["cut"] for h in hits})
    details = {
        "n_bars": n, "signal_bars": int(len(active)), "n_cuts": int(len(cuts)),
        "cuts_with_change": len(cut_hit), "rewrites": list(_REWRITES),
        "elapsed_sec": round(time.time() - t0, 2), "examples": hits[:5],
    }
    if not hits:
        return AuditResult(
            "lookahead", "pass",
            f"No signal at or before a cut changed when later bars were rewritten "
            f"({len(cuts)} cuts x {len(_REWRITES)} rewrites, {len(active)} signal bars). "
            f"This does not rule out a leak that rarely flips a signal.", details)
    first = hits[0]
    details["diagnosis"] = _diagnose(hits)
    return AuditResult(
        "lookahead", "fail",
        f"LOOK-AHEAD: signals changed after rewriting only later bars at "
        f"{len(cut_hit)}/{len(cuts)} cuts. Example: cut at bar {first['cut']} "
        f"({first['rewrite']}) changed {first['bars_changed']} earlier signal(s) "
        f"(bars {first['earliest_changed']}..{first['latest_changed']}). "
        f"Likely cause: {details['diagnosis']}.", details)


# ── Test 2: null data ───────────────────────────────────────────────────────

def _gross_pnl(r) -> np.ndarray:
    """Per-trade R before costs, NO_FILL removed. Accepts TradeResults, any
    structured array with pnl_r (and cost_r, exit_type), or a plain array."""
    arr = np.asarray(r)
    if arr.dtype.names and "pnl_r" in arr.dtype.names:
        pnl = np.asarray(arr["pnl_r"], dtype=np.float64)
        if "cost_r" in arr.dtype.names:
            pnl = pnl + np.asarray(arr["cost_r"], dtype=np.float64)
        if "exit_type" in arr.dtype.names:
            pnl = pnl[np.asarray(arr["exit_type"]) != EXIT_NO_FILL]
        return pnl
    return np.asarray(arr, dtype=np.float64).ravel()


def _cluster_t(tot: np.ndarray, cnt: np.ndarray) -> tuple[float, float]:
    """Mean R per trade over all paths and its t-statistic, with paths as
    clusters (trades within a path overlap; paths are independent). Unlike
    the mean of per-path means, it is not biased when the number of trades
    depends on how a path went."""
    n = cnt.sum()
    mu = tot.sum() / n
    k = len(tot)
    resid = tot - mu * cnt
    se = np.sqrt(k / (k - 1) * np.sum(resid ** 2)) / n
    if se > 1e-12 * max(1.0, abs(mu)):
        return float(mu), float(mu / se)
    return float(mu), (0.0 if abs(mu) <= 1e-12 else float(np.copysign(np.inf, mu)))


@contextmanager
def _coin_flip_directions(seed_seq):
    """While active (in this thread only), simulate_trades multiplies the
    trade directions by random +1/-1 and keeps the SL/TP distances: the same
    entries with coin-flip directions."""
    from backtest_engine import core
    state = core._DIRECTION_FLIP
    prev = getattr(state, "rng", None)
    state.rng = np.random.default_rng(seed_seq)
    try:
        yield
    finally:
        state.rng = prev


def check_null_edge(
    backtest_fn: Callable[[dict], Any],
    n_paths: int = 120,
    n_bars: int = 5000,
    bar_minutes: int = 60,
    vol: float = 0.002,
    seed: int = 0,
    t_threshold: float = 3.5,
    placebo: bool = True,
    n_placebo: int = 2,
) -> AuditResult:
    """Null-data test: the full backtest on driftless random walks.

    Parameters
    ----------
    backtest_fn : ``backtest_fn(data) -> TradeResults`` (or any array with a
                  ``pnl_r`` field, or a plain array of per-trade R). Runs the
                  whole pipeline: signals, simulate_trades, costs. Costs are
                  added back (``cost_r``) before testing.
    n_paths : Independent random walks (at least 10). Paths are spawned from
              ``SeedSequence(seed)``, so different seeds share no path. The
              t-statistics treat each path as a cluster of trades. 120 paths
              catch weak leaks (a fraction of trades affected) that 30 miss.
    placebo : Also run the backtest with the direction of every trade passed
              to ``simulate_trades`` / ``simulate_trades_hires`` replaced by a
              coin flip (SL/TP distances unchanged): the same entries with
              random directions, under the same exit rules. On a driftless
              walk this keeps every fill-rule bias (SL-first) and removes any
              information in the directions. Only calls in the auditing
              thread are flipped; if the results do not change, the placebo
              is dropped (``details``).
    t_threshold : |t| above which a result is flagged (see bench/README.md
                  for measured false-alarm rates).

    Verdicts
    --------
    fail : the strategy makes money on random data (look-ahead somewhere);
           or differs from its placebo (the directions carry information
           about later prices, or entries are misaligned in time); or,
           without an effective placebo, loses clearly more than chance.
    warn : it loses on random data and so does its placebo, by about as much:
           the SL-first rule (a bar touching both SL and TP counts as SL)
           biases this backtest; resolve the order with simulate_trades_hires.
    """
    if n_paths < _MIN_PATHS:
        return AuditResult("null_edge", "error", f"n_paths must be >= {_MIN_PATHS}")
    tot, cnt, ptot, pcnt, all_pnl = [], [], [], [], []
    same_as_real = True
    t0 = time.time()
    path_seeds = np.random.SeedSequence(seed).spawn(n_paths)
    for k in range(n_paths):
        path_seq, *flip_seqs = path_seeds[k].spawn(1 + n_placebo)
        d = random_walk_bars(n_bars, bar_minutes, vol, seed=path_seq)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pnl = _gross_pnl(backtest_fn({kk: v.copy() for kk, v in d.items()}))
                pl = []
                if placebo:
                    for j in range(n_placebo):
                        with _coin_flip_directions(flip_seqs[j]):
                            pl.append(_gross_pnl(backtest_fn({kk: v.copy() for kk, v in d.items()})))
        except Exception as e:
            return AuditResult("null_edge", "error", f"backtest function raised {type(e).__name__}: {e}")
        tot.append(float(pnl.sum()))
        cnt.append(len(pnl))
        all_pnl.append(pnl)
        if placebo:
            ptot.append(float(np.mean([x.sum() for x in pl])))
            pcnt.append(float(np.mean([len(x) for x in pl])))
            if any(len(x) != len(pnl) or not np.allclose(x, pnl, rtol=0, atol=1e-12) for x in pl):
                same_as_real = False
    tot, cnt = np.array(tot), np.array(cnt)
    n_trades = int(cnt.sum())
    if n_trades < 30 or np.sum(cnt > 0) < _MIN_PATHS:
        return AuditResult("null_edge", "error",
                           f"Too few trades on null data ({n_trades} in {n_paths} paths) to test. "
                           f"Raise NULL_BARS or use a looser entry rule for the test.")
    mean, t_raw = _cluster_t(tot, cnt)
    x = np.concatenate(all_pnl)
    gp, gl = float(x[x > 0].sum()), float(-x[x < 0].sum())
    details = {
        "n_paths": n_paths, "n_bars": n_bars, "n_trades": n_trades,
        "mean_r_per_trade": round(mean, 4),
        "t_across_paths": round(t_raw, 2),
        "gross_profit_factor": round(gp / gl, 3) if gl > 0 else None,
    }
    t_diff = t_plc = None
    if placebo and same_as_real:
        details["placebo"] = ("no effect, dropped: simulate_trades was not called in the "
                              "auditing thread, or the results do not depend on direction")
    elif placebo:
        ptot, pcnt = np.array(ptot), np.array(pcnt)
        plc_mean, t_plc = _cluster_t(ptot, pcnt)
        _, t_diff = _cluster_t(tot - ptot, cnt)
        details.update({
            "placebo": "same entries, coin-flip directions",
            "placebo_mean_r_per_trade": round(plc_mean, 4),
            "t_placebo": round(t_plc, 2),
            "t_vs_placebo": round(t_diff, 2),
        })
    details["elapsed_sec"] = round(time.time() - t0, 2)
    if t_raw > t_threshold:
        return AuditResult(
            "null_edge", "fail",
            f"EDGE ON RANDOM DATA: {mean:+.3f}R per trade before costs (t={t_raw:.1f} across "
            f"{n_paths} paths) on driftless random walks, where no strategy can have an "
            f"edge. The pipeline sees the future (signal look-ahead, or entries/exits/"
            f"targets that use bars after the decision).", details)
    if t_diff is not None and abs(t_diff) > t_threshold:
        return AuditResult(
            "null_edge", "fail",
            f"DIFFERS FROM RANDOM DIRECTIONS: {mean:+.3f}R per trade against "
            f"{details['placebo_mean_r_per_trade']:+.3f}R for the same entries with coin-flip "
            f"directions (t={t_diff:.1f}). On random data the direction cannot carry "
            f"information about later prices, so the signals use the future or are "
            f"misaligned in time (e.g. entering at the open of the bar whose close produced "
            f"the signal).", details)
    if t_raw < -t_threshold:
        if t_plc is not None and t_plc < -t_threshold:
            return AuditResult(
                "null_edge", "warn",
                f"LOSS ON RANDOM DATA from the fill rules: {mean:+.3f}R per trade (t={t_raw:.1f}), "
                f"about the same as with coin-flip directions. The usual cause: bars often touch "
                f"both SL and TP, and the SL-first rule makes this backtest pessimistic; re-run "
                f"with simulate_trades_hires (1m fills) before trusting the result. (Only costs "
                f"passed as entry_costs are added back; costs subtracted from pnl_r by hand "
                f"also show up as this loss.)", details)
        return AuditResult(
            "null_edge", "fail",
            f"LOSS ON RANDOM DATA: {mean:+.3f}R per trade before costs (t={t_raw:.1f}) on "
            f"driftless random walks. Either entries are misaligned in time (e.g. entering "
            f"before the bar whose close produced the signal), or the SL-first rule biases "
            f"this backtest (bars often touch both SL and TP; then re-run with "
            f"simulate_trades_hires)." + (
                f" The same entries with coin-flip directions: "
                f"{details['placebo_mean_r_per_trade']:+.3f}R (t={t_plc:.1f}); difference "
                f"t={t_diff:.1f}. A loss the coin-flip version does not share points to "
                f"misaligned entries; if the coin-flip t is close to -{t_threshold:.1f}, re-run "
                f"with simulate_trades_hires and another seed (--seed / seed=)."
                if t_plc is not None else
                " The coin-flip comparison that tells them apart needs the backtest to use "
                "backtest_engine's simulate_trades.")
            + " (Only costs passed as entry_costs are added back; costs subtracted from "
              "pnl_r by hand also show up as a loss.)", details)
    return AuditResult(
        "null_edge", "pass",
        f"No edge on random data: {mean:+.3f}R per trade before costs (t={t_raw:.1f} across "
        f"{n_paths} paths)" + (f"; versus coin-flip directions t={t_diff:.1f}" if t_diff is not None else "")
        + ".", details)


# ── Strategy files and CLI ──────────────────────────────────────────────────

def _installed(f) -> bool:
    """True for files of installed packages (e.g. a .venv inside the folder)."""
    parts = Path(f).parts
    return "site-packages" in parts or "dist-packages" in parts


def _under(f, folder: Path) -> bool:
    try:
        Path(f).resolve().relative_to(folder)
        return True
    except (ValueError, OSError):
        return False


@contextmanager
def _strategy_module(path: Path):
    """Import a strategy file roughly as ``python path`` would: its folder is
    first on sys.path while the audit runs. Afterwards sys.path is restored
    (undoing the strategy's own edits) and modules first loaded from files
    under the folder (except installed packages) are dropped.

    This does not isolate audits from each other or from the caller; for
    that, ``audit_strategy`` runs each audit in a fresh Python process.
    """
    path = path.resolve()
    folder = path.parent
    name = f"_audit_target_{path.stem}_{abs(hash(str(path)))}"
    before = set(sys.modules)
    saved_path = list(sys.path)
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # needed by dataclasses, pickle, etc.
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.path[:] = saved_path
        for mod_name in set(sys.modules) - before:
            f = getattr(sys.modules.get(mod_name), "__file__", None)
            if mod_name == name or (f and _under(f, folder) and not _installed(f)):
                sys.modules.pop(mod_name, None)


def _audit_in_process(path: Path, data, n_cuts: int, n_paths: int, seed: int,
                      rtol: float) -> dict:
    from backtest_engine.bug_guard import (
        check_bfill_in_source, check_expanding_quantile, check_entry_price_type,
    )
    results: list[AuditResult] = []
    try:
        for chk in (check_bfill_in_source, check_expanding_quantile, check_entry_price_type):
            r = chk(path)
            verdict = "pass" if r.passed else ("fail" if r.severity == "ERROR" else "warn")
            results.append(AuditResult(f"source:{r.check_id}", verdict, r.message))
    except Exception as e:  # unreadable file, a folder, a non-UTF-8 source
        return _report(path, [AuditResult(
            "strategy_file", "error", f"could not read {path}: {type(e).__name__}: {e}")])
    try:
        with _strategy_module(path) as mod:
            try:
                bar_minutes = int(getattr(mod, "BAR_MINUTES", 60))
                vol = float(getattr(mod, "NULL_VOL", 0.002))
                null_bars = int(getattr(mod, "NULL_BARS", 5000))
                has_sig, has_bt = hasattr(mod, "signals"), hasattr(mod, "backtest")
                if has_sig:
                    d = data if data is not None else random_walk_bars(null_bars, bar_minutes, vol, seed=seed)
                    results.append(check_lookahead(mod.signals, d, n_cuts=n_cuts, seed=seed, rtol=rtol))
                if has_bt:
                    results.append(check_null_edge(
                        mod.backtest, n_paths=n_paths, n_bars=null_bars, bar_minutes=bar_minutes,
                        vol=vol, seed=seed))
                if not has_sig and not has_bt:
                    results.append(AuditResult("strategy_file", "error",
                                               "Define signals(data) and/or backtest(data) in the file."))
            except Exception as e:
                results.append(AuditResult(
                    "audit", "error", f"the audit failed: {type(e).__name__}: {e}",
                    {"traceback": traceback.format_exc(limit=5)}))
    except Exception as e:
        results.append(AuditResult(
            "strategy_file", "error", f"could not load {path.name}: {type(e).__name__}: {e}",
            {"traceback": traceback.format_exc(limit=5)}))
    return _report(path, results)


def _report(path, results: list[AuditResult], seed: Optional[int] = None) -> dict:
    order = {"fail": 0, "error": 1, "warn": 2, "pass": 3}
    overall = min((r.verdict for r in results), key=order.__getitem__)
    rep = {"file": str(path), "overall": overall, "results": [r.to_dict() for r in results]}
    if seed is not None:
        rep["seed"] = int(seed)
    return rep


def audit_strategy(path: str | Path, data=None, n_cuts: int = 32, n_paths: int = 120,
                   seed: Optional[int] = 0, rtol: float = 1e-9, isolate: bool = True,
                   timeout: Optional[float] = None) -> dict:
    """Run every applicable test on a strategy file.

    The file defines ``signals(data)`` (for the look-ahead test) and/or
    ``backtest(data)`` (for the null-data test, with the coin-flip placebo).
    Optional module constants: ``BAR_MINUTES`` (bar size of ``data``,
    default 60), ``NULL_VOL`` (per-bar volatility of the random walk relative
    to price, default 0.002) and ``NULL_BARS`` (bars per random walk and for
    the default look-ahead data, default 5000; raise it for 1m data that is
    resampled to coarser signal bars inside the strategy).

    With ``isolate=True`` (default) the audit runs in a fresh Python process
    (this copy of the library's CLI, ``backtest_engine.audit.main``), so
    strategy files and their helpers never meet another audit's modules or
    the caller's. That costs about a second per call, and ``data`` is copied
    to a temporary file. ``isolate=False`` runs in this process and reuses any
    module the caller already imported under the same name. ``seed=None``
    picks a random seed (reported in ``seed``). ``timeout`` (seconds) turns a
    hanging strategy into an ``error`` report (the child and any process it
    started are killed). On Linux/macOS, processes the strategy leaves running
    are also killed 2 s after the audit ends.

    The audit is not a sandbox: strategy code runs with your permissions and
    could fake its own report. Audit code you are willing to run.
    """
    path = Path(path)
    if seed is None:
        seed = int(np.random.default_rng().integers(2**31))
    if data is not None:
        try:
            data = _as_data(data)
        except (ValueError, TypeError, IndexError) as e:
            return _report(path, [AuditResult("data", "error", f"bad data: {e}")], seed)
    if not isolate:
        rep = _audit_in_process(path, data, int(n_cuts), int(n_paths), int(seed), float(rtol))
        rep["seed"] = int(seed)
        return rep
    import os
    import pickle
    import signal
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix="bt_audit_") as tmp:  # mode 0700
        out = Path(tmp) / "report.json"
        args = [str(path.resolve()), "--json-out", str(out), "--cuts", str(int(n_cuts)),
                "--paths", str(int(n_paths)), "--seed", str(int(seed)), "--rtol", repr(float(rtol))]
        if data is not None:
            pkl = Path(tmp) / "data.pkl"
            try:
                with open(pkl, "wb") as fh:
                    pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                return _report(path, [AuditResult(
                    "data", "error",
                    f"data could not be passed to the audit process ({type(e).__name__}: {e}); "
                    f"use plain numpy columns, or isolate=False")], seed)
            args += ["--data-pkl", str(pkl)]
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        # Empty PYTHONPATH entries (usually accidental, e.g. ":/x") would put
        # the cwd back, so they are dropped; relative entries (PYTHONPATH=src,
        # or ".") are kept, made absolute against the cwd as Python would.
        if env.get("PYTHONPATH"):
            keep = [os.path.abspath(x) for x in env["PYTHONPATH"].split(os.pathsep) if x]
            if keep:
                env["PYTHONPATH"] = os.pathsep.join(keep)
            else:
                env.pop("PYTHONPATH")
        # Run this copy of the library: load the package from its real path
        # (so numba's on-disk cache, keyed by file path, is reused), then run
        # the CLI. -P keeps the current folder off the child's sys.path (a
        # backtest_engine/ or json.py there would be imported instead).
        # Worker processes a strategy spawns import from sys.path as usual.
        pkg = Path(__file__).resolve().parent
        boot = (
            "import sys, importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('backtest_engine', {str(pkg / '__init__.py')!r}, "
            f"submodule_search_locations=[{str(pkg)!r}])\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "sys.modules['backtest_engine'] = mod\n"
            "spec.loader.exec_module(mod)\n"
            "from backtest_engine.audit import main\n"
            "sys.exit(main(sys.argv[1:]))\n"
        )
        cmd = [sys.executable, "-P", "-c", boot] + args
        err_path = Path(tmp) / "stderr.txt"
        popen_kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                    if os.name == "nt" else {"start_new_session": True})

        def kill_group() -> None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)  # the child and anything it started
                except OSError:
                    pass

        def stderr_tail() -> str:
            try:
                return err_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            except OSError:
                return ""

        with open(err_path, "w", encoding="utf-8") as err_fh:
            # No pipes: a strategy that keeps them open cannot make us wait.
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=err_fh, env=env, **popen_kw)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                kill_group()
                proc.wait()
                return _report(path, [AuditResult(
                    "strategy_file", "error", f"the audit did not finish within {timeout} s",
                    {"stderr": stderr_tail()})], seed)
            except BaseException:  # Ctrl-C, SystemExit, ...: do not leave the child running
                kill_group()
                proc.wait()
                raise
            if os.name != "nt":
                # Processes the strategy started and left running: give them
                # (e.g. multiprocessing's resource tracker, which frees leaked
                # shared memory) up to 2 s to finish, then kill the rest.
                # POSIX only: on Windows the exited child's tree cannot be
                # found reliably.
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    try:
                        os.killpg(proc.pid, 0)
                    except OSError:
                        break
                    time.sleep(0.05)
                else:
                    kill_group()
        stderr = stderr_tail()
        if out.exists():
            try:
                rep = json.loads(out.read_text(encoding="utf-8"))
                rep["file"] = str(path)
                rep["seed"] = int(seed)
                return rep
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    return _report(path, [AuditResult(
        "strategy_file", "error",
        f"the audit process exited with code {proc.returncode} without a report",
        {"stderr": (stderr or "")[-2000:]})], seed)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backtest_engine.audit",
                                 description="Audit a strategy file for look-ahead and null-data edge.")
    ap.add_argument("strategy", help="Python file defining signals(data) and/or backtest(data)")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    ap.add_argument("--csv", help="OHLC(V) CSV for the look-ahead test (default: random walk)")
    ap.add_argument("--cuts", type=int, default=32)
    ap.add_argument("--paths", type=int, default=120,
                    help="random walks for null_edge (default 120)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=1e-9,
                    help="relative tolerance for float signals (default 1e-9)")
    ap.add_argument("--json-out", help=argparse.SUPPRESS)   # used by audit_strategy
    ap.add_argument("--data-pkl", help=argparse.SUPPRESS)   # used by audit_strategy
    a = ap.parse_args(argv)
    data = None
    if a.csv:
        from backtest_engine.utils import load_ohlcv
        data = load_ohlcv(a.csv)
    report = None
    if a.data_pkl:
        import pickle
        try:
            with open(a.data_pkl, "rb") as fh:  # written by audit_strategy in a private temp dir
                data = pickle.load(fh)
        except Exception as e:
            report = _report(Path(a.strategy), [AuditResult(
                "data", "error",
                f"data could not be read in the audit process ({type(e).__name__}: {e}); "
                f"use plain numpy columns, or isolate=False")])
    if report is None:
        report = _audit_in_process(Path(a.strategy), data, a.cuts, a.paths, a.seed, a.rtol)
    report["seed"] = a.seed
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(report), encoding="utf-8")  # ASCII-escaped
    if a.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif not a.json_out:
        print(f"Audit: {report['file']}  ->  {report['overall'].upper()}")
        for r in report["results"]:
            print(f"  [{r['verdict'].upper():>5}] {r['check']}: {r['summary']}")
    return {"pass": 0, "warn": 0, "fail": 1, "error": 2}[report["overall"]]


if __name__ == "__main__":
    sys.exit(main())
