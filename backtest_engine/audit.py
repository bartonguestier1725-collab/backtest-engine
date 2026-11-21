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
   from the SL-first rule when a bar touches both SL and TP. Optionally the
   same trades are run with coin-flip directions (a placebo with the same
   entry bars, exits and sizing), which cancels that bias.

A strategy file for the CLI defines ``signals(data)`` and/or
``backtest(data)``; ``data`` is a dict of numpy arrays with keys
``time, open, high, low, close`` (and ``volume`` if available). ``backtest``
should get its signals by calling ``signals(data)``, so the audit can swap in
the placebo::

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
    if isinstance(data, dict):
        d = {k: np.asarray(v) for k, v in data.items()}
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
    seed: int = 0,
) -> dict:
    """Driftless random-walk OHLC bars (plus a random volume column).

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
    near = min(h["cut"] - h["latest_changed"] for h in hits)
    far = max(h["cut"] - h["earliest_changed"] for h in hits)
    if far <= 2:
        return "the signal uses the next bar(s)"
    if near <= 2 and far <= 60:
        return f"a window reaching up to {far} bars into the future (e.g. a centered window or a pivot)"
    return (f"changes reach up to {far} bars before the cut: a statistic, filter or model "
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
    atol = 1e-12 * float(np.nanmax(np.abs(base))) if np.isfinite(base).any() else 0.0
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
                    "reach_bars": int(cut - idx.min()),
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


def _t(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    if sd > 0:
        return float(x.mean() / (sd / np.sqrt(len(x))))
    return 0.0 if x.mean() == 0 else float(np.copysign(np.inf, x.mean()))


def check_null_edge(
    backtest_fn: Callable[[dict], Any],
    n_paths: int = 20,
    n_bars: int = 5000,
    bar_minutes: int = 60,
    vol: float = 0.002,
    seed: int = 0,
    t_threshold: float = 3.0,
    placebo_fn: Optional[Callable[[dict, int], Any]] = None,
    n_placebo: int = 2,
) -> AuditResult:
    """Null-data test: the full backtest on driftless random walks.

    Parameters
    ----------
    backtest_fn : ``backtest_fn(data) -> TradeResults`` (or any array with a
                  ``pnl_r`` field, or a plain array of per-trade R). Runs the
                  whole pipeline: signals, simulate_trades, costs. Costs are
                  added back (``cost_r``) before testing.
    n_paths : Independent random walks (at least 10). The t-statistics are
              computed across paths, since trades within a path overlap.
    placebo_fn : Optional ``placebo_fn(data, k) -> trade results``: the same
                 backtest with the same entry bars but random directions
                 (k = 0, 1, ... selects the coin flips). ``audit_strategy``
                 builds it from the file's ``signals``. It has the same
                 entries, exits and sizing, so the difference to it cancels
                 fill-rule biases.
    t_threshold : |t| above which a result is flagged.

    Verdicts
    --------
    fail : the strategy makes money on random data (look-ahead somewhere), or
           differs from its own placebo (entries carry information about
           later prices, or are misaligned in time), or, without a placebo,
           loses clearly more than chance.
    warn : it loses on random data but no more than its placebo: the SL-first
           rule (a bar touching both SL and TP counts as SL) biases this
           backtest; resolve the order with simulate_trades_hires.
    """
    if n_paths < _MIN_PATHS:
        return AuditResult("null_edge", "error", f"n_paths must be >= {_MIN_PATHS}")
    m, p, n_trades, all_pnl = [], [], 0, []
    t0 = time.time()
    for k in range(n_paths):
        d = random_walk_bars(n_bars, bar_minutes, vol, seed=seed + k)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pnl = _gross_pnl(backtest_fn(d))
                if placebo_fn is not None:
                    pl = [_gross_pnl(placebo_fn(d, k * n_placebo + j)) for j in range(n_placebo)]
        except Exception as e:
            return AuditResult("null_edge", "error", f"backtest function raised {type(e).__name__}: {e}")
        n_trades += len(pnl)
        all_pnl.append(pnl)
        m.append(float(pnl.mean()) if len(pnl) else np.nan)
        if placebo_fn is not None:
            pv = np.concatenate(pl) if pl else np.array([])
            p.append(float(pv.mean()) if len(pv) else np.nan)
    m = np.array(m)
    ok = ~np.isnan(m)
    if placebo_fn is not None:
        p = np.array(p)
        ok &= ~np.isnan(p)
    if ok.sum() < _MIN_PATHS or n_trades < 30:
        return AuditResult("null_edge", "error",
                           f"Too few trades on null data ({n_trades} in {n_paths} paths) to test. "
                           f"Raise NULL_BARS or use a looser entry rule for the test.")
    t_raw = _t(m[ok])
    x = np.concatenate(all_pnl)
    gp, gl = float(x[x > 0].sum()), float(-x[x < 0].sum())
    details = {
        "n_paths": n_paths, "n_bars": n_bars, "n_trades": int(n_trades),
        "mean_r_per_path": round(float(m[ok].mean()), 4),
        "t_across_paths": round(t_raw, 2),
        "gross_profit_factor": round(gp / gl, 3) if gl > 0 else None,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    t_diff = None
    if placebo_fn is not None:
        diff = m[ok] - p[ok]
        t_diff = _t(diff)
        details.update({"placebo_mean_r_per_path": round(float(p[ok].mean()), 4),
                        "mean_minus_placebo": round(float(diff.mean()), 4),
                        "t_vs_placebo": round(t_diff, 2)})
    mean = float(m[ok].mean())
    if t_raw > t_threshold:
        return AuditResult(
            "null_edge", "fail",
            f"EDGE ON RANDOM DATA: {mean:+.3f}R per trade before costs (t={t_raw:.1f} across "
            f"{ok.sum()} paths) on driftless random walks, where no strategy can have an "
            f"edge. The pipeline sees the future (signal look-ahead, or entries/exits/"
            f"targets that use bars after the decision).", details)
    if t_diff is not None and abs(t_diff) > t_threshold:
        return AuditResult(
            "null_edge", "fail",
            f"DIFFERS FROM RANDOM DIRECTIONS: {details['mean_minus_placebo']:+.3f}R per trade "
            f"versus the same trades with coin-flip directions (t={t_diff:.1f}). On random "
            f"data the direction cannot carry information about later prices, so the "
            f"signals use the future or are misaligned in time (e.g. entering at the open "
            f"of the bar whose close produced the signal).", details)
    if t_raw < -t_threshold:
        if t_diff is not None:
            return AuditResult(
                "null_edge", "warn",
                f"LOSS ON RANDOM DATA from the fill rules: {mean:+.3f}R per trade (t={t_raw:.1f}), "
                f"the same as with coin-flip directions. Bars often touch both SL and TP, and the "
                f"SL-first rule makes this backtest pessimistic. Re-run with "
                f"simulate_trades_hires (1m fills) before trusting the result.", details)
        return AuditResult(
            "null_edge", "fail",
            f"LOSS ON RANDOM DATA: {mean:+.3f}R per trade before costs (t={t_raw:.1f}) on "
            f"driftless random walks. Either entries are misaligned in time (e.g. entering "
            f"before the bar whose close produced the signal), or the SL-first rule biases "
            f"this backtest (bars often touch both SL and TP; then re-run with "
            f"simulate_trades_hires). Define signals() and call it from backtest() so the "
            f"audit can tell the two apart.", details)
    return AuditResult(
        "null_edge", "pass",
        f"No edge on random data: {mean:+.3f}R per trade before costs (t={t_raw:.1f} across "
        f"{ok.sum()} paths)" + (f"; versus coin-flip directions t={t_diff:.1f}" if t_diff is not None else "")
        + ".", details)


# ── Strategy files and CLI ──────────────────────────────────────────────────

@contextmanager
def _strategy_module(path: Path):
    """Import a strategy file as if run with ``python path``: its folder is on
    sys.path for the whole audit (lazy imports inside functions work), and
    afterwards every module loaded from that folder is dropped, so a later
    audit of another folder does not reuse a same-named helper."""
    path = path.resolve()
    folder = path.parent
    name = f"_audit_target_{path.stem}_{abs(hash(str(path)))}"
    before = set(sys.modules)
    # A same-named helper cached from another (user) folder would shadow the
    # one next to this file. Installed packages and the stdlib are left alone.
    protected = tuple(str(Path(x).resolve()) for x in
                      {sys.prefix, sys.base_prefix, sys.exec_prefix, Path(__file__).parent})
    for mod_name in list(sys.modules):
        if "." in mod_name or not (folder / f"{mod_name}.py").exists():
            continue
        f = getattr(sys.modules[mod_name], "__file__", None)
        if f and Path(f).resolve().parent != folder and not str(Path(f).resolve()).startswith(protected):
            del sys.modules[mod_name]
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
        try:
            sys.path.remove(str(folder))
        except ValueError:
            pass
        for mod_name in set(sys.modules) - before:
            f = getattr(sys.modules.get(mod_name), "__file__", None)
            if mod_name == name or (f and Path(f).resolve().parent == folder):
                sys.modules.pop(mod_name, None)


def _make_placebo(mod) -> Callable[[dict, int], Any]:
    """The file's backtest with the same entry bars but each signal's
    direction set by a coin flip. Works when backtest() gets its signals by
    calling the module-level signals().

    The entry bars and exits stay as they are, so fill-rule biases (SL-first)
    are the same; the directions carry no information about prices. (Shuffling
    the signals across bars would not be a fair placebo: a direction moved to
    an earlier bar was decided with prices that come after it.)"""
    real = mod.signals

    def placebo(d: dict, k: int):
        out = real(d)
        rng = np.random.default_rng(1_000_003 + k)
        if isinstance(out, tuple):
            dirs = np.asarray(out[1])
            fake = (out[0], dirs * rng.choice(np.array([-1, 1], dtype=np.int8), len(dirs)), *out[2:])
        else:
            arr = np.asarray(out)
            if arr.dtype == np.bool_:
                arr = arr.astype(np.int8)
            fake = arr * rng.choice(np.array([-1, 1], dtype=np.int8), len(arr))
        mod.signals = lambda _d: fake
        try:
            return mod.backtest(d)
        finally:
            mod.signals = real
    return placebo


def audit_strategy(path: str | Path, data=None, n_cuts: int = 32, n_paths: int = 20,
                   seed: int = 0, rtol: float = 1e-9) -> dict:
    """Run every applicable test on a strategy file.

    The file defines ``signals(data)`` (for the look-ahead test) and/or
    ``backtest(data)`` (for the null-data test; if it calls ``signals(data)``
    the placebo comparison is added). Optional module constants:
    ``BAR_MINUTES`` (bar size of ``data``, default 60), ``NULL_VOL``
    (per-bar volatility of the random walk relative to price, default 0.002)
    and ``NULL_BARS`` (bars per random walk and for the default look-ahead
    data, default 5000; raise it for 1m data that is resampled to coarser
    signal bars inside the strategy).
    """
    from backtest_engine.bug_guard import (
        check_bfill_in_source, check_expanding_quantile, check_entry_price_type,
    )
    path = Path(path)
    results: list[AuditResult] = []
    for chk in (check_bfill_in_source, check_expanding_quantile, check_entry_price_type):
        r = chk(path)
        verdict = "pass" if r.passed else ("fail" if r.severity == "ERROR" else "warn")
        results.append(AuditResult(f"source:{r.check_id}", verdict, r.message))
    try:
        with _strategy_module(path) as mod:
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
                    vol=vol, seed=seed, placebo_fn=_make_placebo(mod) if has_sig else None))
            if not has_sig and not has_bt:
                results.append(AuditResult("strategy_file", "error",
                                           "Define signals(data) and/or backtest(data) in the file."))
    except Exception as e:
        results.append(AuditResult(
            "strategy_file", "error", f"could not load {path.name}: {type(e).__name__}: {e}",
            {"traceback": traceback.format_exc(limit=5)}))
    order = {"fail": 0, "error": 1, "warn": 2, "pass": 3}
    overall = min((r.verdict for r in results), key=order.__getitem__)
    return {"file": str(path), "overall": overall, "results": [r.to_dict() for r in results]}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backtest_engine.audit",
                                 description="Audit a strategy file for look-ahead and null-data edge.")
    ap.add_argument("strategy", help="Python file defining signals(data) and/or backtest(data)")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    ap.add_argument("--csv", help="OHLC(V) CSV for the look-ahead test (default: random walk)")
    ap.add_argument("--cuts", type=int, default=32)
    ap.add_argument("--paths", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=1e-9,
                    help="relative tolerance for float signals (default 1e-9)")
    a = ap.parse_args(argv)
    data = None
    if a.csv:
        from backtest_engine.utils import load_ohlcv
        data = load_ohlcv(a.csv)
    report = audit_strategy(a.strategy, data=data, n_cuts=a.cuts, n_paths=a.paths,
                            seed=a.seed, rtol=a.rtol)
    if a.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Audit: {report['file']}  ->  {report['overall'].upper()}")
        for r in report["results"]:
            print(f"  [{r['verdict'].upper():>5}] {r['check']}: {r['summary']}")
    return {"pass": 0, "warn": 0, "fail": 1, "error": 2}[report["overall"]]


if __name__ == "__main__":
    sys.exit(main())
