"""Strategy audit: dynamic checks that BugGuard's pattern checks cannot do.

Two tests, both independent of what the strategy code looks like:

1. **Look-ahead test** (``check_lookahead``) — rewrite every bar after a cut
   point and re-run the signal function. A causal signal at a bar at or before
   the cut cannot change when only later bars change. Four rewrites are used
   (mirror around the cut close, 2x deviations, +25% jump, -20% jump), because
   some leaks survive one of them (a full-period volatility quantile does not
   change when the future is mirrored).

2. **Null-data test** (``check_null_edge``) — run the whole backtest on
   driftless random walks. Bars are cut from a finer random path, so stops and
   targets fill at fair prices and the expected PnL is about zero (slightly
   negative, because a bar touching both SL and TP counts as SL). A
   significantly positive mean means the pipeline sees the future; a
   significantly negative one usually means entries are misaligned in time
   (e.g. entering at the open of the bar whose close produced the signal).

A strategy file for the CLI defines ``signals(data)`` and/or
``backtest(data)``; ``data`` is a dict of numpy arrays with keys
``time, open, high, low, close``::

    python -m backtest_engine.audit my_strategy.py            # human-readable
    python -m backtest_engine.audit my_strategy.py --json     # for AI agents

Passing both tests does not prove a strategy is free of bias (a leak that
never changes a signal, or survivorship in the instrument list, is invisible
to them). Failing either one is strong evidence of a bug.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
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
    """Accept a dict with open/high/low/close (time optional) or the 6-tuple
    returned by fetch_aggvault / load_ohlcv."""
    if isinstance(data, dict):
        d = {k: np.asarray(v) for k, v in data.items()}
    else:
        ts, o, h, l, c = data[:5]
        d = {"time": np.asarray(ts), "open": np.asarray(o), "high": np.asarray(h),
             "low": np.asarray(l), "close": np.asarray(c)}
    for k in ("open", "high", "low", "close"):
        if k not in d:
            raise ValueError(f"data needs '{k}'")
        d[k] = np.asarray(d[k], dtype=np.float64)
    n = len(d["close"])
    if "time" not in d:
        d["time"] = np.arange(n, dtype=np.int64) * 60
    if any(len(d[k]) != n for k in ("open", "high", "low", "close", "time")):
        raise ValueError("time/open/high/low/close must have the same length")
    return d


def random_walk_bars(
    n_bars: int = 5000,
    bar_minutes: int = 60,
    vol: float = 0.002,
    sub_steps: int = 20,
    start_price: float = 100.0,
    seed: int = 0,
) -> dict:
    """Driftless random-walk OHLC bars cut from a finer path.

    Each bar opens at the previous close; high/low are the extremes of
    ``sub_steps`` log-normal steps, so a level inside [low, high] really was
    traded. ``vol`` is the per-bar log-return standard deviation.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, vol / np.sqrt(sub_steps), n_bars * sub_steps)
    path = start_price * np.exp(np.cumsum(steps)).reshape(n_bars, sub_steps)
    o = np.r_[start_price, path[:-1, -1]]
    c = path[:, -1].copy()
    h = np.maximum(path.max(axis=1), o)
    l = np.minimum(path.min(axis=1), o)
    t = np.arange(n_bars, dtype=np.int64) * bar_minutes * 60 + 1_704_067_200
    return {"time": t, "open": o, "high": h, "low": l, "close": c}


def _rewrite_future(d: dict, cut: int, mode: str) -> dict:
    out = {k: v.copy() for k, v in d.items()}
    p = d["close"][cut]
    f = slice(cut + 1, None)
    o, h, l, c = (d[k][f] for k in ("open", "high", "low", "close"))
    if mode == "mirror":
        o2, h2, l2, c2 = 2 * p - o, 2 * p - l, 2 * p - h, 2 * p - c
    elif mode == "scale2":
        o2, h2, l2, c2 = (p + 2 * (x - p) for x in (o, h, l, c))
    elif mode in ("jump_up", "jump_down"):
        k = 1.25 if mode == "jump_up" else 0.8
        o2, h2, l2, c2 = o * k, h * k, l * k, c * k
    else:
        raise ValueError(mode)
    # Mirroring/scaling can push far-away prices to <= 0; keep them positive.
    floor = 1e-6 * abs(p)
    out["open"][f], out["high"][f], out["low"][f], out["close"][f] = (
        np.maximum(x, floor) for x in (o2, h2, l2, c2))
    return out


def _per_bar(sig, n: int) -> np.ndarray:
    """Signal output -> float array per bar. Accepts an array of length n, or
    a tuple whose first two items are (signal_bars, directions)."""
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


def _same(a: np.ndarray, b: np.ndarray, rtol: float) -> np.ndarray:
    eq = (a == b) | (np.isnan(a) & np.isnan(b))
    if rtol > 0:
        eq |= np.isclose(a, b, rtol=rtol, atol=0.0)
    return eq


# ── Test 1: look-ahead ──────────────────────────────────────────────────────

def check_lookahead(
    signal_fn: Callable[[dict], Any],
    data=None,
    n_cuts: int = 32,
    warmup: Optional[int] = None,
    seed: int = 0,
    rtol: float = 0.0,
) -> AuditResult:
    """Look-ahead test: do signals at or before a cut change when only bars
    after the cut change?

    Parameters
    ----------
    signal_fn : ``signal_fn(data) -> array (one value per bar) or
                (signal_bars, directions)``. ``data`` is a dict of numpy arrays
                (time/open/high/low/close). Must not modify ``data``.
    data : Bars to test on (dict or 6-tuple). Default: 5,000 random-walk bars.
           Real data is better when the signal depends on real market
           structure (sessions, volatility regimes).
    n_cuts : Number of cut points (spread over the data after ``warmup``).
             More cuts catch rarer leaks; each cut costs 4 signal runs.
    warmup : First bar a cut may be placed at (default: 10% of the bars).
    rtol : Relative tolerance when comparing float signals (default exact).
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
    if not np.all(_same(base, again, rtol)):
        return AuditResult(
            "lookahead", "error",
            "Signal function is not deterministic (two runs on the same data differ). "
            "Fix the random seed before testing.")
    rng = np.random.default_rng(seed)
    pool = np.arange(warmup, n - 1)
    cuts = np.sort(rng.choice(pool, size=min(n_cuts, len(pool)), replace=False))
    hits = []
    for cut in cuts:
        for mode in _REWRITES:
            try:
                pert = _per_bar(signal_fn(_rewrite_future(d, int(cut), mode)), n)
            except Exception as e:
                return AuditResult("lookahead", "error",
                                   f"signal function raised {type(e).__name__} on rewritten data: {e}")
            diff = ~_same(base[: cut + 1], pert[: cut + 1], rtol)
            if diff.any():
                idx = np.flatnonzero(diff)
                hits.append({
                    "cut": int(cut), "rewrite": mode, "bars_changed": int(len(idx)),
                    "earliest_changed": int(idx.min()), "latest_changed": int(idx.max()),
                    "reach_bars": int(cut - idx.min()),
                })
    cut_hit = sorted({h["cut"] for h in hits})
    details = {
        "n_bars": n, "n_cuts": int(len(cuts)), "cuts_with_change": len(cut_hit),
        "rewrites": list(_REWRITES), "elapsed_sec": round(time.time() - t0, 2),
        "examples": hits[:5],
    }
    if not hits:
        return AuditResult(
            "lookahead", "pass",
            f"No signal at or before a cut changed when later bars were rewritten "
            f"({len(cuts)} cuts x {len(_REWRITES)} rewrites).", details)
    reach = max(h["reach_bars"] for h in hits)
    first = hits[0]
    return AuditResult(
        "lookahead", "fail",
        f"LOOK-AHEAD: signals changed after rewriting only later bars at "
        f"{len(cut_hit)}/{len(cuts)} cuts. Example: cut at bar {first['cut']} "
        f"({first['rewrite']}) changed {first['bars_changed']} earlier signal(s), "
        f"up to {first['reach_bars']} bars before the cut. Largest reach: {reach} bars "
        f"({'uses the next bar(s)' if reach <= 2 else 'a window or statistic over future bars'}).",
        details)


# ── Test 2: null data ───────────────────────────────────────────────────────

def check_null_edge(
    backtest_fn: Callable[[dict], Any],
    n_paths: int = 20,
    n_bars: int = 5000,
    bar_minutes: int = 60,
    vol: float = 0.002,
    seed: int = 0,
    t_threshold: float = 3.0,
) -> AuditResult:
    """Null-data test: the full backtest on driftless random walks.

    Parameters
    ----------
    backtest_fn : ``backtest_fn(data) -> TradeResults`` (or any array with a
                  ``pnl_r`` field, or a plain array of per-trade R). Runs the
                  whole pipeline: signals, simulate_trades, costs.
    n_paths : Independent random walks. The t-statistic is computed across
              paths (trades within a path overlap and are correlated).
    t_threshold : |t| above which the verdict is "fail" (positive) or
                  "warn" (negative).
    """
    path_means, n_trades, all_pnl = [], 0, []
    t0 = time.time()
    for k in range(n_paths):
        d = random_walk_bars(n_bars, bar_minutes, vol, seed=seed + k)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = backtest_fn(d)
        except Exception as e:
            return AuditResult("null_edge", "error", f"backtest function raised {type(e).__name__}: {e}")
        arr = np.asarray(r)
        if arr.dtype.names and "pnl_r" in arr.dtype.names:
            pnl = np.asarray(arr["pnl_r"], dtype=np.float64)
            if "exit_type" in arr.dtype.names:
                pnl = pnl[np.asarray(arr["exit_type"]) != EXIT_NO_FILL]
        else:
            pnl = np.asarray(arr, dtype=np.float64).ravel()
        n_trades += len(pnl)
        all_pnl.append(pnl)
        path_means.append(float(pnl.mean()) if len(pnl) else np.nan)
    m = np.array([x for x in path_means if not np.isnan(x)])
    if len(m) < 3 or n_trades < 30:
        return AuditResult("null_edge", "error",
                           f"Too few trades on null data ({n_trades} in {n_paths} paths) to test.")
    sd = m.std(ddof=1)
    if sd > 0:
        t = float(m.mean() / (sd / np.sqrt(len(m))))
    else:  # identical path means
        t = 0.0 if m.mean() == 0 else float(np.copysign(np.inf, m.mean()))
    x = np.concatenate(all_pnl)
    gp, gl = float(x[x > 0].sum()), float(-x[x < 0].sum())
    details = {
        "n_paths": n_paths, "n_bars": n_bars, "n_trades": int(n_trades),
        "mean_r": round(float(x.mean()), 4), "t_across_paths": round(t, 2),
        "profit_factor": round(gp / gl, 3) if gl > 0 else None,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    if t > t_threshold:
        return AuditResult(
            "null_edge", "fail",
            f"EDGE ON RANDOM DATA: mean {x.mean():+.3f}R per trade (t={t:.1f} across "
            f"{len(m)} paths, PF {details['profit_factor']}) on driftless random walks, "
            f"where no strategy can have an edge. The pipeline sees the future "
            f"(signal look-ahead, or entries/exits placed before the bar that "
            f"triggers them).", details)
    if t < -t_threshold:
        return AuditResult(
            "null_edge", "warn",
            f"LOSS ON RANDOM DATA: mean {x.mean():+.3f}R per trade (t={t:.1f}) on driftless "
            f"random walks, larger than costs/fill rules explain. Entries are often "
            f"misaligned in time (e.g. entering at the open of the bar whose close "
            f"produced the signal), which flips the sign of a real edge too.", details)
    return AuditResult(
        "null_edge", "pass",
        f"No edge on random data: mean {x.mean():+.3f}R per trade (t={t:.1f} across "
        f"{len(m)} paths).", details)


# ── Strategy files and CLI ──────────────────────────────────────────────────

def _load_module(path: Path):
    """Import a strategy file. Its directory is put on sys.path while it
    loads, as when running ``python path``, so it can import its helpers."""
    spec = importlib.util.spec_from_file_location(f"_audit_target_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    folder = str(path.resolve().parent)
    sys.path.insert(0, folder)
    try:
        spec.loader.exec_module(mod)
    finally:
        try:
            sys.path.remove(folder)
        except ValueError:
            pass
    return mod


def audit_strategy(path: str | Path, data=None, n_cuts: int = 32, n_paths: int = 20,
                   seed: int = 0) -> dict:
    """Run every applicable test on a strategy file.

    The file defines ``signals(data)`` (for the look-ahead test) and/or
    ``backtest(data)`` (for the null-data test). Optional module constants:
    ``BAR_MINUTES`` (bar size of ``data``, default 60), ``NULL_VOL``
    (per-bar volatility of the random walk, default 0.002) and ``NULL_BARS``
    (bars per random walk, default 5000; raise it for 1m data that is
    resampled to coarser signal bars inside the strategy).
    """
    from backtest_engine.bug_guard import (
        check_bfill_in_source, check_expanding_quantile, check_entry_price_type,
    )
    path = Path(path)
    mod = _load_module(path)
    results: list[AuditResult] = []
    for chk in (check_bfill_in_source, check_expanding_quantile, check_entry_price_type):
        r = chk(path)
        verdict = "pass" if r.passed else ("fail" if r.severity == "ERROR" else "warn")
        results.append(AuditResult(f"source:{r.check_id}", verdict, r.message))
    bar_minutes = int(getattr(mod, "BAR_MINUTES", 60))
    vol = float(getattr(mod, "NULL_VOL", 0.002))
    null_bars = int(getattr(mod, "NULL_BARS", 5000))
    if hasattr(mod, "signals"):
        d = data if data is not None else random_walk_bars(5000, bar_minutes, vol, seed=seed)
        results.append(check_lookahead(mod.signals, d, n_cuts=n_cuts, seed=seed))
    if hasattr(mod, "backtest"):
        results.append(check_null_edge(mod.backtest, n_paths=n_paths, n_bars=null_bars,
                                       bar_minutes=bar_minutes, vol=vol, seed=seed))
    if not hasattr(mod, "signals") and not hasattr(mod, "backtest"):
        results.append(AuditResult("strategy_file", "error",
                                   "Define signals(data) and/or backtest(data) in the file."))
    order = {"fail": 0, "error": 1, "warn": 2, "pass": 3}
    overall = min((r.verdict for r in results), key=order.__getitem__)
    return {"file": str(path), "overall": overall, "results": [r.to_dict() for r in results]}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backtest_engine.audit",
                                 description="Audit a strategy file for look-ahead and null-data edge.")
    ap.add_argument("strategy", help="Python file defining signals(data) and/or backtest(data)")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    ap.add_argument("--csv", help="OHLC CSV for the look-ahead test (default: random walk)")
    ap.add_argument("--cuts", type=int, default=32)
    ap.add_argument("--paths", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    data = None
    if a.csv:
        from backtest_engine.utils import load_ohlcv
        data = load_ohlcv(a.csv)
    report = audit_strategy(a.strategy, data=data, n_cuts=a.cuts, n_paths=a.paths, seed=a.seed)
    if a.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Audit: {report['file']}  ->  {report['overall'].upper()}")
        for r in report["results"]:
            print(f"  [{r['verdict'].upper():>5}] {r['check']}: {r['summary']}")
    return {"pass": 0, "warn": 0, "fail": 1, "error": 2}[report["overall"]]


if __name__ == "__main__":
    sys.exit(main())
