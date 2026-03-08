"""GateKeeper — Fast-kill system for trading strategy evaluation.

Kill bad strategies in 5 minutes, not 7 hours.

Gate 0: Data validation (1 min)    — Do we have enough data? Correct costs?
Gate 1: Quick feasibility (5 min)  — Run 10 combos on full data. PF < 1.05 → dead.
Gate 2: Coarse screen (20 min)     — Run 100 combos. PF < 1.15 → dead.
Gate 3: Full grid + WFA (1 hour)   — 1000+ combos. WFA on top 10. PBO > 0.40 → dead.
Gate 4: MC + Fundora check (30 min) — Monte Carlo + prop firm rules.

Usage:
    from backtest_engine import GateKeeper

    gk = GateKeeper(
        strategy_name="Currency Strength v6",
        n_bars=7022,
        bar_minutes=60,
        resolution_minutes=1,
        spreads_used=SPREADS,
        source_path=__file__,
    )

    # Gate 0: auto-runs BugGuard
    gk.gate0_validate()

    # Gate 1: provide a function that takes param_dict → metric_dict
    gk.gate1_quick(run_func, quick_params)

    # Gate 2: expanded grid
    gk.gate2_screen(run_func, screen_params)

    # Gate 3: full grid + WFA
    gk.gate3_wfa(run_func, full_params, data_array)

    # Gate 4: Monte Carlo
    gk.gate4_mc(best_trades, balance=50000)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from backtest_engine.bug_guard import run_all_checks, GuardReport


@dataclass
class GateResult:
    """Result of a single gate."""
    gate: str
    passed: bool
    message: str
    best_metric: Optional[dict] = None
    elapsed_sec: float = 0.0


class GateKeeper:
    """Fast-kill strategy evaluator.

    Parameters
    ----------
    strategy_name : Human-readable name for logging.
    n_bars : Total bars in dataset.
    bar_minutes : Bar size in minutes (for data period check).
    resolution_minutes : Execution simulation resolution.
    spreads_used : Cost dict for BugGuard validation.
    source_path : Script path for source code checks.
    broker : Broker name for cost registry.
    """

    # Gate thresholds
    GATE1_MIN_PF = 1.05
    GATE1_MIN_TRADES = 30
    GATE2_MIN_PF = 1.10
    GATE2_MIN_RF = 1.5
    GATE2_MIN_TRADES = 50
    GATE3_MAX_PBO = 0.40
    GATE3_MIN_WFA_WINRATE = 0.55
    GATE4_MIN_MC_PASS = 0.70

    def __init__(
        self,
        strategy_name: str,
        n_bars: int = 0,
        bar_minutes: int = 60,
        resolution_minutes: int = 1,
        spreads_used: Optional[dict[str, float]] = None,
        source_path: Optional[str | Path] = None,
        broker: str = "fundora",
    ):
        self.strategy_name = strategy_name
        self.n_bars = n_bars
        self.bar_minutes = bar_minutes
        self.resolution_minutes = resolution_minutes
        self.spreads_used = spreads_used
        self.source_path = source_path
        self.broker = broker
        self.gates: list[GateResult] = []

    def _header(self, gate: str, desc: str):
        print(f"\n{'='*70}")
        print(f"[{gate}] {self.strategy_name} — {desc}")
        print(f"{'='*70}")

    def _footer(self, gate: str, result: GateResult):
        status = "PASS ✓" if result.passed else "KILLED ✗"
        print(f"\n[{gate}] {status} ({result.elapsed_sec:.1f}s) — {result.message}")
        if not result.passed:
            print(f"\n{'!'*70}")
            print(f"  STRATEGY KILLED AT {gate}. Do NOT proceed.")
            print(f"  Move on to the next strategy idea.")
            print(f"{'!'*70}")

    def gate0_validate(self) -> GateResult:
        """Gate 0: Data validation + BugGuard.

        Checks all known bug patterns. Fails if any ERROR-level check fails.
        """
        self._header("Gate 0", "Data Validation + BugGuard")
        t0 = time.time()

        report = run_all_checks(
            source_path=self.source_path,
            spreads_used=self.spreads_used,
            broker=self.broker,
            resolution_minutes=self.resolution_minutes,
            n_bars=self.n_bars,
            bar_minutes=self.bar_minutes,
            strict=False,  # We handle the result ourselves
        )

        elapsed = time.time() - t0
        result = GateResult(
            gate="Gate 0",
            passed=report.passed,
            message=f"{report.n_errors} errors, {report.n_warnings} warnings",
            elapsed_sec=elapsed,
        )
        self.gates.append(result)
        self._footer("Gate 0", result)

        if not result.passed:
            raise RuntimeError(
                f"Gate 0 FAILED: {report.n_errors} bug guard errors. "
                "Fix the issues before proceeding."
            )
        return result

    def gate1_quick(
        self,
        run_func: Callable[[dict], Optional[dict]],
        param_list: list[dict],
    ) -> GateResult:
        """Gate 1: Quick feasibility — run a small set of params on full data.

        run_func: Takes a param dict, returns a metric dict with at least
                  'pf', 'total_r', 'n_trades', 'max_dd_r'.
                  Returns None if no trades.
        param_list: ~10-20 diverse parameter combos to test.
        """
        self._header("Gate 1", f"Quick Feasibility ({len(param_list)} combos)")
        t0 = time.time()

        results = []
        for i, params in enumerate(param_list):
            metrics = run_func(params)
            if metrics is not None and metrics.get("n_trades", 0) >= self.GATE1_MIN_TRADES:
                results.append({**params, **metrics})
            print(f"  [{i+1}/{len(param_list)}] "
                  f"PF={metrics.get('pf', 0):.3f} "
                  f"n={metrics.get('n_trades', 0)} "
                  f"R={metrics.get('total_r', 0):.1f}"
                  if metrics else f"  [{i+1}/{len(param_list)}] No trades")

        elapsed = time.time() - t0

        if not results:
            result = GateResult("Gate 1", False, "No combos produced enough trades",
                                elapsed_sec=elapsed)
            self.gates.append(result)
            self._footer("Gate 1", result)
            return result

        best = max(results, key=lambda x: x.get("pf", 0))
        best_pf = best.get("pf", 0)
        best_r = best.get("total_r", 0)
        n_profitable = sum(1 for r in results if r.get("pf", 0) > 1.0)

        msg = (f"Best PF={best_pf:.3f}, Best R={best_r:.1f}, "
               f"{n_profitable}/{len(results)} profitable")

        passed = best_pf >= self.GATE1_MIN_PF
        result = GateResult("Gate 1", passed, msg, best_metric=best,
                            elapsed_sec=elapsed)
        self.gates.append(result)
        self._footer("Gate 1", result)
        return result

    def gate2_screen(
        self,
        run_func: Callable[[dict], Optional[dict]],
        param_list: list[dict],
    ) -> GateResult:
        """Gate 2: Coarse screening — ~100 combos, check PF and RF."""
        self._header("Gate 2", f"Coarse Screen ({len(param_list)} combos)")
        t0 = time.time()

        results = []
        for i, params in enumerate(param_list):
            metrics = run_func(params)
            if metrics is not None and metrics.get("n_trades", 0) >= self.GATE2_MIN_TRADES:
                rf = (metrics.get("total_r", 0) / metrics.get("max_dd_r", 1e9)
                      if metrics.get("max_dd_r", 0) > 0 else 0)
                results.append({**params, **metrics, "rf": rf})
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(param_list)}]")

        elapsed = time.time() - t0

        if not results:
            result = GateResult("Gate 2", False, "No combos passed minimum trades",
                                elapsed_sec=elapsed)
            self.gates.append(result)
            self._footer("Gate 2", result)
            return result

        best_pf = max(results, key=lambda x: x.get("pf", 0))
        best_rf = max(results, key=lambda x: x.get("rf", 0))
        max_pf = best_pf.get("pf", 0)
        max_rf = best_rf.get("rf", 0)
        n_profitable = sum(1 for r in results if r.get("total_r", 0) > 0)

        msg = (f"Best PF={max_pf:.3f}, Best RF={max_rf:.2f}, "
               f"{n_profitable}/{len(results)} profitable")

        passed = max_pf >= self.GATE2_MIN_PF and max_rf >= self.GATE2_MIN_RF
        result = GateResult("Gate 2", passed, msg,
                            best_metric=best_pf, elapsed_sec=elapsed)
        self.gates.append(result)
        self._footer("Gate 2", result)

        if passed:
            print(f"\n  Top 5 by PF:")
            top5 = sorted(results, key=lambda x: x.get("pf", 0), reverse=True)[:5]
            for r in top5:
                print(f"    PF={r['pf']:.3f} RF={r.get('rf',0):.2f} "
                      f"R={r['total_r']:.1f} n={r['n_trades']}")

        return result

    def summary(self):
        """Print summary of all gates passed."""
        print(f"\n{'='*70}")
        print(f"GateKeeper Summary: {self.strategy_name}")
        print(f"{'='*70}")
        for g in self.gates:
            status = "PASS" if g.passed else "KILL"
            print(f"  [{status}] {g.gate}: {g.message} ({g.elapsed_sec:.1f}s)")
        total_time = sum(g.elapsed_sec for g in self.gates)
        print(f"\n  Total time: {total_time:.1f}s")
        all_passed = all(g.passed for g in self.gates)
        if all_passed:
            print(f"  STATUS: All gates passed. Proceed to WFA/MC validation.")
        else:
            failed = [g.gate for g in self.gates if not g.passed]
            print(f"  STATUS: KILLED at {', '.join(failed)}. Strategy is not viable.")
        print(f"{'='*70}")
