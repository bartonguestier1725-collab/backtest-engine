"""GateKeeper — Fast-kill system for trading strategy evaluation.

Kill bad strategies in 5 minutes, not 7 hours.

Gate 0: Data validation (1 min)    — Do we have enough data? Correct costs?
Gate 1: Quick feasibility (5 min)  — Run ~20 combos on full data. PF < 1.05 → dead.
Gate 2: Coarse screen (20 min)     — Run ~100 combos. PF < 1.10, RF < 1.5 → dead.
Gate 3: WFA + CSCV (30 min)       — PBO < 0.40, OOS win rate >= 0.55 → dead.
Gate 4: Monte Carlo (5 min)        — MC pass rate >= 0.70 → dead.

Default thresholds are tuned for FX. Override them via class variables
(e.g. ``gk.GATE1_MIN_PF = 1.10``) for other asset classes.

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

    # Gate 3: WFA + CSCV overfitting check
    gk.gate3_validate(wfa_result, cscv_result)

    # Gate 4: Monte Carlo drawdown check
    gk.gate4_montecarlo(mc)

    gk.summary()
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from backtest_engine.bug_guard import run_all_checks, GuardReport, _fundora_expected_costs


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
    expected_costs : {pair: cost_in_price_units} for cost validation.
                     Use BrokerCost.cost_prices() to generate this.
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
        expected_costs: Optional[dict[str, float]] = None,
        **kwargs,
    ):
        # Backward compat: accept deprecated broker kwarg
        if "broker" in kwargs:
            warnings.warn(
                "broker parameter is deprecated. "
                "Use expected_costs=BrokerCost.cost_prices() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            broker = kwargs.pop("broker")
            if expected_costs is None and broker == "fundora":
                expected_costs = _fundora_expected_costs()
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")

        self.strategy_name = strategy_name
        self.n_bars = n_bars
        self.bar_minutes = bar_minutes
        self.resolution_minutes = resolution_minutes
        self.spreads_used = spreads_used
        self.source_path = source_path
        self.expected_costs = expected_costs
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
        Warns if key inputs (source_path, spreads_used, n_bars) are missing,
        as this causes important checks to be silently skipped.
        """
        self._header("Gate 0", "Data Validation + BugGuard")
        t0 = time.time()

        # Build GK-00 check before running BugGuard so it's included
        # in print_report() and n_warnings count is consistent.
        from backtest_engine.bug_guard import CheckResult
        gk00 = None
        missing = []
        if self.source_path is None:
            missing.append("source_path (BG-04/07/09 skipped)")
        if self.spreads_used is None:
            missing.append("spreads_used (BG-02 skipped)")
        if self.n_bars == 0:
            missing.append("n_bars (BG-06 skipped)")
        if missing:
            gk00 = CheckResult(
                check_id="GK-00",
                passed=False,
                message=(
                    "INCOMPLETE INPUTS: "
                    + ", ".join(missing)
                    + ". Gate 0 may pass with insufficient validation."
                ),
                severity="WARN",
            )

        report = run_all_checks(
            source_path=self.source_path,
            spreads_used=self.spreads_used,
            expected_costs=self.expected_costs,
            resolution_minutes=self.resolution_minutes,
            n_bars=self.n_bars,
            bar_minutes=self.bar_minutes,
            strict=False,  # We handle the result ourselves
            _suppress_print=True,  # We'll print after adding GK-00
        )

        # Append GK-00 before printing so counts are correct
        if gk00 is not None:
            report.results.append(gk00)

        report.print_report()

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
                dd = metrics.get("max_dd_r", 0)
                total = metrics.get("total_r", 0)
                if dd > 0:
                    rf = total / dd
                elif total > 0:
                    rf = float("inf")
                else:
                    rf = 0.0
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

    def gate3_validate(
        self,
        wfa_result: dict,
        cscv_result: dict | None = None,
    ) -> GateResult:
        """Gate 3: Walk-Forward + CSCV overfitting check.

        Parameters
        ----------
        wfa_result : Dict returned by ``WalkForward.run()``.
                     Must contain 'oos_positive_frac' (and optionally 'oos_mean').
        cscv_result : Dict returned by ``CSCV.run()``, or None to skip PBO check.
                      Must contain 'pbo' if provided.

        Checks:
          - WFA OOS win rate >= GATE3_MIN_WFA_WINRATE
          - CSCV PBO <= GATE3_MAX_PBO (if provided)
        """
        self._header("Gate 3", "WFA + CSCV Overfitting Check")
        t0 = time.time()

        checks = []
        oos_frac = wfa_result.get("oos_positive_frac", 0.0)
        wfa_ok = oos_frac >= self.GATE3_MIN_WFA_WINRATE
        checks.append(
            f"WFA OOS win rate={oos_frac:.2f} "
            f"({'OK' if wfa_ok else 'FAIL'}, min={self.GATE3_MIN_WFA_WINRATE})"
        )

        pbo_ok = True
        if cscv_result is not None:
            pbo = cscv_result.get("pbo", 1.0)
            pbo_ok = pbo <= self.GATE3_MAX_PBO
            checks.append(
                f"CSCV PBO={pbo:.2f} "
                f"({'OK' if pbo_ok else 'FAIL'}, max={self.GATE3_MAX_PBO})"
            )
        else:
            checks.append("CSCV PBO=skipped (not provided)")

        passed = wfa_ok and pbo_ok
        msg = "; ".join(checks)
        elapsed = time.time() - t0

        best_metric = {
            "oos_positive_frac": oos_frac,
            "oos_mean": wfa_result.get("oos_mean", 0.0),
            "cscv_skipped": cscv_result is None,
        }
        if cscv_result is not None:
            best_metric["pbo"] = cscv_result.get("pbo", 1.0)

        result = GateResult("Gate 3", passed, msg,
                            best_metric=best_metric, elapsed_sec=elapsed)
        self.gates.append(result)
        self._footer("Gate 3", result)
        return result

    def gate4_montecarlo(
        self,
        mc: object,
        dd_limit: float = 0.20,
        confidence: float = 95.0,
    ) -> GateResult:
        """Gate 4: Monte Carlo drawdown validation.

        Parameters
        ----------
        mc : A ``MonteCarloDD`` instance (already run, or will auto-run).
        dd_limit : Max acceptable DD at *confidence* percentile.
        confidence : Percentile for DD check (default 95th).

        Checks:
          - MC pass rate (fraction of sims with max DD < dd_limit) >= GATE4_MIN_MC_PASS
          - Reports DD at confidence percentile and 99th percentile.
        """
        self._header("Gate 4", "Monte Carlo Drawdown")
        t0 = time.time()

        # Access max_dds (triggers auto-run if needed)
        max_dds = mc.max_dds
        pass_rate = float(np.mean(max_dds < dd_limit))
        dd_conf = float(np.percentile(max_dds, confidence))
        dd_99 = float(np.percentile(max_dds, 99.0))

        rate_ok = pass_rate >= self.GATE4_MIN_MC_PASS
        conf_ok = dd_conf <= dd_limit
        passed = rate_ok and conf_ok
        msg = (
            f"MC pass rate={pass_rate:.2f} "
            f"({'OK' if rate_ok else 'FAIL'}, min={self.GATE4_MIN_MC_PASS}), "
            f"DD@{confidence:.0f}%={dd_conf*100:.1f}% "
            f"({'OK' if conf_ok else 'FAIL'}, limit={dd_limit*100:.0f}%), "
            f"DD@99%={dd_99*100:.1f}%"
        )
        elapsed = time.time() - t0

        result = GateResult("Gate 4", passed, msg,
                            best_metric={
                                "pass_rate": pass_rate,
                                "dd_confidence": dd_conf,
                                "dd_99": dd_99,
                                "risk_pct": mc.risk_pct,
                            },
                            elapsed_sec=elapsed)
        self.gates.append(result)
        self._footer("Gate 4", result)
        return result

    TOTAL_GATES = 5  # Gate 0 through Gate 4

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
        ran = {g.gate for g in self.gates}
        expected = {f"Gate {i}" for i in range(self.TOTAL_GATES)}
        missing = sorted(expected - ran)
        if all_passed and not missing:
            # Check if any gate skipped optional checks
            caveats = []
            for g in self.gates:
                if g.best_metric and g.best_metric.get("cscv_skipped"):
                    caveats.append("CSCV/PBO not checked")
            if caveats:
                print(f"  STATUS: All gates passed. "
                      f"Strategy is validated ({', '.join(caveats)}).")
            else:
                print(f"  STATUS: All gates passed. Strategy is validated.")
        elif all_passed:
            print(f"  STATUS: {len(ran)}/{self.TOTAL_GATES} gates passed, "
                  f"but validation is INCOMPLETE ({', '.join(missing)} not run).")
        else:
            failed = [g.gate for g in self.gates if not g.passed]
            print(f"  STATUS: KILLED at {', '.join(failed)}. Strategy is not viable.")
        print(f"{'='*70}")
