"""Tests for backtest_engine.gatekeeper — GateKeeper pipeline."""

import numpy as np
import pytest

from backtest_engine.gatekeeper import GateKeeper, GateResult
from backtest_engine.montecarlo import MonteCarloDD


# ── Helpers ──────────────────────────────────────────────────────────────────

def _profitable_run_func(params):
    """Dummy run func returning profitable metrics."""
    return {
        "pf": params.get("pf", 1.20),
        "total_r": params.get("total_r", 15.0),
        "n_trades": params.get("n_trades", 100),
        "max_dd_r": params.get("max_dd_r", 5.0),
    }


def _unprofitable_run_func(params):
    """Dummy run func returning unprofitable metrics."""
    return {
        "pf": 0.90,
        "total_r": -5.0,
        "n_trades": 100,
        "max_dd_r": 10.0,
    }


def _no_trades_run_func(params):
    """Dummy run func returning None (no trades)."""
    return None


# ── Gate 0 Tests ─────────────────────────────────────────────────────────────

class TestGate0:
    def test_minimal_input_passes_with_warning(self, capsys):
        """Gate 0 with no inputs should pass but emit GK-00 warning."""
        gk = GateKeeper(strategy_name="test")
        result = gk.gate0_validate()
        assert result.passed
        output = capsys.readouterr().out
        assert "INCOMPLETE INPUTS" in output
        # GK-00 must be counted in n_warnings (not "0 warnings")
        assert "0 warnings" not in result.message
        # GK-00 is included in BugGuard report (no numerical contradiction)
        assert "GK-00" in output

    def test_full_input_no_gk00(self, tmp_path, capsys):
        """Gate 0 with all inputs should not emit GK-00."""
        # Create a minimal source file
        src = tmp_path / "strategy.py"
        src.write_text("# no bfill, no quantile, no Close entry\n")

        gk = GateKeeper(
            strategy_name="test",
            n_bars=300_000,  # ~13 months of 1min FX data
            bar_minutes=1,
            resolution_minutes=1,
            spreads_used={"EURUSD": 0.00010},
            expected_costs={"EURUSD": 0.00010},
            source_path=str(src),
        )
        result = gk.gate0_validate()
        assert result.passed
        output = capsys.readouterr().out
        assert "INCOMPLETE" not in output

    def test_gate0_log_counts_consistent(self, capsys):
        """BugGuard report and Gate 0 footer must show the same warning count."""
        gk = GateKeeper(strategy_name="test")  # triggers GK-00
        result = gk.gate0_validate()
        output = capsys.readouterr().out
        # BugGuard "ALL CLEAR" line must include the GK-00 warning count
        # (no "0 warnings" when GK-00 adds a warning)
        assert "ALL CLEAR (0 warnings)" not in output

    def test_gate0_fails_on_bug_guard_error(self, tmp_path):
        """Gate 0 should raise RuntimeError on BugGuard ERROR."""
        src = tmp_path / "bad.py"
        src.write_text("df.bfill()\n")  # triggers BG-04

        gk = GateKeeper(
            strategy_name="test",
            resolution_minutes=60,  # triggers BG-05 ERROR
            source_path=str(src),
        )
        with pytest.raises(RuntimeError, match="Gate 0 FAILED"):
            gk.gate0_validate()


# ── Gate 1 Tests ─────────────────────────────────────────────────────────────

class TestGate1:
    def test_profitable_passes(self, capsys):
        gk = GateKeeper(strategy_name="test")
        params = [{"pf": 1.20}, {"pf": 1.10}]
        result = gk.gate1_quick(_profitable_run_func, params)
        assert result.passed
        assert result.best_metric is not None

    def test_unprofitable_killed(self, capsys):
        gk = GateKeeper(strategy_name="test")
        params = [{}]
        result = gk.gate1_quick(_unprofitable_run_func, params)
        assert not result.passed

    def test_no_trades_killed(self, capsys):
        gk = GateKeeper(strategy_name="test")
        params = [{}]
        result = gk.gate1_quick(_no_trades_run_func, params)
        assert not result.passed


# ── Gate 2 Tests ─────────────────────────────────────────────────────────────

class TestGate2:
    def test_profitable_passes(self, capsys):
        gk = GateKeeper(strategy_name="test")
        params = [{"pf": 1.20, "total_r": 20.0, "n_trades": 100, "max_dd_r": 5.0}]
        result = gk.gate2_screen(_profitable_run_func, params)
        assert result.passed

    def test_low_rf_killed(self, capsys):
        def low_rf(params):
            return {"pf": 1.15, "total_r": 3.0, "n_trades": 100, "max_dd_r": 10.0}

        gk = GateKeeper(strategy_name="test")
        result = gk.gate2_screen(low_rf, [{}])
        assert not result.passed  # RF = 3/10 = 0.3, below 1.5

    def test_zero_dd_passes(self, capsys):
        """max_dd_r=0 (perfect strategy) should give RF=inf → PASS."""
        def zero_dd(params):
            return {"pf": 1.20, "total_r": 20.0, "n_trades": 100, "max_dd_r": 0.0}

        gk = GateKeeper(strategy_name="test")
        result = gk.gate2_screen(zero_dd, [{}])
        assert result.passed  # RF=inf >= 1.5

    def test_zero_dd_negative_total_killed(self, capsys):
        """max_dd_r=0 but total_r<=0 should give RF=0 → depends on PF."""
        def zero_dd_neg(params):
            return {"pf": 0.90, "total_r": -5.0, "n_trades": 100, "max_dd_r": 0.0}

        gk = GateKeeper(strategy_name="test")
        result = gk.gate2_screen(zero_dd_neg, [{}])
        assert not result.passed  # PF=0.90 < 1.10


# ── Gate 3 Tests ─────────────────────────────────────────────────────────────

class TestGate3:
    def test_good_wfa_and_cscv_pass(self, capsys):
        wfa_result = {"oos_positive_frac": 0.80, "oos_mean": 5.0}
        cscv_result = {"pbo": 0.10}
        gk = GateKeeper(strategy_name="test")
        result = gk.gate3_validate(wfa_result, cscv_result)
        assert result.passed
        assert "WFA OOS win rate=0.80" in result.message
        assert "CSCV PBO=0.10" in result.message

    def test_bad_wfa_killed(self, capsys):
        wfa_result = {"oos_positive_frac": 0.40}  # below 0.55
        gk = GateKeeper(strategy_name="test")
        result = gk.gate3_validate(wfa_result)
        assert not result.passed

    def test_bad_pbo_killed(self, capsys):
        wfa_result = {"oos_positive_frac": 0.80}
        cscv_result = {"pbo": 0.60}  # above 0.40
        gk = GateKeeper(strategy_name="test")
        result = gk.gate3_validate(wfa_result, cscv_result)
        assert not result.passed

    def test_wfa_only_no_cscv(self, capsys):
        """Gate 3 without CSCV should pass but note PBO was skipped."""
        wfa_result = {"oos_positive_frac": 0.70}
        gk = GateKeeper(strategy_name="test")
        result = gk.gate3_validate(wfa_result, cscv_result=None)
        assert result.passed
        assert "skipped" in result.message
        assert result.best_metric["cscv_skipped"] is True

    def test_wfa_exact_threshold_passes(self, capsys):
        """OOS frac exactly at threshold (0.55) should pass."""
        gk = GateKeeper(strategy_name="test")
        result = gk.gate3_validate({"oos_positive_frac": 0.55})
        assert result.passed

    def test_wfa_just_below_threshold_fails(self, capsys):
        """OOS frac just below threshold should fail."""
        gk = GateKeeper(strategy_name="test")
        result = gk.gate3_validate({"oos_positive_frac": 0.54})
        assert not result.passed

    def test_pbo_exact_threshold_passes(self, capsys):
        """PBO exactly at threshold (0.40) should pass."""
        gk = GateKeeper(strategy_name="test")
        result = gk.gate3_validate(
            {"oos_positive_frac": 0.80},
            {"pbo": 0.40},
        )
        assert result.passed

    def test_pbo_just_above_threshold_fails(self, capsys):
        """PBO just above threshold should fail."""
        gk = GateKeeper(strategy_name="test")
        result = gk.gate3_validate(
            {"oos_positive_frac": 0.80},
            {"pbo": 0.41},
        )
        assert not result.passed

    def test_best_metric_populated(self, capsys):
        wfa_result = {"oos_positive_frac": 0.75, "oos_mean": 3.5}
        cscv_result = {"pbo": 0.15}
        gk = GateKeeper(strategy_name="test")
        result = gk.gate3_validate(wfa_result, cscv_result)
        assert result.best_metric["oos_positive_frac"] == 0.75
        assert result.best_metric["pbo"] == 0.15


# ── Gate 4 Tests ─────────────────────────────────────────────────────────────

class TestGate4:
    @pytest.fixture
    def profitable_mc(self):
        """MonteCarloDD with profitable system → low DDs."""
        np.random.seed(42)
        n = 200
        pnl = np.where(np.random.rand(n) < 0.6, 1.5, -1.0).astype(np.float64)
        mc = MonteCarloDD(pnl, n_sims=500, risk_pct=0.005, seed=42)
        mc.run()
        return mc

    @pytest.fixture
    def losing_mc(self):
        """MonteCarloDD with losing system → high DDs."""
        pnl = np.full(100, -0.5, dtype=np.float64)
        mc = MonteCarloDD(pnl, n_sims=500, risk_pct=0.02, seed=42)
        mc.run()
        return mc

    def test_profitable_passes(self, profitable_mc, capsys):
        gk = GateKeeper(strategy_name="test")
        result = gk.gate4_montecarlo(profitable_mc, dd_limit=0.20)
        assert result.passed
        assert result.best_metric["pass_rate"] >= 0.70

    def test_losing_killed(self, losing_mc, capsys):
        gk = GateKeeper(strategy_name="test")
        result = gk.gate4_montecarlo(losing_mc, dd_limit=0.05)
        assert not result.passed

    def test_auto_run(self, capsys):
        """Gate 4 should auto-run MC if not already run."""
        np.random.seed(42)
        pnl = np.where(np.random.rand(200) < 0.6, 1.5, -1.0).astype(np.float64)
        mc = MonteCarloDD(pnl, n_sims=100, risk_pct=0.005, seed=42)
        # NOT calling mc.run() — gate4 should trigger auto-run via max_dds
        gk = GateKeeper(strategy_name="test")
        result = gk.gate4_montecarlo(mc, dd_limit=0.30)
        assert isinstance(result, GateResult)

    def test_best_metric_keys(self, profitable_mc, capsys):
        gk = GateKeeper(strategy_name="test")
        result = gk.gate4_montecarlo(profitable_mc)
        assert "pass_rate" in result.best_metric
        assert "dd_confidence" in result.best_metric
        assert "dd_99" in result.best_metric
        assert "risk_pct" in result.best_metric

    def test_high_pass_rate_but_bad_confidence_dd_killed(self, capsys):
        """High pass rate but DD@confidence > dd_limit → must FAIL."""
        # 80% of sims at 10% DD, 20% at 30% DD
        max_dds = np.concatenate([
            np.full(80, 0.10),
            np.full(20, 0.30),
        ])
        np.random.seed(42)
        pnl = np.where(np.random.rand(200) < 0.6, 1.5, -1.0).astype(np.float64)
        mc = MonteCarloDD(pnl, n_sims=100, risk_pct=0.01, seed=42)
        mc.run()
        mc._max_dds = max_dds  # Override with controlled distribution

        gk = GateKeeper(strategy_name="test")
        result = gk.gate4_montecarlo(mc, dd_limit=0.20, confidence=95.0)
        # pass_rate = 80% >= 70% but DD@95% = 30% > 20% → FAIL
        assert not result.passed

    def test_dd_at_confidence_within_limit_passes(self, profitable_mc, capsys):
        """DD@confidence within limit + pass rate OK → PASS."""
        gk = GateKeeper(strategy_name="test")
        result = gk.gate4_montecarlo(profitable_mc, dd_limit=0.50, confidence=99.0)
        assert result.passed


# ── Summary Tests ────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_all_5_gates_pass(self, capsys):
        """All 5 gates passed → 'Strategy is validated'."""
        gk = GateKeeper(strategy_name="test")
        for i in range(5):
            gk.gates.append(GateResult(f"Gate {i}", True, "ok"))
        gk.summary()
        output = capsys.readouterr().out
        assert "All gates passed" in output
        assert "Strategy is validated" in output

    def test_summary_partial_gates_incomplete(self, capsys):
        """Only 2/5 gates → 'INCOMPLETE', not 'validated'."""
        gk = GateKeeper(strategy_name="test")
        gk.gates.append(GateResult("Gate 0", True, "ok"))
        gk.gates.append(GateResult("Gate 1", True, "ok"))
        gk.summary()
        output = capsys.readouterr().out
        assert "INCOMPLETE" in output
        assert "Strategy is validated" not in output
        # Verify missing gate names are listed
        assert "Gate 2" in output
        assert "Gate 3" in output
        assert "Gate 4" in output

    def test_summary_duplicate_gates_not_fooled(self, capsys):
        """Duplicate gate runs must not fool the completeness check."""
        gk = GateKeeper(strategy_name="test")
        # Run Gate 0 and Gate 1 three times each (6 entries, but only 2 unique)
        for _ in range(3):
            gk.gates.append(GateResult("Gate 0", True, "ok"))
            gk.gates.append(GateResult("Gate 1", True, "ok"))
        gk.summary()
        output = capsys.readouterr().out
        assert "INCOMPLETE" in output
        assert "Strategy is validated" not in output

    def test_summary_cscv_skipped_caveat(self, capsys):
        """CSCV skipped → 'validated' but with caveat, not plain 'validated'."""
        gk = GateKeeper(strategy_name="test")
        gk.gates.append(GateResult("Gate 0", True, "ok"))
        gk.gates.append(GateResult("Gate 1", True, "ok"))
        gk.gates.append(GateResult("Gate 2", True, "ok"))
        gk.gates.append(GateResult("Gate 3", True, "ok",
                                    best_metric={"cscv_skipped": True}))
        gk.gates.append(GateResult("Gate 4", True, "ok"))
        gk.summary()
        output = capsys.readouterr().out
        assert "CSCV/PBO not checked" in output
        # Must NOT show plain "Strategy is validated." without caveat
        assert "Strategy is validated." not in output.replace(
            "Strategy is validated (CSCV/PBO not checked).", "")

    def test_summary_cscv_provided_no_caveat(self, capsys):
        """CSCV provided → plain 'Strategy is validated.' without caveat."""
        gk = GateKeeper(strategy_name="test")
        gk.gates.append(GateResult("Gate 0", True, "ok"))
        gk.gates.append(GateResult("Gate 1", True, "ok"))
        gk.gates.append(GateResult("Gate 2", True, "ok"))
        gk.gates.append(GateResult("Gate 3", True, "ok",
                                    best_metric={"cscv_skipped": False}))
        gk.gates.append(GateResult("Gate 4", True, "ok"))
        gk.summary()
        output = capsys.readouterr().out
        assert "Strategy is validated." in output
        assert "CSCV/PBO not checked" not in output

    def test_summary_with_kill(self, capsys):
        gk = GateKeeper(strategy_name="test")
        gk.gates.append(GateResult("Gate 0", True, "ok"))
        gk.gates.append(GateResult("Gate 1", False, "PF too low"))
        gk.summary()
        output = capsys.readouterr().out
        assert "KILLED" in output
        assert "Gate 1" in output

    def test_threshold_override(self):
        """Thresholds should be overridable."""
        gk = GateKeeper(strategy_name="test")
        gk.GATE3_MAX_PBO = 0.20
        wfa_result = {"oos_positive_frac": 0.80}
        cscv_result = {"pbo": 0.30}  # above 0.20 → FAIL
        result = gk.gate3_validate(wfa_result, cscv_result)
        assert not result.passed
