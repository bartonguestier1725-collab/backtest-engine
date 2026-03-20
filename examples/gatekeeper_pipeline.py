"""Example: Full GateKeeper pipeline (Gate 0–4).

Demonstrates how a new user would validate a strategy through all 5 gates.
Uses synthetic data with relaxed thresholds so all gates are exercised.

NOTE: This example lowers Gate 2 thresholds for demonstration purposes.
      With real data, always use the default thresholds or stricter.

The GateKeeper kills bad strategies fast:
  Gate 0 (1 min)  — BugGuard: data quality + code bug detection
  Gate 1 (5 min)  — Quick feasibility: ~20 param combos, PF >= 1.05?
  Gate 2 (20 min) — Coarse screen: ~100 combos, PF >= 1.10, RF >= 1.5?
  Gate 3 (30 min) — WFA + CSCV: overfitting detection
  Gate 4 (5 min)  — Monte Carlo: drawdown resilience
"""

import numpy as np

from backtest_engine import (
    simulate_trades, LONG, SHORT,
    sma, atr, BrokerCost, MonteCarloDD, StressTest,
    WalkForward, CSCV, GateKeeper,
)


# ── Synthetic data (replace with your own) ────────────────────────────────────

def generate_data(n_bars=300_000, seed=42):
    np.random.seed(seed)
    returns = np.random.randn(n_bars) * 0.001 + 0.00005
    close = 1.10000 + np.cumsum(returns)
    spread = np.abs(np.random.randn(n_bars)) * 0.0003
    high = close + spread
    low = close - spread
    open_price = close + np.random.randn(n_bars) * 0.0001
    return high, low, close, open_price


# ── Signal generator ──────────────────────────────────────────────────────────

def find_signals(close, fast_period=10, slow_period=50):
    fast = sma(close, fast_period)
    slow = sma(close, slow_period)
    bars, dirs = [], []
    for i in range(slow_period, len(close) - 1):
        if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
            bars.append(i)
            dirs.append(LONG)
        elif fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
            bars.append(i)
            dirs.append(SHORT)
    return np.array(bars, dtype=np.int32), np.array(dirs, dtype=np.int8)


# ── run_func: the interface GateKeeper expects ────────────────────────────────
# Takes a param dict, returns {'pf', 'total_r', 'n_trades', 'max_dd_r'} or None.

HIGH, LOW, CLOSE, OPEN = generate_data()
ATR_VALS = atr(HIGH, LOW, CLOSE, 14)
COST_MODEL = BrokerCost.tradeview_ilc()


def run_func(params):
    """Single backtest run with given parameters."""
    fast = params.get("fast", 10)
    slow = params.get("slow", 50)
    atr_mult = params.get("atr_mult", 1.5)
    rr_ratio = params.get("rr_ratio", 2.0)

    sig, dirs = find_signals(CLOSE, fast, slow)
    if len(sig) < 10:
        return None

    sl = ATR_VALS[sig] * atr_mult
    tp = sl * rr_ratio
    instruments = ["EURUSD"] * len(sig)
    costs = COST_MODEL.per_trade_cost(instruments, sl)

    results = simulate_trades(
        HIGH, LOW, CLOSE, sig, dirs, sl, tp,
        max_hold=100, open_prices=OPEN, entry_costs=costs, preflight=False,
    )
    if len(results) == 0:
        return None

    return {
        "pf": results.profit_factor,
        "total_r": float(np.sum(results["pnl_r"])),
        "n_trades": len(results),
        "max_dd_r": results.max_drawdown_r,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    # ── Gate 0: BugGuard ──────────────────────────────────────────────────
    gk = GateKeeper(
        strategy_name="SMA Crossover Demo",
        n_bars=len(CLOSE),
        bar_minutes=1,
        resolution_minutes=1,
        spreads_used=COST_MODEL.cost_prices(),
        expected_costs=COST_MODEL.cost_prices(),
        source_path=__file__,
    )
    # Relax thresholds for demo (SMA crossover on synthetic data is weak)
    # With real strategies, always use the defaults or stricter values
    gk.GATE2_MIN_PF = 1.05   # default: 1.10
    gk.GATE2_MIN_RF = 0.5    # default: 1.5
    gk.GATE3_MAX_PBO = 0.60  # default: 0.40

    gk.gate0_validate()

    # ── Gate 1: Quick feasibility (~20 combos) ────────────────────────────
    quick_params = [
        {"fast": f, "slow": s, "atr_mult": 1.5, "rr_ratio": 2.0}
        for f in [5, 10, 20]
        for s in [30, 50, 100, 200]
        if f < s
    ]
    result1 = gk.gate1_quick(run_func, quick_params)
    if not result1.passed:
        gk.summary()
        return

    # ── Gate 2: Coarse screen (~100 combos) ───────────────────────────────
    screen_params = [
        {"fast": f, "slow": s, "atr_mult": a, "rr_ratio": r}
        for f in [5, 10, 15, 20]
        for s in [30, 50, 75, 100, 150]
        for a in [1.0, 1.5, 2.0]
        for r in [1.5, 2.0]
        if f < s
    ]
    result2 = gk.gate2_screen(run_func, screen_params)
    if not result2.passed:
        gk.summary()
        return

    # ── Gate 3: WFA + CSCV ────────────────────────────────────────────────
    def evaluate(params, start, end):
        sig, dirs = find_signals(CLOSE[start:end], params["fast"], params["slow"])
        if len(sig) < 10:
            return 0.0
        sl = ATR_VALS[start:end][sig] * params.get("atr_mult", 1.5)
        tp = sl * params.get("rr_ratio", 2.0)
        instruments = ["EURUSD"] * len(sig)
        costs = COST_MODEL.per_trade_cost(instruments, sl)
        r = simulate_trades(
            HIGH[start:end], LOW[start:end], CLOSE[start:end],
            sig, dirs, sl, tp, max_hold=100,
            open_prices=OPEN[start:end], entry_costs=costs, preflight=False,
        )
        return float(np.mean(r["pnl_r"])) if len(r) > 0 else 0.0

    param_grid = [
        {"fast": 5, "slow": 50, "atr_mult": 1.5, "rr_ratio": 2.0},
        {"fast": 10, "slow": 50, "atr_mult": 1.5, "rr_ratio": 2.0},
        {"fast": 10, "slow": 100, "atr_mult": 1.5, "rr_ratio": 2.0},
    ]

    wf = WalkForward(n_bars=len(CLOSE), is_ratio=0.7, n_splits=5)
    wfa_result = wf.run(param_grid, evaluate)

    cscv = CSCV(n_splits=10)
    cscv_result = cscv.run(param_grid, evaluate, n_bars=len(CLOSE))

    result3 = gk.gate3_validate(wfa_result, cscv_result)
    if not result3.passed:
        gk.summary()
        return

    # ── Gate 4: Monte Carlo DD ────────────────────────────────────────────
    # Run best params to get pnl for MC
    best = result2.best_metric or {"fast": 10, "slow": 50}
    sig, dirs = find_signals(CLOSE, best.get("fast", 10), best.get("slow", 50))
    sl = ATR_VALS[sig] * best.get("atr_mult", 1.5)
    tp = sl * best.get("rr_ratio", 2.0)
    instruments = ["EURUSD"] * len(sig)
    costs = COST_MODEL.per_trade_cost(instruments, sl)
    results = simulate_trades(
        HIGH, LOW, CLOSE, sig, dirs, sl, tp,
        max_hold=100, open_prices=OPEN, entry_costs=costs, preflight=False,
    )

    mc = MonteCarloDD(results["pnl_r"], n_sims=10_000, risk_pct=0.01, seed=42)
    mc.run()
    result4 = gk.gate4_montecarlo(mc, dd_limit=0.20)

    # ── Summary ───────────────────────────────────────────────────────────
    gk.summary()

    # ── Bonus: StressTest ─────────────────────────────────────────────────
    if result4.passed:
        print("\n--- Stress Test (post-validation) ---")
        st = StressTest(results["pnl_r"], n_sims=1000, seed=42)
        report = st.run_all(block_size=10)
        print(f"Baseline DD@95%: {report['baseline']['dd_95']*100:.1f}%")
        print(f"Block bootstrap DD@95%: {report['block_bootstrap']['dd_95']*100:.1f}%")
        for name, scenario in report["degraded"].items():
            print(f"  {name}: DD@95%={scenario['dd_95']*100:.1f}%, "
                  f"expectancy={scenario['expectancy_r']:.3f}R")


if __name__ == "__main__":
    main()
