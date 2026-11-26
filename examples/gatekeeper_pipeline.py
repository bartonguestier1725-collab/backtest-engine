"""Example: Full GateKeeper pipeline (Gate 0–4).

Demonstrates how a new user would validate a strategy through the 5 gates.
Gates run in order and the pipeline stops at the first failed gate. The
synthetic data is a random walk with a small upward drift, which long trades
of a trend-following rule profit from. With the default thresholds it passes
Gates 0-3 and stops at Gate 4: at 1% risk per trade, thousands of trades reach
a drawdown above the 20% limit.

The backtests are GROSS (no costs): the gates ask whether an edge exists.
Whether it survives your broker's costs is a later step (BrokerCost).

The GateKeeper kills bad strategies fast:
  Gate 0 — BugGuard: data quality + code bug detection
  Gate 1 — Quick feasibility: ~20 param combos, PF >= 1.05?
  Gate 2 — Coarse screen: ~100 combos, one combo with PF >= 1.10 and RF >= 1.5?
  Gate 3 — WFA + CSCV: overfitting detection
  Gate 4 — Monte Carlo: drawdown resilience
"""

import numpy as np

from backtest_engine import (
    simulate_trades, LONG, SHORT,
    sma, atr, MonteCarloDD, StressTest,
    WalkForward, CSCV, GateKeeper,
)


# ── Synthetic data (replace with your own) ────────────────────────────────────

def generate_data(n_bars=300_000, seed=42):
    np.random.seed(seed)
    returns = np.random.randn(n_bars) * 0.001 + 0.00005
    close = 1.10000 + np.cumsum(returns)
    # Each bar opens near the previous close and its range contains both
    # the open and the close, as in real OHLC data.
    open_price = np.r_[close[0], close[:-1]] + np.random.randn(n_bars) * 0.00002
    spread = np.abs(np.random.randn(n_bars)) * 0.0003
    high = np.maximum(open_price, close) + spread
    low = np.minimum(open_price, close) - spread
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

    results = simulate_trades(   # GROSS: no entry_costs
        HIGH, LOW, CLOSE, sig, dirs, sl, tp,
        max_hold=100, open_prices=OPEN, preflight=False,
    )
    if len(results) == 0:
        return None

    return {
        "pf": results.profit_factor,
        "total_r": float(np.sum(results["pnl_r"])),
        "n_trades": results.n_filled,  # len(results) also counts NO_FILL rows
        "max_dd_r": results.max_drawdown_r,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    # ── Gate 0: BugGuard ──────────────────────────────────────────────────
    gk = GateKeeper(
        strategy_name="SMA Crossover Demo",
        n_bars=len(CLOSE),
        bar_minutes=5,           # the synthetic bars stand for 5m bars
        resolution_minutes=5,
        source_path=__file__,
        # GROSS: no spreads_used / expected_costs (GK-00 lists spreads_used as
        # missing; for a GROSS backtest that is expected)
    )

    # Pass one real simulation so the trade-level checks run too
    # (entry on the previous exit bar, trade count, no-SL, open-price entry,
    # fixed cost). Gate 0 does not look for look-ahead: run the audit for that.
    sig, dirs = find_signals(CLOSE, 10, 50)
    sl = ATR_VALS[sig] * 1.5
    baseline = simulate_trades(
        HIGH, LOW, CLOSE, sig, dirs, sl, sl * 2.0, max_hold=100, open_prices=OPEN,
        preflight=False,
    )
    # Bar times of the synthetic 5m bars: 300,000 bars = 2.9 years. BG-06 takes
    # the data period from them (as 1m bars this would be 7 months: too short).
    bar_times = 1577836800 + 300 * np.arange(len(CLOSE))
    gk.gate0_validate(trades=baseline, sl_distances=sl, avg_price=float(np.median(CLOSE)),
                      timestamps=bar_times)

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
        r = simulate_trades(
            HIGH[start:end], LOW[start:end], CLOSE[start:end],
            sig, dirs, sl, tp, max_hold=100,
            open_prices=OPEN[start:end], preflight=False,
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
    results = simulate_trades(
        HIGH, LOW, CLOSE, sig, dirs, sl, tp,
        max_hold=100, open_prices=OPEN, preflight=False,
    )

    mc = MonteCarloDD(results.filled["pnl_r"], n_sims=10_000, risk_pct=0.01, seed=42)
    mc.run()
    result4 = gk.gate4_montecarlo(mc, dd_limit=0.20)

    # ── Summary ───────────────────────────────────────────────────────────
    gk.summary()

    # ── Bonus: StressTest ─────────────────────────────────────────────────
    if result4.passed:
        print("\n--- Stress Test (post-validation) ---")
        st = StressTest(results.filled["pnl_r"], n_sims=1000, seed=42)
        report = st.run_all(block_size=10)
        print(f"Baseline DD@95%: {report['baseline']['dd_95']*100:.1f}%")
        print(f"Block bootstrap DD@95%: {report['block_bootstrap']['dd_95']*100:.1f}%")
        for name, scenario in report["degraded"].items():
            print(f"  {name}: DD@95%={scenario['dd_95']*100:.1f}%, "
                  f"expectancy={scenario['expectancy_r']:.3f}R")


if __name__ == "__main__":
    main()
