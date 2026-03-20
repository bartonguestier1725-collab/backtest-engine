"""Example: SMA crossover strategy using backtest_engine.

This demonstrates the full workflow:
1. Generate synthetic OHLCV data
2. Compute indicators (SMA)
3. Generate entry signals
4. Run simulation with simulate_trades
5. Inspect TradeResults metrics
6. Analyze results with MonteCarloDD
7. Stress-test with StressTest (block bootstrap + degradation)
8. Validate with Walk-Forward
"""

import numpy as np

from backtest_engine import (
    simulate_trades, LONG, SHORT,
    EXIT_SL, EXIT_TP, EXIT_TIME, EXIT_BE,
    sma, atr, BrokerCost, MonteCarloDD, StressTest, WalkForward,
)


def generate_synthetic_data(n_bars=5000, seed=42):
    """Generate synthetic OHLCV data with trending behavior."""
    np.random.seed(seed)
    # Random walk with slight upward drift
    returns = np.random.randn(n_bars) * 0.001 + 0.00005
    close = 1.10000 + np.cumsum(returns)

    # Generate realistic OHLC from close
    spread = np.abs(np.random.randn(n_bars)) * 0.0003
    high = close + spread
    low = close - spread
    open_price = close + np.random.randn(n_bars) * 0.0001

    return high, low, close, open_price


def find_signals(close, fast_period=10, slow_period=50):
    """SMA crossover signal generation."""
    fast_sma = sma(close, fast_period)
    slow_sma = sma(close, slow_period)

    signal_bars = []
    directions = []

    for i in range(slow_period, len(close) - 1):
        # Golden cross: fast crosses above slow → LONG
        if fast_sma[i - 1] <= slow_sma[i - 1] and fast_sma[i] > slow_sma[i]:
            signal_bars.append(i)
            directions.append(LONG)
        # Death cross: fast crosses below slow → SHORT
        elif fast_sma[i - 1] >= slow_sma[i - 1] and fast_sma[i] < slow_sma[i]:
            signal_bars.append(i)
            directions.append(SHORT)

    return np.array(signal_bars, dtype=np.int32), np.array(directions, dtype=np.int8)


def main():
    print("=" * 60)
    print("backtest_engine Example: SMA Crossover Strategy")
    print("=" * 60)

    # 1. Generate data
    high, low, close, open_price = generate_synthetic_data(5000)
    print(f"\nData: {len(close)} bars, price range {close.min():.5f} — {close.max():.5f}")

    # 2. Compute ATR for dynamic SL/TP
    atr_vals = atr(high, low, close, 14)

    # 3. Generate signals
    signal_bars, directions = find_signals(close, fast_period=10, slow_period=50)
    print(f"Signals: {len(signal_bars)} trades ({np.sum(directions == LONG)} long, {np.sum(directions == SHORT)} short)")

    if len(signal_bars) == 0:
        print("No signals generated!")
        return

    # 4. Set SL/TP based on ATR
    sl_distances = atr_vals[signal_bars] * 1.5  # 1.5 ATR SL
    tp_distances = atr_vals[signal_bars] * 3.0  # 3.0 ATR TP (2:1 RR)

    # 5. Per-trade cost (ATR-varying SL → cost varies per trade)
    cost_model = BrokerCost.tradeview_ilc()
    instruments = ["EURUSD"] * len(signal_bars)
    cost_r_array = cost_model.per_trade_cost(instruments, sl_distances)

    # 6. Simulate trades (Grade A: open_prices + entry_costs)
    results = simulate_trades(
        high, low, close,
        signal_bars, directions, sl_distances, tp_distances,
        max_hold=100,
        exit_mode="rr",
        be_trigger_pct=0.5,  # Move SL to breakeven at 50% of TP
        open_prices=open_price,
        entry_costs=cost_r_array,
    )

    # 7. Quality grade
    print(f"\nQuality Grade: {results.quality.grade}")

    # 8. Analyze results using TradeResults convenience metrics
    pnl = results["pnl_r"]
    print(f"\n--- Trade Results (net of costs) ---")
    print(f"Total trades: {len(pnl)}")
    print(f"Profit Factor: {results.profit_factor:.2f}")
    print(f"Win rate: {results.win_rate:.1%}")
    print(f"Expectancy: {results.expectancy_r:.3f}R")
    print(f"Sharpe (R): {results.sharpe_r:.2f}")
    print(f"Sortino (R): {results.sortino_r:.2f}")
    print(f"Max Drawdown: {results.max_drawdown_r:.1f}R")
    print(f"Recovery Factor: {results.recovery_factor:.2f}")
    print(f"Avg cost: {np.mean(results['cost_r']):.4f}R")

    # Exit type breakdown
    for exit_type, name in [(EXIT_SL, "SL"), (EXIT_TP, "TP"), (EXIT_TIME, "Timeout"), (EXIT_BE, "BE")]:
        count = np.sum(results["exit_type"] == exit_type)
        if count > 0:
            print(f"  {name}: {count} ({count/len(pnl)*100:.0f}%)")

    # 9. Monte Carlo analysis
    mc = MonteCarloDD(pnl, n_sims=10000, risk_pct=0.01, seed=42)
    mc.run()
    print(f"\n--- Monte Carlo (1% risk, 10K sims) ---")
    print(f"DD 50th pct: {mc.dd_percentile(50)*100:.1f}%")
    print(f"DD 95th pct: {mc.dd_percentile(95)*100:.1f}%")
    print(f"DD 99th pct: {mc.dd_percentile(99)*100:.1f}%")
    print(f"Ruin prob (>30% DD): {mc.ruin_probability(0.30)*100:.1f}%")
    print(f"Kelly fraction: {mc.kelly_fraction()*100:.1f}%")

    # 10. Optimal risk sizing
    optimal = mc.optimal_risk_pct(max_dd=0.15, target_pct=95.0)
    print(f"Optimal risk% (15% DD @ 95th): {optimal*100:.2f}%")

    # 11. Stress test (block bootstrap + degradation)
    st = StressTest(pnl, n_sims=1000, seed=42)
    report = st.run_all(block_size=10)
    print(f"\n--- Stress Test ---")
    print(f"Baseline DD@95%: {report['baseline']['dd_95']*100:.1f}%")
    print(f"Block bootstrap DD@95%: {report['block_bootstrap']['dd_95']*100:.1f}%")
    for name, scenario in report["degraded"].items():
        print(f"  {name}: DD@95%={scenario['dd_95']*100:.1f}%, "
              f"expectancy={scenario['expectancy_r']:.3f}R")

    # 12. Walk-Forward validation
    def evaluate(params, start, end):
        sig, dirs = find_signals(close[start:end], params["fast"], params["slow"])
        if len(sig) == 0:
            return 0.0
        sl_d = atr_vals[start:end][sig] * 1.5
        tp_d = atr_vals[start:end][sig] * 3.0
        # Clamp to valid bars
        valid = sig < (end - start)
        if not np.any(valid):
            return 0.0
        instruments = ["EURUSD"] * int(np.sum(valid))
        costs = cost_model.per_trade_cost(instruments, sl_d[valid])
        r = simulate_trades(
            high[start:end], low[start:end], close[start:end],
            sig[valid], dirs[valid], sl_d[valid], tp_d[valid],
            max_hold=100,
            open_prices=open_price[start:end], entry_costs=costs,
            preflight=False,  # suppress per-iteration warnings
        )
        return float(np.mean(r["pnl_r"])) if len(r) > 0 else 0.0

    param_grid = [
        {"fast": 5, "slow": 30},
        {"fast": 10, "slow": 50},
        {"fast": 20, "slow": 100},
    ]

    wf = WalkForward(n_bars=len(close), is_ratio=0.7, n_splits=3)
    wf_result = wf.run(param_grid, evaluate)
    print(f"\n--- Walk-Forward (3 splits) ---")
    print(f"OOS mean: {wf_result['oos_mean']:.3f}R")
    print(f"OOS positive frac: {wf_result['oos_positive_frac']*100:.0f}%")
    for i, split in enumerate(wf_result["splits"]):
        print(f"  Split {i+1}: IS={split['is_metric']:.3f}, OOS={split['oos_metric']:.3f}, params={split['best_params']}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
