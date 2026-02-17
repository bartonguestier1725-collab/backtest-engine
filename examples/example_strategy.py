"""Example: SMA crossover strategy using backtest_engine.

This demonstrates the full workflow:
1. Generate synthetic OHLCV data
2. Compute indicators (SMA)
3. Generate entry signals
4. Run simulation with simulate_trades
5. Analyze results with MonteCarloDD
6. Validate with Walk-Forward and CSCV
"""

import numpy as np

from backtest_engine import (
    simulate_trades, LONG, SHORT,
    EXIT_SL, EXIT_TP, EXIT_TIME, EXIT_BE,
    sma, atr, BrokerCost, MonteCarloDD, WalkForward,
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
    high, low, close, _ = generate_synthetic_data(5000)
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

    # 5. Simulate trades
    results = simulate_trades(
        high, low, close,
        signal_bars, directions, sl_distances, tp_distances,
        max_hold=100,
        exit_mode="rr",
        be_trigger_pct=0.5,  # Move SL to breakeven at 50% of TP
    )

    # 6. Analyze results
    pnl = results["pnl_r"]
    print(f"\n--- Trade Results ---")
    print(f"Total trades: {len(pnl)}")
    print(f"Win rate: {np.mean(pnl > 0) * 100:.1f}%")
    print(f"Avg PnL: {np.mean(pnl):.3f}R")
    print(f"Total PnL: {np.sum(pnl):.1f}R")
    print(f"Max MFE: {np.max(results['mfe_r']):.2f}R")
    print(f"Max MAE: {np.min(results['mae_r']):.2f}R")

    # Exit type breakdown
    for exit_type, name in [(EXIT_SL, "SL"), (EXIT_TP, "TP"), (EXIT_TIME, "Timeout"), (EXIT_BE, "BE")]:
        count = np.sum(results["exit_type"] == exit_type)
        if count > 0:
            print(f"  {name}: {count} ({count/len(pnl)*100:.0f}%)")

    # 7. Cost analysis
    cost_model = BrokerCost.tradeview_ilc()
    cost_r = cost_model.as_r("EURUSD", np.mean(sl_distances))
    pnl_after_costs = pnl - cost_r
    print(f"\n--- After Costs (Tradeview ILC) ---")
    print(f"Cost per trade: {cost_r:.4f}R")
    print(f"Avg PnL after costs: {np.mean(pnl_after_costs):.3f}R")
    print(f"Total PnL after costs: {np.sum(pnl_after_costs):.1f}R")

    # 8. Monte Carlo analysis
    mc = MonteCarloDD(pnl_after_costs, n_sims=10000, risk_pct=0.01, seed=42)
    mc.run()
    print(f"\n--- Monte Carlo (1% risk, 10K sims) ---")
    print(f"DD 50th pct: {mc.dd_percentile(50)*100:.1f}%")
    print(f"DD 95th pct: {mc.dd_percentile(95)*100:.1f}%")
    print(f"DD 99th pct: {mc.dd_percentile(99)*100:.1f}%")
    print(f"Ruin prob (>30% DD): {mc.ruin_probability(0.30)*100:.1f}%")
    print(f"Kelly fraction: {mc.kelly_fraction()*100:.1f}%")

    # 9. Optimal risk sizing
    optimal = mc.optimal_risk_pct(max_dd=0.15, target_pct=95.0)
    print(f"Optimal risk% (15% DD @ 95th): {optimal*100:.2f}%")

    # 10. Walk-Forward validation
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
        r = simulate_trades(
            high[start:end], low[start:end], close[start:end],
            sig[valid], dirs[valid], sl_d[valid], tp_d[valid],
            max_hold=100,
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
