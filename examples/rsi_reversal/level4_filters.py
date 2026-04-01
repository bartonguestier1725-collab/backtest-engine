"""RSI Reversal Strategy — Level 4: Structural Filters.

Level 3 proved the optimized RSI alone is overfit (PBO=80%).
Can structural filters create genuine edge?

Filters tested:
  1. SMA trend filter — only trade in the direction of the trend
  2. Time-of-day filter — avoid low-liquidity hours (Asian session for EURUSD)
  3. Volatility filter — skip low-volatility bars (ATR < threshold)

Key difference from Level 2:
  - Level 2 optimized WHAT to trade (RSI params)
  - Level 4 filters WHEN to trade (market conditions)
  This is a structural improvement, not parameter fitting.

Then re-run WFA + CSCV: does PBO improve?
"""

import itertools
import numpy as np
import pandas as pd
from backtest_engine import (
    fetch_aggvault, simulate_trades,
    LONG, SHORT, rsi, sma, atr, BrokerCost,
    WalkForward, CSCV, MonteCarloDD, StressTest,
)

# ── Step 1: データ取得 ────────────────────────────────────────────────
timestamps, opens, highs, lows, closes, _ = fetch_aggvault(
    "EURUSD", "1h", "2021-04-01", "2026-03-31",
)
dt_index = pd.to_datetime(timestamps, unit="s", utc=True)
n_bars = len(timestamps)
hours = np.array([dt.hour for dt in dt_index])
print(f"Loaded: {n_bars} bars, {dt_index[0]} → {dt_index[-1]}")

# ── Pre-compute indicators ──────────────────────────────────────────
atr_vals = atr(highs, lows, closes, 14)
sma_200 = sma(closes, 200)  # Trend filter
cost_model = BrokerCost.tradeview_ilc()


def evaluate_fn(params: dict, start_idx: int = 0, end_idx: int = 0) -> float:
    """RSI backtest with structural filters. Returns total_r."""
    if end_idx == 0:
        end_idx = n_bars

    rsi_period = params.get("rsi_period", 14)
    oversold = params.get("oversold", 30)
    overbought = params.get("overbought", 70)
    sl_mult = params.get("sl_atr_mult", 2.0)
    rr_ratio = params.get("rr_ratio", 2.0)
    max_hold = params.get("max_hold", 48)

    # Filter flags
    use_trend = params.get("trend_filter", False)
    use_time = params.get("time_filter", False)
    use_vol = params.get("vol_filter", False)

    rsi_vals = rsi(closes, rsi_period)
    warmup = max(rsi_period + 1, 201)  # Need 200 bars for SMA

    signal_bars = []
    directions = []
    for j in range(max(warmup, start_idx), min(end_idx, len(closes) - 1)):
        if np.isnan(rsi_vals[j]) or np.isnan(atr_vals[j]):
            continue
        if np.isnan(sma_200[j]):
            continue

        # Time filter: skip Asian session (21:00-07:00 UTC)
        if use_time and (hours[j] >= 21 or hours[j] < 7):
            continue

        # Volatility filter: skip low-vol bars (ATR < 50% of 14-period avg)
        if use_vol:
            atr_avg = np.mean(atr_vals[max(0, j-100):j+1])
            if atr_vals[j] < atr_avg * 0.5:
                continue

        if rsi_vals[j] < oversold:
            # Trend filter: only buy above SMA(200)
            if use_trend and closes[j] < sma_200[j]:
                continue
            signal_bars.append(j)
            directions.append(LONG)
        elif rsi_vals[j] > overbought:
            # Trend filter: only sell below SMA(200)
            if use_trend and closes[j] > sma_200[j]:
                continue
            signal_bars.append(j)
            directions.append(SHORT)

    if len(signal_bars) < 10:
        return -999.0

    signal_bars = np.array(signal_bars, dtype=np.int32)
    directions = np.array(directions, dtype=np.int8)
    sl_distances = atr_vals[signal_bars] * sl_mult
    tp_distances = sl_distances * rr_ratio
    instruments = ["EURUSD"] * len(signal_bars)
    entry_costs = cost_model.per_trade_cost(instruments, sl_distances)

    res = simulate_trades(
        highs, lows, closes,
        signal_bars, directions, sl_distances, tp_distances,
        max_hold=max_hold, exit_mode="rr",
        open_prices=opens, entry_costs=entry_costs,
        preflight=False,
    )

    pnl = res["pnl_r"]
    if len(pnl) < 10:
        return -999.0
    return float(np.sum(pnl))


def run_detail(params: dict):
    """Run full backtest and return detailed results."""
    rsi_period = params["rsi_period"]
    rsi_vals = rsi(closes, rsi_period)
    warmup = max(rsi_period + 1, 201)
    use_trend = params.get("trend_filter", False)
    use_time = params.get("time_filter", False)
    use_vol = params.get("vol_filter", False)

    signal_bars = []
    directions = []
    for j in range(warmup, len(closes) - 1):
        if np.isnan(rsi_vals[j]) or np.isnan(atr_vals[j]) or np.isnan(sma_200[j]):
            continue
        if use_time and (hours[j] >= 21 or hours[j] < 7):
            continue
        if use_vol:
            atr_avg = np.mean(atr_vals[max(0, j-100):j+1])
            if atr_vals[j] < atr_avg * 0.5:
                continue
        if rsi_vals[j] < params["oversold"]:
            if use_trend and closes[j] < sma_200[j]:
                continue
            signal_bars.append(j)
            directions.append(LONG)
        elif rsi_vals[j] > params["overbought"]:
            if use_trend and closes[j] > sma_200[j]:
                continue
            signal_bars.append(j)
            directions.append(SHORT)

    signal_bars = np.array(signal_bars, dtype=np.int32)
    directions = np.array(directions, dtype=np.int8)
    sl_distances = atr_vals[signal_bars] * params["sl_atr_mult"]
    tp_distances = sl_distances * params["rr_ratio"]
    instruments = ["EURUSD"] * len(signal_bars)
    entry_costs = cost_model.per_trade_cost(instruments, sl_distances)

    return simulate_trades(
        highs, lows, closes,
        signal_bars, directions, sl_distances, tp_distances,
        max_hold=params["max_hold"], exit_mode="rr",
        open_prices=opens, entry_costs=entry_costs,
    ), directions


# ── Step 2: フィルターの効果比較 ─────────────────────────────────────
print("\n" + "=" * 70)
print("Step 2: Filter Impact Comparison (RSI 14/30/70, default params)")
print("=" * 70)

base = {"rsi_period": 14, "oversold": 30, "overbought": 70,
        "sl_atr_mult": 2.0, "rr_ratio": 2.0, "max_hold": 48}

configs = [
    ("No filter",       {**base}),
    ("+ Trend (SMA200)", {**base, "trend_filter": True}),
    ("+ Time (EU/US)",  {**base, "time_filter": True}),
    ("+ Volatility",    {**base, "vol_filter": True}),
    ("+ All filters",   {**base, "trend_filter": True, "time_filter": True, "vol_filter": True}),
]

print(f"\n{'Config':>20} | {'PF':>5} {'WR':>5} {'Exp':>7} {'Trades':>6} {'MaxDD':>7} {'Sharpe':>6}")
print(f"{'-'*70}")

for name, cfg in configs:
    res, dirs = run_detail(cfg)
    pnl = res["pnl_r"]
    if len(pnl) < 10:
        print(f"{name:>20} | insufficient trades")
        continue
    print(f"{name:>20} | {res.profit_factor:>5.2f} {res.win_rate:>4.0%} "
          f"{res.expectancy_r:>+7.3f} {len(pnl):>6} {res.max_drawdown_r:>7.1f} "
          f"{res.sharpe_r:>6.2f}")

# ── Step 3: フィルター付きで最適化 ───────────────────────────────────
print("\n" + "=" * 70)
print("Step 3: Optimization WITH filters (structural improvement)")
print("=" * 70)

# Smaller grid (filters reduce the parameter space importance)
param_grid_filtered = [
    {"rsi_period": p, "oversold": os_val, "overbought": ob_val,
     "sl_atr_mult": sl, "rr_ratio": rr, "max_hold": 48,
     "trend_filter": True, "time_filter": True, "vol_filter": True}
    for p in [7, 14, 21]
    for os_val in [20, 25, 30]
    for ob_val in [65, 70, 75]
    for sl in [2.0, 2.5]
    for rr in [2.0, 3.0]
]
print(f"Grid: {len(param_grid_filtered)} combos (all filters ON)")

# Find best
best_total = -999.0
best_params = None
for params in param_grid_filtered:
    total = evaluate_fn(params)
    if total > best_total:
        best_total = total
        best_params = params

if best_params:
    res, dirs = run_detail(best_params)
    pnl = res["pnl_r"]
    print(f"\nBest: RSI({best_params['rsi_period']}) "
          f"{best_params['oversold']}/{best_params['overbought']}, "
          f"SL={best_params['sl_atr_mult']}×ATR, RR={best_params['rr_ratio']}")
    print(f"  PF={res.profit_factor:.2f}, WR={res.win_rate:.0%}, "
          f"Exp={res.expectancy_r:+.3f}R, n={len(pnl)}, "
          f"DD={res.max_drawdown_r:.1f}R")

    # Yearly
    entry_bars_arr = res["entry_bar"]
    print(f"\n  {'Year':>6} {'PnL(R)':>10} {'n':>5} {'WR':>6} {'PF':>6}")
    for year in range(2021, 2027):
        year_mask = np.array([dt_index[b].year == year for b in entry_bars_arr])
        if not np.any(year_mask):
            continue
        y_pnl = pnl[year_mask]
        y_total = np.sum(y_pnl)
        y_wr = (y_pnl > 0).mean()
        y_wins = y_pnl[y_pnl > 0]
        y_losses = y_pnl[y_pnl < 0]
        y_pf = np.sum(y_wins) / abs(np.sum(y_losses)) if len(y_losses) > 0 and np.sum(y_losses) != 0 else float('inf')
        print(f"    {year}  {y_total:>+9.1f}R  {len(y_pnl):>4}  {y_wr:>5.0%}  {y_pf:>5.2f}")

# ── Step 4: WFA + CSCV (フィルター付き) ──────────────────────────────
print("\n" + "=" * 70)
print("Step 4: WFA + CSCV — Is the filtered strategy still overfit?")
print("=" * 70)

# WFA
print(f"\n  Walk-Forward: {len(param_grid_filtered)} params, 5 splits...")
wf = WalkForward(n_bars=n_bars, is_ratio=0.7, n_splits=5, anchored=False)
wfa_result = wf.run(param_grid_filtered, evaluate_fn)
ratio = wfa_result['is_oos_ratio']
print(f"  OOS mean PnL:      {wfa_result['oos_mean']:.3f}R")
print(f"  OOS positive frac: {wfa_result['oos_positive_frac']:.1%}")
print(f"  OOS/IS ratio:      {ratio:.2f}")

# CSCV
print(f"\n  CSCV: {len(param_grid_filtered)} params, 10 splits...")
cscv = CSCV(n_splits=10)
cscv_result = cscv.run(param_grid_filtered, evaluate_fn, n_bars=n_bars)
pbo = cscv_result['pbo']
print(f"  PBO:    {pbo:.1%}")
print(f"  Logit:  {cscv_result['logit_mean']:.2f}")

# ── Step 5: Monte Carlo (if we have a viable strategy) ────────────────
if best_params:
    print("\n" + "=" * 70)
    print("Step 5: Monte Carlo & Stress Test (filtered best)")
    print("=" * 70)

    res_full, _ = run_detail(best_params)
    pnl_full = res_full["pnl_r"]

    mc = MonteCarloDD(pnl_full, n_sims=10_000, risk_pct=0.01, seed=42)
    mc.run()
    print(f"\n  Monte Carlo ({len(pnl_full)} trades):")
    print(f"    DD 95th: {mc.dd_percentile(95)*100:.1f}%")
    print(f"    DD 99th: {mc.dd_percentile(99)*100:.1f}%")
    print(f"    Kelly:   {mc.kelly_fraction()*100:.1f}%")

    st = StressTest(pnl_full, n_sims=1000, seed=42)
    report = st.run_all(block_size=10)
    print(f"\n  Stress Test:")
    print(f"    Baseline DD@95%:       {report['baseline']['dd_95']*100:.1f}%")
    print(f"    Block bootstrap DD@95%: {report['block_bootstrap']['dd_95']*100:.1f}%")
    print(f"    Combined deg DD@95%:   {report['degraded']['combined']['dd_95']*100:.1f}%")

# ── Summary: Level 1-4 comparison ────────────────────────────────────
print(f"\n{'='*70}")
print("Final Summary: Levels 1-4")
print(f"{'='*70}")
print(f"""
  Level 1: RSI(14) 30/70, no filter        → PF=1.05, no edge
  Level 2: RSI(21) 20/70, no filter        → PF=1.33, "looks good" (in-sample)
  Level 3: WFA+CSCV on Level 2             → PBO=80%, OVERFIT
  Level 4: RSI + trend/time/vol filters    → PBO={pbo:.0%}

  {'✓ Filters reduced overfitting!' if pbo < 0.40 else '⚠ Filters helped but PBO still high' if pbo < 0.60 else '✗ Even with filters, RSI reversal on EURUSD lacks robust edge'}

  Key takeaways:
  1. Parameter optimization alone ≠ real edge (Level 2→3)
  2. Structural filters (trend/time/vol) improve robustness
  3. PBO is the ultimate arbiter: <40% to trust a strategy
  4. RSI reversal as a standalone system has fundamental limits
     on trending pairs like EURUSD
""")
