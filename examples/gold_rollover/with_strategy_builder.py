"""Gold Rollover — re-run with Strategy Builder.

Validates:
  1. MatchReport accounts for every entry candidate
  2. Timezone conversion is correct
  3. pip_size API works
  4. BG-14 fires on no-SL strategy
"""

import numpy as np
import pandas as pd
from backtest_engine.strategy_builder import build_time_based_signals
from backtest_engine.costs import BrokerCost
from backtest_engine.bug_guard import run_all_checks

# ── Load data ──────────────────────────────────────────────────────────
df = pd.read_csv("/tmp/xauusd_1h_5year.csv")
timestamps = df["time"].values.astype(np.int64)
opens = df["open"].values.astype(np.float64)
closes = df["close"].values.astype(np.float64)
highs = df["high"].values.astype(np.float64)
lows = df["low"].values.astype(np.float64)

print(f"Loaded: {len(df)} bars")

# ── Strategy Builder ───────────────────────────────────────────────────
trades, report = build_time_based_signals(
    timestamps, opens, closes,
    entry_hour_local=2,      # 02:00 JST
    exit_hour_local=14,      # 14:00 JST
    tz_offset_hours=9,       # JST = UTC+9
    direction=1,             # BUY only
    skip_weekdays_local=[3], # Skip Thursday JST
    entry_price_mode="open",
)

print("\n" + "=" * 60)
report.print_report()
print("=" * 60)

# ── Results ────────────────────────────────────────────────────────────
if len(trades) > 0:
    bc = BrokerCost.tradeview_ilc()
    trades["pnl_price"] = trades["exit_price"] - trades["entry_price"]
    trades["pnl_pips"] = bc.price_to_pips("XAUUSD", trades["pnl_price"])

    total_pips = trades["pnl_pips"].sum()
    avg_pips = trades["pnl_pips"].mean()
    win_rate = (trades["pnl_pips"] > 0).mean()
    cum = trades["pnl_pips"].cumsum()
    max_dd = (cum - cum.cummax()).min()

    print(f"\nTrades:    {len(trades)}")
    print(f"Total:     {total_pips:+,.1f} pips (GROSS)")
    print(f"Avg:       {avg_pips:+,.1f} pips/trade")
    print(f"Win Rate:  {win_rate:.1%}")
    print(f"Max DD:    {max_dd:,.1f} pips")

    # Yearly
    trades["year"] = trades["entry_time"].dt.year
    print(f"\n{'Year':>6} {'PnL':>10} {'n':>5} {'WR':>6} {'Avg':>8}")
    for year in sorted(trades["year"].unique()):
        yt = trades[trades["year"] == year]
        y_pnl = yt["pnl_pips"].sum()
        y_wr = (yt["pnl_pips"] > 0).mean()
        y_avg = yt["pnl_pips"].mean()
        print(f"  {year}  {y_pnl:>+9.1f}  {len(yt):>4}  {y_wr:>5.0%}  {y_avg:>+7.1f}")

# ── BugGuard BG-14 test ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("BugGuard check (should fire BG-14 for no-SL strategy)")
print("=" * 60)

# Simulate what a user would do: set SL to huge value
avg_price = float(np.median(closes))
fake_sl = np.full(len(trades), avg_price * 0.5)  # 50% of price = effectively no SL

run_all_checks(
    resolution_minutes=60,
    n_bars=len(df),
    bar_minutes=60,
    n_trades=len(trades),
    sl_distances=fake_sl,
    avg_price=avg_price,
    strict=False,
)
