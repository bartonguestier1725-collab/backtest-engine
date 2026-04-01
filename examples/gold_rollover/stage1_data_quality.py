"""Stage 1: Gold Rollover Strategy — AggVault data quality check.

ルール:
  - XAUUSD BUY only
  - Entry: 02:00 JST (= 17:00 UTC) at bar OPEN
  - Exit:  14:00 JST (= 05:00 UTC) at bar OPEN (= close of previous bar)
  - Skip: Thursday JST entries (= Wednesday 17:00 UTC)
  - No SL/TP, no costs (Stage 1)

比較対象: 記事の2024-2025バックテスト
  - 403 trades, +8,439 pips net (est. ~13,275 pips gross)
  - 54% win rate, max DD -1,811 pips
"""

import pandas as pd
import numpy as np
from datetime import timezone
from backtest_engine import fetch_aggvault

# ── Load data ──────────────────────────────────────────────────────────
timestamps, opens, highs, lows, closes, _ = fetch_aggvault(
    "XAUUSD", "1h", "2024-01-01", "2026-01-01",
)
df = pd.DataFrame({
    "open": opens, "high": highs, "low": lows, "close": closes,
}, index=pd.to_datetime(timestamps, unit="s", utc=True))
df.index.name = "datetime"
print(f"Loaded: {len(df)} bars, {df.index[0]} → {df.index[-1]}")

# ── Debug: check Sunday 17:00 bars ────────────────────────────────────
for dow in range(7):
    dow_17 = df[(df.index.hour == 17) & (df.index.dayofweek == dow)]
    dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    if len(dow_17) > 0:
        print(f"  {dow_names[dow]} 17:00 UTC bars: {len(dow_17)}")

# ── Find entry bars: 17:00 UTC, excluding Wednesday (= Thursday JST) ──
# JST Thursday entry (02:00 JST Thu) = 17:00 UTC Wednesday → skip dayofweek==2
entry_mask = (df.index.hour == 17) & (df.index.dayofweek != 2)  # 2 = Wednesday UTC
entry_bars = df[entry_mask].copy()
print(f"\nEntry candidates (17:00 UTC, excl Wed): {len(entry_bars)}")

# Check JST day distribution
for _, row_data in entry_bars.iterrows():
    pass  # just iterate to check
jst_entries = entry_bars.index + pd.Timedelta(hours=9)
print("  JST day distribution:")
for dow in range(7):
    n = (jst_entries.dayofweek == dow).sum()
    dow_names_jp = ["月Mon","火Tue","水Wed","木Thu","金Fri","土Sat","日Sun"]
    if n > 0:
        print(f"    {dow_names_jp[dow]}: {n}")

# ── Match exits: 05:00 UTC on the next day ─────────────────────────────
exit_mask = df.index.hour == 5
exit_bars = df[exit_mask]

trades = []
for entry_time, entry_row in entry_bars.iterrows():
    # Find the 05:00 UTC bar that comes after this entry (should be ~12h later)
    next_day_05 = entry_time.normalize() + pd.Timedelta(hours=29)  # +29h from midnight = next day 05:00
    # Find closest exit bar
    exit_candidates = exit_bars[exit_bars.index >= next_day_05 - pd.Timedelta(hours=1)]
    exit_candidates = exit_candidates[exit_candidates.index <= next_day_05 + pd.Timedelta(hours=1)]

    if len(exit_candidates) == 0:
        continue

    exit_row = exit_candidates.iloc[0]
    exit_time = exit_candidates.index[0]

    # BUY: entry at open of 17:00 bar, exit at open of 05:00 bar
    entry_price = entry_row["open"]
    exit_price = exit_row["open"]

    pnl_price = exit_price - entry_price
    pnl_pips = pnl_price / 0.10  # XAUUSD: 1 pip = $0.10

    # Day of week in JST (entry at 02:00 JST)
    jst_time = entry_time + pd.Timedelta(hours=9)

    trades.append({
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_price": pnl_price,
        "pnl_pips": pnl_pips,
        "jst_weekday": jst_time.strftime("%A"),
        "jst_weekday_num": jst_time.dayofweek,
        "hold_hours": (exit_time - entry_time).total_seconds() / 3600,
    })

trades_df = pd.DataFrame(trades)
print(f"\nCompleted trades: {len(trades_df)}")

# ── Overall Results ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Stage 1: Gold Rollover (GROSS, no costs)")
print("=" * 60)

total_pips = trades_df["pnl_pips"].sum()
avg_pips = trades_df["pnl_pips"].mean()
win_rate = (trades_df["pnl_pips"] > 0).mean()
n_trades = len(trades_df)

# Max drawdown in pips (cumulative)
cum_pips = trades_df["pnl_pips"].cumsum()
running_max = cum_pips.cummax()
drawdown = cum_pips - running_max
max_dd_pips = drawdown.min()

print(f"Trades:      {n_trades}")
print(f"Total PnL:   {total_pips:+,.1f} pips")
print(f"Avg PnL:     {avg_pips:+,.1f} pips/trade")
print(f"Win Rate:    {win_rate:.1%}")
print(f"Max DD:      {max_dd_pips:,.1f} pips")
print(f"Hold hours:  {trades_df['hold_hours'].mean():.1f}h avg")

# ── Article comparison ─────────────────────────────────────────────────
print("\n" + "-" * 60)
print("Article comparison (Gross estimate)")
print("-" * 60)
article_net = 8439
article_trades = 403
article_cost_per_trade = 12  # pips
article_gross_est = article_net + article_trades * article_cost_per_trade
print(f"Article:     {article_trades} trades, +{article_gross_est:,} pips gross (est)")
print(f"AggVault:    {n_trades} trades, {total_pips:+,.1f} pips gross")
print(f"Diff trades: {n_trades - article_trades:+d}")
print(f"Diff pips:   {total_pips - article_gross_est:+,.1f}")

# ── Day-of-week breakdown (JST) ───────────────────────────────────────
print("\n" + "-" * 60)
print("Day-of-week breakdown (JST)")
print("-" * 60)
day_names = {0: "月曜 Mon", 1: "火曜 Tue", 2: "水曜 Wed", 3: "木曜 Thu", 4: "金曜 Fri"}

# Article reference
article_dow = {"月曜 Mon": (27.9, 0.48), "火曜 Tue": (10.8, 0.56), "水曜 Wed": (34.6, 0.59), "金曜 Fri": (30.1, 0.60)}

for dow_num in sorted(trades_df["jst_weekday_num"].unique()):
    day_label = day_names.get(dow_num, f"Day{dow_num}")
    mask = trades_df["jst_weekday_num"] == dow_num
    day_trades = trades_df[mask]
    day_avg = day_trades["pnl_pips"].mean()
    day_wr = (day_trades["pnl_pips"] > 0).mean()
    day_n = len(day_trades)

    # Article reference
    art = article_dow.get(day_label, None)
    art_str = f" (article: {art[0]:+.1f}pips, {art[1]:.0%})" if art else ""

    print(f"  {day_label}: {day_avg:+6.1f} pips, WR {day_wr:.0%}, n={day_n}{art_str}")

# ── Monthly breakdown ─────────────────────────────────────────────────
print("\n" + "-" * 60)
print("Monthly breakdown")
print("-" * 60)
trades_df["month"] = trades_df["entry_time"].dt.to_period("M")
monthly = trades_df.groupby("month").agg(
    n=("pnl_pips", "count"),
    total_pips=("pnl_pips", "sum"),
    avg_pips=("pnl_pips", "mean"),
    win_rate=("pnl_pips", lambda x: (x > 0).mean()),
).reset_index()

for _, row in monthly.iterrows():
    bar = "+" * int(max(0, row["total_pips"] / 20)) if row["total_pips"] > 0 else "-" * int(max(0, abs(row["total_pips"]) / 20))
    print(f"  {row['month']}: {row['total_pips']:+7.1f} pips, n={row['n']:2d}, WR={row['win_rate']:.0%} {bar}")

# ── Yearly breakdown ──────────────────────────────────────────────────
print("\n" + "-" * 60)
print("Yearly breakdown")
print("-" * 60)
trades_df["year"] = trades_df["entry_time"].dt.year
yearly = trades_df.groupby("year").agg(
    n=("pnl_pips", "count"),
    total_pips=("pnl_pips", "sum"),
    avg_pips=("pnl_pips", "mean"),
    win_rate=("pnl_pips", lambda x: (x > 0).mean()),
    max_dd=("pnl_pips", lambda x: (x.cumsum() - x.cumsum().cummax()).min()),
).reset_index()

for _, row in yearly.iterrows():
    print(f"  {row['year']}: {row['total_pips']:+8.1f} pips, n={row['n']}, WR={row['win_rate']:.0%}, MaxDD={row['max_dd']:,.1f} pips")

print("\n" + "=" * 60)
print("Done. Next: Stage 2 (add Tradeview ILC costs + swap)")
