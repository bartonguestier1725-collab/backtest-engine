"""Gold Rollover Strategy — 5-year backtest (2021-04 to 2026-03).

Question: Does the edge persist beyond the 2024-2025 gold bull market?
"""

import pandas as pd
import numpy as np
from backtest_engine import fetch_aggvault

# ── Load data ──────────────────────────────────────────────────────────
timestamps, opens, highs, lows, closes, _ = fetch_aggvault(
    "XAUUSD", "1h", "2021-04-01", "2026-03-31",
)
df = pd.DataFrame({
    "open": opens, "high": highs, "low": lows, "close": closes,
}, index=pd.to_datetime(timestamps, unit="s", utc=True))
df.index.name = "datetime"
print(f"Loaded: {len(df)} bars, {df.index[0]} → {df.index[-1]}")
print(f"Price range: ${closes.min():.0f} → ${closes.max():.0f}")

# ── Find entry bars: 17:00 UTC on Mon/Tue/Thu (= Tue/Wed/Fri JST) ────
# Skip Wed UTC (= Thu JST, 3x swap)
# Skip Fri UTC (= Sat JST, exit would be on closed market)
# Skip Sun UTC (= Mon JST, market may not be open yet at 17:00)
entry_mask = (
    (df.index.hour == 17)
    & (df.index.dayofweek.isin([0, 1, 3]))  # Mon, Tue, Thu UTC only
)
entry_bars = df[entry_mask].copy()
print(f"\nEntry candidates: {len(entry_bars)}")

# ── Match exits: 05:00 UTC on the next calendar day ──────────────────
exit_mask = df.index.hour == 5
exit_bars = df[exit_mask]

trades = []
unmatched = 0
for entry_time, entry_row in entry_bars.iterrows():
    # Target: next day 05:00 UTC (exactly 12h later for 17:00 entry)
    target_exit = entry_time.normalize() + pd.Timedelta(hours=29)
    # Exact match: find the bar at that exact hour
    exit_match = exit_bars[
        (exit_bars.index >= target_exit - pd.Timedelta(minutes=30))
        & (exit_bars.index <= target_exit + pd.Timedelta(minutes=30))
    ]

    if len(exit_match) == 0:
        unmatched += 1
        continue

    exit_row = exit_match.iloc[0]
    exit_time = exit_match.index[0]

    entry_price = entry_row["open"]
    exit_price = exit_row["open"]
    pnl_price = exit_price - entry_price
    pnl_pips = pnl_price / 0.10  # XAUUSD: 1 pip = $0.10

    jst_time = entry_time + pd.Timedelta(hours=9)

    trades.append({
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_pips": pnl_pips,
        "jst_weekday_num": jst_time.dayofweek,
        "year": entry_time.year,
    })

trades_df = pd.DataFrame(trades)
print(f"Completed trades: {len(trades_df)}")
print(f"Unmatched entries: {unmatched} (holidays etc.)")

# ── Overall Results ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Gold Rollover — 5-Year Backtest (GROSS, no costs)")
print("=" * 70)

total_pips = trades_df["pnl_pips"].sum()
avg_pips = trades_df["pnl_pips"].mean()
win_rate = (trades_df["pnl_pips"] > 0).mean()
n_trades = len(trades_df)

cum_pips = trades_df["pnl_pips"].cumsum()
running_max = cum_pips.cummax()
drawdown = cum_pips - running_max
max_dd_pips = drawdown.min()

print(f"Period:      2021-04 → 2026-03 (5 years)")
print(f"Trades:      {n_trades}")
print(f"Total PnL:   {total_pips:+,.1f} pips (GROSS)")
print(f"Avg PnL:     {avg_pips:+,.1f} pips/trade")
print(f"Win Rate:    {win_rate:.1%}")
print(f"Max DD:      {max_dd_pips:,.1f} pips")
print(f"Recovery:    {abs(total_pips / max_dd_pips):.1f}x")

# ── Yearly breakdown (THE KEY TABLE) ──────────────────────────────────
print("\n" + "=" * 70)
print("Yearly breakdown — Does the edge persist?")
print("=" * 70)

# Gold price at year boundaries for context
yearly_prices = df.resample("YS").first()["close"]

for year in sorted(trades_df["year"].unique()):
    yt = trades_df[trades_df["year"] == year]
    y_total = yt["pnl_pips"].sum()
    y_avg = yt["pnl_pips"].mean()
    y_wr = (yt["pnl_pips"] > 0).mean()
    y_n = len(yt)
    y_cum = yt["pnl_pips"].cumsum()
    y_dd = (y_cum - y_cum.cummax()).min()

    # Gold price context
    price_start = df[df.index.year == year]["close"].iloc[0] if len(df[df.index.year == year]) > 0 else 0
    price_end = df[df.index.year == year]["close"].iloc[-1] if len(df[df.index.year == year]) > 0 else 0
    gold_return = (price_end - price_start) / price_start * 100

    bar_len = int(max(0, y_total / 100))
    bar = "█" * bar_len if y_total > 0 else "░" * int(max(0, abs(y_total) / 100))

    print(f"\n  {year}: {y_total:+8.1f} pips | n={y_n:3d} | WR={y_wr:.0%} | DD={y_dd:>8.1f} | Gold {gold_return:+.0f}%")
    print(f"         avg={y_avg:+.1f} pips/trade | {bar}")

# ── Regime analysis: bull vs sideways/bear ────────────────────────────
print("\n" + "=" * 70)
print("Regime analysis")
print("=" * 70)

# Monthly breakdown with gold price change
trades_df["month"] = trades_df["entry_time"].dt.to_period("M")
months = trades_df.groupby("month").agg(
    n=("pnl_pips", "count"),
    total=("pnl_pips", "sum"),
    avg=("pnl_pips", "mean"),
    wr=("pnl_pips", lambda x: (x > 0).mean()),
).reset_index()

# Count profitable / losing months
profitable_months = (months["total"] > 0).sum()
losing_months = (months["total"] <= 0).sum()
print(f"Profitable months: {profitable_months}/{len(months)} ({profitable_months/len(months):.0%})")
print(f"Losing months:     {losing_months}/{len(months)}")
print(f"Best month:        {months.loc[months['total'].idxmax(), 'month']} ({months['total'].max():+,.1f} pips)")
print(f"Worst month:       {months.loc[months['total'].idxmin(), 'month']} ({months['total'].min():+,.1f} pips)")

# ── Pre-2024 vs 2024-2025 comparison ─────────────────────────────────
print("\n" + "-" * 70)
print("Pre-2024 (横ばい期) vs 2024-2025 (上昇期)")
print("-" * 70)

pre = trades_df[trades_df["year"] < 2024]
post = trades_df[trades_df["year"] >= 2024]

for label, subset in [("2021-2023", pre), ("2024-2025", post)]:
    if len(subset) == 0:
        continue
    s_total = subset["pnl_pips"].sum()
    s_avg = subset["pnl_pips"].mean()
    s_wr = (subset["pnl_pips"] > 0).mean()
    s_n = len(subset)
    s_cum = subset["pnl_pips"].cumsum()
    s_dd = (s_cum - s_cum.cummax()).min()
    years = subset["year"].nunique()
    print(f"  {label}: {s_total:+8.1f} pips | n={s_n} ({s_n/years:.0f}/yr) | WR={s_wr:.0%} | avg={s_avg:+.1f} | DD={s_dd:.1f}")

# ── Day-of-week breakdown (full 5 years) ─────────────────────────────
print("\n" + "-" * 70)
print("Day-of-week breakdown (JST, 5 years)")
print("-" * 70)
day_names = {1: "火Tue", 2: "水Wed", 4: "金Fri"}
for dow_num in sorted(trades_df["jst_weekday_num"].unique()):
    day_label = day_names.get(dow_num, f"Day{dow_num}")
    mask = trades_df["jst_weekday_num"] == dow_num
    day_trades = trades_df[mask]
    print(f"  {day_label}: {day_trades['pnl_pips'].mean():+6.1f} pips/trade, WR {(day_trades['pnl_pips'] > 0).mean():.0%}, n={len(day_trades)}")

# ── Monthly heatmap ───────────────────────────────────────────────────
print("\n" + "-" * 70)
print("Monthly heatmap (pips)")
print("-" * 70)
print(f"{'':>8}", end="")
for m in range(1, 13):
    print(f"  {m:>6}", end="")
print()

for year in sorted(trades_df["year"].unique()):
    print(f"  {year}", end="")
    for month in range(1, 13):
        period = pd.Period(f"{year}-{month:02d}", freq="M")
        match = months[months["month"] == period]
        if len(match) > 0:
            val = match.iloc[0]["total"]
            if val > 200:
                marker = f"  {val:>5.0f}+"
            elif val > 0:
                marker = f"  {val:>5.0f} "
            elif val > -200:
                marker = f"  {val:>5.0f} "
            else:
                marker = f"  {val:>5.0f}-"
            print(marker, end="")
        else:
            print("      -", end="")
    print()

print("\n" + "=" * 70)
print("Done.")
