#!/usr/bin/env python3
"""report.py demo — generate an HTML report from synthetic data.

No API key required. Uses random data to demonstrate report layout.
"""
import datetime
from pathlib import Path

import numpy as np

from backtest_engine import (
    TradeResults, TRADE_RESULT_DTYPE,
    EXIT_SL, EXIT_TP, EXIT_TIME,
)
from backtest_engine.report import (
    ReportConfig, SummaryCell, GateRow, WfaRow, CscvResult, Section,
    generate_report,
)


def main():
    rng = np.random.RandomState(42)
    N = 200
    trades = np.zeros(N, dtype=TRADE_RESULT_DTYPE)
    trades["pnl_r"] = rng.randn(N) * 0.8 + 0.05
    trades["hold_bars"] = rng.randint(1, 80, N)
    trades["exit_type"] = rng.choice([EXIT_SL, EXIT_TP, EXIT_TIME], N, p=[0.4, 0.35, 0.25])
    trades["mfe_r"] = np.abs(rng.randn(N))
    trades["mae_r"] = -np.abs(rng.randn(N))
    trades["entry_bar"] = np.sort(rng.randint(0, 8000, N))
    trades["exit_bar"] = trades["entry_bar"] + trades["hold_bars"]
    results = TradeResults(trades)

    base = int(datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
    timestamps = np.arange(base, base + 9000 * 3600, 3600, dtype=np.int64)

    exit_bars = results["exit_bar"]
    sort_order = np.argsort(exit_bars, kind="stable")
    pnl_sorted = results["pnl_r"][sort_order]
    equity = np.cumsum(pnl_sorted)
    equity_ts = timestamps[exit_bars[sort_order]]

    cfg = ReportConfig(
        title="Demo Strategy Report",
        title_tooltip="Synthetic data for layout demonstration.",
        subtitle="DEMO | Random signals | 1H | 2022-2023 | GROSS",
        verdict="PASS",
        verdict_tooltip=f"PF={results.profit_factor:.2f}",
        equity_curve=equity,
        equity_timestamps=equity_ts,
        equity_ylabel="Cumulative PnL (R)",
        summary=[
            SummaryCell("Trades", str(len(results)), "Total trades."),
            SummaryCell("Win Rate", f"{results.win_rate:.1%}", "Win rate.", "pos"),
            SummaryCell("PF", f"{results.profit_factor:.2f}", "Profit Factor."),
            SummaryCell("Expectancy", f"{results.expectancy_r:.3f}R", "Mean R/trade.",
                        "pos" if results.expectancy_r > 0 else "neg"),
            SummaryCell("Sharpe", f"{results.sharpe_r:.2f}", "Sharpe ratio (R)."),
            SummaryCell("Total R", f"{float(np.sum(results['pnl_r'])):+.1f}", "Cumulative R.",
                        "pos" if np.sum(results['pnl_r']) > 0 else "neg"),
            SummaryCell("Max DD", f"{results.max_drawdown_r:.1f}R", "Max drawdown.", "warn"),
            SummaryCell("RF", f"{results.recovery_factor:.2f}", "Recovery Factor."),
        ],
        strategy_html=(
            "<b>Entry:</b> Random signal (demo only)<br>"
            "<b>SL:</b> 1.0R / <b>TP:</b> 2.0R<br>"
            "<b>Purpose:</b> Demonstrate report.py layout and auto-generated sections"
        ),
        analysis_html=(
            "<b>Note:</b> This report uses synthetic random data. "
            "All metrics are meaningless — this is a layout demo only."
        ),
        trades=results,
        trade_timestamps=timestamps,
        gates=[
            GateRow("BugGuard", "PASS", "14/14 checks", "Data quality checks."),
            GateRow("Gate 1", "PASS", "PF > 1.05", "Quick feasibility."),
            GateRow("Gate 2", "SKIP", "Demo data", "Parameter screening."),
        ],
        wfa=[
            WfaRow("2022-01~06 → 07~09", "+5.2R", "+1.1R", True),
            WfaRow("2022-04~09 → 10~12", "+3.8R", "-0.5R", False),
            WfaRow("2022-07~12 → 2023-01~03", "+6.1R", "+2.3R", True),
        ],
        wfa_summary="OOS positive: 2/3 (67%) | Demo values",
        cscv=CscvResult(pbo=0.22, logit_mean=-0.31, n_combinations=126),
        footer="Backtest Engine | Demo Report",
    )

    out_path = Path(__file__).with_name("report_demo.html")
    out = generate_report(cfg, out_path)
    print(f"Generated: {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
