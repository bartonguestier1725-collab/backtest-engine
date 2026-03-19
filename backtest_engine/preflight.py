"""Pre-flight quality check for simulate_trades().

Inspects inputs BEFORE simulation and assigns a quality grade (A/B/C).
Grade C backtests silently produce over-optimistic results — this module
makes the gap visible.

Suppress warnings:
    warnings.filterwarnings("ignore", category=BacktestQualityWarning)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


class BacktestQualityWarning(UserWarning):
    """Issued by simulate_trades() when inputs are incomplete.

    Suppress with:
        warnings.filterwarnings("ignore", category=BacktestQualityWarning)
    """
    pass


@dataclass
class PreflightReport:
    """Result of a pre-flight quality check."""

    grade: str                                    # "A", "B", "C"
    has_entry_costs: bool
    has_open_prices: bool
    items: list[tuple[str, bool, str]] = field(default_factory=list)
    # (name, provided, detail_if_missing)

    def format_message(self) -> str:
        """Format as a single multi-line warning message."""
        lines = [f"Backtest Quality: {self.grade}"]
        for name, provided, detail in self.items:
            if provided:
                lines.append(f"  {name}:  provided")
            else:
                lines.append(f"  {name}:  NOT PROVIDED — {detail}")
        if self.grade == "C":
            lines.append(
                "  Grade C backtests overestimate performance. "
                "See: help(BrokerCost.per_trade_cost)"
            )
        return "\n".join(lines)


def run_preflight(
    open_prices: np.ndarray | None,
    entry_costs: np.ndarray | None,
) -> PreflightReport:
    """Pure function. Inspect arguments and return a PreflightReport.

    Parameters
    ----------
    open_prices : If provided, entry at next-bar open (realistic).
    entry_costs : If provided, per-trade cost applied to pnl_r.

    Returns
    -------
    PreflightReport with grade A (both), B (one), or C (neither).
    """
    has_costs = entry_costs is not None
    has_open = open_prices is not None

    items: list[tuple[str, bool, str]] = [
        (
            "entry_costs",
            has_costs,
            "costs will be 0 (use BrokerCost.per_trade_cost())",
        ),
        (
            "open_prices",
            has_open,
            "entry at signal-bar close (optimistic bias)",
        ),
    ]

    n_provided = int(has_costs) + int(has_open)
    if n_provided == 2:
        grade = "A"
    elif n_provided == 1:
        grade = "B"
    else:
        grade = "C"

    return PreflightReport(
        grade=grade,
        has_entry_costs=has_costs,
        has_open_prices=has_open,
        items=items,
    )
