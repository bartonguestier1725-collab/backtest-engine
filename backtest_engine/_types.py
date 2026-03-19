"""Shared constants and dtypes used across all modules."""

import numpy as np

# --- Trade direction ---
LONG: int = 1
SHORT: int = -1

# --- Exit types ---
EXIT_SL: int = 0
EXIT_TP: int = 1
EXIT_TIME: int = 2
EXIT_BE: int = 3
EXIT_CUSTOM: int = 4
EXIT_TRAIL: int = 5
EXIT_NO_FILL: int = 6

# --- Structured array dtype for trade results ---
TRADE_RESULT_DTYPE = np.dtype([
    ("pnl_r", np.float64),
    ("hold_bars", np.int32),
    ("exit_type", np.int8),
    ("mfe_r", np.float64),
    ("mae_r", np.float64),
    ("entry_bar", np.int32),
    ("exit_bar", np.int32),
    ("cost_r", np.float64),
])
