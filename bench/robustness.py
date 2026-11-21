"""Robustness of the audit verdicts on the bench cases.

1. Look-ahead test on real EURUSD 1h bars (2024) instead of a random walk.
   Needs AGGVAULT_KEY; skipped without it.
2. Look-ahead test with 5 other seeds (cut placements, random-walk data).
3. Null-data test with 3 other seeds (different random walks).

    uv run python bench/robustness.py      # writes bench/robustness.json
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np

from backtest_engine import audit

HERE = Path(__file__).resolve().parent
warnings.simplefilter("ignore")

real = None
if os.environ.get("AGGVAULT_KEY"):
    from backtest_engine import fetch_aggvault
    ts, o, h, l, c, _ = fetch_aggvault("EURUSD", "1h", "2024-01-01", "2025-01-01")
    real = {"time": ts, "open": o, "high": h, "low": l, "close": c}

labels = json.loads((HERE / "labels.json").read_text())
out = {}
for cid, lab in sorted(labels.items()):
    mod = audit._load_module(HERE / "cases" / f"{cid}.py")
    row = {"name": lab["name"], "leak": lab["leak"]}
    if hasattr(mod, "signals"):
        if real is not None:
            r = audit.check_lookahead(mod.signals, real, n_cuts=32, seed=0)
            row["real_eurusd_1h"] = r.verdict
        seeds = [audit.check_lookahead(mod.signals, None, n_cuts=32, seed=s).verdict for s in range(1, 6)]
        row["lookahead_fail_of_5_seeds"] = sum(v == "fail" for v in seeds)
    if hasattr(mod, "backtest"):
        kw = dict(n_bars=int(getattr(mod, "NULL_BARS", 5000)),
                  bar_minutes=int(getattr(mod, "BAR_MINUTES", 60)),
                  vol=float(getattr(mod, "NULL_VOL", 0.002)))
        nulls = [audit.check_null_edge(mod.backtest, seed=100 * s, **kw).verdict for s in range(1, 4)]
        row["null_flag_of_3_seeds"] = sum(v in ("fail", "warn") for v in nulls)
    out[cid] = row
    print(cid, row, flush=True)
(HERE / "robustness.json").write_text(json.dumps(out, indent=2) + "\n")
