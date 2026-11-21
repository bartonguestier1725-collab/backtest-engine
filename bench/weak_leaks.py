"""Power of null_edge against weak leaks (only some trades affected), 20 seeds.

    uv run python bench/weak_leaks.py      # writes bench/weak_leaks.json
"""
import json
import warnings
from pathlib import Path

import numpy as np

from backtest_engine import simulate_trades, rsi, atr
from backtest_engine.audit import check_null_edge

HERE = Path(__file__).resolve().parent
warnings.simplefilter("ignore")


def rsi_bt(early_frac=0.0):
    def bt(d):
        o, h, l, c = d["open"], d["high"], d["low"], d["close"]
        r = rsi(c, 14)
        a = atr(h, l, c, 14)
        sig = np.where(r < 30, 1, np.where(r > 70, -1, 0))
        sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
        sb = sb[(sb >= 1) & (sb < len(c) - 1)].astype(np.int32)
        dirs = np.sign(sig[sb]).astype(np.int8)
        # a fraction of trades enters one bar early (at the signal bar's open)
        early = (np.arange(len(sb)) % 10) < int(round(10 * early_frac))
        return simulate_trades(h, l, c, sb - early.astype(np.int32), dirs, a[sb] * 2, a[sb] * 3,
                               max_hold=24, open_prices=o)
    return bt


out = {}
for frac in (0.0, 0.2, 0.5):
    for paths in (30, 120):
        v = [check_null_edge(rsi_bt(frac), n_paths=paths, seed=9000 + s).verdict for s in range(20)]
        out[f"early_{frac:.1f}_paths_{paths}"] = {"fail": v.count("fail"), "warn": v.count("warn"),
                                                  "pass": v.count("pass"), "runs": len(v)}
        print(frac, paths, out[f"early_{frac:.1f}_paths_{paths}"], flush=True)
(HERE / "weak_leaks.json").write_text(json.dumps(out, indent=2) + "\n")
