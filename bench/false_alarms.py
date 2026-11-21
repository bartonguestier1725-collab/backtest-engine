"""How often does null_edge flag a clean strategy? 10 seeds per clean case.

    uv run python bench/false_alarms.py      # writes bench/false_alarms.json
"""
import json
import warnings
from pathlib import Path

from backtest_engine import audit

HERE = Path(__file__).resolve().parent
warnings.simplefilter("ignore")
labels = json.loads((HERE / "labels.json").read_text())
runs, flagged = 0, []
for cid, lab in sorted(labels.items()):
    if lab["leak"]:
        continue
    with audit._strategy_module(HERE / "cases" / f"{cid}.py") as mod:
        placebo = audit._make_placebo(mod) if hasattr(mod, "signals") else None
        kw = dict(n_bars=int(getattr(mod, "NULL_BARS", 5000)),
                  bar_minutes=int(getattr(mod, "BAR_MINUTES", 60)),
                  vol=float(getattr(mod, "NULL_VOL", 0.002)))
        for s in range(10):
            r = audit.check_null_edge(mod.backtest, seed=1000 + 37 * s, placebo_fn=placebo, **kw)
            runs += 1
            if r.verdict != "pass":
                flagged.append({"case": cid, "seed": 1000 + 37 * s, "verdict": r.verdict,
                                "t": r.details.get("t_across_paths"),
                                "t_vs_placebo": r.details.get("t_vs_placebo")})
print(f"null_edge flagged {len(flagged)}/{runs} runs on clean cases", flagged)
(HERE / "false_alarms.json").write_text(json.dumps({"runs": runs, "flagged": flagged}, indent=2) + "\n")
