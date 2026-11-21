"""Score the audit tests on labeled strategy cases.

For each case: static source checks (BugGuard BG-04/07/09), the look-ahead
test, the null-data test, and "any" (flagged by at least one). A case counts
as flagged by a test when the verdict is fail or warn.
"""
import json
import sys
import time
from pathlib import Path

from backtest_engine import audit

HERE = Path(__file__).resolve().parent

labels = json.loads((HERE / "labels.json").read_text())
rows = []
for cid, lab in sorted(labels.items()):
    t0 = time.time()
    rep = audit.audit_strategy(HERE / "cases" / f"{cid}.py", n_cuts=int(sys.argv[1]) if len(sys.argv) > 1 else 32)
    by = {r["check"]: r for r in rep["results"]}
    flag = lambda r: r is not None and r["verdict"] in ("fail", "warn")
    static = any(flag(r) for k, r in by.items() if k.startswith("source:"))
    look = flag(by.get("lookahead"))
    null = flag(by.get("null_edge"))
    errs = [k for k, r in by.items() if r["verdict"] == "error"]
    rows.append({"case": cid, **lab, "static": static, "lookahead": look, "null": null,
                 "any": static or look or null, "errors": errs,
                 "lookahead_cuts": by.get("lookahead", {}).get("details", {}).get("cuts_with_change"),
                 "null_t": by.get("null_edge", {}).get("details", {}).get("t_across_paths"),
                 "sec": round(time.time() - t0, 1)})
    r = rows[-1]
    print(f"{cid} {lab['name']:<22} leak={str(lab['leak']):<5} static={r['static']!s:<5} "
          f"look={r['lookahead']!s:<5}({r['lookahead_cuts']}) null={r['null']!s:<5}(t={r['null_t']}) "
          f"err={errs} {r['sec']}s", flush=True)

for test in ("static", "lookahead", "null", "any"):
    tp = sum(r[test] for r in rows if r["leak"])
    fp = sum(r[test] for r in rows if not r["leak"])
    nl = sum(r["leak"] for r in rows)
    print(f"{test:<10} caught {tp}/{nl} leaky, false alarms {fp}/{len(rows) - nl} clean")
(HERE / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
