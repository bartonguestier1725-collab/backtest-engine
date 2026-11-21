"""Score bench/llm/*.sonnet.json against labels.json (the last JSON object in
each reply counts; one reply corrected itself mid-answer)."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
labels = json.loads((HERE / "labels.json").read_text())
rows = {}
for cid, lab in sorted(labels.items()):
    raw = (HERE / "llm" / f"{cid}.sonnet.json").read_text()
    objs = re.findall(r'\{[^{}]*"lookahead"[^{}]*\}', raw)
    rows[cid] = bool(json.loads(objs[-1])["lookahead"]) if objs else None
leaky = [c for c in labels if labels[c]["leak"]]
clean = [c for c in labels if not labels[c]["leak"]]
print(f"caught {sum(rows[c] is True for c in leaky)}/{len(leaky)} leaky, "
      f"false alarms {sum(rows[c] is True for c in clean)}/{len(clean)} clean")
for c in leaky:
    if rows[c] is not True:
        print("missed:", c, labels[c]["name"])
(HERE / "llm" / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
