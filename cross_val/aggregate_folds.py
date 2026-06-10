import sys, os, json
import numpy as np
from collections import Counter

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
RESULTS   = os.path.join(PROJ_ROOT, "results", "fold_results")

rows, missing = [], []
for fold in range(5):
    path = os.path.join(RESULTS, f"fold_{fold}_results.json")
    if not os.path.exists(path):
        missing.append(fold); continue
    with open(path) as f:
        rows.append(json.load(f))

if missing:
    print(f"WARNING: missing folds {missing}")
if not rows:
    print("No results found yet."); sys.exit(1)

metrics = ["f1", "precision", "recall", "acc"]
print(f"\n{'fold':>5}  " + "  ".join(f"{m:>9}" for m in metrics) + "  uncertain  val_datasets")
print("-" * 85)
for r in rows:
    m = r["metrics"]
    print(f"{r['fold']:>5}  " + "  ".join(f"{m[k]:>9.3f}" for k in metrics) +
          f"  {r['uncertain_count']:>9}  {r['val_datasets']}")

print("\n--- Summary ---")
for metric in metrics:
    vals = [r["metrics"][metric] for r in rows]
    print(f"  {metric:>10}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  min={np.min(vals):.3f}  max={np.max(vals):.3f}")

all_uncertain = [dict(**u, fold=r["fold"]) for r in rows for u in r["uncertain_stimuli"]]
print(f"\n  total uncertain stimuli: {len(all_uncertain)}")
if all_uncertain:
    print("  top 10 cells with most uncertain predictions:")
    for cell, cnt in Counter(u["cell"] for u in all_uncertain).most_common(10):
        print(f"    cell {cell:>4}: {cnt}")

summary = {
    "per_fold": rows,
    "mean": {m: float(np.mean([r["metrics"][m] for r in rows])) for m in metrics},
    "std":  {m: float(np.std( [r["metrics"][m] for r in rows])) for m in metrics},
    "total_uncertain": len(all_uncertain),
}
out = os.path.join(RESULTS, "summary.json")
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved → {out}")
