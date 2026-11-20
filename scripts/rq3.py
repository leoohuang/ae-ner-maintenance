"""LLM annotation protocol, RQ3: active learning with LLM labels (README; amendment 2).

    python scripts/rq3.py        # writes results/llm/rq3.json and results/figures/rq3_curves.png

Reads results/gpu/llm/<model>/{random,uncertainty,hybrid}.jsonl (LLM labels) and the GPU human-label
runs in results/gpu/ (amendment 2: GPU is compared with GPU only). As in scripts/compare_all.py,
the active-learning curves take the 5% and 100% points from random (same initial set; all of train_pool).
AULC over 5-60% of the training tokens; paired Wilcoxon over seeds 1-10; budget to reach 90 / 95 / 98%
of the GPU human-label reference (random, 100%), read from the mean curves.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.stats import aulc, budget_to_reach, paired

FR = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]
AULC_FR = [f for f in FR if f <= 0.60]
NAMES = ["random", "uncertainty", "hybrid"]
LLMS = json.load(open(ROOT / "results/llm/rq1.json"))["rq3_models"]       # highest, lowest
LABELS = {"human": "results/gpu", **{m: f"results/gpu/llm/{m}" for m in LLMS.values()}}
SEEDS = list(range(1, 11))


def table(path):
    d = {}
    for r in map(json.loads, open(ROOT / path)):
        d[(r["seed"], r["frac"])] = (r["budget"], r["test_micro_f1"])
    return d


T = {lab: {n: table(f"{d}/{n}.jsonl") for n in NAMES} for lab, d in LABELS.items()}
for lab in T:                       # every curve must be complete before anything is computed
    for n in NAMES:
        need = FR if n == "random" else [f for f in FR if f not in (0.05, 1.00)]
        miss = [(s, f) for s in SEEDS for f in need if (s, f) not in T[lab][n]]
        assert not miss, f"{lab}/{n} is missing {miss[:5]}"


def curve(lab, name, seed):
    src = lambda f: T[lab]["random"] if name == "random" or f in (0.05, 1.00) else T[lab][name]
    pts = [src(f)[(seed, f)] for f in FR]
    return np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


C = {lab: {n: [curve(lab, n, s) for s in SEEDS] for n in NAMES} for lab in T}
A = {lab: {n: [aulc(x[:len(AULC_FR)], y[:len(AULC_FR)]) for x, y in C[lab][n]] for n in NAMES} for lab in T}
mean = {lab: {n: (np.mean([c[0] for c in C[lab][n]], 0), np.mean([c[1] for c in C[lab][n]], 0))
              for n in NAMES} for lab in T}
REF = float(mean["human"]["random"][1][-1])

res = {"reference_f1": REF, "seeds": SEEDS, "models": LLMS, "aulc_range": [0.05, 0.60],
       "mean_curves": {lab: {n: {"frac": FR, "f1": mean[lab][n][1].tolist()} for n in NAMES} for lab in T},
       "aulc": {lab: {n: {"mean": float(np.mean(A[lab][n])), "sd": float(np.std(A[lab][n], ddof=1)),
                          "per_seed": A[lab][n]} for n in NAMES} for lab in T},
       "reach": {}, "tests": {}}
for lab in T:
    res["reach"][lab] = {}
    for n in NAMES:
        x, y = mean[lab][n]
        res["reach"][lab][n] = {}
        for p in (0.90, 0.95, 0.98):
            b = budget_to_reach(x, y, p * REF)
            res["reach"][lab][n][str(p)] = None if b is None else float(100 * b / x[-1])

# pre-registered: per LLM, uncertainty vs random (1), each strategy LLM vs human (3), AL gain LLM vs human (1)
gain = lambda lab, n: [u - r for u, r in zip(A[lab][n], A[lab]["random"])]
for m in LLMS.values():
    t = {"uncertainty - random": paired(A[m]["uncertainty"], A[m]["random"])}
    for n in NAMES:
        t[f"{n}: LLM - human"] = paired(A[m][n], A["human"][n])
    t["gain (uncertainty - random): LLM - human"] = paired(gain(m, "uncertainty"), gain("human", "uncertainty"))
    t["hybrid - random (not pre-registered)"] = paired(A[m]["hybrid"], A[m]["random"])
    res["tests"][m] = t
res["n_preregistered_tests"] = 5 * len(LLMS)

out = ROOT / "results/llm"; out.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out / "rq3.json", "w"), indent=1)

print(f"reference (GPU human labels, random 100%): {REF:.4f}\n")
print(f"{'labels':18s} " + "  ".join(f"{n:>17s}" for n in NAMES) + "   F1 at 100%")
for lab in T:
    a = res["aulc"][lab]
    print(f"{lab:18s} " + "  ".join(f"{a[n]['mean']:.4f} ± {a[n]['sd']:.4f}" for n in NAMES)
          + f"   {mean[lab]['random'][1][-1]:.4f}")
for m, t in res["tests"].items():
    print(f"\n{m}")
    for k, v in t.items():
        print(f"  {k:44s} {v['diff']:+.4f}, better in {v['better']} of {v['n']}, p = {v['p']:.4f}")
print()
for lab in T:
    for n in NAMES:
        r = res["reach"][lab][n]
        print(f"reach {lab:16s} {n:12s} " + "  ".join(
            f"{int(100 * float(p))}%: " + ("not reached" if v is None else f"{v:.1f}%") for p, v in r.items()))

# ---- figure: one panel per label source ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e6e5e0"
COL = {"random": "#6b6a63", "uncertainty": "#2a78d6", "hybrid": "#d9822b"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, len(T), figsize=(12, 4), sharey=True)
for ax, lab in zip(axes, T):
    for n in NAMES:
        x, y = mean[lab][n]
        ax.plot(100 * x / x[-1], y, marker="o", ms=4, lw=2, color=COL[n], label=n)
    ax.axhline(REF, color=INK2, ls=":", lw=1)
    ax.set_title(f"{lab} labels", loc="left", fontsize=10)
    ax.set_xlabel("labelled text (% of training tokens)")
    ax.grid(color=GRID); ax.set_axisbelow(True)
axes[0].set_ylabel("test micro-F1 (human test labels)")
axes[0].legend(frameon=False, loc="lower right")
fig.suptitle(f"RQ3: learning curves, mean of seeds 1-10 (dotted: human-label reference {REF:.3f})",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(ROOT / "results/figures/rq3_curves.png", dpi=150)
