"""LLM annotation protocol, RQ3c (amendment 10, exploratory): the k reviewed records alone vs RQ3b.

    python scripts/rq3c.py       # writes results/llm/rq3c.json and results/figures/rq3c_alone.png

Reads results/gpu/llm/review.jsonl (RQ3b: k reviewed + LLM labels on the rest) and
results/gpu/llm/review_only.jsonl (the k reviewed records alone, human labels). The random choice does not
depend on the LLM and was run once (rows carry model gemini-4-carbon); it is used for both LLMs.
Paired Wilcoxon over seeds 1-10: RQ3b minus alone, per LLM, k and choice (12 tests). Per-type F1 descriptive.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.evaluate import TYPES
from src.stats import paired

KS = [50, 100, 200]
SEEDS = list(range(1, 11))
LLMS = json.load(open(ROOT / "results/llm/rq1.json"))["rq3_models"]
RANDOM_RUN_UNDER = "gemini-4-carbon"
REF = float(np.mean([r["test_micro_f1"] for r in map(json.loads, open(ROOT / "results/gpu/random.jsonl"))
                     if r["frac"] == 1.0]))

mixed = {(r["model"], r["k"], r["choice"], r["seed"]): r
         for r in map(json.loads, open(ROOT / "results/gpu/llm/review.jsonl"))}
alone = {(r["model"], r["k"], r["choice"], r["seed"]): r
         for r in map(json.loads, open(ROOT / "results/gpu/llm/review_only.jsonl"))}


def alone_row(m, k, c, s):
    return alone[(RANDOM_RUN_UNDER if c == "random" else m, k, c, s)]


for m in LLMS.values():                     # every run must be there, and on the same records, before anything
    for k in KS:
        for c in ["signal", "random"]:
            for s in SEEDS:
                a, b = alone_row(m, k, c, s), mixed[(m, k, c, s)]
                assert a["reviewed"] == b["reviewed"], (m, k, c, s)
                assert not a["with_llm"] and a["n_train"] == k, (m, k, c, s)

res = {"reference_f1": REF, "models": LLMS, "seeds": SEEDS, "exploratory": True, "per_model": {}, "tests": {}}
for m in LLMS.values():
    out, t = {}, {}
    for k in KS:
        for c in ["signal", "random"]:
            A = [alone_row(m, k, c, s) for s in SEEDS]
            B = [mixed[(m, k, c, s)] for s in SEEDS]
            fa, fb = [r["test_micro_f1"] for r in A], [r["test_micro_f1"] for r in B]
            out[f"{k}|{c}"] = {
                "k": k, "choice": c,
                "alone_f1": float(np.mean(fa)), "alone_sd": float(np.std(fa, ddof=1)),
                "with_llm_f1": float(np.mean(fb)), "with_llm_sd": float(np.std(fb, ddof=1)),
                "alone_per_type": {x: float(np.mean([r["test_per_entity"][x]["f1"] for r in A])) for x in TYPES},
                "with_llm_per_type": {x: float(np.mean([r["test_per_entity"][x]["f1"] for r in B])) for x in TYPES},
            }
            t[f"k={k} {c}: with LLM - alone"] = paired(fb, fa)
    out["0|none"] = {"with_llm_f1": float(np.mean([mixed[(m, 0, "none", s)]["test_micro_f1"] for s in SEEDS]))}
    res["per_model"][m], res["tests"][m] = out, t
res["n_tests"] = 6 * len(LLMS)

out_dir = ROOT / "results/llm"; out_dir.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out_dir / "rq3c.json", "w"), indent=1)

SHOW = ["PhysicalObject", "State", "Process", "Activity"]
print(f"reference (GPU human labels, random 100%): {REF:.4f}   [exploratory, amendment 10]")
for m in LLMS.values():
    out = res["per_model"][m]
    print(f"\n{m}   (LLM labels alone, k = 0: {out['0|none']['with_llm_f1']:.4f})")
    print(f"  {'k':>4s} {'choice':7s} {'alone':>15s} {'with LLM':>15s}   per type alone -> with LLM ({', '.join(x[:6] for x in SHOW)})")
    for k in KS:
        for c in ["signal", "random"]:
            v = out[f"{k}|{c}"]
            print(f"  {k:4d} {c:7s} {v['alone_f1']:.4f} ± {v['alone_sd']:.4f} {v['with_llm_f1']:.4f} ± {v['with_llm_sd']:.4f}   "
                  + "  ".join(f"{v['alone_per_type'][x]:.3f}->{v['with_llm_per_type'][x]:.3f}" for x in SHOW))
    for name, v in res["tests"][m].items():
        print(f"  {name:28s} {v['diff']:+.4f}, better in {v['better']} of {v['n']}, p = {v['p']:.4f}")

# ---- figure ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e6e5e0"
COL = {"signal": "#2a78d6", "random": "#6b6a63"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, len(LLMS), figsize=(10, 4.2), sharey=True)
for ax, m in zip(axes, LLMS.values()):
    out = res["per_model"][m]
    for c in ["signal", "random"]:
        ax.plot([0] + KS, [out["0|none"]["with_llm_f1"]] + [out[f"{k}|{c}"]["with_llm_f1"] for k in KS],
                marker="o", ms=5, lw=2, color=COL[c], label=f"{c}: reviewed + LLM labels")
        ax.plot(KS, [out[f"{k}|{c}"]["alone_f1"] for k in KS],
                marker="s", ms=5, lw=2, ls="--", color=COL[c], label=f"{c}: reviewed records alone")
    ax.axhline(REF, color=INK2, ls=":", lw=1)
    ax.set_xticks([0] + KS); ax.set_xlabel("records reviewed (k of 758)")
    ax.set_title(f"{m}", loc="left", fontsize=10)
    ax.grid(color=GRID); ax.set_axisbelow(True)
axes[0].set_ylabel("test micro-F1 (mean of seeds 1-10)")
axes[0].legend(frameon=False, loc="lower right", fontsize=8)
fig.suptitle(f"RQ3c (exploratory): reviewed records with and without the LLM labels (dotted: human labels, {REF:.3f})",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(ROOT / "results/figures/rq3c_alone.png", dpi=150)
