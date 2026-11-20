"""LLM annotation protocol, RQ3b: LLM labels with k records reviewed (README; amendment 8).

    python scripts/rq3b.py       # writes results/llm/rq3b.json and results/figures/rq3b_review.png

Reads results/gpu/llm/review.jsonl. Per model and k: test micro-F1 (mean ± sd over seeds 1-10), share of
the GPU human-label reference (random, 100%, results/gpu/random.jsonl), records and tokens reviewed, and
the share of the model's span errors in train_pool that the reviewed records hold (descriptive).
Paired Wilcoxon over seeds: signal vs random at each k > 0; each k > 0 vs k = 0. Per-type F1 descriptive.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.evaluate import TYPES
from src.llm_analysis import span_errors
from src.stats import paired

KS = [50, 100, 200]
SEEDS = list(range(1, 11))
LLMS = json.load(open(ROOT / "results/llm/rq1.json"))["rq3_models"]
REF = float(np.mean([r["test_micro_f1"] for r in map(json.loads, open(ROOT / "results/gpu/random.jsonl"))
                     if r["frac"] == 1.0]))
recs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed/gold.jsonl")}
pool = json.load(open(ROOT / "data/splits/split_v1.json"))["train_pool"]
pool_tokens = sum(len(recs[i]["tokens"]) for i in pool)

rows = {}
for r in map(json.loads, open(ROOT / "results/gpu/llm/review.jsonl")):
    rows[(r["model"], r["k"], r["choice"], r["seed"])] = r
runs = [(0, "none")] + [(k, c) for k in KS for c in ["signal", "random"]]
for m in LLMS.values():                     # every run must be there before anything is computed
    miss = [(k, c, s) for k, c in runs for s in SEEDS if (m, k, c, s) not in rows]
    assert not miss, f"{m} is missing {miss[:5]}"

res = {"reference_f1": REF, "models": LLMS, "seeds": SEEDS, "per_model": {}, "tests": {}}
for m in LLMS.values():
    llm = {json.loads(l)["id"]: json.loads(l)["spans"] for l in open(ROOT / f"data/llm/{m}.jsonl")}
    err = {i: span_errors(recs[i]["spans"], llm[i]) for i in pool}
    total_err = sum(err.values())
    f1 = lambda k, c: [rows[(m, k, c, s)]["test_micro_f1"] for s in SEEDS]
    out = {}
    for k, c in runs:
        rr = [rows[(m, k, c, s)] for s in SEEDS]
        v = f1(k, c)
        out[f"{k}|{c}"] = {
            "k": k, "choice": c, "f1_mean": float(np.mean(v)), "f1_sd": float(np.std(v, ddof=1)),
            "share_of_reference": float(np.mean(v) / REF),
            "reviewed_tokens_mean": float(np.mean([r["reviewed_tokens"] for r in rr])),
            "reviewed_tokens_share": float(np.mean([r["reviewed_tokens"] for r in rr]) / pool_tokens),
            "errors_fixed_share": float(np.mean([sum(err[i] for i in r["reviewed"]) / total_err for r in rr])),
            "per_type_f1": {t: float(np.mean([r["test_per_entity"][t]["f1"] for r in rr])) for t in TYPES},
        }
    res["per_model"][m] = out
    t = {}
    for k in KS:
        t[f"k={k}: signal - random"] = paired(f1(k, "signal"), f1(k, "random"))
    for k in KS:
        for c in ["signal", "random"]:
            t[f"k={k} {c} - k=0"] = paired(f1(k, c), f1(0, "none"))
    res["tests"][m] = t
res["n_tests"] = 9 * len(LLMS)

out_dir = ROOT / "results/llm"; out_dir.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out_dir / "rq3b.json", "w"), indent=1)

print(f"reference (GPU human labels, random 100%): {REF:.4f}")
for m, out in res["per_model"].items():
    print(f"\n{m}\n  {'k':>4s} {'choice':7s} {'F1':>15s}  of ref  tokens  errors fixed  "
          + " ".join(f"{t[:6]:>6s}" for t in TYPES))
    for v in out.values():
        print(f"  {v['k']:4d} {v['choice']:7s} {v['f1_mean']:.4f} ± {v['f1_sd']:.4f}  {v['share_of_reference']:6.1%}"
              f"  {v['reviewed_tokens_share']:6.1%}  {v['errors_fixed_share']:11.1%}  "
              + " ".join(f"{v['per_type_f1'][t]:6.3f}" for t in TYPES))
    for k, v in res["tests"][m].items():
        print(f"  {k:24s} {v['diff']:+.4f}, better in {v['better']} of {v['n']}, p = {v['p']:.4f}")

# ---- figure ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e6e5e0"
COL = {"signal": "#2a78d6", "random": "#6b6a63"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, len(LLMS), figsize=(10, 4), sharey=True)
for ax, m in zip(axes, LLMS.values()):
    out = res["per_model"][m]
    for c in ["signal", "random"]:
        ks = [0] + KS
        v = [out["0|none"]] + [out[f"{k}|{c}"] for k in KS]
        y, e = np.array([x["f1_mean"] for x in v]), np.array([x["f1_sd"] for x in v])
        ax.fill_between(ks, y - e, y + e, color=COL[c], alpha=0.12, lw=0)
        ax.plot(ks, y, marker="o", ms=5, lw=2, color=COL[c], label=f"{c} choice")
    ax.axhline(REF, color=INK2, ls=":", lw=1)
    ax.set_xticks([0] + KS); ax.set_xlabel("records reviewed (k of 758)")
    ax.set_title(f"{m} labels", loc="left", fontsize=10)
    ax.grid(color=GRID); ax.set_axisbelow(True)
axes[0].set_ylabel("test micro-F1 (mean ± sd, seeds 1-10)")
axes[0].legend(frameon=False, loc="lower right")
fig.suptitle(f"RQ3b: LLM labels with k records reviewed (dotted: human labels, {REF:.3f})",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(ROOT / "results/figures/rq3b_review.png", dpi=150)
