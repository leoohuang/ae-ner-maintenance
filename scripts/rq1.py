"""LLM annotation protocol, RQ1: label quality of the eight LLMs on train_pool (README; amendment 9).

    python scripts/rq1.py        # writes results/llm/rq1.json and results/figures/rq1_f1.png

Exact-match micro-F1 per model with a bootstrap 95% interval over records (1,000 resamples),
per entity type and per record-length tertile; error types; shared errors; the two models for RQ3.
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.evaluate import evaluate, TYPES
from src.llm_analysis import bootstrap_micro_f1, error_types, length_tertiles, shared_errors

MODELS = ["gpt-6-astra", "gpt-6-luna", "claude-opus-5-5", "claude-haiku-5-5", "gemini-4-carbon",
          "qwen4-max", "qwen3.8-27b", "deepseek-v4-pro"]
SMALLER = {"gpt-6-luna", "claude-haiku-5-5", "qwen3.8-27b"}     # tie-break for RQ3: the smaller model
N_BOOT, BOOT_SEED = 1000, 0

recs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed/gold.jsonl")}
ids = json.load(open(ROOT / "data/splits/split_v1.json"))["train_pool"]
gold = {i: recs[i]["spans"] for i in ids}
labels = {m: {json.loads(l)["id"]: json.loads(l)["spans"] for l in open(ROOT / f"data/llm/{m}.jsonl")}
          for m in MODELS}
assert all(sorted(labels[m]) == sorted(ids) for m in MODELS)
tert = length_tertiles({i: len(recs[i]["tokens"]) for i in ids})


def only(spans, t):
    return [s for s in spans if s[2] == t]


def score(sel, filt=lambda s: s):
    g = [filt(gold[i]) for i in sel]
    out = {}
    for m in MODELS:
        p = [filt(labels[m][i]) for i in sel]
        ev = evaluate(g, p)
        lo, hi = bootstrap_micro_f1(g, p, n=N_BOOT, seed=BOOT_SEED)
        out[m] = {"f1": ev["micro_f1"], "ci95": [lo, hi], "precision": ev["precision"], "recall": ev["recall"],
                  "gold_spans": sum(len(x) for x in g)}
    return out


res = {"n_records": len(ids), "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED}}
res["overall"] = score(ids)
res["per_type"] = {t: score(ids, lambda s, t=t: only(s, t)) for t in TYPES}
res["per_type_note"] = "Property has 26 human spans in train_pool and is marked unreliable"
res["length_tertiles"] = {g: {"n_records": sum(1 for i in ids if tert[i] == g),
                              "scores": score([i for i in ids if tert[i] == g])}
                          for g in ["short", "medium", "long"]}
res["error_types"] = {}
for m in MODELS:
    c = Counter()
    for i in ids:
        c.update(error_types(gold[i], labels[m][i]))
    res["error_types"][m] = {"counts": dict(c), "shares": {k: v / sum(c.values()) for k, v in c.items()}}
res["shared_errors"] = shared_errors(gold, labels, min_agree=6)

f1 = {m: res["overall"][m]["f1"] for m in MODELS}
def pick(target):          # ties: the smaller model, then alphabetical ID (pre-registered)
    tied = [m for m in MODELS if round(f1[m], 10) == round(target, 10)]
    return min(tied, key=lambda m: (m not in SMALLER, m))


res["rq3_models"] = {"highest": pick(max(f1.values())), "lowest": pick(min(f1.values()))}

out = ROOT / "results/llm"; out.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out / "rq1.json", "w"), indent=1)

# ---- table ----
order = sorted(MODELS, key=lambda m: -f1[m])
print(f"{'model':18s} {'micro-F1':>8s}  95% CI          " + "  ".join(f"{t[:8]:>8s}" for t in TYPES))
for m in order:
    r = res["overall"][m]
    print(f"{m:18s} {r['f1']:8.3f}  [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  "
          + "  ".join(f"{res['per_type'][t][m]['f1']:8.3f}" for t in TYPES))
print("\nlength tertiles (records):", {g: v["n_records"] for g, v in res["length_tertiles"].items()})
for m in order:
    print(f"{m:18s}", "  ".join(f"{g} {res['length_tertiles'][g]['scores'][m]['f1']:.3f}" for g in ["short", "medium", "long"]))
print("\nerror types (share of errors):")
K = ["boundary", "missed", "extra", "boundary_type", "type"]
for m in order:
    e = res["error_types"][m]
    print(f"{m:18s} n={sum(e['counts'].values()):4d}  " + "  ".join(f"{k} {e['shares'].get(k, 0):.0%}" for k in K))
print("\nshared errors:", res["shared_errors"])
print("RQ3 models:", res["rq3_models"])

# ---- figure ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e6e5e0"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8, 4.2))
y = np.arange(len(order))[::-1]
for yy, m in zip(y, order):
    r = res["overall"][m]
    ax.plot(r["ci95"], [yy, yy], color="#86b6ef", lw=6, solid_capstyle="round")
    ax.plot([r["f1"]], [yy], "o", color="#2a78d6", ms=8)
    ax.text(r["ci95"][1] + 0.006, yy, f"{r['f1']:.3f}", va="center", fontsize=9)
ax.set_yticks(y, order); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
ax.set_xlabel("micro-F1 against human labels (train_pool, 758 records)", color=INK2)
ax.set_title("RQ1: label quality per model, with bootstrap 95% intervals", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(ROOT / "results/figures/rq1_f1.png", dpi=150)
