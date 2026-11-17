"""Overview figures (exploratory, not part of a pre-registered analysis).

    python scripts/plot_overview.py      # writes results/figures/1_..3_*.png

1 gold labels: count per type, type mix per split, label length per type.
2 test F1 per type in the active-learning runs (Mac main results, mean of 10 seeds).
3 the 7 API LLMs on train_pool: label count per type relative to gold, and F1 per type.
  This is a first look; the RQ1 analysis is planned separately.
4 the same LLM labels: what kind of disagreement with gold, per model and per type.
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.evaluate import evaluate, TYPES
OUT = ROOT / "results/figures"; OUT.mkdir(exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e6e5e0"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})

recs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed/gold.jsonl")}
split = json.load(open(ROOT / "data/splits/split_v1.json"))
ORDER = sorted(TYPES, key=lambda t: -sum(1 for r in recs.values() for s in r["spans"] if s[2] == t))

# ---------- Figure 1: gold labels ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
cnt = Counter(s[2] for r in recs.values() for s in r["spans"]); tot = sum(cnt.values())
y = np.arange(len(ORDER))[::-1]
ax[0].barh(y, [cnt[t] for t in ORDER], color=CAT[0], height=0.6)
for yi, t in zip(y, ORDER): ax[0].text(cnt[t] + 25, yi, f"{cnt[t]:,}  ({cnt[t]/tot:.0%})", va="center", fontsize=9, color=INK)
ax[0].set_yticks(y, ORDER); ax[0].set_xlim(0, max(cnt.values()) * 1.35); ax[0].grid(axis="y", visible=False)
ax[0].set_title("How many labels of each type (all 1,076 records)", loc="left", fontsize=11)
# share per split
names = ["train_pool", "val", "test"]
share = np.array([[sum(1 for i in split[n] for s in recs[i]["spans"] if s[2] == t) for t in ORDER] for n in names], float)
share /= share.sum(1, keepdims=True)
left = np.zeros(3)
for k, t in enumerate(ORDER):
    ax[1].barh(range(3)[::-1], share[:, k], left=left, color=CAT[k], height=0.55, edgecolor=SURF, linewidth=2, label=t)
    for j in range(3):
        if share[j, k] > 0.06: ax[1].text(left[j] + share[j, k] / 2, 2 - j, f"{share[j,k]:.0%}", ha="center", va="center", fontsize=8.5, color=INK)
    left += share[:, k]
ax[1].set_yticks(range(3)[::-1], [f"{n} ({len(split[n])})" for n in names]); ax[1].set_xlim(0, 1); ax[1].grid(False)
ax[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax[1].legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax[1].set_title("Type mix is about the same in each split", loc="left", fontsize=11)
# span length per type
for k, t in enumerate(ORDER):
    L = Counter(s[1] - s[0] for r in recs.values() for s in r["spans"] if s[2] == t); n = sum(L.values())
    one, two, more = L[1] / n, L[2] / n, sum(v for l, v in L.items() if l >= 3) / n
    yy = len(ORDER) - 1 - k
    for x0, w, c, lab in [(0, one, CAT[0], "1 word"), (one, two, "#86b6ef", "2 words"), (one + two, more, "#cde2fb", "3+ words")]:
        ax[2].barh(yy, w, left=x0, color=c, height=0.6, edgecolor=SURF, linewidth=2, label=lab if k == 0 else None)
        if w > 0.07: ax[2].text(x0 + w / 2, yy, f"{w:.0%}", ha="center", va="center", fontsize=8.5, color=INK if c != CAT[0] else "white")
ax[2].set_yticks(range(len(ORDER))[::-1], ORDER); ax[2].set_xlim(0, 1); ax[2].grid(False)
ax[2].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax[2].legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax[2].set_title("How long a label is, by type", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "1_gold_labels.png", dpi=150); plt.close(fig)

# ---------- Figure 2: per-type F1 in the active-learning runs (Mac results, mean over seeds) ----------
STRAT = [("random", "Random", CAT[0]), ("uncertainty", "Uncertainty", CAT[1]), ("hybrid", "Hybrid", CAT[2])]
sup = None
fig, axs = plt.subplots(1, 5, figsize=(17, 4), sharey=True)
for fname, lab, c in STRAT:
    rows = [json.loads(l) for l in open(ROOT / f"results/{fname}.jsonl")]
    by = defaultdict(list)
    for r in rows: by[r["frac"]].append(r)
    fr = sorted(by)
    for k, t in enumerate(ORDER):
        m = [np.mean([r["test_per_entity"][t]["f1"] for r in by[f]]) for f in fr]
        axs[k].plot(fr, m, color=c, lw=2, label=lab, marker="o", ms=4)
        sup = rows[0]["test_per_entity"]
for k, t in enumerate(ORDER):
    axs[k].set_title(f"{t}\n(test labels: {sup[t]['support']})", loc="left", fontsize=10)
    axs[k].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); axs[k].set_xlabel("share of train_pool labelled")
axs[0].set_ylabel("test F1 (mean of 10 seeds)"); axs[0].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("Each label type in the active-learning runs (Mac, main results)", x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "2_per_type_learning_curves.png", dpi=150); plt.close(fig)

# ---------- Figure 3: LLM labels on train_pool, by type (exploratory, not the RQ1 analysis) ----------
MODELS = ["gpt-6-astra", "gpt-6-luna", "claude-opus-5-5", "claude-haiku-5-5", "gemini-4-carbon", "qwen4-max", "deepseek-v4-pro"]
ids = split["train_pool"]
gold_cnt = Counter(s[2] for i in ids for s in recs[i]["spans"])
ratio, f1 = np.zeros((len(MODELS), 5)), np.zeros((len(MODELS), 6))
for a, m in enumerate(MODELS):
    P = {json.loads(l)["id"]: json.loads(l)["spans"] for l in open(ROOT / f"data/llm/{m}.jsonl")}
    pc = Counter(s[2] for i in ids for s in P[i])
    ev = evaluate([recs[i]["spans"] for i in ids], [P[i] for i in ids])
    for k, t in enumerate(ORDER):
        ratio[a, k] = pc[t] / gold_cnt[t]; f1[a, k] = ev["per_entity"][t]["f1"]
    f1[a, 5] = ev["micro_f1"]
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [5, 6]})
div = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a0a0", "#c43a3a"])
lr = np.log2(ratio); lim = np.abs(lr).max()
ax[0].imshow(lr, cmap=div, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
for a in range(len(MODELS)):
    for k in range(5): ax[0].text(k, a, f"{ratio[a,k]:.2f}", ha="center", va="center", fontsize=9, color=INK)
ax[0].set_xticks(range(5), [f"{t}\n(gold {gold_cnt[t]})" for t in ORDER], fontsize=8.5); ax[0].set_yticks(range(len(MODELS)), MODELS)
ax[0].grid(False); ax[0].set_title("Labels per type: LLM count ÷ human count\n(red = more than humans, blue = fewer)", loc="left", fontsize=11)
seq = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])
ax[1].imshow(f1, cmap=seq, vmin=0.4, vmax=1, aspect="auto")
for a in range(len(MODELS)):
    for k in range(6): ax[1].text(k, a, f"{f1[a,k]:.2f}", ha="center", va="center", fontsize=9, color="white" if f1[a, k] > 0.82 else INK,
                                  fontweight="bold" if k == 5 else None)
ax[1].set_xticks(range(6), ORDER + ["all (micro)"], fontsize=8.5); ax[1].set_yticks(range(len(MODELS)), [])
ax[1].grid(False); ax[1].set_title("Agreement with human labels (F1), by type", loc="left", fontsize=11)
fig.suptitle("7 LLMs on train_pool (758 records) - first look, not the pre-registered RQ1 analysis", x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "3_llm_by_type.png", dpi=150); plt.close(fig)

# ---------- Figure 4: kinds of disagreement with gold (exploratory) ----------
KINDS = [("bound", "Same type, boundary or nesting differs"), ("missed", "Missed (nothing marked there)"),
         ("bt", "Boundary and type differ"), ("extra", "Extra (no human label there)"), ("type", "Same words, wrong type")]
ov = lambda a, b: a[0] < b[1] and b[0] < a[1]
per_model, per_type = {}, defaultdict(Counter)
for m in MODELS:
    P = {json.loads(l)["id"]: json.loads(l)["spans"] for l in open(ROOT / f"data/llm/{m}.jsonl")}
    c = Counter()
    for i in ids:
        G = {tuple(x) for x in recs[i]["spans"]}; Q = {tuple(x) for x in P[i]}
        for p in Q - G:
            k = ("type" if any(g[:2] == p[:2] for g in G) else "bound" if any(ov(g, p) and g[2] == p[2] for g in G)
                 else "bt" if any(ov(g, p) for g in G) else "extra")
            c[k] += 1; per_type[p[2]][k] += 1
        for g in G - Q:
            if not any(ov(g, p) for p in Q): c["missed"] += 1; per_type[g[2]]["missed"] += 1
    per_model[m] = c
fig, ax = plt.subplots(1, 2, figsize=(16, 5.2))
def stack(a, rows, labels, title):
    y = np.arange(len(rows))[::-1]
    for r, yy in zip(rows, y):
        tot = sum(r.values()); left = 0
        for k, (key, lab) in enumerate(KINDS):
            w = r[key] / tot
            a.barh(yy, w, left=left, color=CAT[k], height=0.62, edgecolor=SURF, linewidth=2,
                   label=lab if yy == y[0] else None)
            if w >= 0.06: a.text(left + w / 2, yy, f"{w:.0%}", ha="center", va="center", fontsize=8.5,
                                  color="white" if k == 0 else INK)
            left += w
        a.text(1.01, yy, f"n={tot:,}", va="center", fontsize=8.5, color=INK2)
    a.set_yticks(y, labels); a.set_xlim(0, 1); a.grid(False)
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a.set_title(title, loc="left", fontsize=11)
stack(ax[0], [per_model[m] for m in MODELS], MODELS, "Per model: what kind of disagreement")
stack(ax[1], [per_type[t] for t in ORDER], ORDER, "Per type (7 models together): what kind of disagreement")
ax[0].legend(ncol=3, fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.08))
fig.suptitle("Where the 7 LLMs differ from the human labels on train_pool - first look, not the RQ1 analysis\n"
             "(n = LLM spans that do not match exactly, plus human spans with nothing marked there; exact matches left out)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "4_llm_disagreement_kinds.png", dpi=150); plt.close(fig)
