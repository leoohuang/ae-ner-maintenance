"""Summary figures and label examples (descriptive, added 2026-11-21 after all results; no new analysis).

    python scripts/plot_summary.py

Writes into results/figures/:
  5_study_flow.png      how the parts of the study fit together
  6_summary_ladder.png  test micro-F1 of the main training settings on one axis (GPU runs, mean of seeds 1-10)
  rq2_topk.png          RQ2: share of a model's span errors seen when checking the top k records, per signal
  rq3c_per_type.png     RQ3c: PhysicalObject and State F1, reviewed records alone vs with the LLM labels
and results/llm/label_examples.json: gold, gemini-4-carbon and qwen3.8-27b spans for a few train_pool records,
chosen by hand to illustrate the kinds of disagreement in RQ1 (not a sample).
All numbers are read from files that the RQ scripts already wrote; nothing is recomputed.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results/figures"
SURF, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e6e5e0"
BLUE, GREY, ORANGE, GREEN = "#2a78d6", "#6b6a63", "#c8702a", "#1a8a5a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
load = lambda p: json.load(open(ROOT / p))
rows = lambda p: [json.loads(l) for l in open(ROOT / p)]
LLMS = load("results/llm/rq1.json")["rq3_models"]
G, Q = LLMS["highest"], LLMS["lowest"]

# ---- 5 study flow ----
fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")


def box(x, y, w, h, title, body, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=col, lw=1.6))
    ax.text(x + 1, y + h - 1.4, title, fontsize=9.5, fontweight="bold", color=col, va="top")
    ax.text(x + 1, y + h - 5.0, body, fontsize=8, color=INK, va="top", linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))


box(1, 44, 22, 16, "MaintIE gold data", "1,076 records, 5 types\nsplit_v1 (near-duplicates\nkept together):\ntrain_pool 758 / val 103 /\ntest 215 (never sent to LLMs)", INK)
box(29, 47, 30, 13, "Human labels: active learning", "random / uncertainty / hybrid\n9 budgets 5-100%, seeds 1-10\nMac (main), then GPU replication\nresult: uncertainty saves ~40%", BLUE)
box(65, 47, 33, 13, "Model (frozen after pilot)", "bert-base-uncased + span head\nchosen on val only\nsame settings in every experiment", GREY)
box(29, 26, 30, 15, "LLM labels (8 models)", "prompt tuned on 20 val records\neach model labels all 758\ntrain_pool records\nRQ1: agreement with human labels", ORANGE)
box(65, 26, 33, 15, "RQ2: find LLM errors first", "signals: NER uncertainty,\ndisagreement between the LLMs,\ncombined (mean rank)\ncheck the top k records", ORANGE)
box(1, 3, 30, 17, "RQ3: train on LLM labels", "best + worst LLM (gemini, qwen)\nsame active learning as above\nevaluate on human test labels", GREEN)
box(35, 3, 30, 17, "RQ3b: LLM labels + k reviewed", "k = 50 / 100 / 200 records get\nhuman labels; rest keep LLM labels\nchosen by RQ2 signal or at random", GREEN)
box(69, 3, 29, 17, "RQ3c: k reviewed alone", "exploratory (amendment 10)\nsame k records, human labels,\nno LLM labels at all", GREEN)
arrow(23.5, 53, 28.5, 53); arrow(12, 43.5, 28.5, 34)
arrow(59.5, 33.5, 64.5, 33.5); arrow(44, 25.5, 16, 20.5); arrow(81, 25.5, 50, 20.5); arrow(65.5, 11.5, 68.5, 11.5)
ax.text(1, 61.5, "How the study fits together (blue: human labels only; orange: LLM labelling; green: NER trained with LLM labels)",
        fontsize=10.5, color=INK)
fig.savefig(FIG / "5_study_flow.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 6 summary ladder ----
def f1s(path, **cond):
    return [r["test_micro_f1"] for r in rows(path) if all(r[k] == v for k, v in cond.items())]


rq3b, rq3c = load("results/llm/rq3b.json"), load("results/llm/rq3c.json")
b = lambda m, key: (rq3b["per_model"][m][key]["f1_mean"], rq3b["per_model"][m][key]["f1_sd"])
ms = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1)))
ladder = [  # (label, (mean, sd), kind)
    (f"{Q} labels on all 758 records", ms(f1s(f"results/gpu/llm/{Q}/random.jsonl", frac=1.0)), "llm"),
    ("human labels, 5% of text (~39 records, random)", ms(f1s("results/gpu/random.jsonl", frac=0.05)), "human"),
    (f"{G} labels on all 758 records", ms(f1s(f"results/gpu/llm/{G}/random.jsonl", frac=1.0)), "llm"),
    ("human labels, 10% of text (~76 records, random)", ms(f1s("results/gpu/random.jsonl", frac=0.1)), "human"),
    (f"{Q} labels + 200 reviewed (signal)", b(Q, "200|signal"), "mix"),
    (f"{G} labels + 200 reviewed (signal)", b(G, "200|signal"), "mix"),
    ("200 reviewed records alone (random)", (rq3c["per_model"][G]["200|random"]["alone_f1"],
                                             rq3c["per_model"][G]["200|random"]["alone_sd"]), "human"),
    ("human labels, 30% of text (uncertainty)", ms(f1s("results/gpu/uncertainty.jsonl", frac=0.3)), "human"),
    ("human labels on all 758 records", ms(f1s("results/gpu/random.jsonl", frac=1.0)), "human"),
]
COLK = {"llm": ORANGE, "mix": GREEN, "human": BLUE}
fig, ax = plt.subplots(figsize=(9.5, 4.6))
for i, (lab, (m, s), k) in enumerate(ladder):
    ax.errorbar(m, i, xerr=s, fmt="o", ms=7, color=COLK[k], ecolor=COLK[k], capsize=3, lw=1.4)
    ax.text(m + s + 0.004, i, f"{m:.4f}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(range(len(ladder))); ax.set_yticklabels([l for l, _, _ in ladder], fontsize=8.5)
ax.set_xlim(0.68, 0.92); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
ax.set_xlabel("test micro-F1 (human test labels; GPU, mean ± sd over seeds 1-10)")
for k, lab in [("llm", "LLM labels only"), ("mix", "LLM labels + human review"), ("human", "human labels only")]:
    ax.plot([], [], "o", color=COLK[k], label=lab)
ax.legend(frameon=False, loc="lower right", fontsize=8.5)
ax.set_title("All main settings on one axis", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "6_summary_ladder.png", dpi=150); plt.close(fig)

# ---- RQ2 top k ----
rq2 = load("results/llm/rq2.json")["per_model"]
KS = [50, 100, 200]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
order = sorted(rq2, key=lambda m: rq2[m]["disagreement"]["auroc"])
for sig, col, mk in [("disagreement", BLUE, "o"), ("combined", GREEN, "D"), ("uncertainty", ORANGE, "s")]:
    ax.plot([rq2[m][sig]["auroc"] for m in order], range(len(order)), mk, color=col, ms=6, label=sig)
ax.axvline(0.5, color=INK2, ls=":", lw=1); ax.text(0.505, -0.6, "chance", fontsize=8, color=INK2)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8.5); ax.set_xlim(0.45, 0.95)
ax.set_xlabel("AUROC for 'record has an error' (mean of seeds)"); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="center left"); ax.set_title("How well each signal separates records", loc="left", fontsize=10)
ax = axes[1]
for sig, col in [("disagreement", BLUE), ("combined", GREEN), ("uncertainty", ORANGE), ("random", INK2)]:
    v = np.array([[rq2[m][sig][f"top{k}_share"] for k in KS] for m in rq2])
    ax.fill_between(KS, v.min(0), v.max(0), color=col, alpha=0.12, lw=0)
    ax.plot(KS, v.mean(0), marker="o", ms=5, lw=2, color=col, ls="--" if sig == "random" else "-",
            label=f"{sig} (8-model mean, band: min-max)" if sig != "random" else "random records")
ax.set_xticks(KS); ax.set_xlabel("records checked (of about 719)"); ax.set_ylim(0, 0.6)
ax.set_ylabel("share of the model's span errors seen"); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.set_title("Errors found when checking the top k", loc="left", fontsize=10)
fig.suptitle("RQ2: finding LLM errors before a human checks them", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "rq2_topk.png", dpi=150); plt.close(fig)

# ---- RQ3c per type ----
TY = ["PhysicalObject", "State"]
fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
for j, m in enumerate([G, Q]):
    pm, k0 = rq3c["per_model"][m], rq3b["per_model"][m]["0|none"]["per_type_f1"]
    for i, t in enumerate(TY):
        ax = axes[i, j]
        for c, col in [("signal", BLUE), ("random", GREY)]:
            ax.plot([0] + KS, [k0[t]] + [pm[f"{k}|{c}"]["with_llm_per_type"][t] for k in KS], marker="o", ms=5, lw=2,
                    color=col, label=f"{c}: reviewed + LLM labels")
            ax.plot(KS, [pm[f"{k}|{c}"]["alone_per_type"][t] for k in KS], marker="s", ms=5, lw=2, ls="--",
                    color=col, label=f"{c}: reviewed records alone")
        ax.set_title(f"{t}, {m}", loc="left", fontsize=9.5); ax.grid(color=GRID); ax.set_axisbelow(True)
        ax.set_xticks([0] + KS)
        if j == 0: ax.set_ylabel("test F1 (mean of seeds)")
        if i == 1: ax.set_xlabel("records reviewed (k of 758)")
axes[1, 0].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.suptitle("RQ3c (exploratory): the LLM labels help State at every k; they hurt PhysicalObject from k = 100 (gemini) or at every k (qwen)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "rq3c_per_type.png", dpi=150); plt.close(fig)

# ---- label examples ----
EXAMPLES = {79: "boundary: the LLMs join a modifier and an object into one name",
            92: "boundary: compare with 79, where the human labels split 'steering hose'",
            0: "extra nesting: gemini adds the longer object name as well",
            38: "boundary and type: the human labels mark 'temperature fault' as one State",
            41: "type: the human guideline marks 'to be fixed' as State, the LLMs mark an Activity",
            85: "missed span: the human labels mark the trade ('boilermaker') as PhysicalObject",
            12: "extra span: gemini marks 'awaiting' as State"}
recs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed/gold.jsonl")}
lab = {m: {json.loads(l)["id"]: json.loads(l)["spans"] for l in open(ROOT / f"data/llm/{m}.jsonl")} for m in (G, Q)}
show = lambda toks, sp: [[" ".join(toks[s:e]), t] for s, e, t in sorted(map(tuple, sp))]
out = []
for i, note in EXAMPLES.items():
    t = recs[i]["tokens"]
    out.append({"id": i, "text": " ".join(t[1:]), "note": note, "gold": show(t, recs[i]["spans"]),
                G: show(t, lab[G][i]), Q: show(t, lab[Q][i])})
json.dump(out, open(ROOT / "results/llm/label_examples.json", "w"), indent=1)
for e in out:
    print(e["id"], e["text"], "|", e["note"])
    for k in ("gold", G, Q):
        print(f"   {k:16s}", "; ".join(f"{s}={t}" for s, t in e[k]))
print("ladder:", [(l, round(m, 4)) for l, (m, s), _ in ladder])
