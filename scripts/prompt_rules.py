"""Amendment 11: boundary rules from the MaintIE guidelines in the prompt (exploratory).

    python scripts/prompt_rules.py      # writes results/llm/prompt_rules.json

Compares gemini-4-carbon with prompts v2 and v3 on the 78 held-out val records, 3 runs each:
micro-F1 and per-type F1, mean over runs, paired bootstrap over records for v3 - v2.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.evaluate import TYPES, evaluate
from src.llm_analysis import error_types, prompt_diff
from src.llm_label import val_heldout

MODEL, RUNS, N_BOOT, SEED = "gemini-4-carbon", [1, 2, 3], 1000, 0
recs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed/gold.jsonl")}
ids = val_heldout(json.load(open(ROOT / "data/splits/split_v1.json")), 5, 20, 2026)
gold = [recs[i]["spans"] for i in ids]


def load(prompt, run):
    rows = {json.loads(l)["id"]: json.loads(l)["spans"]
            for l in open(ROOT / f"data/llm/heldout/{prompt}/run{run}/{MODEL}.jsonl")}
    assert sorted(rows) == sorted(ids), f"{prompt} run{run}: not all 78 records"
    return [rows[i] for i in ids]


runs = {p: [load(p, r) for r in RUNS] for p in ["label_v2", "label_v3"]}
failed = {p: sum(json.loads(l)["failed"] for r in RUNS
                 for l in open(ROOT / f"data/llm/heldout/{p}/run{r}/raw/{MODEL}.jsonl")) for p in runs}
support = evaluate(gold, gold)["per_entity"]
res = {"n_records": len(ids), "runs": RUNS, "failed_records": failed,
       "gold_spans": {t: support[t]["support"] for t in TYPES},
       "per_run": {p: [{"micro_f1": evaluate(gold, x)["micro_f1"],
                        **{t: evaluate(gold, x)["per_entity"][t]["f1"] for t in TYPES}} for x in runs[p]]
                   for p in runs},
       "micro": prompt_diff(gold, runs["label_v2"], runs["label_v3"], n=N_BOOT, seed=SEED),
       "per_type": {t: prompt_diff(gold, runs["label_v2"], runs["label_v3"], n=N_BOOT, seed=SEED, etype=t)
                    for t in TYPES},
       "n_spans": {p: sum(len(s) for x in runs[p] for s in x) / len(RUNS) for p in runs},
       "n_physical_object_spans": {p: sum(1 for x in runs[p] for s in x for y in s if y[2] == "PhysicalObject")
                                   / len(RUNS) for p in runs}}

kinds = {}
for p in runs:
    c = {}
    for x in runs[p]:
        for g, q in zip(gold, x):
            for k, v in error_types(g, q).items():
                c[k] = c.get(k, 0) + v / len(RUNS)
    kinds[p] = c
res["error_types_mean_per_run"] = kinds

po, mi = res["per_type"]["PhysicalObject"], res["micro"]
others = {t: res["per_type"][t]["diff"] for t in TYPES if t != "PhysicalObject" and res["gold_spans"][t] >= 30}
res["decision"] = {"a_physical_object": po["diff"] >= 0.03 and po["ci95"][0] > 0,
                   "b_micro_not_lower": mi["ci95"][1] >= 0,
                   "c_other_types": {t: d >= -0.02 for t, d in others.items()}}
res["decision"]["all_met"] = (res["decision"]["a_physical_object"] and res["decision"]["b_micro_not_lower"]
                              and all(res["decision"]["c_other_types"].values()))

out = ROOT / "results/llm/prompt_rules.json"
json.dump(res, open(out, "w"), indent=2)
for k in ["micro"] + [f"per_type/{t}" for t in TYPES]:
    d = res["micro"] if k == "micro" else res["per_type"][k.split("/")[1]]
    print(f"{k:24s} v2 {d['a']:.4f}  v3 {d['b']:.4f}  diff {d['diff']:+.4f}  CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
print("spans per run", res["n_spans"], "PO", res["n_physical_object_spans"])
print("error kinds", json.dumps(kinds, indent=None))
print("failed", failed, "gold", res["gold_spans"])
print("decision", res["decision"])
