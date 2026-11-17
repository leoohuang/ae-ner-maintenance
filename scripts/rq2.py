"""LLM annotation protocol, RQ2: can a model's labelling errors be found before review? (README; amendment 9)

    python scripts/rq2.py [--unc results/gpu/llm/rq2_uncertainty.jsonl]

Signals per model and seed, over the train_pool records outside the seed's initial 5% set:
1 model uncertainty (from scripts/rq2_uncertainty.py on the NAISS GPU), 2 LLM disagreement
(1 - mean span F1 with the other seven models), 3 combined (mean of the two ranks).
Measures: AUROC for "record in error"; share of the model's span errors in the top k = 50, 100, 200
records, against k / N for k random records. Averaged over seeds, per model.
Writes results/llm/rq2.json and results/llm/rq2_signals.json (the combined signal, used by RQ3b).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.llm_analysis import auroc, combined_rank, disagreement, span_errors, topk_error_share

MODELS = ["gpt-6-astra", "gpt-6-luna", "claude-opus-5-5", "claude-haiku-5-5", "gemini-4-carbon",
          "qwen4-max", "qwen3.8-27b", "deepseek-v4-pro"]
KS = [50, 100, 200]
ap = argparse.ArgumentParser()
ap.add_argument("--unc", default="results/gpu/llm/rq2_uncertainty.jsonl")
ap.add_argument("--out-dir", default="results/llm")
a = ap.parse_args()

recs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed/gold.jsonl")}
ids = json.load(open(ROOT / "data/splits/split_v1.json"))["train_pool"]
labels = {m: {json.loads(l)["id"]: json.loads(l)["spans"] for l in open(ROOT / f"data/llm/{m}.jsonl")}
          for m in MODELS}
unc_rows = [json.loads(l) for l in open(ROOT / a.unc)]
dis_all = {m: {i: disagreement(labels, m, i) for i in ids} for m in MODELS}
err_all = {m: {i: span_errors(recs[i]["spans"], labels[m][i]) for i in ids} for m in MODELS}

per = {}            # (model, seed) -> measures
signals = {}
for r in unc_rows:
    m, seed = r["model"], r["seed"]
    unc = {int(i): v for i, v in r["uncertainty"].items()}
    pool = sorted(unc)
    dis = {i: dis_all[m][i] for i in pool}
    comb = combined_rank(unc, dis)
    errs = {i: err_all[m][i] for i in pool}
    wrong = [int(errs[i] > 0) for i in pool]
    sig = {"uncertainty": unc, "disagreement": dis, "combined": {i: -v for i, v in comb.items()}}
    meas = {"n_pool": len(pool), "records_in_error": sum(wrong), "span_errors": sum(errs.values())}
    for name, s in sig.items():
        meas[name] = {"auroc": auroc([s[i] for i in pool], wrong),
                      **{f"top{k}_share": topk_error_share(s, errs, k) for k in KS}}
    meas["random"] = {f"top{k}_share": k / len(pool) for k in KS}
    per[(m, seed)] = meas
    signals.setdefault(m, {})[str(seed)] = {"combined": {str(i): v for i, v in comb.items()}}

res = {"per_model": {}, "n_seeds": {}}
for m in MODELS:
    rows = [v for (mm, _), v in per.items() if mm == m]
    res["n_seeds"][m] = len(rows)
    if not rows:
        continue
    out = {"records_in_error": float(np.mean([x["records_in_error"] for x in rows])),
           "n_pool": float(np.mean([x["n_pool"] for x in rows]))}
    for name in ["uncertainty", "disagreement", "combined", "random"]:
        keys = rows[0][name].keys()
        out[name] = {k: float(np.mean([x[name][k] for x in rows])) for k in keys}
    res["per_model"][m] = out
res["per_seed"] = {f"{m}|{s}": v for (m, s), v in per.items()}

od = ROOT / a.out_dir; od.mkdir(parents=True, exist_ok=True)
json.dump(res, open(od / "rq2.json", "w"), indent=1)
json.dump(signals, open(od / "rq2_signals.json", "w"))

print(f"{'model':18s} seeds  in-error  " + "  ".join(f"{s[:5]:>5s} AUROC/top50/top100/top200" for s in ["unc", "dis", "comb"]))
for m, v in res["per_model"].items():
    line = f"{m:18s} {res['n_seeds'][m]:5d}  {v['records_in_error']:6.0f}/{v['n_pool']:.0f}  "
    for name in ["uncertainty", "disagreement", "combined"]:
        x = v[name]
        line += f"  {x['auroc']:.3f}/{x['top50_share']:.2f}/{x['top100_share']:.2f}/{x['top200_share']:.2f}"
    print(line, f"  random {v['random']['top50_share']:.2f}/{v['random']['top100_share']:.2f}/{v['random']['top200_share']:.2f}")
