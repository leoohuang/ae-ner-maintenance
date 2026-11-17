"""LLM annotation protocol, RQ2 signal 1 (model uncertainty), run on the NAISS GPU (amendment 2).

    python scripts/rq2_uncertainty.py --model gemini-4-carbon [--seeds 1 2 ...]

Per seed: train the NER model on the seed's initial 5% set with this LLM's labels, then score every
other train_pool record with the existing record score (mean of the L largest span entropies).
Appends one row per (model, seed) to results/gpu/llm/rq2_uncertainty.jsonl.
"""
import argparse, gc, json, time
from pathlib import Path
import torch, yaml
from src.model import finetune, predict
from src.llm_analysis import apply_labels
from src.pool import GRID, SEEDS, budgets, pool_tokens, random_order, random_nested_sets, token_lengths
from src.strategies import record_score

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
ap.add_argument("--out", default="results/gpu/llm/rq2_uncertainty.jsonl")
a = ap.parse_args()

cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
records = [json.loads(l) for l in open(ROOT / "data/processed/gold.jsonl", encoding="utf-8")]
tok_len = token_lengths(records)
split = json.load(open(ROOT / "data/splits/split_v1.json"))
pool = split["train_pool"]
llm = {json.loads(l)["id"]: json.loads(l)["spans"] for l in open(ROOT / f"data/llm/{a.model}.jsonl")}
train_recs = apply_labels(records, llm, split)
bud = budgets(pool_tokens(records, pool))
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

out = ROOT / a.out
out.parent.mkdir(parents=True, exist_ok=True)
done = {(r["model"], r["seed"]) for r in map(json.loads, open(out))} if out.exists() else set()

for seed in a.seeds:
    if (a.model, seed) in done:
        continue
    t0 = time.time()
    # the same initial 5% set as the active-learning runs of this seed
    initial, used = random_nested_sets(random_order(pool, seed), tok_len, [bud[f] for f in GRID])[bud[0.05]]
    model, tok = finetune(cfg["encoder"], [train_recs[i] for i in initial], cfg, seed, device)
    rest = [i for i in pool if i not in set(initial)]
    pred = predict(model, tok, [records[i] for i in rest], cfg["max_len"], device)
    scores = {i: record_score(pr, tok_len[i]) for i, (_, pr) in zip(rest, pred)}
    row = {"model": a.model, "seed": seed, "initial": initial, "tokens_used": used,
           "uncertainty": scores, "sec": round(time.time() - t0)}
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(a.model, seed, len(initial), "initial,", len(scores), "scored,", row["sec"], "s", flush=True)
    del model, tok
    gc.collect()
