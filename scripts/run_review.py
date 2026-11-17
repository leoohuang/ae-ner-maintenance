"""LLM annotation protocol, RQ3b (amendment 8): LLM labels with k records reviewed (human labels),
one NER model on all of train_pool per (model, k, choice, seed). Run on the NAISS GPU.

    python scripts/run_review.py --model gemini-4-carbon [--seeds 1 2 ...]

Needs results/llm/rq2_signals.json (the combined RQ2 signal per model and seed, made by scripts/rq2.py).
Appends rows to results/gpu/llm/review.jsonl.
"""
import argparse, gc, json, time
from pathlib import Path
import torch, yaml
from src.model import finetune, predict, decode
from src.evaluate import evaluate
from src.llm_analysis import apply_labels, choose_review
from src.pool import SEEDS, token_lengths

ROOT = Path(__file__).resolve().parent.parent
KS = [50, 100, 200]
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
ap.add_argument("--out", default="results/gpu/llm/review.jsonl")
a = ap.parse_args()

cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
records = [json.loads(l) for l in open(ROOT / "data/processed/gold.jsonl", encoding="utf-8")]
tok_len = token_lengths(records)
split = json.load(open(ROOT / "data/splits/split_v1.json"))
pool = split["train_pool"]
test = [records[i] for i in split["test"]]
llm = {json.loads(l)["id"]: json.loads(l)["spans"] for l in open(ROOT / f"data/llm/{a.model}.jsonl")}
signals = json.load(open(ROOT / "results/llm/rq2_signals.json"))[a.model]
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

out = ROOT / a.out
out.parent.mkdir(parents=True, exist_ok=True)
done = {(r["model"], r["k"], r["choice"], r["seed"]) for r in map(json.loads, open(out))} if out.exists() else set()

for seed in a.seeds:
    sig = signals[str(seed)]
    combined = {int(i): v for i, v in sig["combined"].items()}
    rq2_pool = sorted(combined)                     # train_pool outside this seed's initial 5% set
    runs = [(0, "none")] + [(k, c) for k in KS for c in ["signal", "random"]]
    for k, choice in runs:
        if (a.model, k, choice, seed) in done:
            continue
        t0 = time.time()
        reviewed = [] if k == 0 else choose_review(rq2_pool, combined, k, choice, seed)
        train = apply_labels(records, llm, split, reviewed=reviewed)
        model, tok = finetune(cfg["encoder"], [train[i] for i in pool], cfg, seed, device)
        res = predict(model, tok, test, cfg["max_len"], device)
        mt = evaluate([r["spans"] for r in test], [decode(sp, pr) for sp, pr in res])
        row = {"model": a.model, "k": k, "choice": choice, "seed": seed, "reviewed": reviewed,
               "reviewed_tokens": sum(tok_len[i] for i in reviewed),
               "test_micro_f1": mt["micro_f1"], "test_precision": mt["precision"], "test_recall": mt["recall"],
               "test_per_entity": mt["per_entity"], "sec": round(time.time() - t0)}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print({x: row[x] for x in ("model", "k", "choice", "seed", "test_micro_f1", "sec")}, flush=True)
        del model, tok
        gc.collect()
