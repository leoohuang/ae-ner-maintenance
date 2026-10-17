import argparse, gc, json, time
from pathlib import Path
import torch, yaml
from src.model import finetune, predict, decode
from src.evaluate import evaluate
from src.pool import (GRID, SEEDS, budgets, pool_tokens, random_order,
                      random_nested_sets, token_lengths)

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--fracs", nargs="+", type=float, default=GRID)
ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
ap.add_argument("--out", default="results/random.jsonl")
a = ap.parse_args()

cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
assert cfg["encoder"] and cfg["encoder_lr"], "configs/base.yaml 还没有填好"
records = [json.loads(l) for l in open(ROOT / "data/processed/gold.jsonl", encoding="utf-8")]
tok_len = token_lengths(records)
split = json.load(open(ROOT / "data/splits/split_v1.json"))
pool = split["train_pool"]
val = [records[i] for i in split["val"]]
test = [records[i] for i in split["test"]]
bud = budgets(pool_tokens(records, pool))
device = "mps" if torch.backends.mps.is_available() else "cpu"

out = ROOT / a.out
out.parent.mkdir(exist_ok=True)
done = set()
if out.exists():
    for l in open(out):
        r = json.loads(l)
        done.add((r["frac"], r["seed"]))


def score(model, tok, recs):
    res = predict(model, tok, recs, cfg["max_len"], device)
    return evaluate([r["spans"] for r in recs], [decode(sp, pr) for sp, pr in res])


for seed in a.seeds:
    sets = random_nested_sets(random_order(pool, seed), tok_len, [bud[f] for f in GRID])
    for frac in a.fracs:
        if (frac, seed) in done:
            continue
        t0 = time.time()
        chosen, used = sets[bud[frac]]
        model, tok = finetune(cfg["encoder"], [records[i] for i in chosen], cfg, seed, device)
        mt, mv = score(model, tok, test), score(model, tok, val)
        row = {"strategy": "random", "frac": frac, "seed": seed,
               "n_records": len(chosen), "tokens_used": used, "budget": bud[frac],
               "test_micro_f1": mt["micro_f1"], "test_precision": mt["precision"],
               "test_recall": mt["recall"], "test_macro_f1": mt["macro_f1"],
               "test_per_entity": mt["per_entity"], "val_micro_f1": mv["micro_f1"],
               "sec": round(time.time() - t0)}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print({k: row[k] for k in ("frac", "seed", "n_records", "tokens_used",
                                   "budget", "test_micro_f1", "val_micro_f1", "sec")}, flush=True)
        del model, tok
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
