import argparse, gc, json, time
from pathlib import Path
import numpy as np
import torch, yaml
from src.model import finetune, predict, decode
from src.evaluate import evaluate
from src.pool import (GRID, SEEDS, budgets, pool_tokens, random_order,
                      random_nested_sets, fill_budget, token_lengths)
from src.strategies import rank_by_uncertainty, record_score
from src.hybrid import hybrid_select
from src.embed import static_embeddings

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
ap.add_argument("--max-frac", type=float, default=0.80)
ap.add_argument("--strategy", choices=["uncertainty", "hybrid"], default="uncertainty")
ap.add_argument("--out", default=None)
a = ap.parse_args()
a.out = a.out or f"results/{a.strategy}.jsonl"

cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
assert cfg["encoder"] and cfg["encoder_lr"], "configs/base.yaml 还没有填好"
records = [json.loads(l) for l in open(ROOT / "data/processed/gold.jsonl", encoding="utf-8")]
tok_len = token_lengths(records)
emb = static_embeddings(records) if a.strategy == "hybrid" else None
split = json.load(open(ROOT / "data/splits/split_v1.json"))
pool = split["train_pool"]
val = [records[i] for i in split["val"]]
test = [records[i] for i in split["test"]]
bud = budgets(pool_tokens(records, pool))
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
AL_FRACS = [f for f in GRID if f < 1.0 and f <= a.max_frac]

out = ROOT / a.out
out.parent.mkdir(exist_ok=True)
rows_old = [json.loads(l) for l in open(out)] if out.exists() else []


def score(model, tok, recs):
    res = predict(model, tok, recs, cfg["max_len"], device)
    return evaluate([r["spans"] for r in recs], [decode(sp, pr) for sp, pr in res])


for seed in a.seeds:
    have = {r["frac"] for r in rows_old if r["seed"] == seed}
    if have >= set(AL_FRACS):
        continue
    if have:    # 这个种子上次没跑完：丢掉残留的行，整个种子重跑
        rows_old = [r for r in rows_old if r["seed"] != seed]
        with open(out, "w") as f:
            for r in rows_old:
                f.write(json.dumps(r) + "\n")

    # 起点：随机基线里同一个种子的 5% 数据
    sets = random_nested_sets(random_order(pool, seed), tok_len, [bud[f] for f in GRID])
    labelled, used = sets[bud[AL_FRACS[0]]]
    labelled = list(labelled)

    for k, frac in enumerate(AL_FRACS):
        t0 = time.time()
        assert len(labelled) == len(set(labelled)) and used <= bud[frac]
        model, tok = finetune(cfg["encoder"], [records[i] for i in labelled], cfg, seed, device)
        mt, mv = score(model, tok, test), score(model, tok, val)
        row = {"strategy": "a.strategy", "frac": frac, "seed": seed,
               "n_records": len(labelled), "tokens_used": used, "budget": bud[frac],
               "test_micro_f1": mt["micro_f1"], "test_precision": mt["precision"],
               "test_recall": mt["recall"], "test_macro_f1": mt["macro_f1"],
               "test_per_entity": mt["per_entity"], "val_micro_f1": mv["micro_f1"],
               "labelled": labelled}

        if k < len(AL_FRACS) - 1:
            # 用同一个模型给没标注的记录打分、排队，补到下一个预算点
            taken = set(labelled)
            remaining = [i for i in pool if i not in taken]
            pred = predict(model, tok, [records[i] for i in remaining], cfg["max_len"], device)
            ranked = rank_by_uncertainty(pred, remaining, records)
            target = bud[AL_FRACS[k + 1]]
            if a.strategy == "hybrid":
                new, used = hybrid_select(ranked, emb, labelled, tok_len, used, target)
            else:
                new, used = fill_budget(ranked, tok_len, used, target)
            scores = {i: record_score(pr, tok_len[i]) for i, (_, pr) in zip(remaining, pred)}
            chosen = set(new)
            rest = [scores[i] for i in remaining if i not in chosen]
            print(f"  选入 {len(new)} 条 | 平均不确定性 选中 {np.mean([scores[i] for i in new]):.3f}"
                  f" vs 未选 {np.mean(rest):.3f} | 平均词数 选中 {np.mean([tok_len[i] for i in new]):.2f}"
                  f" vs 全池 {np.mean([tok_len[i] for i in pool]):.2f}", flush=True)
            labelled = labelled + new

        row["sec"] = round(time.time() - t0)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        rows_old.append(row)
        print({x: row[x] for x in ("frac", "seed", "n_records", "tokens_used", "budget",
                                   "test_micro_f1", "val_micro_f1", "sec")}, flush=True)
        del model, tok
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()