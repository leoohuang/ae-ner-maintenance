import argparse, json, time, statistics as st
from pathlib import Path
import torch, yaml
from src.model import finetune, predict, decode
from src.evaluate import evaluate

ROOT = Path(__file__).resolve().parent.parent
ENCODERS = {"bert": "bert-base-uncased", "roberta": "roberta-base",
            "modernbert": "answerdotai/ModernBERT-base"}
SEEDS = [101, 102, 103]

ap = argparse.ArgumentParser()
ap.add_argument("--encoders", nargs="+", default=["bert"])
ap.add_argument("--lrs", nargs="+", type=float, default=[5e-5])
a = ap.parse_args()

cfg0 = yaml.safe_load(open(ROOT / "configs/base.yaml"))
records = [json.loads(l) for l in open(ROOT / "data/processed/gold.jsonl", encoding="utf-8")]
split = json.load(open(ROOT / "data/splits/split_v1.json"))
train = [records[i] for i in split["train_pool"]]
val = [records[i] for i in split["val"]]            # 测试集不加载
device = "mps" if torch.backends.mps.is_available() else "cpu"

out_path = ROOT / "results/pilot.json"
results = json.load(open(out_path)) if out_path.exists() else []

for enc in a.encoders:
    for lr in a.lrs:
        cfg = {**cfg0, "encoder_lr": lr}
        for seed in SEEDS:
            t0 = time.time()
            try:
                model, tok = finetune(ENCODERS[enc], train, cfg, seed, device)
                res = predict(model, tok, val, cfg["max_len"], device)
                pred = [decode(sp, pr) for sp, pr in res]
                m = evaluate([r["spans"] for r in val], pred)
                row = {"encoder": enc, "lr": lr, "seed": seed, "ok": True,
                       "micro_f1": m["micro_f1"], "precision": m["precision"],
                       "recall": m["recall"], "macro_f1": m["macro_f1"],
                       "sec": round(time.time() - t0)}
            except Exception as e:
                row = {"encoder": enc, "lr": lr, "seed": seed, "ok": False,
                       "error": repr(e)[:300], "sec": round(time.time() - t0)}
            print(row, flush=True)
            results.append(row)
            json.dump(results, open(out_path, "w"), indent=1)
            del_vars = [k for k in ("model", "tok") if k in globals()]
            for k in del_vars: del globals()[k]

print("\n编码器 | lr | 成功次数 | micro-F1 均值 ± std | 平均耗时(秒)")
keys = sorted({(r["encoder"], r["lr"]) for r in results})
for enc, lr in keys:
    rs = [r for r in results if r["encoder"] == enc and r["lr"] == lr]
    ok = [r for r in rs if r["ok"]]
    if ok:
        f = [r["micro_f1"] for r in ok]
        sd = st.stdev(f) if len(f) > 1 else 0.0
        print(f"{enc} | {lr} | {len(ok)}/{len(rs)} | {st.mean(f):.4f} ± {sd:.4f} | "
              f"{st.mean(r['sec'] for r in ok):.0f}")
    else:
        print(f"{enc} | {lr} | 0/{len(rs)} | 全部失败: {rs[0]['error']}")