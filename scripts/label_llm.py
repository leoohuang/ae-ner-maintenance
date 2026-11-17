"""LLM annotation protocol, step 1: one LLM labels train_pool (or the 20 prompt-tuning val records).

    python scripts/label_llm.py --model claude-haiku-5-5 --split tune --limit 1   # smoke test
    python scripts/label_llm.py --model claude-haiku-5-5 --split tune             # prompt tuning
    python scripts/label_llm.py --model claude-haiku-5-5                           # train_pool

Writes data/llm/<model>.jsonl (gold.jsonl format) and data/llm/raw/<model>.jsonl
(tune: data/llm/tune/<prompt>/...). Interrupted runs continue where they stopped.
"""
import argparse, json
from pathlib import Path
import yaml
from src.evaluate import evaluate
from src.llm_clients import load_env, make_client
from src.llm_label import assert_not_test, render_prompt, run_labelling, val_subsets

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_SEED, N_EXAMPLES, N_TUNE = 2026, 5, 20          # preregistered: 5 examples, at most 20 tuning records

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--split", choices=["train_pool", "tune"], default="train_pool")
ap.add_argument("--prompt", default="configs/prompts/label_v2.md")   # frozen, README amendment 6
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

load_env(ROOT / ".env")
cfg = yaml.safe_load(open(ROOT / "configs/llm_models.yaml"))[a.model]
records = [json.loads(l) for l in open(ROOT / "data/processed/gold.jsonl", encoding="utf-8")]
split = json.load(open(ROOT / "data/splits/split_v1.json"))
ex_ids, tune_ids = val_subsets(split, N_EXAMPLES, N_TUNE, EXAMPLE_SEED)
ids = split["train_pool"] if a.split == "train_pool" else tune_ids
ids = ids[:a.limit] if a.limit else ids
assert_not_test(ids + ex_ids, split)

template = (ROOT / a.prompt).read_text()
examples = [records[i] for i in ex_ids]
out = ROOT / "data/llm" if a.split == "train_pool" else ROOT / "data/llm/tune" / Path(a.prompt).stem
info = {"model": a.model, "model_id": cfg["model"], "provider": cfg["provider"], "prompt": a.prompt}
run_labelling(ids, records, make_client(cfg), lambda r: render_prompt(template, r, examples),
              out / f"{a.model}.jsonl", out / "raw" / f"{a.model}.jsonl", info)

rows = {json.loads(l)["id"]: json.loads(l)["spans"] for l in open(out / f"{a.model}.jsonl")}
raw = [json.loads(l) for l in open(out / "raw" / f"{a.model}.jsonl")]
done = [i for i in ids if i in rows]
m = evaluate([records[i]["spans"] for i in done], [rows[i] for i in done])
print(f"{a.model}: {len(done)} records, failed {sum(r['failed'] for r in raw)}, "
      f"dropped spans {sum(sum(r['drops'].values()) for r in raw)}"
      + (f", micro-F1 vs gold {m['micro_f1']:.3f}" if a.split == "tune" else ""))
