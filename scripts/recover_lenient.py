"""README amendment 7: recover the labels of replies that failed the strict JSON format.

    python scripts/recover_lenient.py            # after all train_pool runs are finished

For every record whose reply failed strict parsing (after the retries), the logged replies are
parsed leniently (src.llm_label.parse_reply_lenient). No model is called again. The recovered
spans replace the empty spans in data/llm/<model>.jsonl; data/llm/raw/ keeps the strict result
(`failed`), which is the number reported as format compliance. data/llm/recovered.jsonl lists
every recovered record. Running it again gives the same files.
"""
import json
from pathlib import Path
from src.llm_label import recover_failed

ROOT = Path(__file__).resolve().parent.parent
LLM = ROOT / "data/llm"

log = []
for raw_path in sorted((LLM / "raw").glob("*.jsonl")):
    model = raw_path.stem
    raw = [json.loads(l) for l in open(raw_path)]
    rows = [json.loads(l) for l in open(LLM / f"{model}.jsonl")]
    by_id = {r["id"]: r for r in rows}
    failed = [r for r in raw if r["failed"]]
    n_rec = 0
    for r in failed:
        rec = recover_failed(r, len(by_id[r["id"]]["tokens"]))
        if rec is not None:
            by_id[r["id"]]["spans"] = rec["spans"]
            log.append({"model": model, "id": r["id"], **rec})
            n_rec += 1
    with open(LLM / f"{model}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{model}: {len(raw)} records, strict failures {len(failed)}, recovered {n_rec}, "
          f"still without labels {len(failed) - n_rec}")

with open(LLM / "recovered.jsonl", "w") as f:
    for r in log:
        f.write(json.dumps(r) + "\n")
