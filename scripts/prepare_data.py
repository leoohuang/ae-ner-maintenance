import json
from pathlib import Path
from src.data import to_spans

ROOT = Path(__file__).resolve().parent.parent
raw = json.load(open(ROOT / "data" / "raw" / "gold_release.json", encoding="utf-8"))
out_dir = ROOT / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

records = [{"id": i, "tokens": d["tokens"], "spans": to_spans(d["entities"])}
           for i, d in enumerate(raw)]

total = sum(len(r["spans"]) for r in records)
print("records:", len(records), "| spans:", total, "（预期 3397）")
with open(out_dir / "gold.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("例子:", records[0])
