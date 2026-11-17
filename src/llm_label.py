import hashlib
import json
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone

from src.evaluate import TYPES

FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def parse_reply(text, n_tokens):
    """解析 LLM 的回复 {"spans": [[start, end, type], ...]}（词下标，end 不含）。
    返回 (spans, drops)：spans 按位置排好；不合格的词段丢掉并在 drops 里按原因计数
    （预注册规则：越界、start >= end、类型不对、完全重复、格式不对）。
    整段解析不了时 spans 为 None，由调用方决定重试。嵌套词段保留，与 gold 一致。"""
    drops = Counter()
    try:
        obj = json.loads(FENCE.sub("", text))
    except (json.JSONDecodeError, TypeError):
        return None, drops
    if not isinstance(obj, dict) or not isinstance(obj.get("spans"), list):
        return None, drops

    seen = set()
    for sp in obj["spans"]:
        if not (isinstance(sp, list) and len(sp) == 3 and type(sp[0]) is int
                and type(sp[1]) is int and isinstance(sp[2], str)):
            drops["malformed"] += 1
        elif sp[0] >= sp[1]:
            drops["empty"] += 1
        elif sp[0] < 0 or sp[1] > n_tokens:
            drops["out_of_range"] += 1
        elif sp[2] not in TYPES:
            drops["unknown_type"] += 1
        elif tuple(sp) in seen:
            drops["duplicate"] += 1
        else:
            seen.add(tuple(sp))
    return [list(s) for s in sorted(seen)], drops


def parse_reply_lenient(text, n_tokens):
    """Amendment 7：严格解析失败时，从回复里找出最后一个带 "spans" 的 JSON 对象
    （模型在 JSON 前写了解释，或者后面多了 </s> 之类的结束符）。词段检查规则与 parse_reply 相同。"""
    spans, drops = parse_reply(text, n_tokens)
    if spans is not None or not isinstance(text, str):
        return spans, drops
    dec, found = json.JSONDecoder(), None
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("spans"), list):
                found = obj
    if found is None:
        return None, Counter()
    return parse_reply(json.dumps(found), n_tokens)


def recover_failed(raw, n_tokens):
    """对严格解析失败的记录，按顺序找第一个能宽松解析的回复；都不行返回 None。不重新调用模型。"""
    for k, a in enumerate(raw["attempts"]):
        spans, drops = parse_reply_lenient(a["text"], n_tokens)
        if spans is not None:
            return {"spans": spans, "drops": dict(drops), "attempt": k}
    return None


def assert_not_test(ids, split):
    """预注册：test 集永远不发给任何 LLM。碰到 test 的记录直接报错。"""
    bad = set(ids) & set(split["test"])
    if bad:
        raise ValueError(f"{len(bad)} test records must never be sent to an LLM")


def val_subsets(split, n_examples, n_tune, seed):
    """从 val 里固定抽两组互不重叠的记录：提示词里的示例、调提示词用的记录。"""
    ids = sorted(split["val"])
    random.Random(seed).shuffle(ids)
    return ids[:n_examples], ids[n_examples:n_examples + n_tune]


def format_record(rec):
    """把每个词和它的下标一起给模型看，模型按下标回答。"""
    return " ".join(f"{i}:{t}" for i, t in enumerate(rec["tokens"]))


def render_prompt(template, rec, examples):
    """template 里有 {examples} 和 {record} 两个位置；示例带标准答案，待标记录不带。"""
    shown = "\n\n".join(format_record(e) + "\n" + json.dumps({"spans": e["spans"]}) for e in examples)
    return template.replace("{examples}", shown).replace("{record}", format_record(rec))


def label_one(call, prompt, n_tokens, max_parse_retries=2, max_net_retries=10, sleep=time.sleep):
    """调用一次模型并解析。预注册：解析不了最多重试 2 次，之后记为失败、这条不标实体。
    网络错误（call 抛异常）另算，等一会儿再试，连续失败 max_net_retries 次就把异常抛出去，
    整个运行停下，之后可以接着跑。"""
    attempts = []
    for _ in range(1 + max_parse_retries):
        for k in range(max_net_retries):
            try:
                text, meta = call(prompt)
                break
            except Exception:
                if k == max_net_retries - 1:
                    raise
                sleep(min(2 ** k, 60))          # waits out rate limits (about 6 minutes in all)
        spans, drops = parse_reply(text, n_tokens)
        attempts.append({"text": text, "meta": meta})
        if spans is not None:
            return {"spans": spans, "drops": dict(drops), "failed": False, "attempts": attempts}
    return {"spans": [], "drops": {}, "failed": True, "attempts": attempts}


def done_ids(path):
    """已经标过的记录（中断后接着跑时跳过）。"""
    try:
        return {json.loads(l)["id"] for l in open(path)}
    except FileNotFoundError:
        return set()


def run_labelling(ids, records, call, prompt_for, label_path, raw_path, info, sleep=time.sleep):
    """逐条标注 ids，结果追加写入：label_path 与 gold.jsonl 同格式（id、tokens、spans），
    raw_path 存每次调用的原始回复、丢弃计数和 info（模型、设置等）。"""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    skip = done_ids(label_path)
    for i in ids:
        if i in skip:
            continue
        rec = records[i]
        prompt = prompt_for(rec)
        res = label_one(call, prompt, len(rec["tokens"]), sleep=sleep)
        raw = {"id": i, **info, "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest(),
               "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "failed": res["failed"], "drops": res["drops"], "attempts": res["attempts"]}
        with open(raw_path, "a") as f:
            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
        with open(label_path, "a") as f:
            f.write(json.dumps({"id": i, "tokens": rec["tokens"], "spans": res["spans"]}) + "\n")
