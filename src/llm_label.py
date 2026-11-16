import json
import re
from collections import Counter

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
