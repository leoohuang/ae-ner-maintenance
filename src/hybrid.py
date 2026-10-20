import numpy as np
from src.pool import fill_budget


def hybrid_select(ranked, emb, labelled, tok_len, current_tokens, target_tokens, factor=3):
    need = target_tokens - current_tokens
    cand, tot = [], 0
    for i in ranked:
        cand.append(i)
        tot += tok_len[i]
        if tot >= factor * need:
            break
    cand_set = set(cand)
    mind = 1.0 - (emb[cand] @ emb[list(labelled)].T).max(axis=1)
    mind = dict(zip(cand, mind))
    remaining = list(cand)
    chosen, total = [], current_tokens
    while remaining and total < target_tokens:
        best = min(remaining, key=lambda i: (-mind[i], i))
        remaining.remove(best)
        if total + tok_len[best] > target_tokens:
            continue
        chosen.append(best)
        total += tok_len[best]
        for c in remaining:
            mind[c] = min(mind[c], 1.0 - float(emb[c] @ emb[best]))
    if total < target_tokens:
        rest = [i for i in ranked if i not in cand_set]
        extra, total = fill_budget(rest, tok_len, total, target_tokens)
        chosen.extend(extra)
    return chosen, total