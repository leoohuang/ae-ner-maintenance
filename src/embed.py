import numpy as np
from transformers import AutoTokenizer, AutoModel


def static_embeddings(records, model_name="bert-base-uncased"):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    W = model.embeddings.word_embeddings.weight.detach().cpu().numpy()
    out = np.zeros((len(records), W.shape[1]), dtype=np.float64)
    for i, r in enumerate(records):
        ids = tok(r["tokens"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        v = W[ids].mean(axis=0)
        n = np.linalg.norm(v)
        out[i] = v / n if n > 0 else v
    return out