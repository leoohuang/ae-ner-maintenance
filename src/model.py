import random, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
 
TYPES = ["PhysicalObject", "State", "Process", "Activity", "Property"]
LABELS = ["NONE"] + TYPES
L2I = {l: i for i, l in enumerate(LABELS)}
MAX_W = 16                      # 最长记录 12 个词，留点余量
 
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
 
def get_tokenizer(name):
    # RoBERTa 处理「已分词输入」必须加 add_prefix_space=True
    # ModernBERT 若报同样的错，也照此处理
    kw = {"add_prefix_space": True} if "roberta" in name.lower() else {}
    return AutoTokenizer.from_pretrained(name, **kw)
 
def all_spans(n):
    return [(s, e) for s in range(n) for e in range(s + 1, n + 1)]   # end 不含
 
def encode_batch(tok, records, max_len):
    enc = tok([r["tokens"] for r in records], is_split_into_words=True,
              truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    widx = torch.zeros(len(records), MAX_W, dtype=torch.long)    # 每个词的第一个子词位置
    wmask = torch.zeros(len(records), MAX_W, dtype=torch.bool)
    for b, r in enumerate(records):
        prev, seen = None, 0
        for pos, wid in enumerate(enc.word_ids(b)):
            if wid is not None and wid != prev:
                widx[b, wid] = pos; seen += 1
            prev = wid
        assert seen == len(r["tokens"]), "有词被截断，调大 max_len"
        wmask[b, :len(r["tokens"])] = True
    return enc, widx, wmask
 
def span_labels(r, spans):
    gold = {(s, e): t for s, e, t in r.get("spans", [])}
    return [L2I[gold.get(sp, "NONE")] for sp in spans]

class SpanModel(nn.Module):
    def __init__(self, name, hidden=256, width_dim=16, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        d = self.enc.config.hidden_size
        self.width = nn.Embedding(MAX_W + 1, width_dim)
        self.head = nn.Sequential(nn.Linear(3 * d + width_dim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, len(LABELS)))
 
    def forward(self, enc, widx, spans):
        dev = next(self.parameters()).device
        h = self.enc(input_ids=enc["input_ids"].to(dev),
                     attention_mask=enc["attention_mask"].to(dev)).last_hidden_state
        w = torch.gather(h, 1, widx.to(dev).unsqueeze(-1).expand(-1, -1, h.size(-1)))
        feats = []
        for b, sp in enumerate(spans):
            for s, e in sp:
                feats.append(torch.cat([w[b, s], w[b, e - 1], w[b, s:e].mean(0),
                                        self.width(torch.tensor(e - s, device=dev))]))
        return self.head(torch.stack(feats))        # (这一批的 span 总数, 6)

def finetune(name, records, cfg, seed, device):
    set_seed(seed)
    tok = get_tokenizer(name)
    model = SpanModel(name, cfg["head_hidden"], cfg["width_dim"], cfg["dropout"]).to(device)
    weights = torch.ones(len(LABELS)); weights[0] = cfg["none_weight"]
    lossf = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": cfg["encoder_lr"]},
        {"params": list(model.width.parameters()) + list(model.head.parameters()),
         "lr": cfg["head_lr"]},
    ], weight_decay=0.01)
    n, bs = len(records), cfg["batch_size"]
    steps_per_epoch = -(-n // bs)
    # 按「最少步数」对齐不同预算下的训练量，避免小数据欠训练
    epochs = max(cfg["min_epochs"], -(-cfg["min_steps"] // steps_per_epoch))
    total = steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    model.train()
    for _ in range(epochs):
        order = list(range(n)); random.shuffle(order)        # 由 set_seed 控制
        for i in range(0, n, bs):
            chunk = [records[j] for j in order[i:i + bs]]
            enc, widx, _ = encode_batch(tok, chunk, cfg["max_len"])
            spans = [all_spans(len(r["tokens"])) for r in chunk]
            y = torch.tensor([l for r, sp in zip(chunk, spans)
                              for l in span_labels(r, sp)], device=device)
            loss = lossf(model(enc, widx, spans), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
    return model, tok

def finetune(name, records, cfg, seed, device):
    set_seed(seed)
    tok = get_tokenizer(name)
    model = SpanModel(name, cfg["head_hidden"], cfg["width_dim"], cfg["dropout"]).to(device)
    weights = torch.ones(len(LABELS)); weights[0] = cfg["none_weight"]
    lossf = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": cfg["encoder_lr"]},
        {"params": list(model.width.parameters()) + list(model.head.parameters()),
         "lr": cfg["head_lr"]},
    ], weight_decay=0.01)
    n, bs = len(records), cfg["batch_size"]
    steps_per_epoch = -(-n // bs)
    # 按「最少步数」对齐不同预算下的训练量，避免小数据欠训练
    epochs = max(cfg["min_epochs"], -(-cfg["min_steps"] // steps_per_epoch))
    total = steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    model.train()
    for _ in range(epochs):
        order = list(range(n)); random.shuffle(order)        # 由 set_seed 控制
        for i in range(0, n, bs):
            chunk = [records[j] for j in order[i:i + bs]]
            enc, widx, _ = encode_batch(tok, chunk, cfg["max_len"])
            spans = [all_spans(len(r["tokens"])) for r in chunk]
            y = torch.tensor([l for r, sp in zip(chunk, spans)
                              for l in span_labels(r, sp)], device=device)
            loss = lossf(model(enc, widx, spans), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
    return model, tok

@torch.no_grad()
def predict(model, tok, records, max_len, device, bs=64):
    """每条记录返回 (spans, probs)：spans 是全部候选 (start, end)，probs 形状 (n_spans, 6)"""
    model.eval(); out = []
    for i in range(0, len(records), bs):
        chunk = records[i:i + bs]
        enc, widx, _ = encode_batch(tok, chunk, max_len)
        spans = [all_spans(len(r["tokens"])) for r in chunk]
        probs = torch.softmax(model(enc, widx, spans), -1).cpu().numpy()
        pos = 0
        for sp in spans:
            out.append((sp, probs[pos:pos + len(sp)])); pos += len(sp)
    return out
 
def decode(spans, probs):
    """概率最大的类不是 NONE 就输出；不做重叠抑制"""
    return {(s, e, LABELS[k]) for (s, e), k in zip(spans, probs.argmax(1)) if k > 0}
