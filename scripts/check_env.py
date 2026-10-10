import torch
print("CUDA:", torch.cuda.is_available(), "| MPS:", torch.backends.mps.is_available())
from transformers import AutoTokenizer
for name in ["bert-base-uncased", "roberta-base", "answerdotai/ModernBERT-base"]:
    tok = AutoTokenizer.from_pretrained(name)
    print(name, "OK, vocab =", len(tok))
