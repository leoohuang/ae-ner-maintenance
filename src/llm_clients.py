"""每个服务商一个调用函数，统一成 call(prompt) -> (回复文本, meta)。
meta 记录实际发送的设置和服务商返回的模型版本，写进原始记录里。API 密钥只从环境变量读。"""
import os
from pathlib import Path

KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GEMINI_API_KEY"}
# _anthropic and _google are kept for direct provider access; amendment 4 routes these models through OpenRouter
MAX_TOKENS = 16000     # reasoning tokens count towards this limit; only generated tokens are billed


def load_env(path):
    """读项目根目录的 .env（KEY=VALUE 每行一个），不覆盖已有的环境变量。"""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _key(cfg):
    name = cfg.get("key_env") or KEY_ENV[cfg["provider"]]
    if not os.environ.get(name):
        raise RuntimeError(f"{name} is not set (put it in .env)")
    return os.environ[name]


def _openai(cfg):
    from openai import OpenAI, BadRequestError
    client = OpenAI(api_key=_key(cfg))
    state = {"temperature": cfg.get("temperature")}

    def call(prompt):
        kw = {"model": cfg["model"], "input": prompt, "max_output_tokens": MAX_TOKENS,
              "reasoning": {"effort": cfg["reasoning_effort"]}}
        if state["temperature"] is not None:
            kw["temperature"] = state["temperature"]
        try:
            r = client.responses.create(**kw)
        except BadRequestError as e:
            if "temperature" not in str(e) or state["temperature"] is None:
                raise
            state["temperature"] = None          # amendment 1: temperature 0 if accepted, else default
            kw.pop("temperature")
            r = client.responses.create(**kw)
        sent = {k: v for k, v in kw.items() if k not in ("input", "model")}
        return r.output_text, {"model_version": r.model, "sent": sent,
                               "usage": r.usage.model_dump() if r.usage else None}
    return call


def _compat(cfg, client=None):
    """OpenAI 兼容的 chat 接口：DeepSeek 官方，以及经 OpenRouter 调用的模型（amendment 4）。"""
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=_key(cfg), base_url=cfg["base_url"])

    def call(prompt):
        kw = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS}
        if cfg.get("temperature") is not None:
            kw["temperature"] = cfg["temperature"]
        if cfg.get("extra_body"):
            kw["extra_body"] = cfg["extra_body"]
        r = client.chat.completions.create(**kw)
        msg = r.choices[0].message
        sent = {k: v for k, v in kw.items() if k not in ("messages", "model")}
        return msg.content or "", {"model_version": r.model, "sent": sent,
                                   "served_by": (getattr(r, "model_extra", None) or {}).get("provider"),
                                   "reasoning_returned": bool(getattr(msg, "reasoning_content", None)
                                                              or getattr(msg, "reasoning", None)),
                                   "usage": r.usage.model_dump() if r.usage else None}
    return call


def _anthropic(cfg):
    import anthropic
    client = anthropic.Anthropic(api_key=_key(cfg))

    def call(prompt):
        # 当前 Claude 模型不接受采样参数。没有开启服务端 fallback：换模型会混进别的模型的标注，
        # 拒答（stop_reason "refusal"）按解析失败处理并记录
        kw = {"model": cfg["model"], "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]}
        if cfg.get("thinking") == "disabled":
            kw["thinking"] = {"type": "disabled"}
        if cfg.get("effort"):
            kw["output_config"] = {"effort": cfg["effort"]}
        r = client.messages.create(**kw)
        text = "".join(b.text for b in r.content if b.type == "text")
        sent = {k: v for k, v in kw.items() if k not in ("messages", "model")}
        return text, {"model_version": r.model, "sent": sent, "stop_reason": r.stop_reason,
                      "usage": r.usage.model_dump()}
    return call


def _google(cfg):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=_key(cfg))

    def call(prompt):
        conf = types.GenerateContentConfig(
            max_output_tokens=MAX_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_level=cfg["thinking_level"]))
        r = client.models.generate_content(model=cfg["model"], contents=prompt, config=conf)
        return r.text or "", {"model_version": r.model_version,
                              "sent": {"thinking_level": cfg["thinking_level"], "max_output_tokens": MAX_TOKENS},
                              "usage": r.usage_metadata.model_dump() if r.usage_metadata else None}
    return call


def _local(cfg):
    """在 NAISS GPU 上本地运行开源模型，贪心解码，关闭 thinking。"""
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    # text only: the tokenizer is enough (AutoProcessor would also load the video processor,
    # which needs torchvision)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForImageTextToText.from_pretrained(cfg["model"], dtype=torch.bfloat16,
                                                        device_map="auto")

    def call(prompt):
        msgs = [{"role": "user", "content": prompt}]
        inp = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                      return_tensors="pt", enable_thinking=cfg["enable_thinking"]).to(model.device)
        out = model.generate(**inp, max_new_tokens=MAX_TOKENS, do_sample=False)
        new = out[0][inp["input_ids"].shape[1]:]
        text = tok.decode(new, skip_special_tokens=True)
        return text, {"model_version": cfg["model"],
                      "sent": {"do_sample": False, "enable_thinking": cfg["enable_thinking"],
                               "max_new_tokens": MAX_TOKENS},
                      "usage": {"completion_tokens": len(new), "prompt_tokens": inp["input_ids"].shape[1],
                                "thinking_in_output": "<think>" in tok.decode(new)}}
    return call


def make_client(cfg):
    return {"openai": _openai, "compat": _compat, "anthropic": _anthropic,
            "google": _google, "local": _local}[cfg["provider"]](cfg)
