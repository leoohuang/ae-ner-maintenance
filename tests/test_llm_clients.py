from types import SimpleNamespace
from pathlib import Path
import yaml
from src.llm_clients import _compat

ROOT = Path(__file__).resolve().parent.parent


class FakeChat:
    def __init__(self):
        self.sent = None

    def create(self, **kw):
        self.sent = kw
        msg = SimpleNamespace(content='{"spans": []}', reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], model="m-2026",
                               usage=None, model_extra={"provider": "Anthropic"})


def fake_client():
    chat = FakeChat()
    return SimpleNamespace(chat=SimpleNamespace(completions=chat)), chat


def test_compat_sends_no_temperature_when_not_configured():
    client, chat = fake_client()
    call = _compat({"model": "anthropic/x", "base_url": "u", "extra_body": {"reasoning": {"effort": "low"}}},
                   client=client)
    text, meta = call("p")
    assert "temperature" not in chat.sent
    assert chat.sent["extra_body"] == {"reasoning": {"effort": "low"}}
    assert meta["served_by"] == "Anthropic"


def test_compat_sends_configured_temperature():
    client, chat = fake_client()
    _compat({"model": "x", "base_url": "u", "temperature": 0}, client=client)("p")
    assert chat.sent["temperature"] == 0


def test_openrouter_models_are_pinned_to_one_provider_without_fallbacks():
    cfgs = yaml.safe_load(open(ROOT / "configs/llm_models.yaml"))
    routed = {k: c for k, c in cfgs.items() if "openrouter.ai" in c.get("base_url", "")}
    assert set(routed) == set(cfgs) - {"qwen3.8-27b"}       # every API model; amendment 5
    for c in routed.values():
        p = c["extra_body"]["provider"]
        assert len(p["order"]) == 1 and p["allow_fallbacks"] is False and p["require_parameters"] is True
        assert c["key_env"] == "OPENROUTER_API_KEY"
