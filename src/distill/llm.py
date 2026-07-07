"""A tiny Claude client (Bedrock default) for the teacher, plus token accounting.

Self-contained — this repo depends on none of the others. Credentials come from the standard AWS
chain, populated from `.env` then `~/.env` at import. Every call returns the label text and the
input/output token counts, because the teacher's token spend is the numerator of the whole
cost-vs-quality argument.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _load_env() -> None:
    for path in (".env", os.path.expanduser("~/.env")):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
        except FileNotFoundError:
            continue


_load_env()

DEFAULT_MODEL = os.environ.get("DISTILL_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"


@dataclass
class Reply:
    text: str
    input_tokens: int
    output_tokens: int


class BedrockClaude:
    def __init__(self, model: str = DEFAULT_MODEL, region: str = DEFAULT_REGION,
                 max_tokens: int = 8, temperature: float = 0.0):
        from anthropic import AnthropicBedrock

        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self._client = AnthropicBedrock(aws_region=region)

    def complete(self, system: str, prompt: str) -> Reply:
        msg = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            system=system, messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return Reply(text, msg.usage.input_tokens, msg.usage.output_tokens)


class AnthropicClaude:
    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 8,
                 temperature: float = 0.0):
        from anthropic import Anthropic

        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self._client = Anthropic()

    def complete(self, system: str, prompt: str) -> Reply:
        msg = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            system=system, messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return Reply(text, msg.usage.input_tokens, msg.usage.output_tokens)


def build_teacher_client(backend: str = "bedrock", **kw):
    if backend == "bedrock":
        return BedrockClaude(**kw)
    if backend == "anthropic":
        return AnthropicClaude(**kw)
    raise ValueError(f"unknown backend {backend!r}")
