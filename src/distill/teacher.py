"""The teacher: Claude labeling documents zero-shot, with an on-disk cache.

The teacher is expensive (a network call and real tokens per document), so its labels are cached
to JSON keyed by (split, doc index, model). A committed cache makes the whole benchmark
reproducible for free — but it was produced by real Claude calls, and deleting the cache re-runs
them. The cache also records the token counts, so the cost figures come from measured usage, not
an estimate of what the calls "should" have cost.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .data import LABELS, Example

SYSTEM = (
    "You are a topic classifier. Read the document and reply with exactly one of these labels and "
    f"nothing else: {', '.join(LABELS)}. "
    "'med' = medicine/health, 'space' = spaceflight/astronomy, 'autos' = cars/driving, "
    "'graphics' = computer graphics/image software."
)


def _prompt(text: str) -> str:
    return f"Document:\n{text}\n\nLabel:"


def _parse(reply_text: str) -> str:
    low = reply_text.strip().lower()
    for lab in LABELS:                     # first label mentioned wins
        if lab in low:
            return lab
    return "graphics"                      # deterministic fallback if the model went off-script


@dataclass
class TeacherRun:
    labels: list[str]
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0


class Teacher:
    def __init__(self, client, model_tag: str, cache_path: str | None = None):
        self.client = client
        self.model_tag = model_tag
        self.cache_path = cache_path
        self._cache: dict[str, dict] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self._cache = json.load(f)

    def _key(self, split: str, idx: int) -> str:
        return f"{self.model_tag}:{split}:{idx}"

    def label(self, split: str, examples: list[Example]) -> TeacherRun:
        run = TeacherRun(labels=[])
        for ex in examples:
            key = self._key(split, ex.idx)
            hit = self._cache.get(key)
            if hit is not None:
                run.labels.append(hit["label"])
                run.cache_hits += 1
                continue
            reply = self.client.complete(SYSTEM, _prompt(ex.text))
            label = _parse(reply.text)
            self._cache[key] = {"label": label, "in": reply.input_tokens, "out": reply.output_tokens}
            run.labels.append(label)
            run.input_tokens += reply.input_tokens
            run.output_tokens += reply.output_tokens
            run.calls += 1
        if self.cache_path:
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=0)
        return run

    def cached_tokens(self, split: str, examples: list[Example]) -> tuple[int, int]:
        """Total (input, output) tokens the teacher spent to label these examples, from the cache."""
        tin = tout = 0
        for ex in examples:
            hit = self._cache.get(self._key(split, ex.idx))
            if hit:
                tin += hit.get("in", 0)
                tout += hit.get("out", 0)
        return tin, tout


class FakeTeacher:
    """Offline teacher for tests: returns gold with a fixed, deterministic error rate so the
    distillation gap is exercised without any network."""

    def __init__(self, error_every: int = 7):
        self.error_every = error_every

    def label(self, split: str, examples: list[Example]) -> TeacherRun:
        labels = []
        for ex in examples:
            if self.error_every and ex.idx % self.error_every == 0:
                # deterministic wrong label: shift to the next category
                wrong = LABELS[(LABELS.index(ex.gold) + 1) % len(LABELS)]
                labels.append(wrong)
            else:
                labels.append(ex.gold)
        return TeacherRun(labels=labels, input_tokens=len(examples) * 100,
                          output_tokens=len(examples), calls=len(examples))
