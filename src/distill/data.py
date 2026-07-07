"""The task and its data: topic classification over a 4-category slice of 20 Newsgroups.

Real text (not synthetic), with real gold labels — which is what makes the distillation honest:
we can score the teacher *and* the student against ground truth, and separately ask how much the
student loses by learning from the teacher's labels instead of the gold ones.

Categories are chosen to be clearly separable so the ceiling is high and the interesting variable
is the distillation gap, not task difficulty. Documents are stripped of headers/quotes/footers
(which leak the label) and truncated, both to make it a real content-classification task and to
bound the teacher's per-document token cost.
"""
from __future__ import annotations

from dataclasses import dataclass

CATEGORIES = ("sci.med", "sci.space", "rec.autos", "comp.graphics")
LABELS = ("med", "space", "autos", "graphics")           # short names the teacher must emit
_CAT_TO_LABEL = dict(zip(CATEGORIES, LABELS))


@dataclass(frozen=True)
class Example:
    idx: int
    text: str
    gold: str            # short label from LABELS


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def load_split(n_train: int, n_test: int, max_words: int = 120, seed: int = 0
               ) -> tuple[list[Example], list[Example]]:
    """Return (train_pool, test_set) of cleaned, truncated, gold-labeled examples.

    The train pool is what the teacher will label (for distillation); the test set has gold labels
    and is what both teacher and student are scored on. Downloads 20 Newsgroups once via sklearn.
    """
    from sklearn.datasets import fetch_20newsgroups

    def _build(subset: str, n: int) -> list[Example]:
        bunch = fetch_20newsgroups(
            subset=subset, categories=list(CATEGORIES),
            remove=("headers", "footers", "quotes"), shuffle=True, random_state=seed,
        )
        out: list[Example] = []
        for i, (raw, target) in enumerate(zip(bunch.data, bunch.target)):
            text = _truncate(raw.strip(), max_words)
            if len(text.split()) < 15:                    # skip near-empty docs after cleaning
                continue
            gold = _CAT_TO_LABEL[bunch.target_names[target]]
            out.append(Example(idx=len(out), text=text, gold=gold))
            if len(out) >= n:
                break
        return out

    return _build("train", n_train), _build("test", n_test)
