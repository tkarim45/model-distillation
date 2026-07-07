"""The student: a TF-IDF + logistic-regression classifier trained on the teacher's labels.

Deliberately small and cheap — the whole point of distillation is to move the teacher's behavior
into something with near-zero marginal cost. Training and inference are fully local, deterministic
given a seed, and fast. The student never sees gold labels during distillation; it learns only
from what the teacher said (except in the ``gold`` control, which trains on gold to measure the
ceiling the teacher's imperfect labels cost us).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Example


@dataclass
class Student:
    seed: int = 0

    def __post_init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self._vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                                    stop_words="english")
        self._clf = LogisticRegression(max_iter=1000, random_state=self.seed)

    def fit(self, examples: list[Example], labels: list[str]) -> "Student":
        X = self._vec.fit_transform([e.text for e in examples])
        self._clf.fit(X, labels)
        return self

    def predict(self, examples: list[Example]) -> list[str]:
        X = self._vec.transform([e.text for e in examples])
        return list(self._clf.predict(X))


def accuracy(preds: list[str], golds: list[str]) -> float:
    return round(float(np.mean([p == g for p, g in zip(preds, golds)])), 3)
