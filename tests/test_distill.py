"""Offline tests — FakeTeacher + local student, so no keys, no network.

They verify the distillation mechanics: the student learns the teacher's decision boundary, the
gold-trained ceiling is well-defined, the distillation gap is computed correctly, and the cost
accounting adds up. Real teacher/student accuracy numbers come from a Bedrock run, not these tests.
"""
from __future__ import annotations

from distill.benchmark import PRICING, _dollars, run
from distill.data import LABELS, Example
from distill.student import Student, accuracy
from distill.teacher import FakeTeacher


def _toy_examples(n: int = 60) -> list[Example]:
    """A trivially separable toy task: the label word itself appears in the text, so a TF-IDF
    student can learn it. Deterministic, no dataset download."""
    out = []
    for i in range(n):
        lab = LABELS[i % len(LABELS)]
        out.append(Example(idx=i, text=f"this document is about {lab} {lab} topic number {i}", gold=lab))
    return out


def test_accuracy_helper():
    assert accuracy(["a", "b", "c"], ["a", "x", "c"]) == 0.667


def test_student_learns_a_separable_task():
    train, test = _toy_examples(80), _toy_examples(40)
    student = Student(seed=0).fit(train, [e.gold for e in train])
    assert accuracy(student.predict(test), [e.gold for e in test]) >= 0.9


def test_fake_teacher_has_the_designed_error_rate():
    ex = _toy_examples(70)
    run_ = FakeTeacher(error_every=7).label("train", ex)
    wrong = sum(1 for lbl, e in zip(run_.labels, ex) if lbl != e.gold)
    assert wrong == len(ex) // 7 + (1 if len(ex) % 7 else 0)  # every 7th index is flipped


def test_pipeline_reports_gap_and_ceiling():
    train, test = _toy_examples(120), _toy_examples(40)
    r = run(FakeTeacher(error_every=6), train, test, curve_sizes=[30, 60, 120])
    # ceiling (gold-trained) should be >= distilled, so the gap is non-negative
    assert r["student_gold_accuracy"] >= r["student_distilled_accuracy"] - 1e-9
    assert r["distillation_gap"] == round(r["student_gold_accuracy"] - r["student_distilled_accuracy"], 3)
    assert r["breakeven_docs"] == len(train)
    assert len(r["learning_curve"]) == 3
    assert all(0.0 <= p["accuracy"] <= 1.0 for p in r["learning_curve"])


def test_cost_accounting():
    # 1,000,000 input tokens at $1/M + 1,000,000 output at $5/M = $6.00
    assert _dollars(1_000_000, 1_000_000) == 6.0
    assert PRICING["input_per_m"] == 1.00 and PRICING["output_per_m"] == 5.00
