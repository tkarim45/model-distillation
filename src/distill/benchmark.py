"""The experiment: distill Claude's topic-classification behavior into a local student, then ask
three questions with numbers.

1. **How much quality survives distillation?** Teacher accuracy (Claude zero-shot on the gold test
   set) vs student accuracy (trained on the teacher's labels, evaluated on the same gold test set).
2. **What did learning from an imperfect teacher cost?** The same student trained on *gold* labels
   is the ceiling; the gap to the teacher-labeled student is the distillation tax.
3. **When does distilling pay off?** The teacher costs real tokens per document forever; the
   student costs a one-time labeling of its training set and then ~0 per document. Break-even is
   simply the number of training documents — after that many queries, the student is free.

A learning curve (student accuracy vs number of teacher-labeled training docs) shows how much
teacher labeling you actually need.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .data import Example
from .student import Student, accuracy

# Claude Haiku 4.5 list price, USD per 1M tokens (stated so cost figures are auditable, not guessed).
# Update PRICING if the published price changes; the token counts are measured either way.
PRICING = {"input_per_m": 1.00, "output_per_m": 5.00, "model": "Claude Haiku 4.5"}


def _dollars(input_tokens: int, output_tokens: int) -> float:
    return round(input_tokens / 1e6 * PRICING["input_per_m"]
                 + output_tokens / 1e6 * PRICING["output_per_m"], 6)


@dataclass
class Result:
    teacher_accuracy: float
    teacher_label_accuracy_on_train: float      # how good the teacher's train labels were vs gold
    student_distilled_accuracy: float           # student trained on teacher labels
    student_gold_accuracy: float                # student trained on gold labels (ceiling)
    distillation_gap: float                     # gold - distilled (tax for the noisy teacher)
    student_minus_teacher: float                # student can beat the teacher by denoising it
    learning_curve: list = field(default_factory=list)   # [{"n": N, "accuracy": a}]
    teacher_tokens_input: int = 0
    teacher_tokens_output: int = 0
    teacher_cost_usd: float = 0.0
    breakeven_docs: int = 0
    n_train: int = 0
    n_test: int = 0
    pricing: dict = field(default_factory=lambda: dict(PRICING))


def run(teacher, train: list[Example], test: list[Example],
        curve_sizes: list[int] | None = None, seed: int = 0) -> dict:
    test_gold = [e.gold for e in test]
    train_gold = [e.gold for e in train]

    # Teacher labels the train pool (for distillation) and the test set (to score the teacher).
    train_run = teacher.label("train", train)
    test_run = teacher.label("test", test)

    teacher_test_labels = test_run.labels
    teacher_acc = accuracy(teacher_test_labels, test_gold)
    teacher_train_label_acc = accuracy(train_run.labels, train_gold)

    # Distilled student: trained on the teacher's train labels.
    distilled = Student(seed=seed).fit(train, train_run.labels)
    distilled_preds = distilled.predict(test)
    distilled_acc = accuracy(distilled_preds, test_gold)

    # Ceiling: same student trained on gold labels.
    gold_student = Student(seed=seed).fit(train, train_gold)
    gold_acc = accuracy(gold_student.predict(test), test_gold)

    # Learning curve over the teacher-labeled pool.
    curve = []
    for n in (curve_sizes or []):
        n = min(n, len(train))
        s = Student(seed=seed).fit(train[:n], train_run.labels[:n])
        curve.append({"n": n, "accuracy": accuracy(s.predict(test), test_gold)})

    tin = getattr(train_run, "input_tokens", 0) + getattr(test_run, "input_tokens", 0)
    tout = getattr(train_run, "output_tokens", 0) + getattr(test_run, "output_tokens", 0)
    # if labels came from cache, pull token totals from the cache instead
    if tin == 0 and hasattr(teacher, "cached_tokens"):
        a_in, a_out = teacher.cached_tokens("train", train)
        b_in, b_out = teacher.cached_tokens("test", test)
        tin, tout = a_in + b_in, a_out + b_out

    result = Result(
        teacher_accuracy=teacher_acc,
        teacher_label_accuracy_on_train=teacher_train_label_acc,
        student_distilled_accuracy=distilled_acc,
        student_gold_accuracy=gold_acc,
        distillation_gap=round(gold_acc - distilled_acc, 3),
        student_minus_teacher=round(distilled_acc - teacher_acc, 3),
        learning_curve=curve,
        teacher_tokens_input=tin, teacher_tokens_output=tout,
        teacher_cost_usd=_dollars(tin, tout),
        breakeven_docs=len(train),
        n_train=len(train), n_test=len(test),
    )
    return asdict(result)


def format_report(r: dict) -> str:
    p = r["pricing"]
    lines = [
        f"task: 4-way topic classification | train {r['n_train']} | test {r['n_test']}",
        "",
        f"teacher (Claude zero-shot) accuracy on gold test : {r['teacher_accuracy']:.3f}",
        f"student distilled (teacher labels)     accuracy   : {r['student_distilled_accuracy']:.3f}",
        f"student gold-trained (ceiling)         accuracy   : {r['student_gold_accuracy']:.3f}",
        f"teacher label accuracy on train pool              : {r['teacher_label_accuracy_on_train']:.3f}",
        "",
        f"distillation tax (gold - distilled)               : {r['distillation_gap']:+.3f}",
        f"student - teacher (denoising effect)              : {r['student_minus_teacher']:+.3f}",
        "",
        f"teacher tokens: {r['teacher_tokens_input']} in / {r['teacher_tokens_output']} out",
        f"teacher cost  : ${r['teacher_cost_usd']:.4f}  (at {p['model']} "
        f"${p['input_per_m']}/1M in, ${p['output_per_m']}/1M out)",
        f"break-even    : after ~{r['breakeven_docs']} documents the distilled student is cheaper "
        f"than calling the teacher (and ~free thereafter)",
    ]
    if r["learning_curve"]:
        lines += ["", "learning curve (distilled student):"]
        for pt in r["learning_curve"]:
            lines.append(f"  n={pt['n']:>4}  acc={pt['accuracy']:.3f}")
    return "\n".join(lines)
