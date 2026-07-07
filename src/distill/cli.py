"""Command line for the distillation benchmark.

    model-distillation                        # real Claude teacher (Bedrock), cached labels
    model-distillation --n-train 300 --n-test 150
    model-distillation --backend anthropic    # direct Anthropic API
    model-distillation --offline              # fake teacher (no keys) — pipeline demo only
    model-distillation --no-cache             # ignore the committed label cache (re-pays teacher)

A real run labels the train pool and test set with Claude Haiku (one short call per document),
caching every label + token count to data/teacher_labels.json so re-runs are free and reproducible.
The committed cache means anyone can reproduce the exact numbers without a key; deleting it re-runs
the teacher.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .benchmark import format_report, run
from .data import load_split

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data",
                     "teacher_labels.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Distill Claude's topic classification into a local student.")
    ap.add_argument("--n-train", type=int, default=300, help="documents the teacher labels for training")
    ap.add_argument("--n-test", type=int, default=150, help="held-out gold documents to score on")
    ap.add_argument("--max-words", type=int, default=120, help="truncate docs (bounds teacher tokens)")
    ap.add_argument("--backend", default="bedrock", choices=["bedrock", "anthropic"])
    ap.add_argument("--offline", action="store_true", help="fake teacher (no keys) — demonstrates the pipeline")
    ap.add_argument("--no-cache", action="store_true", help="ignore the committed teacher-label cache")
    ap.add_argument("--curve", default="50,100,150,200,300", help="learning-curve train sizes")
    ap.add_argument("--json", metavar="PATH", help="write full results as JSON")
    args = ap.parse_args(argv)

    train, test = load_split(args.n_train, args.n_test, max_words=args.max_words)

    if args.offline:
        from .teacher import FakeTeacher
        teacher = FakeTeacher()
    else:
        from .llm import DEFAULT_MODEL, build_teacher_client
        from .teacher import Teacher
        client = build_teacher_client(args.backend)
        cache_path = None if args.no_cache else CACHE
        teacher = Teacher(client, model_tag=DEFAULT_MODEL, cache_path=cache_path)

    curve = [int(x) for x in args.curve.split(",") if x.strip()]
    result = run(teacher, train, test, curve_sizes=curve)

    print(format_report(result))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
