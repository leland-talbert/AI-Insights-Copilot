"""Run the evaluation suite from the command line.

    python evals/run_eval.py                 # GPT-4.1-mini, default settings
    python evals/run_eval.py --model gpt-4.1 --k 8
    python evals/run_eval.py --model ollama  # local model (judge stays on OpenAI)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.embeddings import OpenAIEmbedder  # noqa: E402
from copilot.evaluation import run_evaluation  # noqa: E402
from copilot.tracing import init_tracing  # noqa: E402


def pct(value) -> str:
    return "  –  " if value is None else f"{value:5.0%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.MODELS))
    parser.add_argument("--judge", default=config.JUDGE_MODEL)
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--min-similarity", type=float, default=config.MIN_SIMILARITY)
    args = parser.parse_args()

    traced = init_tracing()
    report = run_evaluation(
        model_key=args.model, judge_model=args.judge, embedder=OpenAIEmbedder(), k=args.k,
        min_similarity=args.min_similarity,
        progress=lambda i, n, q: print(f"[{i:>2}/{n}] {q}", flush=True),
    )

    print("\n" + "-" * 96)
    print(f"{'ID':<5}{'Answerable':<12}{'Result':<24}{'Correct':<9}{'Recall':<9}{'Faithful':<10}{'Latency':>8}")
    print("-" * 96)
    for c in report["cases"]:
        print(f"{c['id']:<5}{'yes' if c['answerable'] else 'no':<12}{c['status']:<24}"
              f"{'✓' if c['abstention_correct'] else '✗':<9}{pct(c['retrieval_recall']):<9}"
              f"{pct(c['faithfulness']):<10}{c['latency_s']:>7.2f}s")
    s = report["summary"]
    print("-" * 96)
    print(f"Retrieval recall          {pct(s['retrieval_recall'])}")
    print(f"Faithfulness              {pct(s['faithfulness'])}")
    print(f"Answered when answerable  {s['answered_when_answerable']}")
    print(f"Abstained when not        {s['abstained_when_unanswerable']}")
    print(f"Claims removed by check   {s['claims_rejected_by_citation_check']}")
    print(f"Total cost (est.)         ${s['total_cost_usd']:.4f}")
    print(f"\nSaved to evals/results/latest.json" + (" · traces in Phoenix" if traced else ""))


if __name__ == "__main__":
    main()
