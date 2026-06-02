"""
EchoCoach Correction Model Evaluation Harness
=============================================

Runs the correction prompt against the 100-utterance test set and reports:
  - Precision (of flagged cases, how many were correct)
  - Recall (of actual errors, how many were caught)
  - F1 score
  - Error type accuracy (for flagged correct cases)
  - False positive rate (valid utterances incorrectly flagged)
  - Per-error-type breakdown

Usage:
  cd backend
  python -m eval.run_eval

  # Run only a subset:
  python -m eval.run_eval --limit 20

  # Run only error cases:
  python -m eval.run_eval --errors-only

  # Save results to file:
  python -m eval.run_eval --output eval/results.json

Requires GROQ_API_KEY to be set in environment or .env file.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Allow running from backend/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))


async def run_single(client, case: dict, correction_system: str) -> dict:
    """Run correction check on a single test case."""
    utterance = case["utterance"]
    start = time.perf_counter()

    try:
        from groq import AsyncGroq
        response = await client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "system", "content": correction_system},
                {"role": "user", "content": utterance},
            ],
            stream=False,
            max_tokens=180,
            temperature=0.05,
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - start) * 1000
        result = json.loads(response.choices[0].message.content)
        model_flagged = bool(result.get("error"))
        model_error_type = result.get("error_type") if model_flagged else None

        return {
            "id": case["id"],
            "utterance": utterance,
            "should_flag": case["should_flag"],
            "expected_type": case.get("error_type"),
            "model_flagged": model_flagged,
            "model_error_type": model_error_type,
            "model_output": result,
            "latency_ms": round(latency_ms, 1),
            "correct_detection": model_flagged == case["should_flag"],
            "correct_type": (
                model_error_type == case.get("error_type")
                if model_flagged and case["should_flag"]
                else None
            ),
        }
    except Exception as e:
        return {
            "id": case["id"],
            "utterance": utterance,
            "should_flag": case["should_flag"],
            "expected_type": case.get("error_type"),
            "model_flagged": None,
            "model_error_type": None,
            "model_output": {"error": f"API error: {str(e)}"},
            "latency_ms": (time.perf_counter() - start) * 1000,
            "correct_detection": False,
            "correct_type": None,
        }


def compute_metrics(results: list[dict]) -> dict:
    """Compute precision, recall, F1, and per-type accuracy."""
    valid = [r for r in results if r["model_flagged"] is not None]

    # Binary detection metrics
    tp = sum(1 for r in valid if r["should_flag"] and r["model_flagged"])
    fp = sum(1 for r in valid if not r["should_flag"] and r["model_flagged"])
    fn = sum(1 for r in valid if r["should_flag"] and not r["model_flagged"])
    tn = sum(1 for r in valid if not r["should_flag"] and not r["model_flagged"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    accuracy = (tp + tn) / len(valid) if valid else 0

    # Error type accuracy (only for true positives)
    true_positives = [r for r in valid if r["should_flag"] and r["model_flagged"]]
    type_correct = sum(1 for r in true_positives if r["correct_type"])
    type_accuracy = type_correct / len(true_positives) if true_positives else 0

    # Per-type breakdown
    error_types = set(r["expected_type"] for r in valid if r["expected_type"])
    per_type = {}
    for et in sorted(error_types):
        cases = [r for r in valid if r["expected_type"] == et]
        caught = [r for r in cases if r["model_flagged"]]
        per_type[et] = {
            "total": len(cases),
            "caught": len(caught),
            "recall": round(len(caught) / len(cases), 3) if cases else 0,
        }

    # Latency stats
    latencies = [r["latency_ms"] for r in valid if r["latency_ms"] is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    return {
        "total_cases": len(valid),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "type_accuracy_on_true_positives": round(type_accuracy, 4),
        "per_error_type": per_type,
        "latency_avg_ms": round(avg_latency, 1),
        "latency_p95_ms": round(p95_latency, 1),
    }


def print_report(metrics: dict, results: list[dict]):
    """Print a human-readable evaluation report."""
    print("\n" + "═" * 60)
    print("  ECHOCOACH CORRECTION MODEL EVALUATION REPORT")
    print("═" * 60)
    print(f"  Model:     gemma2-9b-it (Groq)")
    print(f"  Cases run: {metrics['total_cases']}")
    print()
    print(f"  Detection Metrics:")
    print(f"    Accuracy:           {metrics['accuracy']*100:.1f}%")
    print(f"    Precision:          {metrics['precision']*100:.1f}%  (flagged → actually wrong)")
    print(f"    Recall:             {metrics['recall']*100:.1f}%  (errors → actually caught)")
    print(f"    F1 Score:           {metrics['f1']*100:.1f}%")
    print(f"    False Positive Rate:{metrics['false_positive_rate']*100:.1f}%  (valid utterances wrongly flagged)")
    print()
    print(f"  Type Classification (on true positives):")
    print(f"    Error type accuracy: {metrics['type_accuracy_on_true_positives']*100:.1f}%")
    print()
    print(f"  Latency:")
    print(f"    Average: {metrics['latency_avg_ms']}ms")
    print(f"    P95:     {metrics['latency_p95_ms']}ms  (budget: 500ms)")
    print()
    print(f"  Per Error Type Recall:")
    for et, stats in metrics["per_error_type"].items():
        bar = "█" * int(stats["recall"] * 20)
        print(f"    {et:<14} {stats['caught']}/{stats['total']}  [{bar:<20}] {stats['recall']*100:.0f}%")

    # Show failures
    failures = [r for r in results if not r.get("correct_detection")]
    if failures:
        print()
        print(f"  Failures ({len(failures)} cases):")
        for r in failures[:10]:
            flag_str = "FLAGGED" if r["model_flagged"] else "MISSED"
            expected_str = f"expected={'FLAG' if r['should_flag'] else 'PASS'}"
            print(f"    [{r['id']:3}] {flag_str} ({expected_str}): {r['utterance'][:70]}")
        if len(failures) > 10:
            print(f"    ... and {len(failures)-10} more (see output file for full list)")

    print("═" * 60 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Run EchoCoach correction eval")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N cases")
    parser.add_argument("--errors-only", action="store_true", help="Run only error cases")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent API calls")
    args = parser.parse_args()

    # Load API key
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Add it to backend/.env")
        sys.exit(1)

    # Load test set
    test_set_path = Path(__file__).parent / "correction_test_set.json"
    with open(test_set_path) as f:
        test_data = json.load(f)

    cases = test_data["test_cases"]

    if args.errors_only:
        cases = [c for c in cases if c["should_flag"]]
    if args.limit:
        cases = cases[:args.limit]

    print(f"Running {len(cases)} test cases (concurrency={args.concurrency})...")

    from groq import AsyncGroq
    from prompts.correction import CORRECTION_SYSTEM

    client = AsyncGroq(api_key=api_key)

    # Run with bounded concurrency
    semaphore = asyncio.Semaphore(args.concurrency)
    results = []

    async def run_with_sem(case):
        async with semaphore:
            result = await run_single(client, case, CORRECTION_SYSTEM)
            status = "✓" if result["correct_detection"] else "✗"
            flag_str = "FLAGGED" if result["model_flagged"] else "CLEAN "
            print(f"  [{result['id']:3}] {status} {flag_str}  {result['latency_ms']:.0f}ms  {case['utterance'][:55]}...")
            return result

    tasks = [run_with_sem(c) for c in cases]
    results = await asyncio.gather(*tasks)

    metrics = compute_metrics(results)
    print_report(metrics, results)

    if args.output:
        output = {
            "metrics": metrics,
            "results": results,
            "test_set_version": test_data.get("version"),
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
