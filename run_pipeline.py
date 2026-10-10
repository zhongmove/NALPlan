"""Public-release pipeline skeleton for NALPlan.

The four contribution stages expose their interfaces in this release, but their
implementation bodies are withheld until paper acceptance. This driver documents
the execution order and provides a clear integration point for the future release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nalplan.config import CANDIDATES_PER_PATH, NUM_PATHS, TEMPERATURE
from nalplan.reranker import select_top1


def load_examples(path: Path, limit: int | None = None) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return records[:limit] if limit else records


def run_one(record: dict) -> dict:
    """Show the NALPlan stage order for one natural-language query."""
    stages = [
        "knowledge preparation",
        f"multi-path generation ({NUM_PATHS} paths x {CANDIDATES_PER_PATH} candidates)",
        "constraint-guided self-consistency pruning",
        "index enumeration and execution-free cost estimation",
        "semantic workload-cache lookup/admission",
        "top-1 reranking and optional surface postprocessing",
    ]
    return {
        "id": record["id"],
        "dataset": record["dataset"],
        "query_type": record["query_type"],
        "status": "interface-only public release",
        "temperature": TEMPERATURE,
        "stages": stages,
        "message": "Core implementation is withheld until paper acceptance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("benchmark/nalplan_benchmark_examples.json"))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = [run_one(row) for row in load_examples(args.input, args.limit)]
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
