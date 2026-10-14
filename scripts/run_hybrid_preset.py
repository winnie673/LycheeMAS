"""仅运行 Hybrid-P0-FIELDS-ONLY 条件并保存 summary。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> None:
    _ensure_pythonpath()

    parser = argparse.ArgumentParser(description="Run Hybrid-P0-FIELDS-ONLY only on a split.")
    parser.add_argument("--split", default="T", help="Split to evaluate (default T).")
    parser.add_argument("--backend", choices=["fake", "api"], default="api")
    parser.add_argument("--model", default=None)
    parser.add_argument("--n", "--num-samples", type=int, default=None, dest="n")
    parser.add_argument(
        "--protocol",
        default="configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml",
        help="Fields-only protocol path.",
    )
    parser.add_argument(
        "--out",
        default="runs/hybrid_comm/hybrid_preset_summary.json",
        help="Where to write the summary JSON.",
    )
    args = parser.parse_args()

    from lychee_mas.hybrid_comm.pipeline.evaluator import ProtocolEvaluator

    evaluator = ProtocolEvaluator()
    summary = evaluator.evaluate(
        protocol_path=args.protocol,
        split=args.split,
        backend_name=args.backend,
        model=args.model,
        n=args.n,
        mode="hybrid",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[Hybrid-P0-FIELDS-ONLY summary]")
    print(f"  n: {summary['n']}")
    print(f"  correct: {summary['correct']}")
    print(f"  accuracy: {summary['accuracy']*100:.1f}%")
    print(f"  avg_communication_tokens: {summary['avg_communication_tokens']:.1f}")
    print(f"  avg_output_tokens: {summary['avg_output_tokens']:.1f}")
    print(f"  avg_model_calls: {summary['avg_model_calls']:.2f}")
    print(f"  run_dir: {summary['run_dir']}")
    print(f"\n[write] summary saved to {out_path}")


if __name__ == "__main__":
    main()

