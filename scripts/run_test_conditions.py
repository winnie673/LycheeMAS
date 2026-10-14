"""在指定数据集上运行 NL 与当前 Hybrid-P0-FIELDS-ONLY 并对比。

用法示例：
    DEEPSEEK_API_KEY=... python scripts/run_test_conditions.py --backend api --split T
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _ensure_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _format_row(label: str, s: dict[str, Any]) -> str:
    return (
        f"{label:<18} "
        f"{s['n']:>4}  "
        f"{s['correct']:>4}  "
        f"{s['accuracy']*100:>6.1f}%  "
        f"{s['avg_communication_tokens']:>10.1f}  "
        f"{s['avg_output_tokens']:>10.1f}  "
        f"{s['avg_model_calls']:>8.2f}  "
        f"{s['run_dir']}"
    )


def main() -> None:
    _ensure_pythonpath()

    parser = argparse.ArgumentParser(
        description="Run NL and Hybrid-P0-FIELDS-ONLY on a split and compare."
    )
    parser.add_argument("--split", default="T", help="Split to evaluate (default T).")
    parser.add_argument("--backend", choices=["fake", "api"], default="api")
    parser.add_argument("--model", default=None)
    parser.add_argument("--n", "--num-samples", type=int, default=None, dest="n")
    parser.add_argument(
        "--protocol",
        default="configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml",
        help="Current fields-only protocol path.",
    )
    parser.add_argument(
        "--out", default="runs/hybrid_comm/preset_comparison.json", help="Comparison output path."
    )
    args = parser.parse_args()

    from lychee_mas.hybrid_comm.pipeline.evaluator import ProtocolEvaluator

    evaluator = ProtocolEvaluator()

    conditions = [
        ("NL", args.protocol, "nl"),
        ("Hybrid-P0-FIELDS-ONLY", args.protocol, "hybrid"),
    ]

    summaries: dict[str, dict[str, Any]] = {}
    for label, protocol_path, mode in conditions:
        print(f"\n[test] Running {label} on {args.split} ...")
        summary = evaluator.evaluate(
            protocol_path=protocol_path,
            split=args.split,
            backend_name=args.backend,
            model=args.model,
            n=args.n,
            mode=mode,
        )
        summaries[label] = summary
        print(_format_row(label, summary))

    # 计算节省率
    nl_comm_out = summaries["NL"]["avg_communication_tokens"] + summaries["NL"]["avg_output_tokens"]
    hybrid_comm_out = (
        summaries["Hybrid-P0-FIELDS-ONLY"]["avg_communication_tokens"]
        + summaries["Hybrid-P0-FIELDS-ONLY"]["avg_output_tokens"]
    )
    savings = 1 - hybrid_comm_out / nl_comm_out if nl_comm_out else 0.0

    print("\n" + "=" * 110)
    print("Comparison (NL / Hybrid-P0-FIELDS-ONLY)")
    print("=" * 110)
    print(
        f"{'condition':<18} "
        f"{'n':>4}  "
        f"{'corr':>4}  "
        f"{'acc':>7}  "
        f"{'avg_comm':>10}  "
        f"{'avg_out':>10}  "
        f"{'avg_calls':>8}  "
        f"{'comm+out_save':>14}"
    )
    print("-" * 110)
    for label in ("NL", "Hybrid-P0-FIELDS-ONLY"):
        s = summaries[label]
        save = "-"
        if label == "Hybrid-P0-FIELDS-ONLY":
            save = f"{savings*100:+.1f}%"
        print(
            f"{label:<18} "
            f"{s['n']:>4}  "
            f"{s['correct']:>4}  "
            f"{s['accuracy']*100:>6.1f}%  "
            f"{s['avg_communication_tokens']:>10.1f}  "
            f"{s['avg_output_tokens']:>10.1f}  "
            f"{s['avg_model_calls']:>8.2f}  "
            f"{save:>14}"
        )
    print("=" * 110)

    comparison = {
        "split": args.split,
        "backend": args.backend,
        "summaries": summaries,
        "savings_vs_nl": {
            "hybrid_p0_fields_only_comm_out": round(savings, 4),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[write] comparison saved to {out_path}")


if __name__ == "__main__":
    main()

