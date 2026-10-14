"""生成 MATH-500 协议开发与冻结评测划分。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", default="data/splits/math500_protocol_v2_seed42.json"
    )
    args = parser.parse_args()

    from lychee_mas.eval.benchmarks.math500 import load_math500

    source = load_math500(n=None)
    indexed = list(enumerate(source))
    random.Random(args.seed).shuffle(indexed)
    sizes = {"D0": 20, "G1": 50, "G2": 50, "G3": 50, "V": 100, "T": 200}
    manifest = {
        "task": "math500",
        "dataset": "HuggingFaceH4/MATH-500 via MASPO math_500.jsonl",
        "seed": args.seed,
        "schema_version": "v2",
        "scorer_kind": "math500",
        "splits": {},
    }
    offset = 0
    for split, size in sizes.items():
        chosen = indexed[offset : offset + size]
        offset += size
        manifest["splits"][split] = [
            {
                "task_id": f"math500:seed{args.seed}:{split}:{i:05d}",
                "source_index": source_index,
                "question": item["question"],
                "gold": item["gold"],
                "kind": "aime",  # preserve v2 split hash; runner uses top-level scorer_kind
                "content_hash": hashlib.sha256(
                    (item["question"] + item["gold"]).encode("utf-8")
                ).hexdigest()[:16],
            }
            for i, (source_index, item) in enumerate(chosen)
        ]
    manifest["reserve_source_indices"] = [i for i, _ in indexed[offset:]]
    material = json.dumps(manifest["splits"], sort_keys=True, ensure_ascii=False)
    manifest["manifest_hash"] = "sha256:" + hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()[:16]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in manifest["splits"].items()}))
    print(manifest["manifest_hash"])


if __name__ == "__main__":
    main()
