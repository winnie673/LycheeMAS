"""生成本地调试用的小型合成 manifest（无需下载 GSM8K）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

SAMPLES = [
    ("What is 9 plus 9? Give the final numeric answer.", "18"),
    ("A baker has 16 eggs, uses 3 and 4. How many are left? Give the final numeric answer.", "9"),
    ("Apples cost $2 each. How much do 5 apples cost? Give the final numeric answer.", "10"),
    ("A train travels 60 mph for 3 hours. How far? Give the final numeric answer.", "180"),
    ("What is 12 minus 7? Give the final numeric answer.", "5"),
    ("A box holds 8 books. How many can 4 boxes hold? Give the final numeric answer.", "32"),
    ("What is 7 multiplied by 6? Give the final numeric answer.", "42"),
    ("A rope is 50 m. If 18 m are cut off, how long? Give the final numeric answer.", "32"),
    ("What is 100 divided by 4? Give the final numeric answer.", "25"),
    ("A store has 80 pens. If 35 are sold, how many remain? Give the final numeric answer.", "45"),
]


def _content_hash(text: str, gold: str) -> str:
    return hashlib.sha256((text + gold).encode("utf-8")).hexdigest()[:16]


def build_manifest(seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed)
    items = list(SAMPLES)
    rng.shuffle(items)

    manifest = {
        "dataset": "synthetic",
        "config": None,
        "seed": seed,
        "splits": {},
    }
    for name, size in (
        ("D0", 6),
        ("G1-dev", 2),
        ("G1-val", 2),
    ):
        manifest["splits"][name] = [
            {
                "task_id": f"synthetic:seed{seed}:{name}:{i:05d}",
                "original_index": i,
                "question": q,
                "gold": a,
                "content_hash": _content_hash(q, a),
            }
            for i, (q, a) in enumerate(items[:size])
        ]
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a small synthetic manifest for local debugging."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/splits/synthetic_seed42.json")
    args = parser.parse_args()

    manifest = build_manifest(seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"Wrote synthetic manifest to {out_path}")


if __name__ == "__main__":
    main()
