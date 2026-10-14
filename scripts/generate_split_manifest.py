"""生成 GSM8K 协议实验的冻结数据划分 manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_manifest(seed: int = 42) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from lychee_mas.eval.benchmarks.gsm8k import load_gsm8k

    train = load_gsm8k(n=None)  # test split in gsm8k.py is actually train parquet
    # gsm8k.load_gsm8k 返回的是官方 test split；我们需要 train split 做开发集。
    # 因此直接复用加载器并额外加载 train split。
    from lychee_mas.eval.benchmarks.common import load_parquet

    train_items = []
    for source_index, it in enumerate(load_parquet("gsm8k/main", "train")):
        gold = it["answer"].split("####")[-1].strip().replace(",", "")
        train_items.append(
            {
                "task": "gsm8k",
                "kind": "exact",
                "question": it["question"].strip() + "\nGive the final numeric answer.",
                "gold": gold,
                "context": None,
                "original_index": source_index,
            }
        )

    test_items = train  # load_gsm8k 实际加载的是官方 test split

    rng = random.Random(seed)
    shuffled_train = list(train_items)
    rng.shuffle(shuffled_train)

    splits_spec = {
        "D0": 20,
        "G1": 50,
        "H1": 50,
        "G2": 50,
        "H2": 50,
        "G3": 50,
        "H3": 50,
        "V": 200,
    }

    required_train = sum(splits_spec.values())
    if len(shuffled_train) < required_train:
        raise RuntimeError(
            f"GSM8K train has {len(shuffled_train)} items, "
            f"need at least {required_train}. Run data preparation first."
        )

    manifest: dict[str, Any] = {
        "dataset": "openai/gsm8k",
        "config": "main",
        "seed": seed,
        "splits": {},
    }

    offset = 0
    for name, size in splits_spec.items():
        items = shuffled_train[offset : offset + size]
        offset += size
        manifest["splits"][name] = [
            {
                "task_id": f"gsm8k:train:seed{seed}:{name}:{i:05d}",
                "original_index": it["original_index"],
                "question": it["question"],
                "gold": it["gold"],
                "content_hash": _content_hash(it["question"] + it["gold"]),
            }
            for i, it in enumerate(items)
        ]

    # T 来自官方 test split（load_gsm8k 返回的）
    manifest["splits"]["T"] = [
        {
            "task_id": f"gsm8k:test:seed{seed}:T:{i:05d}",
            "original_index": i,
            "question": it["question"],
            "gold": it["gold"],
            "content_hash": _content_hash(it["question"] + it["gold"]),
        }
        for i, it in enumerate(test_items[:200])
    ]

    manifest["manifest_hash"] = _manifest_hash(manifest)
    return manifest


def _manifest_hash(manifest: dict[str, Any]) -> str:
    material = json.dumps(manifest["splits"], sort_keys=True, ensure_ascii=False)
    return f"sha256:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GSM8K split manifest for hybrid_comm.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default="data/splits/gsm8k_protocol_seed42.json",
        help="Output manifest path.",
    )
    args = parser.parse_args()

    manifest = build_manifest(seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    total = sum(len(v) for v in manifest["splits"].values())
    print(f"Wrote manifest with {total} items to {out_path}")
    print(f"Manifest hash: {manifest['manifest_hash']}")


if __name__ == "__main__":
    main()
