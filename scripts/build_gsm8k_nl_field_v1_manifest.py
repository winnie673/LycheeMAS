"""Build the GSM8K manifest for the optional-NL protocol lineage v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MOVED_T_POSITIONS = (12, 62, 85, 119, 162)
# random.Random(42).sample(range(200, 1319), 5), frozen for reproducibility.
REPLACEMENT_ORIGINAL_INDICES = (428, 251, 763, 701, 657)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _manifest_hash(splits: dict[str, list[dict[str, Any]]]) -> str:
    material = json.dumps(splits, sort_keys=True, ensure_ascii=False)
    return f"sha256:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"


def _load_official_test() -> list[dict[str, Any]]:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from lychee_mas.eval.benchmarks.gsm8k import load_gsm8k

    return load_gsm8k(n=None)


def build_manifest(base: dict[str, Any]) -> dict[str, Any]:
    splits = {name: list(cases) for name, cases in base["splits"].items()}
    old_t = splits["T"]
    if len(old_t) != 200:
        raise ValueError(f"Expected frozen T size 200, got {len(old_t)}")

    moved_ids = {f"gsm8k:test:seed42:T:{pos:05d}" for pos in MOVED_T_POSITIONS}
    moved = [case for case in old_t if case["task_id"] in moved_ids]
    if len(moved) != len(MOVED_T_POSITIONS):
        raise ValueError("Frozen T does not contain every requested semantic-risk case")

    moved_for_d0 = []
    for i, case in enumerate(moved):
        updated = dict(case)
        updated["task_id"] = f"gsm8k:test:seed42:D0_NL:{i:05d}"
        moved_for_d0.append(updated)
    splits["D0"] = splits["D0"] + moved_for_d0

    official_test = _load_official_test()
    replacements = []
    for original_index in REPLACEMENT_ORIGINAL_INDICES:
        item = official_test[original_index]
        question = item["question"]
        gold = item["gold"]
        replacements.append(
            {
                "task_id": f"gsm8k:test:seed42:T:{original_index:05d}",
                "original_index": original_index,
                "question": question,
                "gold": gold,
                "content_hash": _content_hash(question + gold),
            }
        )

    splits["T"] = [case for case in old_t if case["task_id"] not in moved_ids]
    splits["T"].extend(replacements)

    output = {
        "dataset": base["dataset"],
        "config": base["config"],
        "seed": base["seed"],
        "lineage": "gsm8k_nl_field_v1",
        "parent_manifest_hash": base.get("manifest_hash", ""),
        "split_change": {
            "moved_from_T_to_D0": list(MOVED_T_POSITIONS),
            "T_replacement_original_indices": list(REPLACEMENT_ORIGINAL_INDICES),
            "reason": (
                "Post-hoc semantic-risk development cases for optional NL; "
                "new T remains 200 cases."
            ),
        },
        "splits": splits,
    }
    output["manifest_hash"] = _manifest_hash(splits)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="data/splits/gsm8k_protocol_schema_seed42.json",
    )
    parser.add_argument(
        "--out",
        default="data/splits/gsm8k_protocol_nl_field_v1_seed42.json",
    )
    args = parser.parse_args()

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    manifest = build_manifest(base)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"D0={len(manifest['splits']['D0'])} T={len(manifest['splits']['T'])}")
    print(f"Manifest hash: {manifest['manifest_hash']}")


if __name__ == "__main__":
    main()
