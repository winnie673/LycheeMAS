"""Resume task-adaptive Card evolution, fixed-V selection, and frozen-T comparison.

The D0 Free-NL traces and Card_0 must already exist. Each completed API batch is
checkpointed under the selected run root for safe resumption.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lychee_mas.hybrid_comm.protocol import load_task_profile  # noqa: E402
from run_hybrid_compare import _load_manifest, _run_mode, _summarize  # noqa: E402


def _save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _outputs(run_dir: str) -> str:
    return str(Path(run_dir) / "outputs.jsonl")


def _call(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="data/splits/math500_protocol_v2_seed42.json"
    )
    parser.add_argument(
        "--task-profile", default="configs/hybrid_comm/tasks/math500.yaml"
    )
    parser.add_argument(
        "--card0",
        default="configs/hybrid_comm/generated/math500/initial/card_0_math500.yaml",
    )
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--run-root", default="runs/hybrid_comm/math500")
    parser.add_argument(
        "--generated-root", default="configs/hybrid_comm/generated/math500"
    )
    parser.add_argument("--version-tag", default="")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    manifest["_path"] = args.manifest
    task_profile = load_task_profile(args.task_profile)
    task_name = task_profile["task"]
    manifest_task = manifest.get("task")
    if manifest_task is not None and manifest_task != task_name:
        raise ValueError(
            f"manifest task {manifest_task!r} does not match task profile {task_name!r}"
        )
    manifest["task"] = task_name
    expected = {
        "D0": 20,
        "G1": 50,
        "G2": 50,
        "G3": 50,
        "V": 100 if task_name == "math500" else 200,
        "T": 200,
    }
    if {k: len(manifest["splits"][k]) for k in expected} != expected:
        raise ValueError(
            f"manifest split sizes do not match the frozen {task_name} design"
        )

    root = Path(args.run_root)
    state_path = root / "pipeline_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("manifest_hash") != manifest["manifest_hash"]:
            raise ValueError("pipeline state belongs to a different manifest")
    else:
        state = {
            "task": task_name,
            "manifest": args.manifest,
            "manifest_hash": manifest["manifest_hash"],
            "model": args.model,
            "tokenizer": args.tokenizer,
            "workers": args.workers,
            "runs": {},
            "rounds": {},
        }
        _save(state_path, state)

    if not Path(args.card0).exists():
        raise FileNotFoundError(args.card0)

    def run_once(key: str, mode: str, split: str, card: str, directory: str) -> str:
        prior = state["runs"].get(key)
        if prior and Path(_outputs(prior)).exists():
            with Path(_outputs(prior)).open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            expected_ids = [case["task_id"] for case in manifest["splits"][split]]
            if [row.get("task_id") for row in rows] != expected_ids:
                raise ValueError(f"{key}: checkpoint has incomplete or mismatched cases")
            return prior
        run_dir, samples, _ = _run_mode(
            mode=mode,
            backend_name="api",
            model=args.model,
            manifest=manifest,
            split=split,
            n=None,
            run_root=str(root / directory),
            protocol_path=card,
            tokenizer_name=args.tokenizer,
            workers=args.workers,
        )
        if len(samples) != len(manifest["splits"][split]):
            raise ValueError(f"{key}: incomplete batch")
        state["runs"][key] = run_dir
        _save(state_path, state)
        return run_dir

    incumbent_card = args.card0
    incumbent_v = run_once("card0_v", "hybrid", "V", incumbent_card, "validation/card0")

    for round_no in range(1, 4):
        split = f"G{round_no}"
        round_key = f"round_{round_no}"
        saved = state["rounds"].get(round_key)
        if saved:
            incumbent_card = saved["winner_card"]
            incumbent_v = saved["winner_v_run"]
            continue

        nl_run = run_once(
            f"{split}_nl", "nl", split, incumbent_card, f"evolution/{round_key}/nl"
        )
        hybrid_run = run_once(
            f"{split}_hybrid",
            "hybrid",
            split,
            incumbent_card,
            f"evolution/{round_key}/hybrid",
        )
        round_root = root / "evolution" / round_key
        feedback = round_root / "feedback.json"
        if not feedback.exists():
            _call(
                "scripts/build_evolution_feedback.py",
                "--nl", _outputs(nl_run),
                "--hybrid", _outputs(hybrid_run),
                "--out", str(feedback),
            )

        candidate_card = str(
            Path(args.generated_root)
            / "evolution"
            / f"card_{round_no}_{task_name}_candidate.yaml"
        )
        artifact = round_root / "evolution.json"
        if not Path(candidate_card).exists() or not artifact.exists():
            _call(
                "scripts/evolve_protocol_card.py",
                "--task-profile", args.task_profile,
                "--card", incumbent_card,
                "--feedback", str(feedback),
                "--version", f"CARD-{round_no}-{task_name.upper()}{args.version_tag}",
                "--out", candidate_card,
                "--artifact", str(artifact),
                "--model", args.model,
            )
        evolution = json.loads(artifact.read_text(encoding="utf-8"))
        winner_card = str(
            Path(args.generated_root)
            / "evolution"
            / f"card_{round_no}_{task_name}_winner.yaml"
        )
        selection = round_root / "selection.json"

        if evolution["action"] == "KEEP":
            Path(winner_card).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(incumbent_card, winner_card)
            selection.write_text(
                json.dumps(
                    {"winner": "incumbent", "criterion": "evolver_keep",
                     "winner_card": winner_card, "incumbent_v_run": incumbent_v},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
            winner_v = incumbent_v
        else:
            candidate_v = run_once(
                f"card{round_no}_candidate_v", "hybrid", "V", candidate_card,
                f"validation/card{round_no}_candidate",
            )
            _call(
                "scripts/select_evolved_card.py",
                "--incumbent-card", incumbent_card,
                "--incumbent-outputs", _outputs(incumbent_v),
                "--candidate-card", candidate_card,
                "--candidate-outputs", _outputs(candidate_v),
                "--winner-card", winner_card,
                "--out", str(selection),
            )
            selected = json.loads(selection.read_text(encoding="utf-8"))
            winner_v = candidate_v if selected["winner"] == "candidate" else incumbent_v

        state["rounds"][round_key] = {
            "split": split,
            "input_card": incumbent_card,
            "candidate_card": candidate_card,
            "winner_card": winner_card,
            "winner_v_run": winner_v,
            "feedback": str(feedback),
            "evolution": str(artifact),
            "selection": str(selection),
        }
        _save(state_path, state)
        incumbent_card = winner_card
        incumbent_v = winner_v

    final_card = Path(args.generated_root) / "final" / f"card_{task_name}_evolved.yaml"
    if not final_card.exists():
        final_card.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(incumbent_card, final_card)
    state["final_card"] = str(final_card)
    _save(state_path, state)

    nl_t = run_once("test_nl", "nl", "T", str(final_card), "final_test/nl")
    hybrid_t = run_once(
        "test_hybrid", "hybrid", "T", str(final_card), "final_test/hybrid"
    )
    comparison = {
        "task": task_name,
        "split": "T",
        "manifest_hash": manifest["manifest_hash"],
        "final_card": str(final_card),
        "nl_run_dir": nl_t,
        "hybrid_run_dir": hybrid_t,
    }

    def load_rows(run_dir: str) -> list[dict]:
        with Path(_outputs(run_dir)).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    comparison["nl_summary"] = _summarize(load_rows(nl_t))
    comparison["hybrid_summary"] = _summarize(load_rows(hybrid_t))
    comparison_path = root / "final_test" / "comparison.json"
    _save(comparison_path, comparison)
    state["comparison"] = str(comparison_path)
    _save(state_path, state)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
