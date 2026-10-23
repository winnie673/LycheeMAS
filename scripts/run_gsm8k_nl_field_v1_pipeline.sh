#!/usr/bin/env bash
set -euo pipefail

# Continue the NL-field v1 experiment after D0 NL traces and formal Card-0 exist.
# DEEPSEEK_API_KEY must already be exported by the caller. No secret is stored here.

PY="${LYCHEE_PYTHON:-.venv/bin/python}"
ROOT="${LYCHEE_RUN_ROOT:-runs/server_qwen38_27b_nl_field_v1}"
MANIFEST="data/splits/gsm8k_protocol_nl_field_v1_seed42.json"
SCHEMA="configs/hybrid_comm/generated/gsm8k_nl_field_v1/protocol_schema_nl_field_v1.yaml"
LINEAGE="configs/hybrid_comm/generated/gsm8k_nl_field_v1"
TOKENIZER="${LYCHEE_TOKENIZER:-/home/u2023311a15/models/Qwen3.8-27B-AWQ-4bit}"
LOCAL_BASE_URL="${LYCHEE_LOCAL_BASE_URL:-http://127.0.0.1:8000/v1}"
LOCAL_MODEL="${LYCHEE_LOCAL_MODEL:-qwen38-27b-q4km}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY must be exported before starting the pipeline" >&2
  exit 2
fi

export PYTHONPATH=src
mkdir -p "$ROOT" "$LINEAGE/evolution" "$LINEAGE/final"

use_local_executor() {
  export DEEPSEEK_BASE_URL="$LOCAL_BASE_URL"
  export DEEPSEEK_MODEL="$LOCAL_MODEL"
}

use_design_api() {
  unset DEEPSEEK_BASE_URL
  unset DEEPSEEK_MODEL
}

latest_output() {
  local directory="$1"
  local mode="$2"
  find "$directory" -type f -name outputs.jsonl | grep "_api_${mode}_" | sort | tail -n 1
}

run_hybrid() {
  local split="$1"
  local card="$2"
  local run_root="$3"
  shift 3
  "$PY" scripts/run_hybrid_compare.py \
    --backend api --protocol "$card" --manifest "$MANIFEST" --split "$split" \
    --run-root "$run_root" --model "$LOCAL_MODEL" --tokenizer "$TOKENIZER" \
    --workers 1 "$@"
}

start_round="${LYCHEE_START_ROUND:-1}"
if ((start_round > 1)); then
  incumbent="$LINEAGE/evolution/card_$((start_round - 1))_gsm8k_winner.yaml"
else
  incumbent="$LINEAGE/initial/card_0_gsm8k_nl_field.yaml"
fi

for ((round = start_round; round <= 3; round++)); do
  split="G${round}"
  round_root="$ROOT/evolution/round${round}"
  candidate="$LINEAGE/evolution/card_${round}_gsm8k_candidate.yaml"
  winner="$LINEAGE/evolution/card_${round}_gsm8k_winner.yaml"
  mkdir -p "$round_root/evidence" "$round_root/validation/incumbent" \
    "$round_root/validation/candidate"

  if [[ -f "$round_root/feedback.json" ]]; then
    echo "[round${round}] reuse existing evidence $(date -Is)"
  else
    echo "[round${round}] evidence start $(date -Is)"
    use_local_executor
    run_hybrid "$split" "$incumbent" "$round_root/evidence" --modes nl hybrid
    nl_outputs="$(latest_output "$round_root/evidence" nl)"
    hybrid_outputs="$(latest_output "$round_root/evidence" hybrid)"
    "$PY" scripts/build_evolution_feedback.py \
      --nl "$nl_outputs" --hybrid "$hybrid_outputs" \
      --out "$round_root/feedback.json"
  fi

  echo "[round${round}] evolve start $(date -Is)"
  use_design_api
  "$PY" scripts/evolve_gsm8k_card.py \
    --card "$incumbent" --feedback "$round_root/feedback.json" \
    --version "CARD-${round}-GSM8K-NL-FIELD-V1" --out "$candidate" \
    --artifact "$round_root/evolution.json" --schema "$SCHEMA"

  echo "[round${round}] validation start $(date -Is)"
  use_local_executor
  run_hybrid V "$incumbent" "$round_root/validation/incumbent" --modes hybrid
  run_hybrid V "$candidate" "$round_root/validation/candidate" --modes hybrid
  incumbent_outputs="$(latest_output "$round_root/validation/incumbent" hybrid)"
  candidate_outputs="$(latest_output "$round_root/validation/candidate" hybrid)"
  "$PY" scripts/select_evolved_card.py \
    --incumbent-card "$incumbent" --incumbent-outputs "$incumbent_outputs" \
    --candidate-card "$candidate" --candidate-outputs "$candidate_outputs" \
    --winner-card "$winner" --out "$round_root/selection.json"

  incumbent="$winner"
  echo "[round${round}] complete $(date -Is)"
done

cp "$incumbent" "$LINEAGE/final/card_gsm8k_evolved.yaml"

echo "[test] NL and final Hybrid start $(date -Is)"
use_local_executor
run_hybrid T "$LINEAGE/final/card_gsm8k_evolved.yaml" "$ROOT/test" --modes nl hybrid
echo "[pipeline] complete $(date -Is)"
