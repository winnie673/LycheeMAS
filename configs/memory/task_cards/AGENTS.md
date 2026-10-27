# Task card conventions

One frozen task card per YAML, named `<task>_<card>.yaml`. Preserve the source
wire version and rules; use metadata for task scope and role/edge restrictions.
Historical generated cards remain in the separate experiment branch. Never
overwrite a frozen card with an evolved candidate. New task/schema lineages
must use new files. No traces, credentials or model weights belong here.

Current default: `gsm8k_psv_nl_field_card2_20261027.yaml` (task + architecture +
field lineage + selected evolution round + promotion date). Explicitly promoted
from gsm8k_review_boundary_api_20261027_173531/final/gsm8k_card.yaml after V
selection; preserve semantic content, source wire version and embedded contract.
Add name and relative promotion-source path/hash metadata. Default selection is
in ../gsm8k_hybrid_protocol.yaml, not inferred from file order or newest date.

Historical card: `gsm8k_card2.yaml`; keep it unchanged, no longer the default.
It is a legacy artifact requiring an explicit ProtocolContract.
Do not edit it to retrofit validation metadata. Newly generated cards in runs/
embed validation_contract and record architecture/contract SHA256; promotion into
this directory remains an explicit separate operation, never an automatic overwrite.
Each matching config passes the same identifier as `task_card.task` to CDM.
For uncommitted experiment sources, record source_path and source_sha256
instead of pretending the generated card exists in a source commit. Exclude
task_card metadata when comparing semantic fingerprints with the source.
