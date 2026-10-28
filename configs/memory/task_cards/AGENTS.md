# Task card conventions

One frozen task card per YAML, named `<task>_<card>.yaml`. Preserve the source
wire version and rules; use metadata for task scope and role/edge restrictions.
Historical generated cards remain in the separate experiment branch. Never
overwrite a frozen card with an evolved candidate. New task/schema lineages
must use new files. No traces, credentials or model weights belong here.

Only published card: `gsm8k_psv_hybrid_card_20261028.yaml`, using the user's
chosen task/architecture/protocol/date naming. It is the final V-selected Card-0
from gsm8k_schema_derived_api_20261027_212336/final/gsm8k_card.yaml; all three
evolution candidates lost V selection. Keep its CARD-0-gsm8k wire version and
embedded contract. Add only name and relative promotion-source path/hash metadata.
Default selection lives in ../gsm8k_hybrid_protocol.yaml, not file ordering.

User approved removing the two older configuration cards on 2026-10-28; they
are recoverable in the external cleanup archive, not runtime/test dependencies.
Legacy compatibility tests must construct synthetic temporary cards rather
than require a deleted or Git-history file. Newly generated cards in runs/
embed validation_contract and record architecture/contract SHA256; promotion into
this directory remains an explicit separate operation, never an automatic overwrite.
Each matching config passes the same identifier as `task_card.task` to CDM.
For uncommitted experiment sources, record source_path and source_sha256
instead of pretending the generated card exists in a source commit. Exclude
task_card metadata when comparing semantic fingerprints with the source.
