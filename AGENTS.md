# Minimal benchmark task-card text channel integration

Base: origin/LycheeMASv0.3 (9189cc27abc6f1440b45b36d90be1999aa9ab957).
Branch: memory_gsm8k_channel. Use the LycheeMAS conda environment and
PYTHONPATH=src for this checkout; do not import the old editable experiment tree.

Read README.md, CLAUDE.md, docs/DESIGN.md and the memory README/base before
editing. Follow existing NLMemory/LatentMemory strategy delegation. CDM stays
the registered manager; no new registry category or plugin-interface edits.

Layout: one helper in methods/memory/channels/hybrid_protocol.py; frozen card
and matching config in configs/memory; one focused test and one example using
the original static graph/node_factory and core.Message. Keep implementation
notes in docs/plans/memory-hybrid-protocol.md. Except for the approved offline
scope below, do not add a runner, custom graph, backend, role templates,
evolution scripts or extra record types.

Approved GSM8K offline protocol scope (2026-10-27): `methods/memory/protocol/`
contains internal architecture/callback contracts in base.py and an injected
generation/evolution algorithm in evolution.py, exported by __init__.py.
`scripts/run_memory_protocol_evolution.py` assembles existing backends, frozen
GSM8K D0/G1/G2/G3/V splits and the SAME example run_demo; no MATH-500 evolution.
The algorithm never imports an SDK, creates a graph or calls a backend directly.
Card-0 is model-generated from D0 NL traces; each G round proposes UPDATE/KEEP,
then evaluates parent/candidate on the same V cases. Select higher accuracy,
then fewer exact communication tokens, otherwise retain parent. No T access
in generation/selection; final T evaluation is an explicit separate action.
Use runs/memory_protocol_evolution/<new_run>/ for all candidates and audit files;
never overwrite frozen cards. New protocol/ directory follows these conventions:
no model weights/data/templates, only method-local contracts and algorithm.
Extend existing channel/example/evaluation with explicit card selection and
architecture checks; do not create a second executor or change plugins/registry.
Fake runs are plumbing tests, never efficacy or publishable frozen protocols.
Add tests/test_memory_protocol_evolution.py and update the existing plan/READMEs.

Approved complete decoupling (2026-10-27): core memory protocol contracts,
declarative validation and evolution must contain no benchmark/role/field names.
`scripts/hybrid_gsm8k_spec.py` is the sole experiment specification: PSV roles,
fields, constraints, guidance and D0/G1-G3/V split schedule. Examples/scripts
inject it; src must never import scripts. Existing frozen cards are unchanged:
legacy loading REQUIRES an explicit contract, new cards embed their contract.
No silent loss of semantic guards or permissive fallback for missing contracts.
Keep the current example as the GSM8K executor, not a universal graph builder.
Test a second synthetic non-PSV/non-numeric task and renamed fields end-to-end
through CDM/evolution (no network) to prove the core actually accepts other specs.
No file deletion or real API execution is authorized by this refactor.

Approved behavior restoration (2026-10-27): preserve the original LycheeMAS
GSM8K role instructions, independent initial semantic review, evidence-bound
diagnosis and <=3-path evolution patches, runtime feedback aliases and recovery
boundaries. `scripts/hybrid_gsm8k_design.py` may vendor ONLY the required pure
design functions with provenance and expose an injected design strategy. Keep
the frozen source schema at configs/memory/gsm8k_protocol_schema.yaml; this file
is schema, not a learned initial card. No production imports of the sibling repo.
Core orchestration may accept a generic design strategy and must remain role-free.
Keep NL natural text, the second verifier and current V selection objective.
Do not copy the old SDK max_retries plumbing bug: explicitly disable SDK retries.
Preserve actual finish_reason and usage_unknown; do not pretend token budget
alone is an observed truncation. Unexpected program errors still fail explicitly.
Add parity tests against original-source function fixtures when available, and
self-contained offline tests so the sibling repository is not a test dependency.
No deletes, historical-card overwrite, real API run, push or global install.

Approved live experiment (2026-10-27): user explicitly authorizes the restored
GSM8K pipeline with deepseek-chat on full D0/G1-G3/V, followed only on success
by paired NL/Hybrid T evaluation of the newly selected card. Inject the supplied
credential only into this run's process environment; never persist or print it.
Use new run directories and the cached matching tokenizer; preserve old results,
cards and data. This authorization does not permit configuration changes in
response to evaluation outcomes, external publication or repository pushes.

Approved review clarification and rerun (2026-10-27): clarify both induction and
review prompts that CHK remains ACCEPT/REVISE and missing evidence is represented
by CHK:REVISE with E:INSUFFICIENT_INFO, not another CHK status. Preserve strict
review, six initial attempts, frozen schema, runtime, splits and scoring. Record
these two deliberate prompt deviations in parity tests and run a new full API
pipeline after offline regression checks; do not alter or resume old artifacts.

Approved example-grounding repair and retry (2026-10-27): distinguish failed D0
reasoning from valid teaching examples; require new coherent checkable examples
and replacement of unrepairable examples rather than invented answers. Reviewer
fixes must respect frozen task answer policy and distinguish missing message
evidence from an unsolvable problem. Keep temperature, retry limits, strict
review, schema, roles and splits unchanged. Implement task-specific guidance in
the injected GSM8K adapter, generic replacement guidance in initial retry only,
test offline, then rerun the authorized full API pipeline in new directories.

Approved review-boundary repair (2026-10-27): semantic review must receive only
design material, excluding program-injected task_card and validation_contract;
the full saved card still undergoes deterministic runtime-contract validation.
Check reviewer JSON consistency, concrete existing finding paths and optional
versus blocking findings. Inconsistent reviews receive at most one separately
audited recheck of the SAME candidate, never automatic approval or silent finding
deletion. Persistent reviewer inconsistency stops explicitly. Keep six initial
generation attempts, strict semantic gate, task data and execution unchanged.
Before the full new API pipeline, re-review a saved D0 candidate using the same
production review path in a new preflight directory; only proceed if this check
returns a consistent, design-scoped verdict. A legitimate rejection of that old
candidate stays a rejection; it is never frozen or substituted for fresh Card-0.
Old cards/runs stay intact, secrets stay process-only.

Approved evaluation scope: `scripts/run_memory_protocol_compare.py` may load a
frozen split manifest, run paired NL/Hybrid tasks and persist scored results.
Approved Hybrid batch evaluation benchmark is GSM8K only. Keep its numeric
scoring unchanged; use its frozen split and selected card. Never edit cards/data
in response to test outputs. Existing GSM8K results stay intact.
It must reuse `run_demo` from the existing example (one node_factory/static
graph implementation), not add a second execution loop or bypass CDM memory.
The shared executor supports at most one semantic revision followed by a second
verification, with the same per-node budgets and stop rules in both modes.
Keep NL output natural language, not Hybrid fields in a JSON wrapper. Gold is
only passed to the scorer after execution. Persist every attempt including
failed tasks; never overwrite runs or approximate missing communication tokens.
Add focused evaluation tests; offline validation must not call external APIs.
Real API evaluation is an explicit separate action, not a default smoke test.

Approved default promotion (2026-10-27): promote the final V-selected card from
runs/memory_protocol_evolution/gsm8k_review_boundary_api_20261027_173531/final/
gsm8k_card.yaml into configs/memory/task_cards/gsm8k_psv_nl_field_card2_20261027.yaml
and switch gsm8k_hybrid_protocol.yaml to this file. Preserve its wire version,
rules, examples and embedded validation contract; add only provenance metadata.
Keep legacy gsm8k_card2.yaml byte-for-byte for historical tests/compatibility,
and preserve all run artifacts. No deletes or new real API evaluation are needed.
Validate both implicit embedded-contract loading and the default example/config.
Never guess benchmark from question text or introduce a second registry/channel.

Approved GitHub snapshot (2026-10-27): create memory_hybrid_gsm8k_20261027 from
the current implementation and push ONLY to winnie673/LycheeMAS (remote winnie).
Exception to the usual artifact exclusion: include only the completed evolution
and comparison directories named gsm8k_review_boundary_api_20261027_173531 under
runs/memory_protocol_evolution/ and runs/memory_protocol_compare/. Preserve local
history without deletion; do not include failed runs, preflight runs, caches,
credentials or unrelated generated uv.lock. Scan the exact upload set for secrets
and conflict markers, verify tests and push without rebase/force or upstream PR.

Approved cleanup (2026-10-27): remove the Hybrid-specific MATH-500 config/card,
comparison wrapper, strict scorer and focused test, and their shared-code/docs
branches. Preserve the framework's original math500 benchmark, MASPO, shared
math parser and all historical runs. Keep a recoverable copy of the five removed
files outside this checkout. Do not interpret this as permission for other deletes.

Approved schema single-source refactor: load the fixed YAML once in the experiment
spec and derive edge interfaces, field semantics, enums, optional-value guards and
conditional requirements through generic pure helpers in protocol/base.py. Keep
roles, task-specific answer guidance and receive-only aliases in the experiment.
Preserve edge-specific meanings, do not flatten by overwriting. Add a real forbidden
condition rather than treating every forbidden list as an exclusive field set.
Old serialized contracts/cards must keep their fingerprints: omit new empty rule
attributes from legacy serialization. Frozen embedded contracts may be selected
only by explicit historical fingerprint and runtime compatibility checks; new
cards use the derived contract. No frozen card/run edits, deletions, live API,
commit or push are authorized for this refactor. Reuse existing files/tests.

Approved post-refactor API experiment: user requests full GSM8K generation,
evolution and evaluation on v5-schema-derived. Use deepseek-chat and the same
frozen D0/G1-G3/V/T manifest, cached tokenizer and process-only credential as
the earlier authorized live runs. Start fresh output directories, run T paired
NL/Hybrid only after successful V selection, and stop downstream on failure.
Do not replace the published default card, edit splits, publish or push results
as part of this experiment. Record source hashes and retain prior artifacts.

Previous drafts are preserved in separate branches/checkouts; do not copy or
delete them. Preserve unrelated files. Verify full tests, lint, dependency
selfcheck, registry snapshot and both offline examples before reporting.

Approved final promotion/cleanup (2026-10-28, supersedes older retention rules):
publish the final v5-schema-derived V-selected Card-0 as
configs/memory/task_cards/gsm8k_psv_hybrid_card_20261028.yaml, keeping source wire
version/rules/examples/contract unchanged and adding only provenance metadata.
Keep only the two successful run directories named
gsm8k_schema_derived_api_20261027_212336 under memory_protocol_evolution and
memory_protocol_compare, plus the matching launcher log. User explicitly approved
deleting the other 22 inventoried run directories, their seven old launcher logs,
gsm8k_card2.yaml and gsm8k_psv_nl_field_card2_20261027.yaml. Preserve schema and
original framework memory configs. Archive exact deletion targets outside this
checkout in E:/毕设/cleanup_archives with a verified per-file SHA256 manifest before
deleting. Validate absolute targets remain inside this checkout, reject reparse
points and any additions beyond the approved inventory. Adapt legacy tests to
synthetic temporary cards instead of retaining obsolete configuration cards.
Do not modify kept raw run records, Git history, credentials, or push changes.

Retention correction (latest user instruction, 2026-10-28): do NOT remove older
successful results. Restore from the verified cleanup archive the completed real
API GSM8K comparison 20261025_162017, review-boundary evolution/comparison
20261027_173531, MATH500 v2 comparison 20261025_192422, successful review preflight
and wire-v2 functional smoke records, with their corresponding launcher logs.
Keep old card snapshots INSIDE those result directories for audit, but retain
only the new published YAML in configs/memory/task_cards. Failed/incomplete and
fake/offline runs remain removed. This overrides the earlier blanket 22-run
deletion approval. Preserve successful records byte-for-byte after restoration.

Approved upload after cleanup (2026-10-28): user confirmed commit and push to
winnie673/LycheeMAS, existing memory_hybrid_gsm8k_20261027 branch. Include current
code/tests/docs, the sole new default card, latest two successful runs and the
six retained successful older result/validation directories plus their four
combined launcher logs. Old success records must not be deleted on GitHub.
Exclude external archives, credentials, caches, failed/offline runs and unrelated
uv.lock. Scan actual staged blobs, verify tests and push without force or rebase.

Latest upload-scope correction: keep older successful results on the local disk,
but publish ONLY the two gsm8k_schema_derived_api_20261027_212336 run directories
and matching evolution launcher log in the branch snapshot. Remove older result
paths from the Git index with --cached only, never from disk. Existing commits
remain recoverable; do not rewrite Git history. This supersedes the broader
eight-directory upload scope above, not the local preservation requirement.
