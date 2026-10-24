# Minimal GSM8K text channel integration

Base: origin/LycheeMASv0.3 (9189cc27abc6f1440b45b36d90be1999aa9ab957).
Branch: memory_gsm8k_channel. Use the LycheeMAS conda environment and
PYTHONPATH=src for this checkout; do not import the old editable experiment tree.

Read README.md, CLAUDE.md, docs/DESIGN.md and the memory README/base before
editing. Follow existing NLMemory/LatentMemory strategy delegation. CDM stays
the registered manager; no new registry category or plugin-interface edits.

Layout: one helper in methods/memory/channels/hybrid_protocol.py; frozen card
and matching config in configs/memory; one focused test and one example using
the original static graph/node_factory and core.Message. Keep implementation
notes in docs/plans/memory-hybrid-protocol.md. Do not add a runner, custom graph,
backend, role templates, experiment/evolution scripts or extra record types.

Previous drafts are preserved in separate branches/checkouts; do not copy or
delete them. Preserve unrelated files. Verify full tests, lint, dependency
selfcheck, registry snapshot and both offline examples before reporting.
