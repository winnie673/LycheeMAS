# Origin-based GSM8K Hybrid text strategy: channel-only revision

## Base and scope

Start directly from origin/LycheeMASv0.3 at 9189cc27abc6f1440b45b36d90be1999aa9ab957.
Use the frozen V-selected GSM8K Card-2 inside the existing NL channel. Follow
NLMemory / LatentMemory: a strategy's internal helper loads only when selected,
the CDM manager depends on nl/latent interfaces, and core.Message supplies the
message contract. Do not change plugins for a new internal channel strategy.

## Layout contract

- configs/memory/task_cards/gsm8k_card2.yaml: task-scoped frozen Card-2.
- methods/memory/channels/hybrid_protocol.py: load card, render instructions,
  validate and retain the latest protocol message; no model calls.
- channels/nl.py: delegate observe/recall/reset and output_instructions to helper.
- managers/DualChannelMemory.py: pass sender/receiver/ID when observing Hybrid;
  return the standard MemoryBundle for nl/both/none/latent routes.
- tests/test_memory_hybrid_protocol.py: scope, wire, prompt, feedback, counting,
  lifecycle, old strategy compatibility and standard graph integration.
- examples/03_gsm8k_memory_protocol.py: existing graph builder + registered CDM
  and fixed router + injected model callback, using existing core types.

## Execution and compatibility

Keep nl/latent/both/none routing; hybrid_protocol is a strategy inside nl.
No LLM transcoding call. The sending node requests output_instructions before
generation, then delivers core.Message through manager.observe. The receiver
calls manager.recall and receives the original payload under PREV_OUTPUT_HEADER.
Do not inject the parsed fields a second time. Enforce card task/role/edge scope
and strict fields, including ACCEPT/REVISE. Counting uses backend.tok when
available, otherwise communication_tokens is explicitly unavailable, not a
word-count estimate. Per-message metadata is distinct from model output usage.

No task-specific attach_memory implementation. Restore the original plugin
files and document that generic node injection is still unimplemented in
origin. End-to-end use goes through the existing node_factory contract; the
example uses ordinary static build_langgraph and the existing model interface.
The protocol does not schedule roles or perform retries; those are caller duties.

## Validation

Use PYTHONPATH=src and the LycheeMAS conda environment. Run focused and full
tests, lint, dependency selfcheck, registry snapshot and both offline demos.
Keep tests for the frozen card fingerprint and rejected C/U fields. No API
credentials or new real-model experiment are required for this refactor.

## Cleanup approval

The previous draft is preserved in its existing checkout. Publish the slim
implementation in a fresh origin-based memory_gsm8k_channel branch/check-out
instead of deleting old files without approval. Copy only the allowlisted
channel/card/config/test/example/docs files; do not copy task_protocol,
protocol_graph or role templates. No deletion is needed for the final branch.

## Verification result

The clean origin-based checkout passes 216 tests (2 skipped), source lint,
dependency selfcheck and both offline examples. The registry still exposes
the original cdm manager and existing routers. plugins/ has no diff from
origin; task_protocol/, protocol_graph.py and role-template directory are
absent. No real inference run or remote push is part of this revision.
