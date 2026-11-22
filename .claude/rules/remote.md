---
paths:
  - "src/naics_embedder/remote/**"
  - "src/naics_embedder/cli/commands/remote.py"
  - "tests/unit/test_remote_*.py"
  - "tests/integration/test_remote_workflow.py"
  - "tests/fixtures/remote.py"
  - "conf/remote.yaml"
  - "docs/remote_workflow.md"
  - "docs/api/remote.md"
---

# Remote Workflow

The root `CLAUDE.md` keeps the remote guardrails: inputs are never rebuilt on an instance, only
safe=true finish is safe to terminate, held refs are never pushed, and exports, QCEW, stores and
decisions stay on the Mac. This file holds the adapters' contracts.

## Adapters and Continuation

Transport config is separate in `conf/remote.yaml`; later operations use the complete persisted
session config. Help never bootstraps or reads state/config. Committed `conf/config.yaml` keeps
its manifest null, and the Mac's gitignored `conf/local.yaml` names it. That file is never pushed:
launch passes the manifest explicitly. Every kept checkpoint, last and both histories travel
together. Exact continuation only uses last under the same absolute checkpoint path/user,
settings/seed and LoRA/active-MoE controls. Empty/new fresh directories pass; finished runs skip;
budgets and checkpoints are never rewritten.

## Launch Requirements and Model Cache

Bootstrap and immediate launch require NTP and native BF16 on logical CUDA 0 using
`including_emulation=False`, identical CUDA visibility and one device. No GPU whitelist, VRAM
threshold, workload-fit inference or precision fallback. Status GPU evidence is observational.
Mac tampering, pending promotions, active training and unhandled edits withhold readiness.
A CUDA `up` copies the Mac's cached Hub snapshots into `.remote/model-cache/hub` and verifies
their SHA-256 inventory on both hosts (`remote/model_cache.py`; MiniLM's revision is pinned).
Launch rechecks it, and the wrappers run with `HF_HUB_OFFLINE=1`: nothing downloads or repairs.

## Plan 10 History

Plan 10's Phase 2 (Task 18 onward) is finished history. Its qualification experiment, ten-seed
Lambda campaign and Mac reads fixed the reference configuration and δ = 3 SD
(`specs/findings/reference-configuration.md`); Stage 7 is stamped COMPLETE (2026-10-07) in
`specs/naics-embedding-roadmap.md`, and Plan 10 is under `specs/plans/completed/`. Phase 1
evidence/counts remain historical, including its queued lint truth.
