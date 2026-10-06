# Lambda Remote Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default), or executing-plans if the user chooses inline
> implementation. Independent task reviews remain required in either mode. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Status: COMPLETE (2026-10-06)** — executed via subagent-driven-development; nothing deferred.

The user approved the complete draft on 2026-10-05, including all twelve clarifications and the
native-BF16 guard. Implementation PR [#126](https://github.com/lowmason/naics-embedder/pull/126)
merged on 2026-10-06 at 11:50:29 UTC as `15de7a0310ff3bbfa9157f98ea9459c45741cf34`.
Its tree equals reviewed `270bc72252077565cbd54e2cf1e86665e47c890f`; substantive runtime
SHA is `304bc34f07b462380e9d587cf5e9282ebcf511fc`. Lambda launch, real-instance qualification
and the campaign remain outside this completed implementation plan.

> Workspace recovery: the planning chat archived its checkout during implementation preflight.
> The native app recovered the approved draft from snapshot `19511544` into managed worktree
> `/Users/lowell/.codex/worktrees/lambda-remote-workflow-1280/naics-embedder`, on the same
> `codex/lambda-remote-workflow` branch and unchanged `origin/main` base. The primary is untouched.

**Goal:** Land the five approved remote commands, with verified exact continuation and result
transport, so Plan 10 Task 18 can qualify the merged implementation before Phase 2.

**Architecture:** Keep Typer adapters thin. A workflow controller coordinates typed local state,
canonical-input and checkpoint gates, reproducible code records, and a transport interface whose
SSH implementation is the only module that invokes SSH/rsync. A directory-backed local transport
and injected process controls exercise the same orchestration without a GPU or network. Pulls
stage and verify coherent checkpoint generations before promoting files to the Mac.

**Tech Stack:** Python 3.10/3.12, existing Pydantic/Typer/Rich/PyTorch/Lightning/pytest packages;
standard-library subprocess, pathlib, hashlib, json, tarfile, fcntl and shlex; GNU rsync >=3.2,
SSH, Linux tmux/timedatectl and Mac caffeinate. No new Python dependencies or lock changes.

## Completion evidence and boundary

All implementation task reviews and final evidence audits approved. Independent GPT-6.1 Ultra
whole-branch coverage, WB1–WB3 fixes and portability round 2 approved the final reviewed head.
Fresh locked local gates there passed: Python 3.12 **2,999 passed / 2 skipped / 171 warnings**;
Python 3.10 **2,999 passed / 2 skipped / 642 warnings**. Required GNU tests passed **47 / zero
skips / 82 warnings** on each version; four rendered preparation cases passed on each without
warnings. Ruff, YAPF, strict docs, shell syntax, source/protected gates and five help commands
passed. Quiet local output did not measure the two skip identities; no identity is invented.
The extra 471 Python 3.10 numerical warnings retain their original source/message/count profile;
no numerical assertion, tolerance or precision was changed.

Corresponding PR CI run **37420123506**, attempt 1, at reviewed head `270bc722` passed docs,
lint and both Python jobs. Each full job measured **2,990 passed / 11 skipped / 80 warnings**,
**89.90% coverage** and successful XML/Codecov upload; each GNU step measured **47 passed /
zero skips / 41 warnings**. GitHub used its PR merge ref for checkout. Earlier run 37416387201
failed five Python 3.12 tests and cancelled Python 3.10; that failure remains historical.
Phase 1's queued lint at merge is likewise not rewritten as a pass.

Ignored evidence is preserved in primary `logs/plan11_worktree/`: the original `.sdd` workspace,
execution ledger and review evidence comprise **906 files / zero symlinks / 23,172,194 bytes**.
Receipt SHA256 is `a3d8494c3792fe685c120ddd318c1e8f64a9cddfffc7a45d42e747511b990a67`.
Independent preservation audit approved; `postmerge-admin/archive-confirmation.json` confirms
native archival of the implementation checkout and absence of its source path. The original
116 protected files (198,306,791 bytes) and seven selection records remained unchanged.

Resolve-before-defer found no skipped implementation work, unresolved review finding or needed
human input. Real Lambda/Mac/image qualification was explicitly out of scope and remains the
live specification's prerequisite, not deferred implementation work. The required existing-item
ticking pass found no item implemented by Plan 11. Backlog: **28 open / 13 ever closed / 32%
closure / zero aged >45 days**; a grouped read-only proposal is retained in completion evidence.
No disposition was executed and no empty Plan 11 deferred section was appended.

This retirement records completed implementation and archival in a separate post-merge
checkout. Its independent documentation review, publication and new documentation PR CI/merge
are administrative follow-ups; they are not claimed passed by this record. Plan 10 Task 18,
Phase 2, its manual qualification and Stage 7 remain open. The normative task commands below
are preserved as historical requirements, with actual results and deviations recorded here.

## Global Constraints

- Implement [the approved specification](../../lambda-remote-workflow.md), read from merged
  `origin/main`, plus [Plan 10 Task 18](../10-objective-anchors-and-live-radius.md) and this brief.
- Preserve R1–R11 in `specs/objective-anchors-and-live-radius.md`. Ten reference seeds;
  δ = 3 SD; no configuration selection before δ. This plan implements transport, not selection.
- Transport every kept checkpoint, `last.ckpt`, `monitor_reads.jsonl`, and
  `epoch_summary.jsonl` together. Restore the full run directory on replacement instances.
- One stable absolute checkpoint directory per run across instances and resume segments.
- Resume only from `last.ckpt`. Never relaunch a finished run. Fresh runs require an empty/new
  checkpoint directory on both machines. Never change a saved epoch budget to extend a run.
- Launch training in tmux with stdin from `/dev/null`; verify NTP synchronization at bootstrap
  and immediately before each launch.
- Require native BF16 support on the selected CUDA device at bootstrap and immediately before
  each training launch. Refuse unsupported or uninspectable devices; no precision fallback.
  GPU eligibility is capability-based, with no model whitelist or invented VRAM threshold.
- No decision-panel reads on Lambda. Only the existing within-run outcome validation monitor
  runs there. Evaluation, exports, QCEW reads, artifact-store operations and decisions stay on
  the Mac. Never open sealed test/outer splits in implementation or campaign preparation.
- Remote selection logs remain separate from the Mac's append-only selection log.
- Resume pre-checks must not require the dropped fields `supervision_mode`,
  `structural_preference_loss_version`, or `mining_contract_version`.
- Honor the current checkpoint contract and supplementary constructor-setting guards,
  including active LoRA/MoE controls. Keep the literal 21-key settings identity unchanged.
- No weights-only migration/start. No checkpoint rewriting or saved-path migration.
- Do not launch Lambda or begin the campaign in the planning or implementation session.
  Lambda instance launch and termination remain the user's actions.
- Execution and task reviews: `gpt-6.1-sol`, `medium`. Final whole-branch review:
  `gpt-6.1-sol`, `ultra`. These explicit user choices replace legacy Claude model routing.
- Red → green → refactor; independent task reviews; full local Python 3.10/3.12 verification,
  Ruff/YAPF, strict docs, and all corresponding CI gates. GitHub Codex review is unavailable
  and is not a gate. Independent agent reviews remain gates.
- Never push `main` or the held private commits `177899c` (config), `a030dfb` (graph config),
  or their later replayed descendants. Do not replay either during this plan.
- Primary checkout remains `/Users/lowell/Projects/naics-embedder`, currently on
  `claude/stage-7-reference-configuration`. All implementation edits/commits use this plan's
  managed worktree and `codex/lambda-remote-workflow` branch, based on `origin/main`.
- Preserve `checkpoints/plan10_smoke/` and `logs/plan10_worktree/`. Never relaunch the finished
  smoke. `logs/selection_log.jsonl` already has seven records; never append the four Phase 1
  exit reads again. Tests write only under `tmp_path` and retain the existing log guard.
- Keep committed `conf/config.yaml`'s manifest null. Canonical inputs travel explicitly and are
  never rebuilt on an instance. Do not download Census/QCEW or operate the artifact store here.
- Keep Python single quotes, type hints, semantic dividers and logging conventions. YAPF owns
  formatting; never use `ruff format`. Format only touched files, then check the entire tree.
- Do not retire Plan 10 or mark Stage 7 complete. That waits for Plan 10 Phase 2.

## Inspection and design decisions

### Verified starting state

Fetch with prune on 2026-10-05 returned `origin/main` at
`87d4e2a1b6efadc1e1c60394cb3cd79a786e6fc3`, PR #125's merge. It removed the Phase 1 remote
branch. No path under `src/naics_embedder/remote/`, no `cli/commands/remote.py`, and no remote
implementation plan exists there. Plan numbering is 1–10, so this is Plan 11.

The primary checkout was clean at `b5f31d6` on its retained Claude branch. Local `main` was
`a030dfb`, ahead of its old base by the two private commits and behind the fetched remote by
24 commits. A native managed worktree was created from `origin/main`, then branched as
`codex/lambda-remote-workflow`:

```text
/Users/lowell/.codex/worktrees/lambda-remote-workflow/naics-embedder
```

No suitable attached active worktree existed before creation. The new worktree has neither
held commit in its ancestry. Do not infer the implementation exists merely because this plan
or the remote specification exists.

Read preservation evidence in the primary checkout:
`logs/plan10_worktree/plan10_execution.md` and `review_evidence/progress.md`.
They record independent task and Ultra reviews, output preservation and archive completion.
Local Phase 1 Python runs: 2,314 passed / one skipped on both versions. GitHub Python runs:
2,304 passed / 11 skipped on both versions. Lint was queued at merge after hosted-runner
acquisition failures; it was not confirmed passed. These are historical counts, not Plan 11
verification or predictions of new counts.

Preservation hashes measured during this planning session:

| Primary artifact | SHA-256 |
|---|---|
| `logs/selection_log.jsonl` (7 records) | `dcbbf4fb690ae41841cea68a79f7239c233114c0adc92b956dbf21400eceb953` |
| `checkpoints/plan10_smoke/last.ckpt` | `71bdd986a5e972fce69766e69779d8ddad67de290bc93748cb81cb674e9c2bd5` |
| `checkpoints/plan10_smoke/monitor_reads.jsonl` | `1fce2261e11a23f513f96366f2e0d25a51c1321d4c7769b0b4f7331e8ef1c31a` |
| `checkpoints/plan10_smoke/epoch_summary.jsonl` | `0b1b497d6ee3d9c05ef362116637fba0d48dff2faa5907119c0f53e43757b30a` |

### Approaches considered

1. **Chosen:** typed Python orchestration over SSH/GNU rsync, with durable manifests and a local
   transport. Fits the approved spec and existing CLI, supplies testable guards, and keeps
   instance ownership with the user.
2. Shell-only orchestration would be smaller but duplicate bundle/checkpoint parsing and make
   state recovery and cross-platform tests fragile. Already rejected by the approved spec.
3. SkyPilot/cloud storage would add instance lifecycle and credential/storage dependencies.
   Already rejected; unnecessary for this single-instance workflow.

No new model, loss, sampling, panel or decision behavior belongs in this implementation.

### Clarifications embodied in this draft

Approval includes these concrete interpretations of otherwise ambiguous spec behavior:

1. **Complete restore:** upload the entire verified kept-checkpoint set with both histories.
   The current user contract supersedes Plan 10 Task 19's older paragraph saying only
   `last.ckpt` is uploaded. The continuation source remains exclusively `last.ckpt`.
2. **Fresh directories:** permit absent or actually empty directories, matching the current
   training guard and user brief; refuse any existing file in either run directory. This
   refines the remote spec's stricter “neither directory exists” wording.
3. **Finish tampering:** before the final pull, compare Mac files against the hashes recorded
   by the most recent successful pull. Missing or changed previously synced files stop finish
   before repair. New remote files are normal pending data. This meets the spec's tampering
   test even though final-pull-first alone would repair a changed Mac file silently.
4. **Active training:** same-host `up`, code pushes, bootstrap and input replacement refuse
   while `naics-train` is running. `--force` cannot override this. Rerunning `up` is safe after
   training stops; it reuses a ready session and repairs incomplete preparation.
5. **Config ownership:** `up` also accepts `--config PATH [OVERRIDES...]` so a checkout with
   the committed null manifest can prepare inputs without editing tracked config. Default
   invocation still works from future local main with its private manifest setting.
6. **Resolved remote paths:** resolve the repository/home/checkpoint paths on the instance.
   Compare these literal canonical paths in Mac preflight, rather than resolving `/home/...`
   on macOS. Extend the shared directory guard with a keyword for a trusted resolved path;
   its existing local callers retain their current behavior.
7. **Provenance includes filesystem type:** hash symlink targets without following them, retain
   executable bits, and preserve relative, existing in-repo symlinks (`AGENTS.md -> CLAUDE.md`).
   Refuse external/broken links and symlink ancestors that escape a transport root. Known
   credential paths cause a named refusal, not silent omission from Git's pushed set.
8. **IDs and concurrency:** second-resolution IDs keep the spec's format; collisions refuse
   rather than overwrite records. Use a single local advisory lock for mutable workflow state
   and pulls, plus an atomic instance launch lock. A sync loop is tied to session and process
   identity; finish quiesces it before final verification.
9. **Partial preparation:** journal a `preparing` session after read-only gates and before the
   first remote mutation; mark `ready` only after bootstrap/input verification. A failed push
   keeps the old successful hash baseline and a recoverable pending-push journal. Never allow
   training in a partially prepared state.
10. **No arbitrary remote commands:** the controller requests fixed operations through the
    transport. Metadata probes/scripts are rendered from validated inputs, not shell fragments
    from user overrides. Training argv is rendered once with `shlex.join`.
11. **Transport prerequisite order:** a host without rsync cannot receive bootstrap.sh by
    rsync. The initial SSH prerequisite operation installs/qualifies remote GNU rsync before
    code upload, using the already approved noninteractive apt behavior. Full uv/GPU/NTP
    bootstrap still runs from the pushed script afterward. A prerequisite failure uploads no code.
12. **GPU capability guard:** the user approved adding an explicit native-BF16 gate after sharing
    an availability snapshot containing H100, A10 and A100 instances. The snapshot is context,
    not a fixed SKU requirement or authorization to launch. Use
    `torch.cuda.is_bf16_supported(including_emulation=False)` on logical CUDA device 0, the
    first visible device used by the existing one-device Trainer. Bootstrap and launch must use
    the same explicit `CUDA_VISIBLE_DEVICES` value (or preserve its unset state); capture it in
    records and pass it to the tmux wrapper so probe and training visibility cannot diverge.
    A host with multiple GPUs still trains on one. Record device name, compute capability and
    total VRAM; do not infer that available VRAM proves workload fit. A replacement instance
    may use a different compatible GPU without changing run settings or the checkpoint identity.

Implementation must make these refinements explicit in the remote specification and guide.
It must not change Plan 10's numbering, historical evidence, or completed checkpoint contents.

### Existing interfaces to reuse (merged base line references)

| Interface | Base location | Use |
|---|---|---|
| `Config.from_yaml(path)`, `Config.override(dict)` | `utils/config.py:1368,1385` | Exact config resolution, without silently defaulting a missing file |
| `parse_config_overrides`, `run_settings`, `effective_precision` | `utils/training.py:189,252,234` | Same dotted values, settings and effective CUDA precision as `train` |
| `runtime_contract_for(cfg, bundle)` | `cli/commands/training.py:199` | Bundle, objective, encoder, summaries contract |
| `load_validated_bundle(path)` | `supervision/artifacts.py` | Existing immutable bundle validation; no rebuilding |
| `validate_exact_resume(path, runtime)` | `supervision/checkpoints.py:206` | Current objective/contract check; no retired-key lookup |
| `read_checkpoint(path)` | `utils/training.py:359` | Deserialize on CPU |
| Directory/settings/constructor/stopped-run guards | `utils/training.py:397,435,505,529` | Current exact identity and completion semantics |
| `read_monitor_records(path)` | `text_model/monitor.py:435` | Parse retained monitor records without scoring a panel |
| `read_epoch_summary(path)` | `text_model/epoch_summary.py:69` | Parse health history without a panel read |
| `build_reference_bundle(root)` | `tests/fixtures/supervision.py:607` | Tiny real validated bundle under `tmp_path` |
| `trained_seeds`, `TrainedSeeds.directory(seed)` | `tests/fixtures/checkpoint_runs.py:40,37` | Existing actual tiny CPU checkpoints/histories |
| `guard_the_repository_selection_log` | `tests/conftest.py:121` | Retain the real-log no-write test gate |

The checkpoint contract fields are `contract_version`, `bundle_id`, `codebook_fingerprint`,
`objective`, `encoder`, `summaries`. Missing objective means `pre-req11` and is refused.
The 21 settings remain:

```text
fusion, dimension, radius_bound, code_code_weight, radial_weight, target_temperature,
radial_step, logit_scale_init, logit_scale_range, learning_rate, weight_decay, warmup_epochs,
lr_plateau_factor, lr_plateau_patience, early_stopping_patience, max_epochs, queries_per_step,
accumulate_grad_batches, gradient_clip_val, accelerator, precision
```

Seed is checked separately. Supplementary controls are `lora_r`, `lora_alpha`, `lora_dropout`
always, plus `num_experts`, `top_k`, `moe_hidden_dim`, `load_balancing_coef` only for active MoE.
Missing active controls fail closed; changes to inactive MoE controls do not change identity.

## Workspace, verification and review protocol

Implementation starts in a **fresh GPT-6.1 Medium session after approval**, in the managed
worktree above. Reuse it through using-git-worktrees; do not create another or reset from local
main. Re-read `AGENTS.md` (a symlink to `CLAUDE.md`), the approved spec, this plan, and relevant
test-driven-development/clean-code/clean-coder/execution/review skills before editing Python.
Use user routing instead of incompatible legacy Opus/Sonnet names.

Create ignored `logs/plan11_execution.md` and `logs/plan11_review_evidence/`. Record base/head,
exact command, exit status and output location at every milestone. Each task below requires:

1. Write its tests before production edits; capture the expected behavioral/import failure.
2. Implement the task, capture green, format touched paths, inspect its actual diff, commit.
3. Dispatch a fresh read-only task-reviewer at GPT-6.1 Medium with the task brief, implementer
   report, exact BASE..HEAD and diff file. First check spec compliance, then code quality.
4. Resolve Critical/Important findings, run new regression red/green, re-review the fixed range.
   Minor findings require explicit disposition. Advance only after approval of that task.

If inline execution is chosen, the independent task reviewer remains a separate agent.
Parallelize genuinely independent read-only reviews or tests; do not dispatch simultaneous
writers to shared workflow/state files. A worker owns only its task's listed files and must be
told other agents are present and that their edits must not be reverted.

### Pre-flight (controller, before Task 1)

- [x] Re-fetch origin, inspect branch/worktree state and open PRs. If origin gained overlapping
  changes, reconcile the plan before implementation. Do not bring the held commits into HEAD.

> Deviation: native snapshot recovery restored the approved worktree after planning archival;
> the approved branch/base and protected primary stayed unchanged.

```bash
git -c core.fsmonitor=false fetch --prune origin
pwd
git -c core.fsmonitor=false status --short --branch
git worktree list --porcelain
git rev-parse HEAD origin/main
git log --oneline origin/main..HEAD
git merge-base --is-ancestor 177899c HEAD
git merge-base --is-ancestor a030dfb HEAD
```

Expected: this managed worktree/branch, clean approved-plan commit(s) only; both ancestry checks
exit **1**. Never treat their expected exit 1 as a setup failure or chain them with `&&`.
Record a refreshed `BASE` equal to the remote merge base and the first implementation HEAD.

- [x] Record `uv.lock` hash. Set up `uv sync --locked` without re-locking. Run baseline gates:

```bash
shasum -a 256 uv.lock
uv sync --locked
HF_HUB_OFFLINE=1 uv run --locked pytest -n auto -q
UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/private/tmp/naics-plan11-py310 HF_HUB_OFFLINE=1 uv run --locked pytest -n auto -q
uv run --locked ruff check src/ tests/
./scripts/format_code.sh --check --all
uv run --locked mkdocs build --strict
```

Expected: zero failures; record actual counts/skips/warnings. An isolated worktree lacks real
data, so skips/counts can differ from preserved Phase 1 results. Do not copy real data to alter
counts. Stop on baseline failures and report the evidence before changing unrelated code.

- [x] Resolve GNU rsync from `RemoteConfig.rsync_path` or PATH. Current PATH supplies
  `/usr/bin/rsync` (`openrsync`, protocol 29), which must be rejected. Check installed optional
  `/opt/homebrew/bin/rsync` or `/usr/local/bin/rsync` if present. If none meets >=3.2, document
  `brew install rsync`; implementation unit tests can proceed, but real transport qualification
  cannot be reported passed with skipped tests. Installation is an execution prerequisite,
  not an action performed in this planning session.

- [x] Snapshot primary evidence hashes/counts above and hashes of all files under the preserved
  smoke/evidence directories. Do not modify or copy them into implementation tests.

## File structure and public interfaces

| File | Responsibility |
|---|---|
| `remote/__init__.py` | Package documentation; no eager operations |
| `remote/config.py` | Strict config loading, effective training config, protected paths |
| `remote/session.py` | State/record models, ID allocation, atomic JSON, advisory lock |
| `remote/transport.py` | SSH/rsync and LocalTransport; argv, probes, safe file operations |
| `remote/bootstrap.sh` | Instance uv/dependencies/GPU/tmux/rsync/NTP preparation |
| `remote/code_manifest.py` | Git file entries, hashes/modes/links, deletions and edits |
| `remote/provenance.py` | Reconstructible immutable push record |
| `remote/canonical.py` | Canonical inputs, resume identity, histories, finished predicate |
| `remote/push.py` | Verified pushes and canonical uploads with pending journal |
| `remote/sync.py` | Staged verified pulls, final checksum and Mac-tampering check |
| `remote/loop.py` | Owned detached caffeinate worker, liveness and shutdown |
| `remote/launch.py` | Exact training argv, segment record, launch gate |
| `remote/workflow.py` | Five operations; sequencing, up/finish/status |
| `remote/worker.py` | Internal JSON probe operations and sync-loop entry point |
| `cli/commands/remote.py` | Five Typer command adapters |
| `utils/config.py` | `RemoteConfig` model, separate from model training configuration |
| `utils/training.py` | Small backward-compatible resolved-directory guard extension |
| `conf/remote.yaml` | Approved transport settings |
| `.gitignore` | Ignore `.remote/` and `outputs/remote/` |
| `docs/remote_workflow.md`, `docs/api/remote.md` | Operator and API contracts |

All source paths in this table start at `src/naics_embedder/` unless otherwise stated.
Do not split existing CLI or model modules beyond the listed integration edits.

Shared data contracts, defined by their owning tasks and consumed unchanged by later tasks.
These are constructor signatures; implement each as a frozen dataclass:

```text
# frozen dataclasses unless stated as a Pydantic state model
FileEntry(path: str, sha256: str, size: int, mode: int, kind: str, target: str | None)
GpuEvidence(logical_index: int, name: str, compute_capability: tuple[int, int],
            total_memory_bytes: int, native_bf16: bool, cuda_visible_devices: str | None)
RemoteInfo(repo: str, checkpoint_base: str, uv: str, python: str, ntp: bool,
           accelerator: str, gpu: str, gpu_evidence: GpuEvidence | None = None)
PullMapping(source: str, destination: Path)
InputSet(manifest: Path, descriptions: Path, bundle: ValidatedSupervisionBundle,
         paths: tuple[str, ...], hashes: dict[str, str])
ResumePlan(directory: Path, remote_directory: str, last: Path, epoch: int,
           training_run: str, files: tuple[str, ...], hashes: dict[str, str],
           finished: bool, finish_reason: str | None)
PushRecord(push_id: str, directory: Path, entries: tuple[FileEntry, ...], head_sha: str,
           dirty: bool)
SyncResult(pulled: int, pending: int, last_sha256: dict[str, str])
LaunchResult(segment_id: str | None, skipped: bool, reason: str | None)
FinishResult(safe: bool, abandoned: bool, latest_checkpoint: str | None,
             local_sha256: str | None, remote_sha256: str | None)
```

`RemoteState` stores schema version 1, status (`preparing`, `ready`, `finished`, `abandoned`),
host, session_id, started_utc, remote_info, successful push_id, pending_push_id,
active_segment_id, last_sync_utc, unreachable_since and last sync manifest reference.
`RunRecord` is separately persistent at `.remote/runs/<experiment>.json`: experiment, stable
remote_directory, canonical bundle fingerprints, seed/settings/constructor controls,
training_run once known and originating session/segment. New sessions never erase run records.
Records and returned types use JSON-safe primitives; local paths serialize as strings.
Define shared dataclasses in `remote/session.py`; `InputSet`/`ResumePlan` live in
`remote/canonical.py` because they depend on bundles/checkpoints, `PushRecord` lives in
`remote/provenance.py`, and sync/launch/workflow return dataclasses live with their owners.
Tasks define/import only the types they consume; there must be no forward dependency cycle.

Transport methods, implemented in Task 2, are synchronous and injected into the workflow:

```text
probe(operation: str, payload: dict[str, object]) -> dict[str, object]
push(source: Path, destination: str, files: tuple[str, ...]) -> None
pull(source: str, destination: Path, files: tuple[str, ...]) -> None
remove_code(paths: tuple[str, ...]) -> None
checksum(mapping: PullMapping) -> tuple[str, ...]
launch(script: str, segment_id: str) -> None
interrupt_training(segment_id: str) -> None
```

`probe` permits only `identity`, `transport_prerequisites`, `bootstrap`, `canonical`,
`inventory`, `training`, `gpu`,
`write_record`, `launch_lock`, `edits`, and `clock`. The pre-bootstrap identity probe uses
system Python; post-bootstrap probes use the recorded absolute uv/Python path. LocalTransport
maps instance paths to its temp root and injects process/bootstrap/clock/GPU responses, while
using real rsync for filesystem operations when GNU rsync is available.

## Task 1: Typed state, config, paths and test fixture foundation

**Files:** Create `remote/__init__.py`, `remote/config.py`, `remote/session.py`,
`conf/remote.yaml`, `tests/fixtures/remote.py`, `tests/unit/test_remote_session.py`,
`tests/unit/test_remote_config.py`. Modify `utils/config.py`, `.gitignore`,
`tests/conftest.py` (register fixture module).

**Interfaces:** Produce FileEntry, RemoteInfo, PullMapping, RemoteState and RunRecord;
`load_remote_config(path: Path) -> RemoteConfig`,
`effective_config(root: Path, path: str, overrides: list[str]) -> Config`,
`relative_path(root: Path, value: str) -> str`, `new_id(kind: str, now: datetime,
existing: set[str], head: str = '') -> str`, `read_state(root: Path) -> RemoteState | None`,
`write_state(root: Path, state: RemoteState) -> None`, `state_lock(root: Path)` context manager,
`pull_mappings(root: Path, session_id: str, info: RemoteInfo) -> tuple[PullMapping, ...]`.

- [x] Write tests first. Core cases:

```python
def test_pull_logs_cannot_replace_the_mac_selection_log(tmp_path):
    info = RemoteInfo('/home/ubuntu/naics-embedder', '/home/ubuntu/naics-embedder/checkpoints',
                      '/home/ubuntu/.local/bin/uv', '/usr/bin/python3', True, 'cuda', 'fixture')
    mappings = pull_mappings(tmp_path, '20261005T210000Z', info)
    assert [(m.source.rsplit('/', 1)[-1], m.destination.relative_to(tmp_path).as_posix())
            for m in mappings] == [
        ('checkpoints', 'checkpoints'),
        ('outputs', 'outputs/remote/20261005T210000Z'),
        ('logs', 'logs/remote/20261005T210000Z'),
        ('segments', 'outputs/remote/20261005T210000Z/segments'),
    ]

@pytest.mark.parametrize('name', ['/tmp/escape', '../escape', 'a/../../escape'])
def test_repo_relative_paths_refuse_escape(tmp_path, name):
    with pytest.raises(ValueError, match='repo-relative'):
        relative_path(tmp_path, name)
```

Also pin defaults; unknown keys/missing config refuse; `--config` cannot escape; experiment is
a single nonempty component excluding `.`, `..`, slash/backslash/control characters; duplicate
override values follow existing last-wins parsing; invalid non-`=` tokens refuse explicitly;
atomic state save survives injected failure; ID collision refuses; concurrent lock acquisition
fails with a named busy error; same session mapping and separate run records survive reopening.

- [x] Run the focused red test command:

```bash
uv run --locked pytest tests/unit/test_remote_config.py tests/unit/test_remote_session.py -q
```
  Expected red: absent remote imports or behavior. Capture it before creating production files.

- [x] Implement models with `extra='forbid'`; UTC timestamps; ID `strftime('%Y%m%dT%H%M%SZ')`,
  push suffix `-<shortsha>`; atomic sibling temp-file JSON write, fsync and `os.replace`.
  Use `fcntl.flock(LOCK_EX | LOCK_NB)` held for the transaction. Never unlink someone else's lock.
  Define the session-owned dataclass/state fields above. No persistence of secret contents.

- [x] Add the exact defaults:

```yaml
repo_dir: ~/naics-embedder
sync_interval_seconds: 600
in_flight_seconds: 120
untracked_cap_bytes: 10000000
pulled_directories: [checkpoints, outputs, logs, .remote/segments]
instance_scan_ignore: [__pycache__/, '*.pyc', .pytest_cache/, .ipynb_checkpoints/]
rsync_path: null
```

The four mappings are fixed contractual roots; validate configured directories as that set,
not an opportunity to pull arbitrary instance paths. Ignore `.remote/`, `outputs/remote/`.
Initialize test-only `remote_repo` fixture in `tests/fixtures/remote.py`: a temp Git repo with
tracked `conf/config.yaml`, `conf/remote.yaml`, tiny source and `AGENTS.md -> CLAUDE.md`;
use `build_reference_bundle(tmp_path / 'inputs')`, copy its entire directory and descriptions
under that repo, set relative config paths there. Fixture returns root/config/manifest and does
not operate real inputs. Add `recorded_transport` fake with calls list and queued probe replies.

- [x] Run the focused tests to green; format listed Python paths; task commit
  `feat(remote): define safe session state and configuration`; independent task review.

## Task 2: Transport operations and instance bootstrap

**Files:** Create `remote/transport.py`, `remote/bootstrap.sh`, `remote/worker.py`,
`tests/unit/test_remote_transport.py`, `tests/unit/test_remote_bootstrap.py`;
modify `tests/fixtures/remote.py` for the recorded runner.

**Interfaces:** Produce `SshTransport(host: str, repo: str, rsync_path: str | None,
runner: Callable = subprocess.run)` and `LocalTransport(root: Path, process: object,
rsync_path: str)` implementing the transport methods above. Produce
`gnu_rsync_version(banner: str) -> tuple[int, int, int]`,
`run_probe(operation: str, payload: dict[str, object], root: Path) -> dict[str, object]`.
Add a fixed GPU-capability probe in `remote/worker.py`, reused by bootstrap and launch;
produce `GpuEvidence` from the locked PyTorch environment. Real CUDA transport always requires
non-null qualified evidence; optional evidence supports injected CPU transport fixtures only.

- [x] Write command-construction/bootstrap tests first:

```python
def test_openrsync_is_refused():
    with pytest.raises(ValueError, match='brew install rsync'):
        gnu_rsync_version('openrsync: protocol version 29\nrsync version 2.6.9 compatible')

def test_pull_uses_partial_files_without_deleting(recorded_transport_runner, tmp_path):
    runner = recorded_transport_runner
    transport = SshTransport('ubuntu@192.0.2.1', '/home/ubuntu/naics-embedder', 'rsync', runner)
    transport.pull('/home/ubuntu/naics-embedder/checkpoints', tmp_path, ('run/last.ckpt',))
    argv = runner.calls[-1].args
    assert '--from0' in argv and '--files-from=-' in argv
    assert any(a.startswith('--partial-dir=') for a in argv)
    assert not any(a.startswith(('--delete', '--inplace', '--append')) for a in argv)
```

Provide `recorded_transport_runner` in the test fixture module as a callable returning queued
`subprocess.CompletedProcess` values and retaining args/kwargs/stdin bytes. Cases cover GNU
3.1 refusal and 3.2+ pass, SSH options `BatchMode=yes`, `StrictHostKeyChecking=accept-new`,
connect timeout, quoting spaces/metacharacters as literal argv, changed host key refusal with
manual `ssh-keygen -R` instruction, bounded command timeouts, no credential upload, checksum
itemize parsing, `--from0` input, explicit deletion targets, and destination escape rejection.

- [x] Run the focused red test command:

```bash
uv run --locked pytest tests/unit/test_remote_transport.py tests/unit/test_remote_bootstrap.py -q
```
  record red.

- [x] Implement subprocess argv with `shell=False`. Rsync uses `-rlpt`, `--from0`,
  `--files-from=-`, `--partial-dir=.rsync-partial`, argument protection and explicit roots;
  add `--checksum` for result transfers to catch equal-size/equal-mtime rewrites. Never pass
  `--delete`, `--inplace` or append modes. File list comes through stdin as NUL-separated bytes.
  SSH's remote command is one `shlex.join`-rendered fixed operation; JSON payload travels on
  stdin. The transport alone can invoke SSH/rsync. Initial identity probe reports actual
  `Path(...).expanduser().resolve()` remote roots. No speculative `/home/<user>` construction.

- [x] Add a fixed SSH transport-prerequisite operation before the initial code push: inspect
  remote rsync and, if absent/too old, install the distro rsync package with `sudo -n apt-get`;
  qualify GNU >=3.2. It uses system tools and never imports this unpushed package. Do not rely
  on bootstrap.sh to install the very tool needed to upload bootstrap.sh. Test missing tool,
  unavailable sudo, old distro package and successful rerun using injected executables only.

- [x] Implement bootstrap as `bash` with `set -euo pipefail`, from the pushed source tree.
  Ensure tmux/rsync via noninteractive `sudo -n apt-get` only if missing; errors retain step
  output. Use system uv if available, otherwise the official standalone installer with
  `UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1`, downloaded to a temporary script,
  then executed. Record the resulting absolute uv executable. Run its `sync --locked`; require
  CUDA available and native BF16 on logical device 0, GNU rsync >=3.2, and
  `timedatectl show --property=NTPSynchronized --value` equal to `yes`. Check the lock hash before
  and after sync. Return JSON RemoteInfo; diagnostics go to stderr, not the JSON channel.
  Never run data preparation, panel/export/decision tools, uv lock, or a training smoke here.

- [x] The fixed GPU probe checks `torch.cuda.is_available()`, enters
  `with torch.cuda.device(0)`, and requires
  `torch.cuda.is_bf16_supported(including_emulation=False)` to return true. Do not accept the
  default emulation-inclusive result. Read device properties for name, major/minor capability
  and total memory, and capture CUDA visibility. Missing API, initialization/property failure
  or a false result produces an actionable refusal before training, with no FP16/FP32 switch.
  Persist evidence in bootstrap/session records; do not add it to the 21-key run identity.

- [x] Bootstrap tests run only with fake executables/runner under `tmp_path`: already prepared,
  uv missing, no sudo, apt failure, locked sync failure, CUDA false, NTP false/unavailable,
  shell PATH missing uv, and stdout JSON with diagnostic stderr. GPU cases: native-BF16 true
  passes; CUDA unavailable, emulation-only support, missing API and probe exceptions refuse;
  assert the BF16 call uses `including_emulation=False`. On a two-device fake, check device 0
  even if device 1 differs, and retain the captured visibility and memory/name/capability.
  No real installer/network/apt/GPU is needed; inject the CUDA module/device context.
  `bash -n src/naics_embedder/remote/bootstrap.sh` must succeed.

- [x] Run focused tests green; format; task commit
  `feat(remote): add tested SSH transport and bootstrap`;
  independent task review.

## Task 3: Reconstructible code records and instance-edit detection

**Files:** Create `remote/code_manifest.py`, `remote/provenance.py`,
`tests/unit/test_remote_provenance.py`, `tests/unit/test_remote_code_manifest.py`.

**Interfaces:** `code_entries(root: Path) -> tuple[FileEntry, ...]`,
`deletion_set(old: tuple[FileEntry, ...], new: tuple[FileEntry, ...]) -> tuple[str, ...]`,
`instance_edits(expected: tuple[FileEntry, ...], actual: tuple[FileEntry, ...],
ignore: tuple[str, ...]) -> dict[str, tuple[str, ...]]`,
`write_push_record(root: Path, host: str, push_id: str, cap_bytes: int) -> PushRecord`.

- [x] Write reconstruction and edit tests first:

```python
def test_deleted_code_only_is_removed_from_a_push():
    old = (FileEntry('src/old.py', '1' * 64, 1, 0o644, 'file', None),)
    new = (FileEntry('src/new.py', '2' * 64, 1, 0o644, 'file', None),)
    assert deletion_set(old, new) == ('src/old.py',)

def test_credential_file_refuses_the_whole_push(remote_repo):
    (remote_repo.root / '.env').write_text('fixture=value\n')
    with pytest.raises(ValueError, match=r'\.env'):
        code_entries(remote_repo.root)
```

Round-trip test in a temp Git repo: committed text/executable/symlink/binary files; staged
changes, unstaged changes, deletion and a new filename with spaces/newline; create push record,
clean clone at `head_sha`, `git apply` binary patch, safely extract the generated archive; compare
all hashes/types/modes. Test 10 MB total cap with named untracked paths, exactly-at-cap pass,
external/broken links refuse, unexpected symlink/directory changes reported, ignored runtime
files ignored, unexpected source file detected, and failed record write leaves no successful ID.

- [x] Run the focused red test command:

```bash
uv run --locked pytest tests/unit/test_remote_provenance.py tests/unit/test_remote_code_manifest.py -q
```
  record red.

- [x] Enumerate `git ls-files -z --cached --others --exclude-standard`, deduplicate and sort;
  skip absent cached paths (their deletion is represented in `git diff --binary HEAD`), but
  refuse unsupported special files/submodules. Hash regular bytes and symlink target bytes
  separately using `lstat`. Preserve executable permissions and safe relative symlinks.
  Refuse known credential roots/files `.git/`, `.ssh/`, `.aws/`, `.env`, `.env.*`, private-key
  filenames/extensions; name paths without reading their contents. Require commit/ignore before
  push rather than silently excluding them from the reconstruction promise.

- [x] Write immutable `.remote/pushes/<id>/provenance.json`, `uncommitted.patch`, `untracked.tar`,
  `hashes.json` plus `files.json` (types/modes/targets). Capture full HEAD/branch/dirty, UTC/host,
  untracked path/hash/size and file count. Preserve hashes.json's path-to-SHA schema. Stage the
  record directory and rename only after its file contents and pushed file list stay unchanged.
  Scan all code locations, but classify `.git/`, `.venv/`, `data/`, checkpoints/logs/outputs,
  `.remote/` and approved generated patterns as runtime roots for *new* files. Still inspect
  every previously pushed file even if it lives in a runtime root (tracked outputs exist).
  Under explicit `--force`, previously modified/deleted pushed paths are overwritten/restored;
  named unexpected new code files are removed only after recording their paths/hashes in the
  pending-push discard journal. Never discard files in generated/credential roots or a path
  outside the validated code scan. Without force, every edit category stops the push.

- [x] Run green/format; task commit
  `feat(remote): record reconstructible pushes and detect instance edits`;
  independent task review.

## Task 4: Canonical-input and exact-resume gates

**Files:** Create `remote/canonical.py`, `tests/unit/test_remote_canonical.py`;
modify `utils/training.py` and `tests/unit/test_utils_training.py` for the resolved-directory
keyword only; extend `tests/fixtures/remote.py` with `remote_resume_fixture`.
No model/checkpoint schema changes.

**Interfaces:** `canonical_inputs(root: Path, cfg: Config) -> InputSet`,
`resume_plan(root: Path, cfg: Config, inputs: InputSet, remote_directory: str) -> ResumePlan`,
`finished_run(saved: Mapping[str, Any], patience: int, max_epochs: int) -> tuple[bool, str | None]`;
directory guard gains keyword-only `resolved_dirpath: str | None = None`.

- [x] Write tests first using the real bundle builder and existing tiny trained checkpoints.
  For `remote_repo` fixture, copy the tiny checkpoint directory from `trained_seeds` into its
  `checkpoints/<experiment>`; derive effective config from `trained_seeds.cfg` with relocated
  descriptions/manifest and explicit CPU precision for LocalTransport. Tests for production
  CUDA preflight change both saved/current settings in test copies only and use real callbacks.
  No fake checkpoint is evidence of training correctness.

> Deviation: after user pause and an expired agent token, a fresh implementer recovered preserved
> edits; original selective-red exact argv was unretained, while new measured green/reviews passed.

```python
def test_the_shared_guard_accepts_an_instance_resolved_path(monkeypatch, tmp_path):
    expected = '/home/ubuntu/naics-embedder/checkpoints/run'
    key = outcome_checkpoint(tmp_path).state_key
    saved = {'callbacks': {key: {'dirpath': expected}}}
    refuse_a_resume_from_another_directory(saved, tmp_path, resolved_dirpath=expected)

def test_missing_history_refuses_before_transport(remote_resume_fixture):
    env = remote_resume_fixture
    (env.directory / 'epoch_summary.jsonl').unlink()
    with pytest.raises(ValueError, match='epoch_summary.jsonl'):
        resume_plan(env.root, env.cfg, env.inputs, env.remote_directory)
    assert env.transport.calls == []
```

Matrix: unset manifest (build instruction), invalid bundle, changed parquet byte, escaping
member/config paths; matching req11 passes, wrong bundle/objective/encoder/summaries refused;
retired fields absent accepted; 21 settings/seed each mismatch and missing field refused;
each LoRA/active-MoE change/missing value refused and inactive MoE ignored; missing last refuses
without selected-checkpoint fallback; absent callback state refused; different absolute path;
early stop and exhausted budget skip; malformed epoch/run ID/history/gaps/duplicates/wrong
seed/run/panel/split/nonfinite MRR refused. Interrupted rows beyond last's epoch are permitted
only as valid same-run later rows, passed through untouched for existing resume pruning.

- [x] Run the focused red test command:

```bash
uv run --locked pytest tests/unit/test_remote_canonical.py tests/unit/test_utils_training.py -q
```
  capture red before the guard change.

- [x] Extend the shared guard's expected path selection only:

```python
def refuse_a_resume_from_another_directory(
    checkpoint: Mapping[str, Any], checkpoint_dir: Path, *,
    resolved_dirpath: Optional[str] = None
) -> None:
    '''
    Refuse another saved checkpoint directory, including a resolved instance path (P19).

    Args:
        checkpoint: The loaded checkpoint and callback states.
        checkpoint_dir: The expected local path for ordinary training callers.
        resolved_dirpath: An absolute normalized instance path from the transport identity
            probe, compared literally without resolving it on the Mac.

    Raises:
        ValueError: If the remote path is malformed or the saved callback directory differs.
    '''

    callback = outcome_checkpoint(checkpoint_dir)
    state = (checkpoint.get('callbacks') or {}).get(callback.state_key) or {}
    saved = state.get('dirpath')
    if resolved_dirpath is not None:
        path = PurePosixPath(resolved_dirpath)
        if (not path.is_absolute() or '..' in path.parts
                or str(path) != resolved_dirpath or str(path) == '/'):
            raise ValueError('expected an absolute normalized instance checkpoint directory')
    expected = callback.dirpath if resolved_dirpath is None else resolved_dirpath
    if saved is None:
        raise ValueError(
            f'the checkpoint holds no ModelCheckpoint state on {OUTCOME_MRR}, so an exact resume '
            'would restore no kept epoch: resume a checkpoint this training saved'
        )
    if saved != expected:
        raise ValueError(
            f"the checkpoint was saved in another checkpoint directory, {saved}, not this run's "
            f'{expected}: ModelCheckpoint would restore none of its kept epochs. Resume it '
            'from the directory it was saved in'
        )
```

Import `PurePosixPath` alongside the existing `Path` in `utils/training.py`.
Add tests that simulated Mac canonicalization cannot alter supplied remote path; local callers
without the keyword retain exact previous errors and behavior. Document that only a transport
identity probe supplies this keyword, never an arbitrary unverified override.

- [x] Resolve config with the existing parser. Load bundle using `load_validated_bundle`, hash
  configured descriptions bytes and compare `description_fingerprint`; require all canonical
  paths under repo `data/` and upload full bundle contents, including validated unread members.
  Read last checkpoint on CPU, call `validate_exact_resume(runtime_contract_for(...))`, directory,
  settings/seed and constructor guards. CUDA settings use `effective_precision(cfg, 'cuda')`,
  independent of the Mac's hardware. LocalTransport test hardware is injected explicitly.

- [x] Validate both histories through last's epoch: exactly epochs `0..epoch`, matching finite
  MRR, seed/run ID, outcome validation read identity. Retain original bytes, including valid
  interrupted later rows. Inventory all `.ckpt` files plus histories and existing run config/
  summary artifacts; refuse links or unstable/malformed continuation files. SHA-256 all inputs.
  Do not call `CheckpointRunner.run`, panel score/export/store/decision functions.

- [x] Finished detection calls the existing early-stop guard: its actual stopped callback means
  finished; missing state remains an error. Budget detection uses saved completed nonnegative
  integer epoch + 1 >= unchanged max_epochs. Return finished status before uploads/segments/
  launch. No exception-message parsing: inspect the same saved callback fields after validating
  their existence. Unknown/corrupt completion state refuses rather than restarts.

- [x] Run green/format; task commit
  `feat(remote): guard canonical inputs and exact resume identity`;
  independent task review.

## Task 5: Verified code/input pushes and recoverable remote up

**Files:** Create `remote/push.py`, `remote/workflow.py`,
`tests/unit/test_remote_push.py`, `tests/unit/test_remote_up.py`;
extend `tests/fixtures/remote.py` with `remote_workflow_fixture`.

**Interfaces:** `push_code(root: Path, state: RemoteState, transport: object,
cfg: RemoteConfig, force: bool) -> PushRecord`, `upload_inputs(inputs: InputSet,
transport: object) -> None`; `RemoteWorkflow(root: Path, remote_cfg: RemoteConfig,
transport_factory: Callable, clock: Callable)` with
`up(host: str, config_path: str, overrides: list[str], force: bool = False) -> RemoteState`.

- [x] Write order/failure tests first:

```python
def test_invalid_canonical_inputs_upload_nothing(remote_workflow_fixture):
    env = remote_workflow_fixture
    with pytest.raises(ValueError, match='manifest_path'):
        env.workflow.up('ubuntu@192.0.2.1', 'conf/config.yaml',
                        ['supervision.manifest_path=null'])
    assert not any(call.name in {'push', 'bootstrap', 'remove_code'}
                   for call in env.transport.calls)
```

Define `remote_workflow_fixture` by constructing RemoteWorkflow on remote_repo and recorded
transport; fake clock returns a controlled aware UTC datetime; pending probe replies describe
empty remote tree, stopped tmux, valid input hashes and bootstrap info. Cases: different host
unfinished refused with last-sync time, explicit force records loss risk; finished host makes
new session; same ready host reuses session; active tmux refuses even force; remote edits stop
before overwrite/deletion; force handles only named code edits with a discard journal;
input hash failure; bootstrap
failure remains preparing; interrupted push retry retains old baseline; ready only after
remote canonical validation; reused host lacking session marker requires new session/force.

- [x] Run the focused red test command:

```bash
uv run --locked pytest tests/unit/test_remote_push.py tests/unit/test_remote_up.py -q
```
  record red.

- [x] Implement under the state lock: tool/canonical/unfinished-session/SSH/running gates first;
  then journal preparing state, create immutable push record, scan edits against the last
  successful baseline, ensure remote transport prerequisites, push files, delete only
  previous-code minus current-code, verify all
  hashes/types/modes. Verify the local snapshot did not change during transfer. Upload push
  record; make it successful only after full hash comparison. Pending journal permits rerun
  after partial transfer without hiding unrelated instance edits.
  The explicit-force new-code deletion set from Task 3 is the only additional deletion set;
  record it separately from previous-push deletions and never apply it without force.
  On an initial push into a preexisting repo directory, inspect its code inventory too;
  unexpected code outside the intended push requires the same named edit/force handling.
  A new session does not make preexisting unrecorded source trustworthy.

- [x] Run bootstrap, upload canonical inputs to identical repo-relative paths, validate again
  remotely with `canonical_inputs`. This is a validation operation, never generation. Compare
  byte hashes for every uploaded input. Remote paths resolve under the recorded root. Commit
  ready state/session marker only after all checks; refuse future user/root changes when a
  persisted run's absolute checkpoint location would change.

- [x] Run green/format; task commit
  `feat(remote): prepare verified reproducible training sessions`;
  independent task review.

## Task 6: Coherent pulls and owned background synchronization

**Files:** Create `remote/sync.py`, `remote/loop.py`, `tests/unit/test_remote_sync.py`,
`tests/unit/test_remote_loop.py`; extend `remote/worker.py` internal loop entry and
`tests/fixtures/remote.py` with `remote_sync_fixture`.

**Interfaces:** `sync_once(root: Path, state: RemoteState, transport: object,
cfg: RemoteConfig, final: bool = False) -> SyncResult`,
`verify_local_sync_manifest(root: Path, state: RemoteState) -> None`,
`verify_final_mappings(root: Path, state: RemoteState, transport: object) -> None`,
`ensure_loop(root: Path, state: RemoteState) -> None`, `stop_loop(root: Path,
session_id: str) -> None`, `loop_status(root: Path, session_id: str) -> dict[str, object]`.

- [x] Write pull/coherence/process tests first:

```python
def test_failed_transfer_keeps_the_previous_last_and_histories(remote_sync_fixture):
    env = remote_sync_fixture
    before = {p: p.read_bytes() for p in env.run_files}
    env.transport.fail_next_pull = True
    with pytest.raises(OSError):
        sync_once(env.root, env.state, env.transport, env.cfg)
    assert {p: p.read_bytes() for p in env.run_files} == before

def test_a_busy_history_defers_the_whole_run(remote_sync_fixture):
    env = remote_sync_fixture
    env.instance.touch('checkpoints/run/epoch_summary.jsonl', env.now)
    result = sync_once(env.root, env.state, env.transport, env.cfg)
    assert result.pending > 0 and result.pulled == 0
```

Fixture uses byte-valid tiny checkpoint/history files in two temp roots and recorded manifests.
Cover all mappings, no deletion of older kept files, session logs isolated, unstable source
pre/post inventory, equal-size/timestamp rewritten files, truncation, stable run bundle,
JSONL/checkpoint identity mismatch, background 120-second skip/final no skip, tampered/missing
previous Mac file, checksum itemize difference, sparse mappings before first epoch, retry after
network loss, collision of concurrent sync calls, stale PID and PID reuse, no unowned kill,
finished session worker exit and changed-session worker exit.

- [x] Run the focused red test command:

```bash
uv run --locked pytest tests/unit/test_remote_sync.py tests/unit/test_remote_loop.py -q
```
  record red.

- [x] Inventory remote files with size/mtime/hash/type before copying. In background passes,
  omit files modified inside `in_flight_seconds`; if any member of a run's kept checkpoint set,
  last or either history is busy, defer that entire run. Use a NUL file list, not newline rsync
  filters. Stage under `.remote/pulls/<pass-id>/`, with partial-dir; re-inventory source and
  compare staged hashes. Validate checkpoint/history coherence through last's epoch without
  reading a panel. A failed/unstable run never replaces an existing good Mac generation.
  Before every promotion, also verify existing Mac files against the last successful sync
  manifest; a background loop must report local tampering instead of silently repairing it
  and erasing evidence before finish's own check. Before the first last checkpoint exists,
  treat an incomplete run directory as pending, not a valid generation or a finished run.

- [x] Promote verified files only with same-filesystem atomic `os.replace`. Preserve every
  Mac-only file, including earlier checkpoints. Journal promotions so a process crash midway
  can replay/complete them before any resume/finish; such a pending promotion blocks launching.
  Successful sync metadata/hashes are published only after all intended replacements complete.
  Exclude directory mtime/permission-only noise from content checksum comparisons. No Mac-side
  delete/prune, no merge into `logs/selection_log.jsonl`. Mark unreachable_since on first failure,
  retain last good sync and clear unreachable state after successful recovery.

- [x] Implement detached Mac worker as absolute current Python executable under
  `caffeinate -i`, `start_new_session=True`, stdin DEVNULL, stdout/stderr `.remote/sync.log`.
  Store PID/session/token/process start identity, verify command identity before treating PID as
  live or signaling it. The worker takes the lock per pass, reads current state, retries failures
  next interval and stops on finished/abandoned/different session. No cron or automation needed.
  Status/restart fake the runner; tests never launch real caffeinate or leave background workers.

- [x] Run green/format; task commit
  `feat(remote): pull coherent run artifacts with owned sync lifecycle`;
  independent task review.

## Task 7: Fresh/exact tmux launch and immutable segment records

**Files:** Create `remote/launch.py`, `tests/unit/test_remote_launch.py`;
extend `remote/workflow.py`, fixed worker probes and `tests/fixtures/remote.py` with
`remote_launch_fixture`.

**Interfaces:** `training_argv(cfg: Config, config_path: str, overrides: list[str],
inputs: InputSet, info: RemoteInfo, resume: bool) -> tuple[str, ...]`,
`launch_training(root: Path, state: RemoteState, transport: object, cfg: Config,
config_path: str, overrides: list[str], inputs: InputSet, resume: bool) -> LaunchResult`;
`RemoteWorkflow.train(resume: bool, config_path: str, overrides: list[str]) -> LaunchResult`.

- [x] Write tests first:

```python
def test_finished_resume_never_uploads_or_launches(remote_launch_fixture):
    env = remote_launch_fixture
    env.set_finished_budget()
    result = env.workflow.train(True, 'conf/config.yaml', env.overrides)
    assert result.skipped and result.segment_id is None
    assert not any(c.name in {'push', 'launch', 'write_record'} for c in env.transport.calls)

def test_resume_command_can_only_name_last(remote_launch_fixture):
    env = remote_launch_fixture
    argv = training_argv(env.cfg, 'conf/config.yaml', env.overrides,
                         env.inputs, env.info, True)
    assert argv[argv.index('--ckpt-path') + 1] == 'last'
    assert argv[argv.index('--checkpoint-load-mode') + 1] == 'exact'
```

Fixture composes Task 4's actual tiny resume files, Task 5's ready state, fake remote inventory/
GPU/NTP/tmux/process. Cases: fresh local or remote nonempty directory refuses, empty/new passes;
wrong root/user/dir/settings/constructor guard blocks before GPU launch; all kept checkpoints
and histories uploaded with matching SHA; upload failure or missing history no launch; active
  tmux and concurrent instance launch lock refuse; effective CUDA precision/budget/seed forwarded;
override injection literal; missing stale config, mismatched pushed code/input refuses; unsynced
clock refuses; no decision operations; full segment/provenance records; exit-code wrapper uses
DEVNULL, explicit cwd and absolute uv; malformed user checkpoint/load-mode flags refuse.
Add a bootstrap-pass/launch-fail GPU case: a later native-BF16 failure or probe exception must
prevent tmux creation. Verify identical probe/wrapper CUDA visibility, one training device on a
multi-GPU host, recorded GPU evidence, and no precision fallback or changed checkpoint identity.

- [x] Run `uv run --locked pytest tests/unit/test_remote_launch.py -q`; capture red.

- [x] Resolve effective config; require ready session, no pending push/pull, unchanged pushed
  code and canonical inputs, and tmux stopped. If local code changed, stop with rerun-up
  instruction rather than silently launching stale code. Acquire the instance launch lock,
  recheck stopped/NTP under it, and validate exact remote roots and canonical fingerprints.
  Immediately before tmux creation, rerun the fixed native-BF16 probe using the launch
  visibility and logical device 0; cached bootstrap evidence alone cannot authorize training.
  Capture current launch evidence and set/unset `CUDA_VISIBLE_DEVICES` explicitly in the
  wrapper before invoking training. Fail closed if visibility or qualification cannot be assured.
  Before any fresh launch, call the shared fresh-directory guard on the Mac and its instance
  equivalent. Before resume, call resume_plan; finished returns a clear skipped result without
  upload, record, loop, GPU launch recheck or tmux launch. Otherwise restore the entire run
  file inventory and verify
  every SHA on the instance, then run the shared preflight again there under remote canonical
  paths and CUDA settings. Missing last never falls back to fresh.
  Before restore into a nonempty instance run directory, compare its last checkpoint and
  histories with the Mac generation. A differing/newer instance copy must not be overwritten:
  stop with a sync-first instruction, then re-plan resume from the resulting verified Mac copy.
  Add a regression where the Mac has epoch k and the stopped instance has epoch k+1 or a
  finished last checkpoint; require zero uploads/launches and preservation of instance bytes.

- [x] Build one authoritative argv from validated effective config. Reject remote-reserved
  conflicting path overrides; append canonical repo-relative manifest and absolute checkpoint
  base after validating that the current effective values refer to the same locations.
  Keep run settings unchanged; freeze manifest/paths in the segment's effective-config record.
  Effective accelerator is CUDA in real transport, injected CPU in local integration tests.
  Validate output/log roots before launch: outputs remain under repo `outputs/`, checkpoints
  under the persisted absolute checkpoint base, and OutcomePanelConfig.selection_log under
  repo `logs/`. Token-cache paths remain under repo `data/`. Reject configs that would write
  outside the pulled/canonical mappings instead of silently losing those files. A user's
  checkpoint base may be repo-relative `checkpoints` or the identical probed absolute base;
  the absolute canonical value sent to the instance is authoritative.
  No `--ckpt-path` option exposed to the remote CLI. Render shell script once with shlex:

```python
command = shlex.join(argv)
script = (
    f'cd {shlex.quote(info.repo)} || exit 1\n'
    'set +e\n'
    f'{command} < /dev/null\n'
    'status=$?\n'
    f'printf "%s\\n" "$status" > {shlex.quote(exit_code_path)}.tmp\n'
    f'mv {shlex.quote(exit_code_path)}.tmp {shlex.quote(exit_code_path)}\n'
    'exit "$status"\n'
)
```

`argv` starts with the recorded absolute uv executable, `run --locked naics-embedder train`;
resume adds `--ckpt-path last --checkpoint-load-mode exact`. Launch script in tmux
`naics-train` with a fixed segment-specific wrapper file. Do not interpolate overrides into
shell snippets. Wrapper setup/launch failures release lock and leave a named failed segment.

- [x] Write immutable segment.json with spec §10.2 fields, full argv/rendered command, effective
  config, full input/continuation hashes, remote path, lock hash and copied code record.
  Include fresh GPU evidence and exact CUDA visibility as execution metadata, separately from
  the existing settings/constructor identity; preserve the single-device BF16 campaign settings.
  Persist RunRecord stable path/identity before first launch; update training_run only from
  verified pulled checkpoint. Record exit code atomically even for nonzero training exits.
  Release instance launch lock only after tmux has been created or failure cleanup completes.
  Start/restart the owned sync loop only after successful launch.

- [x] Run green/format; task commit
  `feat(remote): launch fresh and exact runs with tmux provenance`;
  independent task review.

## Task 8: Verified finish, edit rescue, abandonment and status

**Files:** Extend `remote/workflow.py`, `remote/transport.py`, `remote/worker.py`;
create `tests/unit/test_remote_finish.py`, `tests/unit/test_remote_status.py`;
extend `tests/fixtures/remote.py` with `remote_finish_fixture`.

**Interfaces:** `RemoteWorkflow.finish(stop_training: bool = False, pull_edits: bool = False,
abandon: bool = False) -> FinishResult`; `RemoteWorkflow.status() -> dict[str, object]`.

- [x] Write tests first:

```python
def test_tampering_is_detected_before_finish_repairs_it(remote_finish_fixture):
    env = remote_finish_fixture
    good = env.last.read_bytes()
    env.last.write_bytes(b'changed fixture')
    with pytest.raises(ValueError, match='Mac copy changed'):
        env.workflow.finish()
    assert env.last.read_bytes() != good
    assert env.state.status != 'finished'
    assert not any(c.name == 'pull' for c in env.transport.calls)
```

Cases: running refuses; stop-training SIGINT waits boundedly for tmux/process exit, no forced
kill/termination and no safe message on timeout; final pull ignores in-flight window; zero
checksum required over all four mappings; unresolved edits/deletions/new files refuse; pull-edits
rescues files/manifest/tombstones without touching source working tree; repeated rescue does not
overwrite earlier evidence; unreachable cannot return safe; abandon records possible data loss
and stops own worker; invalid abandon+stop/pull combinations refuse; GPU/process/exit/pending/
loop/last-sync/unreachable status; stale PID never signals another process; no runs yet finish
may be safe with checkpoint/hash null and explicitly “no checkpoint”, rather than inventing one.

- [x] Run the focused red test command:

```bash
uv run --locked pytest tests/unit/test_remote_finish.py tests/unit/test_remote_status.py -q
```
  capture red.

- [x] Finish coordinates the state lock and worker: quiesce any owned background pass, take
  lock, check running; optional SIGINT via fixed tmux operation, poll until stopped with bounded
  timeout. Verify previous Mac sync hashes before final pull; run final coherent sync, then
  `rsync --dry-run --checksum --itemize-changes` for each mapping, requiring no file content/type
  differences and no pending promotion. Recheck training stopped before declaring safe.

- [x] Scan instance edits against the successful push snapshot. Rescue to immutable
  `.remote/instance-edits/<session>/<rescue-id>/` with expected/actual entries and deletion
  tombstones, hash-verify it, mark that edit snapshot handled. Changed files never enter the
  working tree; a changed snapshot after rescue requires another rescue. Force permits
  overwriting only in `up` after edits are named; finish has no force-safe switch.

- [x] Stop only the owned sync worker; mark session finished only when all checks succeed.
  Include latest segment/run `last.ckpt` local/remote matching SHA in FinishResult. CLI prints
  exactly `Safe to terminate` only for safe=true. Unreachable finish remains unfinished;
  explicit abandon journals timestamp/last good sync/data-loss warning, closes session and
  returns abandoned=true/safe=false. Status is read-only and reports errors without making
  successful-sync/finish claims.

- [x] Run green/format; task commit
  `feat(remote): verify termination readiness and report session status`;
  independent task review.

## Task 9: Five CLI commands and operator documentation

**Files:** Create `cli/commands/remote.py`, `docs/remote_workflow.md`, `docs/api/remote.md`,
`tests/unit/test_remote_cli.py`; modify `cli/__init__.py`, `tests/unit/test_cli_main.py`,
`docs/.nav.yml`, `docs/usage.md`, `docs/text_training.md`, `CLAUDE.md`,
`specs/lambda-remote-workflow.md`. Preserve AGENTS.md symlink.
Extend `tests/fixtures/remote.py` with `fake_workflow` and the existing/new CliRunner fixture.

**Interfaces:** Register `remote.app`; all five adapters invoke RemoteWorkflow only. `up` and
`train` accept `--config` and trailing override tokens; `up` additionally `--host`, `--force`;
`train` `--resume`; `sync` `--once`; `finish` three flags; `status` no mutable arguments.
Transport settings load from `conf/remote.yaml`; optional group `--remote-config PATH` supports
an explicitly selected GNU rsync. Refusals exit 1; finished skips exit 0 with named reason.

- [x] Write Typer CliRunner tests first: help has all commands; no raw arbitrary command or
  weights-only switch; config/overrides preserved exactly; flags forwarded once; safe vs
  abandoned/error text; error Rich markup escaped; whitespace-normalized output expectations
  compatible with CI's 80-column wrapping; missing host/session/config refuses before transport.

```python
def test_remote_finished_skip_is_success_without_a_launch(cli_runner, fake_workflow):
    fake_workflow.train_result = LaunchResult(None, True, 'saved epoch budget exhausted')
    result = cli_runner.invoke(app, ['remote', 'train', '--resume', 'seed=1'])
    assert result.exit_code == 0
    assert 'saved epoch budget exhausted' in result.output.replace('\n', '')
```

Define fake_workflow fixture as a RemoteWorkflow-shaped object with call arguments and typed
return values; monkeypatch the CLI factory, not the domain guards being tested elsewhere.

- [x] Run the focused red test command:

```bash
uv run --locked pytest tests/unit/test_remote_cli.py tests/unit/test_cli_main.py -q
```
  capture red.

- [x] Add Typer group after existing data/tools registration; retain allocator initialization
  before imports and warning setup. Adapters parse/display; state/bootstrap/transport work stays
  in remote modules. `sync` without once ensures detached loop and returns immediately; once
  performs one verified pass. Do not install dependencies from a CLI help invocation.

- [x] Document commands, exact/fresh/finished behavior, same path/user across instances,
  canonical config and override precedence, whole checkpoint-set transport, tampering refusal,
  edit rescue, stale loop/reachability, GNU prerequisite, NTP fail-closed behavior, first-instance
  qualification and Mac-only reads. Explain the native-BF16 guard, capability-based GPU
  eligibility, single-device use on multi-GPU hosts, recorded VRAM without a workload-fit claim,
  and refusal without precision fallback. Include the following operator examples as documentation,
  never execute them during this plan:

```bash
uv run --locked naics-embedder remote up --host ubuntu@INSTANCE_IP supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
uv run --locked naics-embedder remote train seed=1 experiment_name=stage7-reference-s1
uv run --locked naics-embedder remote sync --once
uv run --locked naics-embedder remote status
uv run --locked naics-embedder remote finish
```

Explain that `train` resolves the current config and canonical inputs; after replacement-instance
up, resume uses the same manifest/config/settings with `--resume`. No budget-extension example.
Record approved draft clarifications in remote spec only. Explicitly replace its §12 optional
manual smoke with a post-merge qualification using a new experiment, never plan10_smoke, before
depending on the tool. No campaign or Stage 7 completion claim from fixture tests.

- [x] Build docs `uv run --locked mkdocs build --strict`; check changed rendered anchors/navigation
  and all five help commands; run focused tests green, format, task commit
  `feat(cli): expose and document the verified remote workflow`; independent task review.

## Task 10: Real local transport qualification and Task 18 acceptance evidence

**Files:** Create `tests/integration/test_remote_workflow.py`,
`tests/unit/test_remote_task18_contract.py`; extend `tests/fixtures/remote.py` as needed;
create `specs/findings/lambda-remote-workflow-readiness.md`. No real training inputs or logs.

**Interfaces:** Same production controller with LocalTransport; process responses injected.
Produce a line-cited readiness matrix for Task 18, recorded against the final implementation SHA.

- [x] Write integration tests first; ensure failures demonstrate incomplete orchestration when
  a substantive guard is disabled, rather than merely counting mocked calls. Use real GNU rsync
  and temporary directories/Git repos. If absent, explicitly skip with GNU prerequisite reason;
  obtain a passing GNU-enabled run before merge qualification can be marked complete.

Test names and assertions:

```text
test_push_matches_git_and_deletes_only_previously_pushed_code
test_instance_edits_are_refused_and_rescued_without_working_tree_changes
test_all_kept_checkpoints_and_both_histories_restore_on_instance_b
test_sessions_keep_logs_outputs_and_remote_selection_logs_separate
test_background_defers_busy_run_and_finish_pulls_it
test_partial_or_mutating_transfer_never_replaces_a_good_run
test_clean_finish_has_zero_checksums_and_tamper_withholds_safe
test_resume_only_last_preserves_absolute_path_and_skips_finished
test_resume_preflight_honors_lora_and_active_moe
test_pending_promotion_recovery_blocks_launch_until_coherent
```

Construct a new actual tiny CPU interrupted run under `tmp_path` with the existing reference
Trainer fixture patterns if needed. Inject interruption after a saved completed epoch with
unchanged larger budget, not a completed three-epoch fixture presented as unfinished. For a
replacement instance simulation, both LocalTransports expose the *same logical absolute remote
root*, mapped to different temp physical directories; supplementary path guard uses that
logical root while actual test Trainer files remain under tmp_path. Do not modify a checkpoint
dirpath or move real checkpoints to make a test pass. A separate existing real Trainer resume
test remains the numerical continuation gate; local transport tests prove byte preservation,
identity gates and sequencing.

- [x] Run the focused red test command:

```bash
uv run --locked pytest tests/integration/test_remote_workflow.py tests/unit/test_remote_task18_contract.py -q
```
  capture red by exposing tests before their necessary final qualification support changes.
  If all implementation already supports them, verify sensitivity by disabling one relevant
  guard in a temporary diff, observe its expected failure, then restore before green/commit.

- [x] Complete test fixtures/production fixes only for demonstrated failures. Qualify no-network
  bootstrap/process stubs so they cannot call apt/installers or launch a real tmux session.
  Task 18 tests spy on exported decision APIs and fail if workflow calls any panel/export/store/
  QCEW/decision operation. Loading/validating the bundle and history is allowed.

- [x] Run GNU-enabled green on Python 3.10 and 3.12, and record integration non-skip counts.
  Create readiness finding with each row below citing implementation file/line and test node,
  not a blanket claim that “Task 18 passed”. Task 18 itself runs later from merged local main.

| Plan 10 Task 18 requirement | Delivering tasks | Required proof |
|---|---|---|
| 1: fresh overrides/name/tmux/DEVNULL | 1, 7, 9 | Exact argv and wrapper tests |
| 2: every kept ckpt, last, both JSONLs, logs | 6, 7, 10 | Two-instance byte/hash transport |
| 3: NTP synchronized | 2, 7 | False/unavailable gate, launch recheck |
| 4: Mac-only decisions; separate remote log | 1, 6, 9, 10 | Mapping and prohibited-call spies |
| 5: no weights-only start | 4, 7, 9 | No mode/flags/path and old objective refusal |
| 6: histories restored before last continuation | 4, 7, 10 | Upload-set hash and order |
| 7: one absolute checkpoint path | 1, 4, 5, 7 | Replacement root/path refusal |
| 8: fresh empty/new directory only | 4, 7 | Mac and instance collision tests |
| 9: finished run never relaunched | 4, 7, 10 | Stop/budget skip, no upload/launch |
| 10: dropped fields absent from pre-check | 4, 9 | Current-contract fixture and source inspection |
| Supplementary LoRA/active-MoE guards | 4, 7, 10 | Changed/missing active controls refuse |
| Native BF16 on selected training device | 2, 7, 10 | Emulation-only/probe errors refuse; launch rechecks device 0 with identical visibility |
| R1–R11; ten seeds; δ=3 SD; no early selection | 9, 11 | Documentation/handoff and no decision calls |

- [x] Update only measured file/test counts in CLAUDE.md/tests README if new files change them;
  do not rewrite historical Phase 1 counts. Format/test green; task commit
  `test(remote): qualify complete transport and Phase 2 launch contracts`; independent task review.

## Task 11: Final verification, Ultra branch review and merge handoff

**Files:** No new runtime behavior. `logs/plan11_execution.md` and review evidence ignored;
readiness finding records the implementation SHA/evidence references before verification;
the final reviewed HEAD is recorded in the ignored ledger to avoid a self-referential commit.
Correct test/docs failures only after
reproduction and appropriate independent fix review.

- [x] Run complete local verification on the frozen candidate HEAD:

```bash
HF_HUB_OFFLINE=1 uv run --locked pytest -n auto -q
UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/private/tmp/naics-plan11-py310 HF_HUB_OFFLINE=1 uv run --locked pytest -n auto -q
uv run --locked ruff check src/ tests/
./scripts/format_code.sh --check --all
uv run --locked mkdocs build --strict
bash -n src/naics_embedder/remote/bootstrap.sh
git diff --check origin/main...HEAD
git diff --exit-code origin/main...HEAD -- uv.lock conf/config.yaml conf/graph.yaml
git log --oneline origin/main..HEAD
```

Run GNU integration suite explicitly on both versions, with configured executable in fixture
discovery, to distinguish transport tests actually passing from skips. Record node collection,
actual failures/skips/warnings and changed rendered links. Run all five `remote ... --help`
commands; they must not mutate .remote, inputs or selection logs. Recheck primary branch/HEAD,
two held commits, preserved evidence hashes and selection-log count/hash against pre-flight.

- [x] Audit code boundaries: only transport.py invokes SSH/rsync; no result pull passes delete;
  remote worker has no panel scoring/export/QCEW/store/decision command; no retired resume key
  is required; all kept checkpoint/history sources map coherently; no sealed split call added.
  Inspect actual branch diff, not just log subjects, for local manifest pins or retired settings.

- [x] Dispatch fresh read-only **code-reviewer**, `gpt-6.1-sol`, `ultra`, fork context none.
  Give this full plan/spec/user contracts, precise BASE..HEAD, complete diff package, task-review
  results, local verification and preservation evidence. Review the entire branch, including
  transport race/failure recovery, data boundaries, active constructor controls and docs.
  Hold HEAD/tree steady. GPT-6.1 Medium handles any fixes with red/green and scoped independent
  review; rerun Ultra for changed scope before claiming final branch approval.
  Skip a same-family Codex CLI second opinion per requesting-code-review/codex-review.md;
  GitHub Codex review is not requested or required. Neither replaces the independent Ultra gate.

> Deviation: explicit GPT-6.1 Medium/Ultra routing superseded legacy Claude names; the unavailable
> typed reviewer used an independent read-only default seat with the same review contract.
> Additional WB1–WB3 fixes received regression red/green, Medium review and Ultra re-review.

- [x] Prepare a concrete PR description with problem/behavior, canonical/resume guarantees,
  all measured local gates and the Task 18 readiness matrix. Get the user's go-ahead to push
  only `codex/lambda-remote-workflow`, never main/held commits. Create PR against origin/main,
  attach it with the native artifact tool. No publishing or PR creation happens in planning.

- [x] Require new PR CI `lint`, `test (3.10)`, `test (3.12)` to pass at the reviewed head.
  Queued/cancelled/acquisition-failed is not passed; retain and report infrastructure evidence,
  retry failed infrastructure jobs without changing valid assertions. Do not loosen exact
  equality for host BLAS differences without measured failures and user adjudication.
  User merges, or explicitly authorizes exact-head merge. Check PR still open before follow-ups;
  if already merged, fixes require a new origin/main branch/PR.

> Deviation: actual CI exposed host-tool lookup and Rich wrapping fixture failures; controlled
> local red/green led to a test-only portability fix, independent reviews and new successful CI.

- [x] Apply writing-plans completion protocol for **Plan 11 only** after tasks/reviews/gates
  resolve: explicit leftover dispositions, completed markup, relevant deferred-item ticking,
  backlog report, retirement if appropriate. Do not tick/retire Plan 10 or Stage 7.
  Leave `specs/lambda-remote-workflow.md` at its current path as Plan 10's live contract through
  Phase 2; mark implementation landed there without claiming its manual qualification passed.
  Real-instance qualification remains explicitly pending; do not mark unexecuted manual checks
  passed. Keep the readiness finding clear about fixture qualification versus Lambda readiness.

> Deviation: the user had already merged PR #126, so tracked completion uses a new origin/main
> documentation branch/PR; its pending administrative gates are separate from implementation.

- [x] Before native archival, preserve Plan 11 ignored execution/review evidence into a new
  `logs/plan11_worktree/` directory in the primary checkout with verified file hashes and no
  overwrites. Never append implementation fixture logs to the primary selection log. Archive
  only this managed worktree after merge and evidence preservation; private commits untouched.

> Deviation: complete evidence preservation and independently approved native implementation
> archival preceded final markup in this separate checkout, so archival is now an actual result.

## Post-merge boundary: fresh Plan 10 Phase 2 session

This is a handoff, not a Task 11 action. Start a fresh session at GPT-6.1 Medium after the remote
implementation merges. Run Plan 10 Task 18 onward in order, inline, from the primary checkout on
local main. No campaign agents, no replay in the implementation worktree.

1. Inspect git/worktree state, fetch origin, confirm monitor and remote CLI implementation landed.
2. Sync local main by replaying **only** held config/graph-config commits onto origin/main.
   Verify the actual diff; expected config change is the local manifest path only, plus the
   known graph config edit. Stop on conflicts, additional commits or restored retired keys.
   Never push these commits. Do not assume Phase 1's earlier simulated clean replay proves this
   later replay clean after remote changes.
3. Execute Task 18's ten implementation checks with current line citations; check its lock,
   dependency, artifact nonexistence and bundle/text-only pins; selection log remains seven
   before new authorized campaign reads. Do not rerun the Phase 1 log append or finished smoke.
4. Freeze `uv.lock` from first campaign launch through the last decision using these margins.
   Train exactly ten reference seeds with shipped settings; δ=3 SD fixed before selection.
5. User supplies/launches/terminates each Lambda instance. Real-instance bootstrap/native BF16/NTP,
   tmux and cross-instance continuation must be qualified before relying on unverified image
   behavior. Use a new dedicated qualification experiment if the user chooses the manual
   two-instance smoke; never use plan10_smoke or change a campaign seed's epoch budget.
6. Training/within-run monitor on Lambda; all exported tables, QCEW/regressor reads, artifact
   records, margins and decisions on the Mac. Remote logs stay under session roots. Never open
   test/outer splits. Candidate selection starts only after reference margins exist.
7. Plan 10 retirement and Stage 7 completion wait until Phase 2 and its final gates finish.

## Spec coverage, technical references and approval

Self-review coverage: spec §§1–5 constraints/design -> Global Constraints/decisions; §6 modules
-> file structure/Tasks 1–9; §7 up/push/train/sync/finish/status -> Tasks 5/3/7/6/8/9;
§8 all guards -> Tasks 1–8; §9 failure handling -> Tasks 2/5/6/8;
§10 reconstructible code/segment records -> Tasks 3/7; §11 unit/integration -> every task/10;
§12 manual smoke and §13 image uncertainties -> explicit post-merge qualification boundary.
Task 18's ten checks plus supplementary constructors -> the Task 10 matrix. Related deferred
items were inspected; no unrelated backlog item is promoted by this plan.

Primary-source references checked during planning:

- [Pinned PyTorch 2.9.1 CUDA implementation](
  https://github.com/pytorch/pytorch/blob/v2.9.1/torch/cuda/__init__.py#L171-L193):
  the native-BF16 guard uses `including_emulation=False` on the current selected CUDA device.
- [GNU rsync manual](https://download.samba.org/pub/rsync/rsync.1): NUL file lists,
  partial-directory replacement, checksum/itemize and argument protection.
- [uv installer options](https://docs.astral.sh/uv/reference/installer/): explicit installer
  directory and avoiding shell-profile changes on an ephemeral image.
- [systemd timedate1 source documentation](
  https://github.com/systemd/systemd/blob/main/man/org.freedesktop.timedate1.xml):
  `NTPSynchronized` reports the kernel synchronization state. Installer/image compatibility
  remains a real-instance qualification item, not an assumption certified by fixture tests.

**Approval:** the complete architecture, twelve clarifications, native-BF16 guard, task sequence
and gates were approved on 2026-10-05. The planning statement that no runtime source, canonical
artifact, campaign record or primary branch was changed described draft preparation. Completed
implementation and its measured gates are recorded above; manual qualification remains pending.
