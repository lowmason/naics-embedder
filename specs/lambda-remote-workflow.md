# Lambda Remote Workflow

**Status:** IMPLEMENTATION LANDED (2026-10-06); live contract for Plan 10 Phase 2.
Plan 11 was approved on 2026-10-05 and completed via subagent-driven-development. Implementation
PR [#126](https://github.com/lowmason/naics-embedder/pull/126) merged at
`15de7a0310ff3bbfa9157f98ea9459c45741cf34`, with the same tree as independently Ultra-reviewed
`270bc72252077565cbd54e2cf1e86665e47c890f`. Corresponding CI run 37420123506 passed docs,
lint and both Python jobs: each full job measured 2,990 passed / 11 skipped / 80 warnings,
89.90% coverage; each GNU step passed 47 with zero skips. Fresh local full/GNU/static gates and
independent task/evidence reviews also passed, as recorded in the
[completed implementation plan](plans/completed/11-lambda-remote-workflow.md) and
[fixture readiness finding](findings/lambda-remote-workflow-readiness.md).

**Real Lambda/Mac qualification (§12), Plan 10 Task 18 onward and the campaign remain pending.**
Fixture/source proof does not certify the actual image or process lifecycle. This specification
stays at its current path through Phase 2; Plan 10 and Stage 7 remain open. Post-merge
completion documentation review and its new PR publication/CI/merge are separate administration,
not implementation or manual qualification results.

## 1. Purpose

Text-model training (`naics-embedder train`) runs on fresh Lambda instances reached by SSH from
the Mac. Nothing on an instance survives termination, and one run often spans several instances:
it stops on one and exact-resumes on the next. Today each instance is prepared by hand, `data/`
does not travel with git, and results come back only if someone remembers. That exposes three
failure modes:

1. Stale code silently produces wrong training inputs. On 2026-09-23 the pre-#75 pipeline wrote
   389 of 2,125 codes under the locked environment, and the pre-#76 pipeline left 624
   descriptions ending in a dangling `Illustrative Examples:` marker.
2. A supervision bundle rebuilt on an instance gets a new random `bundle_id` (`uuid.uuid4()` in
   `src/naics_embedder/data/supervision_bundle.py`). Exact resume compares the checkpoint's saved
   contract, including `bundle_id` (`src/naics_embedder/supervision/checkpoints.py`), so the run
   cannot continue on the next instance.
3. Checkpoints, logs, and code edited on the instance are lost at termination unless they are
   copied back first.

This spec adds a `naics-embedder remote` command group. The Mac becomes the single source of
truth for code, the canonical training inputs (descriptions parquet and supervision bundle), and
results. The commands move them to and from each instance and refuse to proceed when an
invariant would break.

## 2. Scope

### 2.1 In scope

- Five commands: `remote up`, `remote train`, `remote sync`, `remote finish`, `remote status`.
- Pushing the Mac's working tree (tracked files plus untracked, non-ignored files).
- Uploading the canonical parquet and supervision bundle, validated on the Mac and the instance.
- Instance bootstrap: uv, `uv sync --locked`, native-BF16 CUDA qualification, NTP, tmux.
- Launching `naics-embedder train` in tmux, fresh or as an exact resume.
- A background pull loop on the Mac and a verified final pull.
- A code record for every push and a record for every training segment.
- The guards in §8, and `.gitignore` entries for `.remote/` and `outputs/remote/`.

### 2.2 Explicitly out of scope

- Launching or terminating instances; the Lambda console keeps that job.
- More than one instance at a time, and multi-node training.
- Cloud-storage transport (§5.4).
- Pruning old checkpoints on the Mac.
- Editing code on the instance as a workflow. Such edits are detected and rescued (§8), not
  supported.
- Training commands other than `naics-embedder train`; the CLI has no other training command.
- Migration from an older objective or supervision bundle. `remote train` supports fresh runs and
  exact resume only; incompatible checkpoints are refused, and nothing migrates (D2).

## 3. Success criteria

1. On a fresh instance, `remote up --host USER@IP` followed by `remote train` (or
   `remote train --resume`) starts training with no other manual step.
2. A run started on one instance exact-resumes on a new instance: training continues at the next
   epoch under the same `bundle_id`.
3. `remote finish` never reports "safe to terminate" while training is running, a synced file
   differs from the Mac's copy, or an edit made on the instance is unhandled.
4. No pull deletes a file on the Mac, overwrites an earlier session's logs or outputs, or leaves a
   truncated checkpoint.
5. Every training segment's code can be rebuilt exactly from its records: the commit, the
   uncommitted diff, and the untracked-file archive.
6. The `remote` commands never regenerate the parquet or the bundle on an instance, and never
   upload a parquet/bundle pair that fails validation.

## 4. Prerequisites on the Mac

- GNU rsync 3.2 or newer: `brew install rsync`. macOS ships `openrsync` (protocol 29, no long
  `--filter` option), which `remote up` rejects with this instruction.
- **The canonical supervision bundle.** Build it once from the main checkout with
  `uv run naics-embedder data supervision` and set `supervision.manifest_path` to its manifest.
  The command prints a repo-relative path,
  `data/supervision/stage3-supervision-v2/<bundle_id>/manifest.json`. The current bundle,
  `301cce28-539c-42ea-8781-496bbdcf511c`, was built on 2026-09-26 by roadmap Stage 5 (plan 7)
  and is pinned to that stage's descriptions parquet (sha256 `fe8c54e3…`). It replaced bundle
  `18403d29-3b23-444e-9e81-371d0ca8b7ea`, whose contract main no longer loads. Rebuild it only
  when the parquet changes; a new bundle requires a fresh run. Stage 7 checkpoints must name
  `objective: req11-v1`; older objectives are refused, and nothing migrates (D2).
- SSH access with the user's existing key: `ssh ubuntu@IP` works without a password prompt.

## 5. Chosen approach and rejected alternatives

### 5.1 Chosen: a `remote` command group over ssh and GNU rsync

Python (Typer) in the existing CLI. `ssh` and `rsync` do the transport; the guards are plain
Python, unit-tested with pytest. It follows the repo's CLI, config, console, and test conventions.

### 5.2 Rejected: shell scripts

Smallest option, but macOS ships bash 3.2, and the hash comparisons, code record, and bundle gate
are fragile in shell and untestable with the repo's test setup.

### 5.3 Rejected: SkyPilot

It would also launch and terminate instances, but it centers on cloud-bucket storage, adds a heavy
dependency and a Lambda API key, and would still need the bundle gate and code record built here.

### 5.4 Rejected for now: cloud storage, or pushing each checkpoint as it is saved

The Mac is usually awake and online during runs. Storage would add credentials to every fresh
instance and a second copy of results to reconcile. Revisit if runs regularly go unattended with
the Mac offline; the transport interface (§6) is where it would plug in.

### 5.5 Rejected: rebuilding inputs on each instance

A rebuilt bundle gets a new random `bundle_id`, which breaks exact resume, and a regenerated
parquet is not guaranteed to be byte-identical to the one the bundle pins.

### 5.6 Rejected: training pushed commits only, or editing and committing on the instance

Pushing the working tree gives the fastest edit-to-GPU loop and keeps git push access off the
instance. The code record (§10.1) keeps uncommitted runs reconstructible, and the instance-edit
guard (§8) covers the habit of editing on the instance.

## 6. Components

| Unit | Responsibility |
|---|---|
| `src/naics_embedder/cli/commands/remote.py` | Typer sub-app registered in `src/naics_embedder/cli/__init__.py` beside `data` and `tools`; parses arguments and prints results, nothing else |
| `src/naics_embedder/remote/transport.py` | The only code that runs `ssh` or `rsync`. `SshTransport` for real use; `LocalTransport`, in which a local directory plays the instance, for tests |
| `src/naics_embedder/remote/canonical.py` | Bundle gate and resume pre-check (§8) |
| `src/naics_embedder/remote/provenance.py` | Code record for each push (§10.1) |
| `src/naics_embedder/remote/code_manifest.py` | Hash lists, deletion sets, and instance-edit detection |
| `src/naics_embedder/remote/session.py` | `.remote/` state; session, segment, and push IDs; the pull path mapping (§7.5) |
| `src/naics_embedder/remote/bootstrap.sh` | Run on the instance from the pushed tree (§7.2 step 6), so no package data is needed |
| `conf/remote.yaml` | Settings, below |
| `.remote/` on the Mac (gitignored) | `state.json`, `pushes/<push_id>/`, `sync.pid`, `sync.log`, `instance-edits/<session_id>/` |

`conf/remote.yaml` settings: remote repo directory (default `~/naics-embedder`), sync interval
(600 s), in-flight window (120 s), untracked-file cap (10 MB), pulled directories, instance-scan
ignore patterns (for example `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ipynb_checkpoints/`),
and an optional rsync path.

`.gitignore` gains `.remote/` and `outputs/remote/`. Without them, pulled results would be pushed
back up as untracked "code" (§7.3) and could be committed by accident; `outputs/` already holds
tracked files, including a TensorBoard event file from a Lambda host.

## 7. Command behavior

### 7.1 Terms

- **Session:** one fresh instance, from `remote up` to `remote finish` (or `--abandon`).
  `session_id` is the session's UTC start time, `YYYYMMDDTHHMMSSZ`.
- **Segment:** one `remote train` launch within a session; `segment_id` is its UTC start time.
- **Push:** one code upload; `push_id` is its UTC time plus the short commit SHA.
- **Canonical inputs:** `data/naics_descriptions.parquet` and the bundle directory that
  `supervision.manifest_path` names.

Commands run from the checkout whose `data/` holds the canonical inputs (normally the main
checkout) and use the host recorded in `.remote/state.json`.

### 7.2 `remote up --host USER@IP [--force] [--config PATH] [OVERRIDES...]`

1. Check tools: `rsync --version` must report GNU rsync 3.2 or newer.
2. Resolve the required repo-relative training YAML plus ordered overrides, then the bundle
   gate on the Mac (§8). Nothing is uploaded until it passes. A missing config never falls back
   to defaults. The group accepts `--remote-config PATH` before the command for transport YAML.
3. Unfinished-session guard (§8). A new host, or a host whose session is finished, starts a new
   session.
4. Check SSH: non-interactive, `StrictHostKeyChecking=accept-new`, short timeout.
5. Journal preparing state, qualify/install remote GNU rsync before the first upload, then
   push code (§7.3). A prerequisite failure uploads no code.
6. Bootstrap by running `src/naics_embedder/remote/bootstrap.sh` from the pushed tree: install uv
   if missing, `uv sync --locked`, require CUDA/native BF16 on logical device 0, and ensure
   tmux and rsync are installed. Check that the instance clock is NTP-synchronized before training: monitor
   read timestamps participate in the margins-first guard.
7. Upload the canonical inputs to the same repo-relative paths, then validate them on the
   instance with the checks from step 2.
8. Record host, session, and push in `.remote/state.json`.

Rerunning `remote up` on the same host after training stops reuses a ready session or repairs
incomplete preparation, pushes changes, reruns bootstrap and re-verifies inputs. Active training
refuses up/bootstrap/code/input replacement even with force. Ready state publishes only after
all verification and the matching session marker. Pending pushes retain the prior successful
baseline and journal; partial preparation cannot launch.

### 7.3 Code push

1. File list: `git ls-files --cached --others --exclude-standard`. Git decides what counts as
   code, and this is the same set the code record describes.
2. Write the code record (§10.1). If untracked files exceed the cap, stop and name them.
3. On a repeat push in the same session, run the instance-edit guard (§8) first.
4. Copy the list with `rsync --files-from`, then delete on the instance the files that were in the
   previous push's hash list but are not in this one. Never use `rsync --delete`.
5. Save the new hash list in `.remote/pushes/<push_id>/` and copy the code record to the same path
   on the instance.

### 7.4 `remote train [--resume] [--config PATH] [OVERRIDES...]`

1. Resolve the effective config (`--config`, default `conf/config.yaml`, plus overrides) with the
   repo's config loader to get `experiment_name`. `--config` must be a repo-relative path, since
   the pushed tree carries it to the instance. Keep one absolute instance checkpoint directory
   per run (`~/naics-embedder/checkpoints/<experiment>/`, resolved under the same user), unchanged
   across segments and instances: exact resume checks the saved ModelCheckpoint `dirpath`.
2. Refuse if a training tmux session is already running on the instance.
3. Without `--resume`: fresh-run collision guard (§8).
4. With `--resume`: resume pre-check (§8); upload `last.ckpt`, `monitor_reads.jsonl` and
   `epoch_summary.jsonl` and every kept checkpoint/remaining run file from the same directory;
   confirm every SHA-256 on the instance before launch. Resume only from `last.ckpt`, never the selected checkpoint or an
   earlier epoch. Skip finished runs before any automatic resume loop: runs ended by early
   stopping or by exhaustion of the saved epoch budget must never be relaunched. Preserve all
   saved run settings, including the epoch budget, and the supplementary saved constructor
   hyperparameters (LoRA always; expert/routing/balancing controls under active MoE).
5. Write the segment record (§10.2) under `.remote/segments/<segment_id>/` on the instance, and
   copy the current push's code record next to it.
6. Recheck NTP and native BF16 with the same CUDA visibility immediately before launch.
   Launch in tmux session `naics-train`, writing the exit code to
   `.remote/segments/<segment_id>/exit_code` when training ends:

   ```bash
   <recorded-absolute-uv> run --locked naics-embedder train [--config PATH] \
     [--ckpt-path last --checkpoint-load-mode exact] \
     supervision.manifest_path=<repo-relative path> [OVERRIDES...] < /dev/null
   ```

7. Start/restart the owned sync loop on the Mac if it is not running. A finished skip exits 0
   with its named reason and does not start a loop or allocate/upload/launch a segment.

`train` resolves its own current config and overrides; up overrides are not implicit train
settings. The final argv appends verified canonical manifest/checkpoint paths, accelerator,
effective precision and devices=1 after user tokens; conflicting reserved paths refuse.
Newer/different instance run generations refuse restore and require sync first. There is no
arbitrary-command, weights migration or checkpoint-path/load-mode switch on the remote CLI.

### 7.5 `remote sync [--once]`

| Instance | Mac |
|---|---|
| `checkpoints/` | `checkpoints/` |
| `outputs/` | `outputs/remote/<session_id>/` |
| `logs/` | `logs/remote/<session_id>/` |
| `.remote/segments/` | `outputs/remote/<session_id>/segments/` |

- Pull each run's entire checkpoint directory: every kept checkpoint, `last.ckpt`,
  `monitor_reads.jsonl` and `epoch_summary.jsonl`. Both JSONL histories travel with checkpoints
  on pulls and resume uploads, preserving the run layout across sessions.
- Pulls add or update files; they never delete anything on the Mac.
- rsync writes to a temporary file (`--partial-dir`) and replaces the Mac's copy only when the
  transfer completes.
- Background passes defer the entire run if any checkpoint/history member is in flight or
  incomplete. Staged hashes, all source inventories and checkpoint/history coherence are
  rechecked before journaled atomic promotion. Failed transfers preserve the good generation.
- Previous successful Mac hashes are checked before every pull/promotion. Changed or missing
  synced bytes refuse before repair; pending promotions recover before ordinary hash checks and
  block launch. Mac-only/older files remain; there is no pruning.
- `logs/train.log` and TensorBoard `version_N/` folders have fixed names that restart on each
  fresh instance, which is why logs and outputs land in per-session folders. Checkpoints keep the
  standard layout because resume looks there; the newest `last.ckpt` replacing the older one is
  intended.
- The loop runs detached under `caffeinate -i` at the sync interval, logging to
  `.remote/sync.log`. A failed pass is logged and retried at the next interval. The loop stops
  on finished/abandoned/changed sessions and cooperative stop. Complete session-owned transport
  config preserves GNU path/intervals; missing/foreign/incomplete snapshots refuse. Without
  `--once`, sync ensures this loop and returns; once performs one public locking pass.

### 7.6 `remote finish [--stop-training] [--pull-edits] [--abandon]`

1. Quiesce only the owned Mac wrapper/utility and prove their exit before the final state lock.
   Stale/reused PID ownership never authorizes unrelated signals. Timeout retains evidence.
2. Recover pending promotions, then verify all previous successful Mac hashes before final pull.
   Changed/missing Mac copies refuse rather than silently repairing evidence.
3. Require training stopped. `--stop-training` sends the fixed interrupt to the owned segment
   and polls bounded tmux/process observations; work after the last checkpoint is lost. No
   forced training kill is permitted.
4. Perform a final coherent pull without in-flight exclusion; pending files/promotions refuse.
5. Check instance edits. `--pull-edits` rescues them under a new immutable
   `.remote/instance-edits/<session_id>/<rescue-id>/`, with verified files, snapshot and deletion
   tombstones. It never touches the working tree. Changed snapshots require a new rescue.
6. Require zero content/type differences across all §7.5 checksum mappings, recheck edits,
   Mac integrity and stopped training, then mark finished.
7. Print exactly **Safe to terminate** only when safe=true, with the latest owned segment's
   checkpoint and matching local/remote SHA-256. Null checkpoint/hash fields explicitly print
   **no checkpoint**, including sessions that trained no completed epoch.

Unreachable finish remains unfinished and never reports safe. `--abandon` stops the owned Mac
worker, journals last good sync/time and possible data loss, closes the session and returns
safe=false/abandoned=true without safe text. It cannot combine with stop-training/pull-edits.
Force only belongs to up; no finish flag bypasses final verification.

### 7.7 `remote status`

Read-only JSON shows host/session, tmux/process observations and exit code, GPU utilization,
last successful sync, pending dry-run paths, pending promotion, loop ownership/errors and
unreachable history. It does not repair evidence. GPU observation cannot authorize a launch.
Refusals/errors exit 1; help reads no config/state, installs nothing and bootstraps nothing.

Only the within-run outcome validation monitor reads on the instance. QCEW slices, the artifact
store, exports and decision-panel reads stay on the Mac. The instance's selection log returns
under `logs/remote/<session_id>/`; never overwrite or merge it into the Mac's decision log during
sync. The Stage 7 campaign waits until its first code PR merges and the remote workflow plan
lands. It uses 10 seeds and fixes each panel's δ at 3 SD on the Mac, with no configuration
selection before δ.

## 8. Guards

| Guard | Runs in | Checks | If it fails |
|---|---|---|---|
| Tool check | `up` | `rsync --version` reports GNU rsync 3.2+ | Stops with `brew install rsync` |
| Bundle gate | `up`, on the Mac and again on the instance | `supervision.manifest_path` is set; the bundle validates (`load_validated_bundle`); the parquet's SHA-256 equals the manifest's `description_fingerprint` | Stops. If unset: "run `uv run naics-embedder data supervision`, then set `supervision.manifest_path`". If mismatched: explains that the parquet changed and a new bundle requires a fresh run; older objectives are refused with nothing migrating (D2) |
| Resume pre-check | `train --resume`, on the Mac | `validate_exact_resume` of the Mac's `last.ckpt` against the canonical bundle's runtime contract (bundle ID, codebook, objective, encoder record and summaries); no reads of the dropped `supervision_mode`, `structural_preference_loss_version` or `mining_contract_version` fields; saved run settings and supplementary LoRA/active-MoE constructor hyperparameters match the current config (missing required values are refused; the 21-key identity is unchanged); absolute checkpoint directory matches; the run is unfinished; both JSONL histories and all kept checkpoints are present and all uploaded SHA-256 values match | Stops before any GPU time is spent |
| Fresh-run collision | `train` without `--resume` | The Mac and instance `checkpoints/<experiment>/` are absent or actually empty | Stops; choose a new `experiment_name` |
| Code record | Every push | Untracked files total at most the cap | Stops and names the files to commit or ignore |
| Instance edits | Repeat pushes in a session, and `finish` | Instance files vs the last push's hash list (modified or deleted), plus new files outside the instance-scan ignore patterns | Stops and lists the files; `--pull-edits` rescues them, `--force` overwrites |
| Unfinished session | `up` on a new host | The previous session was finished or abandoned | Warns with that session's last sync time; continues only with `--force` |
| Finish check | `finish` | Training stopped; final pull done; zero checksum differences; instance edits handled | No "safe to terminate" |

## 9. Failure handling

- **Transfers.** Temporary-file replacement and the in-flight window (§7.5) mean an interrupted
  or mid-write copy never replaces a good file on the Mac.
- **Unreachable instance.** `sync` keeps retrying; `status` shows "unreachable since T" and the
  last good sync; `finish` refuses "safe" and offers `--abandon`.
- **Reused IP.** A changed host key stops the command with the `ssh-keygen -R <IP>` instruction;
  the tool never edits known-host entries itself.
- **Bootstrap failure.** Stops with the failing step's output; `remote up` is safe to rerun.
- **Stale sync loop.** Status reports wrapper identity and stale/reused ownership. Train or sync
  can restart an exited loop; only verified session/process start/command identities authorize
  signals, and successful shutdown proves the wrapper and observed utility exited.
- **Credentials.** None are copied to the instance; nothing there needs git push access.

## 10. Records

### 10.1 Code record, one per push

Stored in `.remote/pushes/<push_id>/` on the Mac and on the instance:

- `provenance.json`: `push_id`, `created_utc`, `host`, `head_sha`, `branch`, `dirty`, the untracked
  files (path, SHA-256, size), and the pushed file count.
- `uncommitted.patch`: `git diff --binary HEAD`.
- `untracked.tar`: the untracked, non-ignored files.
- `hashes.json`: repo-relative path to SHA-256 for every pushed file.

To rebuild: clone, `git checkout <head_sha>`, `git apply uncommitted.patch`, extract
`untracked.tar`. The result must match `hashes.json`.

### 10.2 Segment record, one per `remote train`

`segment.json`: `segment_id`, `session_id`, `host`, `started_utc`, `push_id`, `head_sha`, `dirty`,
`experiment_name`, `bundle_id`, `codebook_fingerprint`, `description_fingerprint`, `resume`,
`resumed_from` (path and SHA-256, or null), and the full training command. It syncs back to
`outputs/remote/<session_id>/segments/<segment_id>/` with a copy of its push's code record.

## 11. Testing

Every test is written first and seen to fail, per the repo's rules.

### 11.1 Unit tests (`unit`; no network or rsync; run in CI)

- **Bundle gate:** build a tiny real bundle in `tmp_path` with the repo's bundle builder. A valid
  bundle passes; an unset `manifest_path` gives the build instruction; changing one byte of the
  parquet gives the mismatch message.
- **Resume pre-check:** a current-objective checkpoint whose contract matches the bundle passes;
  a different `bundle_id` or older objective stops. No check reads the dropped contract fields.
  Tests also enforce identical saved run settings and absolute checkpoint path, resume only from
  `last.ckpt`, both JSONL histories restored before launch, and refusal to relaunch a finished run.
- **Code record round trip:** in a temp git repo, commit, modify a tracked file, and add an
  untracked one. Applying `uncommitted.patch` and `untracked.tar` to a clean checkout of
  `head_sha` must reproduce `hashes.json` exactly. The untracked-file cap has its own test.
- **Guards and state:** hash-list comparison (modified, deleted, new); deletion sets from
  consecutive pushes; unfinished-session and fresh-run collision guards; per-session pull paths;
  the rsync version check (an `openrsync` banner fails, GNU 3.x passes); parsing recorded GNU
  rsync `--itemize-changes` output.
- **Command-line invariants:** pulls never pass `--delete`; pulls always use `--partial-dir`;
  background passes exclude in-flight files and `finish` does not; SSH runs with
  `BatchMode=yes` and `StrictHostKeyChecking=accept-new`.

### 11.2 Local integration tests (`integration`; real GNU rsync; no network)

A temp directory plays the instance and commands run through `LocalTransport`. The tests are
skipped when GNU rsync 3.2+ is not on `PATH`.

- A push matches git's file list; a file deleted on the Mac disappears from the instance;
  `data/` and `checkpoints/` on the instance are untouched.
- An edit on the instance makes the next push stop and name the file; `--pull-edits` copies it to
  `.remote/instance-edits/<session_id>/`.
- New checkpoints and both JSONL histories arrive, older checkpoints on the Mac survive, and
  histories survive transfer to the resumed instance; session 2's `train.log` lands in its own
  folder without overwriting session 1's.
- A file modified moments ago is skipped by a background pass and picked up by `finish`.
- A clean `finish` finds zero differences; after a Mac copy is tampered with, `finish` reports it
  and withholds "safe to terminate".

## 12. Post-merge real-instance qualification (pending)

Review/fixture verification does not certify the actual Lambda image or Mac process lifecycle.
After merge and before relying on the tool, use a **new dedicated qualification experiment**;
never relaunch the finished `plan10_smoke`, use a campaign seed, or extend a saved epoch budget.

1. On the Mac qualify GNU rsync, existing noninteractive SSH and canonical bundle/parquet pins.
   Pass the exact manifest explicitly; do not rebuild the bundle on an instance.
2. User launches A. Up must pass prerequisites, locked bootstrap, NTP, native BF16 and code/input
   verification. Confirm CUDA visibility, one device and recorded name/capability/VRAM without
   interpreting VRAM as workload-fit proof.
3. Train a new experiment with an unchanged larger epoch budget. Confirm tmux, DEVNULL, absolute
   uv/cwd, GPU visibility and exit recording. Interrupt after a completed epoch while unfinished.
4. Finish with stop-training, require all kept checkpoints, last and both histories, matching
   SHA-256 and Safe to terminate. User terminates A.
5. User launches B under the same user/resolved absolute checkpoint directory. Up then exact
   resume uses the same config, seed, manifest, settings and constructor controls. Check the
   entire restored run inventory, last SHA and next-epoch continuation with no duplicate
   surviving history epochs. A newer/different B generation must refuse a stale Mac restore.
6. Finish B, verify session-isolated logs/outputs/remote selection log, and user terminates B.
   Also qualify Mac caffeinate, interrupted transfer recovery, edit rescue, tampering refusal,
   unreachable/abandon boundaries and missing manifest/GPU/NTP fail-closed behavior.

This manual gate remains unexecuted. Plan 10 Task 18 onward starts in a fresh GPT-6.1 Medium
session after merge, inline from primary local main. Fetch current origin/main and replay only
the two held private config/graph-config commits locally. Verify the actual diff, stop on
conflicts/extra commits/retired keys, and never push them. Earlier Phase 1 replay/counts are
historical; queued Phase 1 lint was not confirmed passed. Preserve smoke/evidence and the seven
Mac selection-log records; do not append the four Phase 1 exit reads again. Freeze uv.lock from
first campaign launch through last decision using these margins; ten reference seeds and
δ = 3 SD precede configuration selection. No sealed reads. Plan 10 retirement and Stage 7
completion wait for Phase 2's final gates.

## 13. Open items to confirm during implementation

- How to install uv on the Lambda image. CI uses `pip install uv`; a system Python that enforces
  PEP 668 needs another route. Confirm on the first real instance.
- Whether tmux and rsync ship with the image; the bootstrap installs them with `apt` if not.
- That `uv sync --locked` on the instance installs CUDA-enabled torch wheels that match the
  driver.
- That the tmux command finds uv; call it by absolute path rather than relying on shell startup
  files.
- The first training run downloads `sentence-transformers/all-MiniLM-L6-v2` on the instance;
  optionally prefetch it in the bootstrap so a network problem fails early.

## 14. Approved implementation clarifications (2026-10-05)

These twelve refinements were approved with Plan 11 and govern this live contract:

1. Restore the complete kept-checkpoint/run inventory and both histories; last alone remains
   the continuation source. This supersedes Plan 10 Task 19's older last-only upload wording.
2. Fresh directories may be absent or actually empty; any existing file refuses on either host.
3. Verify previous successful Mac hashes before pull/finish, preserving tampering evidence.
4. Active training blocks same-host up, pushes, bootstrap/input replacement, even with force.
5. Up accepts the same config/overrides as train so committed null manifest need not be edited.
6. Resolve actual remote home/repo/checkpoint paths there, compare literally on the Mac, and
   preserve the stable absolute callback directory. Never invent or rewrite a saved path.
7. Provenance retains filesystem kinds, modes and safe relative existing in-repo symlinks,
   including AGENTS.md -> CLAUDE.md. Refuse escaping/broken links and named credential paths.
8. Second-resolution ID collisions refuse; one local state lock and an atomic instance launch
   lock protect mutations. Sync ownership includes session, token, command and process start.
9. Journal preparing before remote mutation and ready only after verification. Failed pushes
   retain the last successful baseline and recoverable pending journal; no partial launch.
10. Fixed validated transport operations only; metadata is JSON, training argv is rendered once
    with shlex. User overrides never become arbitrary command fragments.
11. Qualify/install remote GNU rsync before uploading bootstrap.sh, then run full uv/GPU/NTP
    bootstrap from the pushed script. A prerequisite failure uploads no code.
12. Require native BF16 via `torch.cuda.is_bf16_supported(including_emulation=False)` on logical
    CUDA 0, the first visible device used by the one-device Trainer, at bootstrap and immediately
    before launch. Use identical explicit CUDA_VISIBLE_DEVICES or preserve its unset state in
    probe and wrapper. Record name, capability and total VRAM as metadata, not workload fit.
    Refuse unsupported/uninspectable CUDA without precision fallback. No whitelist or invented
    VRAM threshold; compatible replacement GPUs may differ without changing the 21-key identity.
