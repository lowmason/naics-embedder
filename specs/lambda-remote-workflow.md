# Lambda Remote Workflow

**Status:** APPROVED (2026-09-23) — ready for an implementation plan

**Next skill after approval:** `writing-plans` in a fresh session

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
- Instance bootstrap: uv, `uv sync --locked`, a GPU check, tmux.
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
- Weights-only starts from an older checkpoint (for example after a bundle rebuild). `remote train`
  supports fresh runs and exact resume only; weights-only starts are run by hand for now.

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
  `data/supervision/stage3-supervision-v1/<bundle_id>/manifest.json`. The current bundle,
  `18403d29-3b23-444e-9e81-371d0ca8b7ea`, was built on 2026-09-23 and is pinned to the
  regenerated parquet. Rebuild it only when the parquet changes; a new bundle means older
  checkpoints can only load with `--checkpoint-load-mode weights_only`.
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

### 7.2 `remote up --host USER@IP [--force]`

1. Check tools: `rsync --version` must report GNU rsync 3.2 or newer.
2. Bundle gate on the Mac (§8). Nothing is uploaded until it passes.
3. Unfinished-session guard (§8). A new host, or a host whose session is finished, starts a new
   session.
4. Check SSH: non-interactive, `StrictHostKeyChecking=accept-new`, short timeout.
5. Push code (§7.3).
6. Bootstrap by running `src/naics_embedder/remote/bootstrap.sh` from the pushed tree: install uv
   if missing, `uv sync --locked`, confirm `torch.cuda.is_available()`, and ensure tmux and rsync
   are installed.
7. Upload the canonical inputs to the same repo-relative paths, then validate them on the
   instance with the checks from step 2.
8. Record host, session, and push in `.remote/state.json`.

Rerunning `remote up` on the same host is safe: it pushes only what changed, reruns the bootstrap
(fast once synced), and re-verifies the inputs.

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
   the pushed tree carries it to the instance.
2. Refuse if a training tmux session is already running on the instance.
3. Without `--resume`: fresh-run collision guard (§8).
4. With `--resume`: resume pre-check (§8); upload `checkpoints/<experiment>/last.ckpt`; confirm its
   SHA-256 on the instance.
5. Write the segment record (§10.2) under `.remote/segments/<segment_id>/` on the instance, and
   copy the current push's code record next to it.
6. Launch in tmux session `naics-train`, writing the exit code to
   `.remote/segments/<segment_id>/exit_code` when training ends:

   ```bash
   uv run naics-embedder train [--config PATH] \
     [--ckpt-path last --checkpoint-load-mode exact] \
     supervision.manifest_path=<repo-relative path> [OVERRIDES...]
   ```

7. Start the sync loop on the Mac if it is not running.

### 7.5 `remote sync [--once]`

| Instance | Mac |
|---|---|
| `checkpoints/` | `checkpoints/` |
| `outputs/` | `outputs/remote/<session_id>/` |
| `logs/` | `logs/remote/<session_id>/` |
| `.remote/segments/` | `outputs/remote/<session_id>/segments/` |

- Pulls add or update files; they never delete anything on the Mac.
- rsync writes to a temporary file (`--partial-dir`) and replaces the Mac's copy only when the
  transfer completes.
- Background passes skip files modified on the instance within the in-flight window.
- `logs/train.log` and TensorBoard `version_N/` folders have fixed names that restart on each
  fresh instance, which is why logs and outputs land in per-session folders. Checkpoints keep the
  standard layout because resume looks there; the newest `last.ckpt` replacing the older one is
  intended.
- The loop runs detached under `caffeinate -i` at the sync interval, logging to
  `.remote/sync.log`. A failed pass is logged and retried at the next interval. The loop stops
  when the session is finished.

### 7.6 `remote finish [--stop-training] [--pull-edits] [--abandon]`

1. If training is still running, stop. `--stop-training` interrupts it first; work since the last
   checkpoint is lost.
2. Final pull, without the in-flight exclusion.
3. A checksum comparison (`rsync --dry-run --checksum --itemize-changes`) over every mapping in
   §7.5 must report zero differences.
4. Instance-edit check (§8). `--pull-edits` copies changed files to
   `.remote/instance-edits/<session_id>/` for review; it never touches the working tree.
5. Stop the sync loop, mark the session finished, and print **Safe to terminate** with the latest
   checkpoint's SHA-256 on both sides.

If the instance is unreachable, `finish` never prints "safe". `--abandon` closes the session and
records that anything after the last successful sync may be lost.

### 7.7 `remote status`

Shows the host and session, whether training is running (or its exit code), GPU utilization, the
last successful sync, the number of files still pending (a dry run), and whether the sync loop is
alive.

## 8. Guards

| Guard | Runs in | Checks | If it fails |
|---|---|---|---|
| Tool check | `up` | `rsync --version` reports GNU rsync 3.2+ | Stops with `brew install rsync` |
| Bundle gate | `up`, on the Mac and again on the instance | `supervision.manifest_path` is set; the bundle validates (`load_validated_bundle`); the parquet's SHA-256 equals the manifest's `description_fingerprint` | Stops. If unset: "run `uv run naics-embedder data supervision`, then set `supervision.manifest_path`". If mismatched: explains that the parquet changed and a new bundle makes older checkpoints weights-only |
| Resume pre-check | `train --resume`, on the Mac | `validate_exact_resume` of the Mac's `last.ckpt` against the canonical bundle's runtime contract (bundle ID, codebook, loss and mining versions); the uploaded copy's SHA-256 matches | Stops before any GPU time is spent |
| Fresh-run collision | `train` without `--resume` | The Mac has no `checkpoints/<experiment>/` | Stops; choose a new `experiment_name` |
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
- **Stale sync loop.** A PID file whose process is gone is reported by `status` and restarted by
  `train`.
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
- **Resume pre-check:** a checkpoint whose contract matches the bundle passes; one with a
  different `bundle_id` stops.
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
- New checkpoints arrive and older ones on the Mac survive; session 2's `train.log` lands in its
  own folder without overwriting session 1's.
- A file modified moments ago is skipped by a background pass and picked up by `finish`.
- A clean `finish` finds zero differences; after a Mac copy is tampered with, `finish` reports it
  and withholds "safe to terminate".

## 12. Real-instance smoke checklist (manual, before relying on the tool)

1. On the Mac: `brew install rsync`, build the canonical bundle, set `supervision.manifest_path`.
2. Launch instance A. `remote up --host ubuntu@<A>` passes the tool check, bundle gate, bootstrap,
   and input validation.
3. `remote train training.trainer.max_epochs=1`; `remote status` shows it running and then
   exited 0. `outputs/remote/<session>/segments/<segment>/segment.json` records the `bundle_id`.
4. `remote finish` prints "Safe to terminate". Terminate instance A.
5. Launch instance B. `remote up --host ubuntu@<B>`, then
   `remote train --resume training.trainer.max_epochs=2`. Training resumes exactly at epoch 1,
   and the segment record's `resumed_from` SHA-256 matches the Mac's `last.ckpt`.
6. `remote finish`; terminate instance B. The Mac holds both sessions' logs in separate folders
   and the final checkpoint.
7. Negative checks: an edit made on the instance stops the next push; with
   `supervision.manifest_path` unset, `remote up` stops with the build instruction.

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
