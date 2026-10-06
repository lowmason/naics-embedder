# Remote Training Workflow

The five `remote` commands prepare one instance, launch text training, pull coherent results,
and verify termination readiness. Run them on the Mac from the checkout that owns the canonical
inputs. The user launches and terminates Lambda instances. These commands use SSH and GNU rsync;
they do not launch instances or perform campaign decisions.

The implementation has local fixture tests. Real Lambda image behavior, tmux, Mac caffeinate,
and two-instance continuation still require the [post-merge qualification](#post-merge-qualification)
below before relying on the tool. This guide does not report that qualification passed or that
Plan 10 Phase 2 or Stage 7 is complete.

## Prerequisites and Configuration

Use GNU rsync 3.2 or newer on the Mac (`brew install rsync`); the bundled macOS openrsync is
refused. SSH must work noninteractively with your existing key. A changed host key stops the
command with a manual `ssh-keygen -R` instruction; the workflow never changes known hosts.
The remote prerequisite operation qualifies or installs GNU rsync before uploading code, since
rsync must exist to receive the bootstrap script. Full bootstrap then runs from the pushed tree.

`conf/remote.yaml` holds transport settings, separately from model settings: remote repo,
sync interval (600 seconds), in-flight window (120 seconds), untracked cap (10 MB), fixed pulled
roots and generated-file ignore patterns. To select an installed GNU executable explicitly,
put `rsync_path: /opt/homebrew/bin/rsync` in a transport YAML and select it before the subcommand:

```bash
uv run --locked naics-embedder remote --remote-config conf/remote-local.yaml up --host ubuntu@INSTANCE_IP
```

That example requires the training config to name the canonical manifest. The committed
`conf/config.yaml` keeps its manifest null. Both `up` and `train` accept `--config PATH` (default
`conf/config.yaml`) and trailing `key=value` tokens. Training YAML paths must be repo-relative
and present in the pushed checkout. YAML is applied over model defaults; overrides apply in
order, with the last repeated key winning. Missing files and unknown or malformed keys refuse.
The CLI forwards argument values literally; overrides are never arbitrary shell commands.

Build canonical inputs once on the Mac through the data workflow, then use the exact manifest
path it prints. Do not rebuild them on an instance. The descriptions parquet and the full
validated bundle travel at identical repo-relative `data/` paths. Their fingerprints must match
on both hosts. A changed parquet or bundle requires a fresh run.

## Five Commands

These are operator examples, not commands executed by implementation tests:

```bash
uv run --locked naics-embedder remote up --host ubuntu@INSTANCE_IP supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
uv run --locked naics-embedder remote train seed=1 experiment_name=stage7-reference-s1
uv run --locked naics-embedder remote sync --once
uv run --locked naics-embedder remote status
uv run --locked naics-embedder remote finish
```

`up`'s overrides prepare inputs; they do not become implicit model overrides for later commands.
`train` resolves the current YAML and its own overrides. For the second command above, the current
local config must name the same manifest, or the manifest override must also be supplied to
`train`. A legitimate config/input change requires another `up` before training. Launch validates
that code, config and canonical inputs still match the successful ready snapshot.
The final launch argv appends the verified repo-relative manifest, literal absolute checkpoint
base, recorded accelerator/effective precision and one device after user overrides. Conflicting
reserved path overrides refuse; these authoritative arguments do not change saved run identity.

| Command | Options | Result |
|---|---|---|
| `remote up` | `--host`, `--force`, `--config`, overrides | Push and verify code/inputs; bootstrap; ready session |
| `remote train` | `--resume`, `--config`, overrides | Fresh launch, exact continuation, or named finished skip |
| `remote sync` | `--once` | One verified pull, or ensure detached loop and return |
| `remote finish` | `--stop-training`, `--pull-edits`, `--abandon` | Verified safe result or explicit abandonment |
| `remote status` | None | Read-only JSON observations and errors |

Help only parses/displays the interface: it reads no config/state, bootstraps nothing and
installs no dependencies. Workflow refusals exit 1. A finished resume skips with its named
reason and exits 0. CLI syntax errors use Typer's usage error. There is no arbitrary command,
checkpoint path or checkpoint-load-mode switch on this group.

## Fresh and Exact Runs

A fresh run requires an absent or actually empty `checkpoints/<experiment>/` on both hosts.
Any existing file refuses; choose a new experiment. Exact resume requires Mac `last.ckpt`, every
kept checkpoint and both histories from one verified run. It restores the entire directory,
verifies every hash remotely, and continues exclusively from `last.ckpt`.

Use the same remote user and resolved absolute repository/checkpoint path on replacement
instances. The path stored by ModelCheckpoint cannot be migrated or rewritten. Persistent
`.remote/runs/<experiment>.json` records survive session replacement. A different remote root,
checkpoint base or run identity refuses. A newer/different instance generation refuses before
upload: sync first rather than overwrite it with an older Mac copy.

After finishing instance A and preparing B under the same user/path, pass the same training
config, manifest, seed and settings with `remote train --resume`. No saved epoch budget can be
extended. The checkpoint's current `req11-v1` objective, bundle/codebook, encoder and summaries,
all 21 settings, seed and supplementary LoRA controls must agree. Active MoE also checks expert,
routing and balancing controls. Missing active controls refuse; inactive MoE controls are
ignored. The retired supervision/mining fields are not required. There is no weights migration.
Early-stopped runs and runs that spent their saved budget skip before uploads, segment creation,
loop start or launch. The finished `plan10_smoke` is never a continuation/qualification target.

## Clock, GPU and Launch Qualification

Bootstrap runs locked dependency synchronization and requires NTP synchronization. NTP is checked
again immediately before each launch; false or unavailable synchronization fails closed because
monitor timestamps affect margins-first ordering. The remote lockfile hash is verified.

Bootstrap and immediate launch checks enter logical CUDA device 0 and call
`torch.cuda.is_bf16_supported(including_emulation=False)`. This must return native support;
missing APIs, initialization/property failures, unavailable CUDA and emulation-only support
refuse. There is no GPU model whitelist, VRAM threshold or precision fallback. Device name,
compute capability, total VRAM and `CUDA_VISIBLE_DEVICES` are execution metadata; recorded VRAM
does not prove that this workload fits. A compatible replacement GPU may differ without changing
the run identity.

Probe and launch use the same explicit CUDA visibility (or preserve its unset state). Logical
0 is the first visible device used by the existing one-device Trainer, even on a multi-GPU host.
The wrapper sets/unsets that visibility, uses the recorded absolute uv executable and explicit
repo working directory, and starts the fixed text command under `naics-train` tmux with stdin
from `/dev/null`. Training writes an atomic exit code under its immutable segment record.
GPU utilization in `status` is observation, not authorization to launch.

## Coherent Pulls and Session Separation

| Instance root | Mac destination |
|---|---|
| `checkpoints/` | `checkpoints/` |
| `outputs/` | `outputs/remote/<session>/` |
| `logs/` | `logs/remote/<session>/` |
| `.remote/segments/` | `outputs/remote/<session>/segments/` |

Every kept checkpoint, `last.ckpt`, `monitor_reads.jsonl` and `epoch_summary.jsonl` travel together.
The loop defers the whole run if a member is recent/in-flight or the run is incomplete. Pulls
stage under `.remote/pulls/`, recheck remote inventories, hashes and checkpoint/history identity,
and journal verified promotions. Pending promotions block launch and recover before ordinary
integrity checks. Failed/unstable copies preserve the previous good generation. Pulls never
delete Mac files or prune older kept checkpoints.

The detached Mac loop uses `caffeinate -i`, logs to `.remote/sync.log`, retries failures at the
persisted interval, and stops on a closed/changed session or matching cooperative stop marker.
One-shot sync owns its pass lock. Starting a loop owns the workflow lock. The complete persisted
session configuration controls later operations, including custom GNU rsync/intervals; missing,
foreign or incomplete snapshots refuse rather than silently use defaults.

`status` reports host/session, training/tmux/process observations and exit code, GPU, pending
dry-run paths, loop ownership, last successful sync, pending promotion and unreachable history.
It never repairs evidence. Stale/reused process ownership is reported; restart/signaling applies
only to verified owned wrapper/utility identities. Finish proves those processes exit before
final verification; a timeout retains evidence and refuses readiness.

Only training and the existing within-run outcome validation monitor run on Lambda. Exports,
QCEW/regressor reads, artifact-store operations, margins and decisions stay on the Mac. The
instance's selection log returns under `logs/remote/<session>/`; never merge it into the Mac's
append-only `logs/selection_log.jsonl`. Never open sealed test or outer splits here.

## Finish, Edits and Failure Boundaries

`finish` quiesces the owned loop, verifies previous successful Mac hashes before pulling, checks
training stopped, performs a final coherent pull including recent files, and requires zero
content/type checksum differences across every mapping. A previously synced Mac file changed
or missing is a refusal before repair; preserve that evidence. Pending files/promotions,
unhandled instance edits and a late training/process observation also withhold readiness.

`--stop-training` sends the fixed interrupt to the owned segment, waits within a bounded timeout,
and loses work since the last checkpoint. It never forcibly kills training. `--pull-edits`
rescues changed/new/deleted code evidence into a new immutable
`.remote/instance-edits/<session>/<rescue-id>/`, with verified bytes and deletion tombstones.
It never writes instance edits into the Mac working tree. A changed edit snapshot needs another
rescue; previous rescue evidence is never overwritten.

Only `safe=true` prints exactly `Safe to terminate`, followed by the latest owned segment's
checkpoint and matching local/remote SHA-256. A verified session with no checkpoint says
`no checkpoint` explicitly. An unreachable instance cannot produce a safe result. Explicit
`--abandon` stops the owned Mac loop, journals the last good sync and closes the session with a
possible-data-loss warning; it never prints the safe phrase. Abandon cannot combine with stop
or edit rescue. The user still terminates the instance.

Same-host `up` refuses while training is active, including with `--force`. Force can record the
risk of leaving an unfinished prior host or a reused host marker and overwrite named instance
code edits after recording discard evidence. It cannot bypass canonical/resume/GPU/NTP checks,
active training or final verification. There is no force-safe finish option.

## Post-merge Qualification

After review, verification and merge, qualify image/bootstrap/native BF16/NTP, tmux, caffeinate
and cross-instance continuation before relying on them. Use a **new dedicated experiment** with
an unchanged larger epoch budget, interrupt after a completed epoch while unfinished, finish A,
then restore/resume on B under the same user/absolute path. Verify next-epoch continuation,
history preservation, all kept files and matching hashes. Never relaunch `plan10_smoke`, alter a
saved budget, or use a campaign seed as the qualification experiment. This is pending operational
work; fixture checks do not mark it passed.

Plan 10 Phase 2 starts in a fresh GPT-6.1 Medium session after merge, inline from the primary
checkout `/Users/lowell/Projects/naics-embedder` on local `main`. Fetch and verify the landed paths,
then replay only the two held private config/graph-config commits locally onto current
`origin/main`. Verify the actual diff and stop on conflicts, extra commits or restored retired
keys; never push those commits. The earlier simulated Phase 1 replay is not proof of this replay.

Execute Task 18 onward in order. Preserve the seven existing selection-log records and finished
smoke/evidence; do not repeat Phase 1 exit reads. Freeze `uv.lock` from the first campaign launch
through the last decision using these margins. Train ten reference seeds and fix each panel's
δ = 3 SD on the Mac before configuration selection. Plan 10 retirement and Stage 7 completion
wait for Phase 2's final gates. See [text training](text_training.md#reference-campaign) and
[remote API](api/remote.md).
