# Lambda remote workflow fixture readiness

Substantive implementation SHA: `304bc34f07b462380e9d587cf5e9282ebcf511fc`. This finding is a later documentation-only
commit and does not claim its own commit hash. Evidence covers fixture qualification of Plan 11
Task 10. **Plan 10 Task 18 has not run. Real Lambda/Mac qualification remains pending.**

Task 11 corrected only a symlink-rescue test's lexical path discovery for Python 3.10; the
runtime and link-type/link-text/source-isolation assertions are unchanged. The original Task 10
qualification below remains historical evidence at `16e2a927b076d0b9942d6b3f8a64692b674e86ba`.
Historical Task 11 GNU requalification at `ca9b384a3eccd183d7db500a74139dd2b59e040e`
passed 38 nodes with zero skips and
72 visible CPU-fixture warnings on each Python version; ignored Task 11 evidence records the
reproduced failure, scoped fix and fresh final branch gates. Manual qualification remains pending.

## Whole-branch WB1–WB3 qualification

The substantive fix preserves a session-owned cumulative Mac manifest with immutable original
manifest references and their SHA256 values. Replacement sessions retain checkpoint/history,
output and earlier session-log paths. Original manifest bytes remain immutable; later successful
pulls may update the current cumulative file hashes. Missing, foreign, changed or unhashed
referenced evidence refuses transfer, exact resume and finish. Manifest reads and publication
use anchored descriptors and no-follow operations, and pending recovery checks that the journal
retains cumulative paths and inherited provenance before promotion.

The fixed production upload preparation uses only system Python and standard-library descriptor
operations before the package exists. Its `prepare_results` operation has the closed four-root
set `checkpoints`, `outputs`, `logs`, `.remote/segments`; all directory creation remains contained
under the recorded repo without following links. Sparse `up` → `finish` creates no segment
records and records null checkpoint/hash while executing all four actual zero GNU checksum gates.
GNU rsync's supported minimum remains 3.2.

A shared continuation-set validator now guards staged pulls and local exact resume. It requires
every authoritative saved ModelCheckpoint reference and the earliest-best monitor epoch, checks
literal directories, containment, run/seed/epoch consistency, and carries every extra present
kept checkpoint. Real Trainer checkpoint paths, epochs, budgets and bytes remain unchanged.

The required GNU 3.5.1 suites actually passed **47 nodes, zero skips** on Python 3.12.12 and
3.10.19: **27 integration** plus **20 contract** nodes. Each emitted 82 unsuppressed Lightning
warnings: 41 available-MPS/CPU-fixture notices and 41 low-worker notices. The affected remote
suite passed **673 nodes, zero skips**, with 90 warnings (45 of each category); its five
pytest-benchmark xdist notices precede the pytest session and are recorded separately. Full Ruff,
YAPF and diff checks passed. Collect-only measured **2,996 nodes**, with unchanged 117 source,
99 unit and two integration file counts. This is historical scoped qualification; the final frozen-head gates and independent
reviews subsequently passed, as recorded below.

Ignored evidence: `wbfix1-affected312-final.log`, `wbfix1-gnu312.log`, `wbfix1-gnu310.log`,
`wbfix1-static-ruff.log`, `wbfix1-static-final.log`, `wbfix1-static-diff.log`, and
`wbfix1-collect.log`. Each matching JSON records exact argv/env/cwd, UTC interval, return code,
and start/end HEAD/status. The ignored whole-branch fix report retains every separate RED,
intermediate failure and GREEN, including sparse missing-source code 23 and Mac `./` checksum
type differences, both manifest ancestor races, and altered pending provenance.

| Boundary | Current source | Regression nodes |
|---|---|---|
| WB1 replacement/integrity/recovery | remote/workflow.py:361; remote/session.py:69; remote/sync.py:69,115,170,233; remote/launch.py:189 | unit/test_remote_launch.py::test_replacement_preserves_verified_mac_bytes; unit/test_remote_launch.py::test_failed_replacement_up_retains_inherited_integrity; unit/test_remote_sync.py::test_inherited_original_manifest_provenance_refuses; unit/test_remote_sync.py::test_inherited_bytes_allow_legitimate_updates_and_recovery; unit/test_remote_sync.py::test_manifest_replacement_symlink_never_follows_external_bytes; unit/test_remote_sync.py::test_inherited_manifest_publication_never_follows_swapped_parent; unit/test_remote_sync.py::test_pending_recovery_cannot_drop_inherited_baseline |
| WB2 production first-upload/sparse finish | remote/transport.py:62,95,314,376; remote/workflow.py:431; remote/sync.py:468 | integration/test_remote_workflow.py::test_production_ssh_first_metadata_upload_with_gnu; integration/test_remote_workflow.py::test_first_workflow_segment_upload_has_no_precreated_parent; integration/test_remote_workflow.py::test_sparse_up_finish_qualifies_empty_owned_roots |
| WB3 authoritative complete set | remote/canonical.py:219,377; remote/sync.py:339,432 | integration/test_remote_workflow.py::test_missing_authoritative_kept_epoch_refuses_before_acceptance; unit/test_remote_canonical.py::test_earliest_best_tie_requires_its_checkpoint_even_when_later_exists; integration/test_remote_workflow.py::test_all_kept_checkpoints_and_both_histories_restore_on_instance_b |

## Historical Task 10 qualification

GNU rsync 3.5.1 at `/opt/homebrew/bin/rsync` ran the production `RemoteWorkflow` with
`LocalTransport`, temporary Git checkouts, real file transfers and injected process boundaries.
Both Python 3.12.12 and 3.10.19 collected and passed **38 nodes, zero skips**: **18 integration**
nodes from the ten specified test names plus **20 contract** nodes. Required-tool mode with an
absent GNU executable fails (one expected setup error), so CI cannot silently skip qualification.
Both matrix jobs install rsync and require this mode; strict locked MkDocs is a PR job. Existing
coverage/lint jobs and documentation deployment are retained.

Each interrupted fixture really fits a tiny CPU Trainer to completed epoch 1, then raises at
the start of epoch 2 under its unchanged five-epoch budget. An actual completed two-epoch
fixture proves finished-run skipping. MoE controls use an actual active-MoE fit. Saved callback
paths, budgets, epochs and checkpoint bytes are never rewritten. Instances A/B expose the same
literal logical root, equal to the actual Mac fixture root, over different physical directories.
Every original kept checkpoint, last and both histories survive by SHA. This tests byte/identity
transport, not numerical continuation: the separate existing real Trainer exact-resume gate
also passed in the affected **302-test** regression.

The final GNU suites each emit 72 unsuppressed Lightning warnings: 36 CPU-fixture notices about
available MPS not being used, and 36 low-worker notices. The affected regression emits 47:
19 of each of those categories, one BF16 summary-size notice, seven expected used-directory
resume notices and one deliberate mid-epoch resume warning. The mid-epoch test verifies refusal.
These describe small fixture execution and tested refusal paths; no warning is hidden.

Evidence in ignored `logs/plan11_review_evidence/`:

- `task10-red-identity.log`: exact focused RED, exit 1, 19 failures/1 pass; bootstrap identity
  exposes the missing logical-root qualification seam.
- `task10-gnu312-final.log`, `task10-gnu310-final.log`: required GNU mode, exit 0, 38 passed each.
- `task10-missing-gnu.log`: required absent tool, exit 1, one expected error/17 deselected.
- `task10-affected.log`: exit 0, 302 passed; numerical resume nodes include
  `tests/integration/test_reference_training.py::test_an_exact_resume_follows_the_uninterrupted_run`
  for both `in-the-warmup` and `while-early-stopping-waits`.
- `task10-acceptance-regression.log`: existing bootstrap/up/CLI qualification, exit 0, 91 passed without warnings.
- `task10-ruff-final.log`, `task10-layout-final.log`, `task10-docs.log`, `task10-bash.log`,
  `task10-yaml.log`: full Ruff/layout, strict docs, shell syntax and workflow YAML gates.
- `task10-collect-final.log`: 2,939 collected nodes. Inventory is 117 source Python files,
  99 unit test files and two integration files; counts are not coverage percentages.

Frozen `uv.lock` SHA256: `4167042e8a5a8caa9af62973151f681fffb50afaa1a7f6d1f801bd9e58bdac21`.
Exact commands, return codes and full warnings are retained in the ignored Task 10 report/evidence.

## Task 18 implementation matrix

Source references start at `src/naics_embedder/`; test nodes start at `tests/`. Line references
are for the substantive SHA above. Every row is fixture/source evidence, not execution of Task 18.

| Requirement | Delivering tasks | Implementation file:line | Qualified test nodes |
|---|---|---|---|
| 1: fresh overrides/name/tmux/DEVNULL | 1, 7, 9 | remote/launch.py:90,238; remote/worker.py:764 | unit/test_remote_task18_contract.py::test_fresh_overrides_own_name_and_devnull; unit/test_remote_launch.py::test_worker_launch_uses_devnull_explicit_cwd_and_recorded_visibility |
| 2: all kept checkpoints, last, both JSONLs, logs | 6, 7, 10 | remote/canonical.py:320; remote/sync.py:367; remote/session.py:161 | integration/test_remote_workflow.py::test_all_kept_checkpoints_and_both_histories_restore_on_instance_b; integration/test_remote_workflow.py::test_sessions_keep_logs_outputs_and_remote_selection_logs_separate |
| 3: synchronized NTP | 2, 7 | remote/bootstrap.sh:42; remote/launch.py:303 | unit/test_remote_task18_contract.py::test_bootstrap_clock_refuses_unavailable_or_unsynchronized; unit/test_remote_task18_contract.py::test_ntp_launch_recheck_refuses; unit/test_remote_bootstrap.py::test_bootstrap_refuses_failed_sync_or_ntp |
| 4: Mac-only scientific operations; separate remote log | 1, 6, 9, 10 | remote/session.py:161; remote/workflow.py:437; remote/worker.py:626 | integration/test_remote_workflow.py::test_sessions_keep_logs_outputs_and_remote_selection_logs_separate; all workflow cases run with exported panel/export/QCEW/store/margin/decision API failure sentinels in tests/fixtures/remote.py:752 |
| 5: exact-only start; old objective refusal | 4, 7, 9 | remote/launch.py:113,131; remote/canonical.py:365 | integration/test_remote_workflow.py::test_resume_only_last_preserves_absolute_path_and_skips_finished[interrupted]; unit/test_remote_launch.py::test_reserved_overrides_refuse; unit/test_remote_canonical.py::test_wrong_contract_is_refused |
| 6: full histories restored before continuation | 4, 7, 10 | remote/launch.py:225,306,395 | integration/test_remote_workflow.py::test_all_kept_checkpoints_and_both_histories_restore_on_instance_b (hashes recorded in segment before injected launch); integration/test_remote_workflow.py::test_resume_only_last_preserves_absolute_path_and_skips_finished[interrupted] (newer instance refuses with sync-first) |
| 7: same literal absolute checkpoint path | 1, 4, 5, 7 | remote/launch.py:37,201; remote/canonical.py:366; remote/transport.py:415 | integration/test_remote_workflow.py::test_all_kept_checkpoints_and_both_histories_restore_on_instance_b; unit/test_remote_canonical.py::test_other_directory_and_absent_callback_refused; unit/test_remote_up.py::test_checkpoint_config_change_refused_for_persisted_run |
| 8: fresh empty/new directories on both sides | 4, 7 | remote/launch.py:292; remote/worker.py:733 | unit/test_remote_launch.py::test_fresh_nonempty_directory_refuses[local]; unit/test_remote_launch.py::test_fresh_nonempty_directory_refuses[remote]; unit/test_remote_task18_contract.py::test_fresh_overrides_own_name_and_devnull |
| 9: finished run skip before uploads/records/GPU/loop | 4, 7, 10 | remote/canonical.py:147; remote/launch.py:286 | integration/test_remote_workflow.py::test_resume_only_last_preserves_absolute_path_and_skips_finished[finished]; unit/test_remote_task18_contract.py::test_actual_finished_trainer_has_no_upload_records_loop_or_gpu; unit/test_remote_canonical.py::test_early_stop_is_finished_and_corruption_refuses |
| 10: dropped fields unrequired | 4, 9 | remote/canonical.py:365; supervision/checkpoints.py:206 | unit/test_remote_task18_contract.py::test_current_contract_omits_dropped_precheck_keys; unit/test_remote_canonical.py::test_retired_contract_fields_are_absent_and_unrequired |
| Supplementary LoRA and active MoE controls | 4, 7, 10 | remote/canonical.py:372; utils/training.py:518 | integration/test_remote_workflow.py::test_resume_preflight_honors_lora_and_active_moe (7 active changes); unit/test_remote_task18_contract.py::test_missing_active_constructor_control_fails_closed (7 missing controls); unit/test_remote_canonical.py::test_inactive_moe_controls_are_ignored |
| Native BF16, logical device 0, identical visibility, one device | 2, 7, 10 | remote/worker.py:25; remote/launch.py:62,321 | unit/test_remote_bootstrap.py::test_native_bf16_selects_zero_and_captures_evidence; unit/test_remote_bootstrap.py::test_unqualified_gpu_refuses; unit/test_remote_task18_contract.py::test_launch_rechecks_native_device_zero_and_visibility; unit/test_remote_task18_contract.py::test_gpu_probe_error_refuses_launch; unit/test_remote_task18_contract.py::test_one_device_contract |
| R1–R11, ten seeds, δ=3 SD before selection | 9, 11 | docs/remote_workflow.md:195; remote/worker.py:626; remote/workflow.py:329 | all workflow cases run with prohibited-call sentinels; documentation handoff. Actual campaign/Task 18 remains pending. |

## Remaining execution boundary

The primary checkout, held private configuration commits, model settings and lock remain outside
this implementation. No SSH, real tmux/caffeinate, installers, cloud operation, real data/log/
panel/QCEW/export/store/campaign operation or sealed split opening ran in these tests. The tiny
Trainer's within-run outcome validation monitor is fixture-only and finishes before scientific
API failure sentinels are installed for workflow operations.

After merge, use a fresh Plan 10 Phase 2 session from primary local main. Replay only the two
held private config commits locally, inspect the actual diff and stop on conflicts, extra commits
or retired keys. Execute Task 18 onward in order; use a fresh qualification experiment, never
`plan10_smoke`, and never extend a saved epoch budget. Freeze the lock across the ten reference
seeds and fix δ = 3 SD before selection. Remote logs stay separate; exports, QCEW, artifact
records, margins and decisions remain on the Mac; sealed test/outer splits stay closed.

Historical Phase 1 truth remains local 2,314 passed/1 skip, CI 2,304 passed/11 skips, with lint
QUEUED at the merge checkpoint. This finding does not alter those historical counts, retire
Plan 10 or mark Stage 7 complete. Final whole-branch and corresponding PR CI gates passed under Task 11, as recorded below.

## CI portability qualification

Test-only correction SHA: `fef570864a252ef9c11c3ef036f2ab26048faa3c`.
PR 126 CI run 37416387201 attempt 1 at `2d29dedd59b56806c0eca18bc6f070a1772ce700`
had five Python 3.12 failures: both missing-tmux apt failure cases, unavailable NTP, and both
missing-training-config CLI cases. Full output measured 2,980 passes, 11 skips and 80 warnings;
Python 3.10 was cancelled by matrix fail-fast and is not a passed gate. Lint/docs and both GNU
qualification steps passed; each GNU step measured 47 passes, zero skips and 41 warnings.

Controlled local RED reproduced the three bootstrap failures with safe populated-host lookup
and the two CLI failures with genuine narrow-terminal wrapping on both Python 3.12.12 and
3.10.19. Bootstrap RED had three passes alongside its three failures; CLI RED had two normal-width
passes alongside its two narrow-width failures. The corrected fixture exposes only fake
bootstrap tools and allowlisted shell utilities. The CLI assertion preserves the complete missing
path across Rich display line breaks, exit 1, the specific config error and zero transport calls.
Runtime, GPU/NTP guards, configuration, lock, CI gates and numerical assertions are unchanged.

Both complete affected modules passed 72 nodes, zero skips and no emitted warnings on each
Python version with locked offline dependencies. Current collection is 3,001 nodes on each
version, five more than the historical WB qualification above. Full Ruff/YAPF and diff checks
passed. Evidence is retained under ignored `logs/plan11_review_evidence/ciport-*`, including
separate RED, GREEN, collection and command metadata; saved failed/cancelled CI logs remain
under `ci-pr126/`. Final frozen-head gates, independent reviews and corresponding CI subsequently passed; PR 126
merged on 2026-10-06. Real Lambda/Mac Task 18 qualification and the campaign remain pending.


## Final implementation evidence and post-merge status (2026-10-06)

Implementation PR [#126](https://github.com/lowmason/naics-embedder/pull/126) merged at
11:50:29 UTC as `15de7a0310ff3bbfa9157f98ea9459c45741cf34`. Its tree equals final reviewed
`270bc72252077565cbd54e2cf1e86665e47c890f`; substantive runtime SHA remains `304bc34f` above.
All task reviews and independent evidence audits approved. Final GPT-6.1 Ultra round 2 retained
original whole-branch and WB1–WB3 review coverage and approved the complete portability scope.
Its corresponding CI and user-merge conditions are now satisfied.

| Final gate at reviewed head | Python 3.12 | Python 3.10 |
|---|---|---|
| Fresh locked local full suite | 2,999 passed / 2 skipped / 171 warnings | 2,999 passed / 2 skipped / 642 warnings |
| Required GNU 3.5.1 local suite | 47 passed / zero skips / 82 warnings | 47 passed / zero skips / 82 warnings |
| Rendered production preparation cases | 4 passed / zero skips / no warnings | 4 passed / zero skips / no warnings |
| Corresponding PR CI full suite | 2,990 passed / 11 skipped / 80 warnings | 2,990 passed / 11 skipped / 80 warnings |
| Corresponding PR CI GNU suite | 47 passed / zero skips / 41 warnings | 47 passed / zero skips / 41 warnings |

Actual pull_request run **37420123506**, attempt 1, associated with reviewed `270bc722`, passed
all four jobs: docs, lint, test (3.10), test (3.12). GitHub checked out its PR merge ref, not the
literal source-head commit. Each coverage result was **89.90%**, with XML and Codecov upload
successful; no subsequent Codecov processing result is inferred. Fresh local Ruff/YAPF, strict
MkDocs, shell syntax, rendered links/anchors, five help/nonmutation, source and protected gates
passed. Local quiet output did not measure the two skip identities. The additional 471 Python
3.10 numerical warnings match the unchanged historical source/message/count profile; assertions,
tolerances and precision were not changed. The failed/cancelled earlier CI and Phase 1 queued
lint remain historical evidence, not passes.

Preserved evidence lives under primary `logs/plan11_worktree/`: `.sdd/11-lambda-remote-workflow`,
`logs/plan11_execution.md` and `logs/plan11_review_evidence`. Authoritative records include
`completion-boundary.md`, `progress.md`, `whole-branch-review-round2.md`,
`final-ci-preservation-audit.md` and `ci-pr126/head270-final-ci-record.json`. The byte-exact
original copy has **906 files / zero symlinks / 23,172,194 bytes**; preservation receipt SHA256
is `a3d8494c3792fe685c120ddd318c1e8f64a9cddfffc7a45d42e747511b990a67`.
`postmerge-admin/preservation-audit.md` approved the complete copy before archival;
`postmerge-admin/archive-confirmation.json` confirms actual native implementation archival and
absence of its checkout. The original protected 116 files / 198,306,791 bytes and seven Mac
selection records remained unchanged. These ignored records are preserved local evidence,
not rendered public documentation links.

[Plan 11](../plans/completed/11-lambda-remote-workflow.md) is complete, with nothing deferred.
Its separate post-merge documentation review/new PR publication/CI/merge remain administrative
follow-ups and are not claimed here. This finding remains fixture/source qualification only:
real Lambda/Mac image, GPU/native-BF16/NTP/tmux/caffeinate and cross-instance qualification,
Plan 10 Task 18 onward, the campaign and Plan 10/Stage 7 completion remain pending.
