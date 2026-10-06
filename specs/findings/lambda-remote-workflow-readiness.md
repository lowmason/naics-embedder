# Lambda remote workflow fixture readiness

Substantive implementation SHA: `16e2a927b076d0b9942d6b3f8a64692b674e86ba`. This finding is a later documentation-only
commit and does not claim its own commit hash. Evidence covers fixture qualification of Plan 11
Task 10. **Plan 10 Task 18 has not run. Real Lambda/Mac qualification remains pending.**

## Measured qualification

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
| 1: fresh overrides/name/tmux/DEVNULL | 1, 7, 9 | remote/launch.py:90,236; remote/worker.py:759 | unit/test_remote_task18_contract.py::test_fresh_overrides_own_name_and_devnull; unit/test_remote_launch.py::test_worker_launch_uses_devnull_explicit_cwd_and_recorded_visibility |
| 2: all kept checkpoints, last, both JSONLs, logs | 6, 7, 10 | remote/canonical.py:221; remote/sync.py:287; remote/session.py:160 | integration/test_remote_workflow.py::test_all_kept_checkpoints_and_both_histories_restore_on_instance_b; integration/test_remote_workflow.py::test_sessions_keep_logs_outputs_and_remote_selection_logs_separate |
| 3: synchronized NTP | 2, 7 | remote/bootstrap.sh:42; remote/launch.py:301 | unit/test_remote_task18_contract.py::test_bootstrap_clock_refuses_unavailable_or_unsynchronized; unit/test_remote_task18_contract.py::test_ntp_launch_recheck_refuses; unit/test_remote_bootstrap.py::test_bootstrap_refuses_failed_sync_or_ntp |
| 4: Mac-only scientific operations; separate remote log | 1, 6, 9, 10 | remote/session.py:160; remote/workflow.py:429; remote/worker.py:626 | integration/test_remote_workflow.py::test_sessions_keep_logs_outputs_and_remote_selection_logs_separate; all workflow cases run with exported panel/export/QCEW/store/margin/decision API failure sentinels in tests/fixtures/remote.py:726 |
| 5: exact-only start; old objective refusal | 4, 7, 9 | remote/launch.py:113,131; remote/canonical.py:266 | integration/test_remote_workflow.py::test_resume_only_last_preserves_absolute_path_and_skips_finished[interrupted]; unit/test_remote_launch.py::test_reserved_overrides_refuse; unit/test_remote_canonical.py::test_wrong_contract_is_refused |
| 6: full histories restored before continuation | 4, 7, 10 | remote/launch.py:223,304,405 | integration/test_remote_workflow.py::test_all_kept_checkpoints_and_both_histories_restore_on_instance_b (hashes recorded in segment before injected launch); integration/test_remote_workflow.py::test_resume_only_last_preserves_absolute_path_and_skips_finished[interrupted] (newer instance refuses with sync-first) |
| 7: same literal absolute checkpoint path | 1, 4, 5, 7 | remote/launch.py:37,199; remote/canonical.py:267; remote/transport.py:353 | integration/test_remote_workflow.py::test_all_kept_checkpoints_and_both_histories_restore_on_instance_b; unit/test_remote_canonical.py::test_other_directory_and_absent_callback_refused; unit/test_remote_up.py::test_checkpoint_config_change_refused_for_persisted_run |
| 8: fresh empty/new directories on both sides | 4, 7 | remote/launch.py:290; remote/worker.py:728 | unit/test_remote_launch.py::test_fresh_nonempty_directory_refuses[local]; unit/test_remote_launch.py::test_fresh_nonempty_directory_refuses[remote]; unit/test_remote_task18_contract.py::test_fresh_overrides_own_name_and_devnull |
| 9: finished run skip before uploads/records/GPU/loop | 4, 7, 10 | remote/canonical.py:145; remote/launch.py:284 | integration/test_remote_workflow.py::test_resume_only_last_preserves_absolute_path_and_skips_finished[finished]; unit/test_remote_task18_contract.py::test_actual_finished_trainer_has_no_upload_records_loop_or_gpu; unit/test_remote_canonical.py::test_early_stop_is_finished_and_corruption_refuses |
| 10: dropped fields unrequired | 4, 9 | remote/canonical.py:266; supervision/checkpoints.py:206 | unit/test_remote_task18_contract.py::test_current_contract_omits_dropped_precheck_keys; unit/test_remote_canonical.py::test_retired_contract_fields_are_absent_and_unrequired |
| Supplementary LoRA and active MoE controls | 4, 7, 10 | remote/canonical.py:273; utils/training.py:518 | integration/test_remote_workflow.py::test_resume_preflight_honors_lora_and_active_moe (7 active changes); unit/test_remote_task18_contract.py::test_missing_active_constructor_control_fails_closed (7 missing controls); unit/test_remote_canonical.py::test_inactive_moe_controls_are_ignored |
| Native BF16, logical device 0, identical visibility, one device | 2, 7, 10 | remote/worker.py:25; remote/launch.py:62,319 | unit/test_remote_bootstrap.py::test_native_bf16_selects_zero_and_captures_evidence; unit/test_remote_bootstrap.py::test_unqualified_gpu_refuses; unit/test_remote_task18_contract.py::test_launch_rechecks_native_device_zero_and_visibility; unit/test_remote_task18_contract.py::test_gpu_probe_error_refuses_launch; unit/test_remote_task18_contract.py::test_one_device_contract |
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
Plan 10 or mark Stage 7 complete. Final whole-branch and real PR CI gates belong to Task 11.
