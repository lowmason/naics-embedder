# Reference configuration: finding

**Status: FINAL (2026-10-07).** Roadmap Stage 7 (`specs/naics-embedding-roadmap.md`).

The reference configuration trained under R7’s stated defaults on ten seeds. The outcome validation
MRR selected the earliest highest-MRR epoch within each run. The Mac then read each selected
checkpoint on D8’s three validation panels and fixed δ at three sample standard deviations across
the ten seeds (R8). No sealed test or outer split was opened. Nothing compares these results with
Stage 6’s floor (R6), and the structural diagnostics have no adoption target.

## Sources

The canonical bundle is `stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c`. The manifest
was passed explicitly to every bundle-reading campaign command. Canonical inputs were transported
and verified, rather than rebuilt on Lambda. The hashes below identify the frozen inputs and
dependency lock.

| Input | SHA-256 |
|---|---|
| Bundle `manifest.json` | 68545c669c4fdbc0de9958df6270ec7426994159da0bc119a0f1fd49177ef40f |
| `data/naics_descriptions.parquet` | fe8c54e36efb7470e46122c0071e16c03c3dba1c909073c84c91ec998a0fdc36 |
| Committed window summaries | dd425eb5ef9a7fa2be1b2e821c02f1b036f74f6ec503ff2fa70256ea7808a9a0 |
| `uv.lock` | 4167042e8a5a8caa9af62973151f681fffb50afaa1a7f6d1f801bd9e58bdac21 |
| `checkpoints/plan9_exit/text_only.parquet` | f4fb4f574f880c940ee8cf44c7f86615638e59f113b527d0b86973cd45ba32ff |

The frozen text-only table’s five D9 provenance fields are copied below. They match the arm’s
backbone, revision, descriptions, summaries and token window. The comparison table uses
attention-masked mean pooling over tokens within each present channel, then the mean over present
channels; its hidden size is 384. It is a frozen comparator, not a trained arm or a source of the
arm’s revision.

| D9 field | Value |
|---|---|
| Backbone | sentence-transformers/all-MiniLM-L6-v2 |
| Backbone revision | 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 |
| Descriptions SHA-256 | fe8c54e36efb7470e46122c0071e16c03c3dba1c909073c84c91ec998a0fdc36 |
| Summaries SHA-256 | dd425eb5ef9a7fa2be1b2e821c02f1b036f74f6ec503ff2fa70256ea7808a9a0 |
| Maximum token length | 128 |
| Table matrix fingerprint | 8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda |

Task 18’s actual local dependency receipt reports torch 2.9.1 and Lightning 2.5.5. The locked Mac
environment uses Python 3.12.12, peft 0.17.1, transformers 4.57.1, polars 1.35.1 and pydantic
2.12.4. The lock stays frozen through the last Stages 8–10 decision that uses these margins, not
merely through this campaign.

Training ran on Lambda instance C (`ubuntu@141.148.162.79`, dashboard type `gpu_1x_a100_sxm4`), with
logical CUDA 0 reporting NVIDIA A100-SXM4-40GB, compute capability 8.0, 42405855232 bytes of total
memory and native BF16 support. CUDA visibility was unset, one device was used, and NTP was checked
at bootstrap and immediately before launch. The backbone used `bf16-mixed`; fusion, projection, the
live-radius head, distances and losses remained float32. Every export, QCEW read, regressor fit,
store write and decision read ran on the Mac (R9). Mac checkpoint query encoding used MPS with
`32-true`; panel Lorentz distances used CPU float64. Radius reports used CPU first-batch gradients
and float64 exported geometry.

All ten training runs used local source commit `b94ecf344cee5ae7f27c343b904455022cfde265`. Mac
sweep, margin, radius and diagnostic commands used `2d1f045dd88d44a0798918a81b43d09ab3ad782d`; the
sole subsequent change was the local QCEW directory. This finding branch starts from public PR127
merge `a3e82f101b87a51540970e35d3f8753c27dec460`. The six private commits remain local on `main` and
are excluded from the finding branch. Thus the real-instance evidence qualifies the recorded private
source, and does not establish identical native readiness for the unchanged public base.

| Private source commit | Scope |
|---|---|
| aa8ebd6098414ed9e956ef2df87713f0bb955d11 | Held text configuration |
| 3fc580a81d839dc1fd4228d03abe43abf00b97bf | Held graph configuration |
| d7927a56e82734701fcde60dc72ee832339f1093 | Verified offline backbone snapshot transport and launch wrapper |
| 5ce248f88dcb986ec09ee216f2eb2a9c605b0153 | Deferred checkpoint-contract import in remote canonical loading |
| b94ecf344cee5ae7f27c343b904455022cfde265 | Native Mac loop ownership and tmux shell quoting |
| 2d1f045dd88d44a0798918a81b43d09ab3ad782d | Machine-local QCEW directory only |

Before seed 1, the dedicated qualification experiment exercised exact continuation from instance A
to B under the same absolute checkpoint directory, user, settings and budget. Its final audit
covered all kept checkpoints, last, both histories, code-edit rescue, an interrupted transfer with
no promotion, a genuine partial-file transfer followed by retry, Mac checksum tampering refusal,
unreachable SSH and unsafe abandon followed by safe recovery. Missing GPU/NTP evidence and missing
canonical inputs were refused. The NTP negative check removed clock evidence; it did not
deliberately desynchronize the host. Qualification and campaign instances were declared safe only
after coherent final pulls and checksum agreement, then terminated by the operator. These checks
used the private source corrections above. Phase 1’s historical lint result at merge remains QUEUED;
these real-instance checks do not rewrite that history.

The first sweep attempt hit macOS access refusal on the original Downloads QCEW directory before any
export or new decision read. After the operator moved the inputs to
`/Users/lowell/Projects/naics-embedder/data/QCEW`, all four exact original SHA-256 pins were
checked, only `qcew_dir` changed locally, and one authorized retry completed. No QCEW content,
held-out draw, folds or regressor settings changed.

| QCEW file | Bytes | SHA-256 |
|---|---|---|
| 2022_US000_annual.csv | 823163 | c45cbb64a1b1eef16bfd743510d9d02792ccad82f60e9df202c5daa3e8c5cc18 |
| 2023_US000_annual.csv | 849504 | fe9ffe874f6e657f6bb1558971965ce6acc015ace45d831ed32c90d97097aee9 |
| 2024_US000_annual.csv | 847197 | 48db086828a01798731242c6d3d4957f80f941afe75463a1ff7d43de774bea46 |
| 2025_US000_annual.csv | 841756 | 0b5528f70d66a84ff9729691f365c667a09f854f0af3d841bdd660ef3cb01811 |

Execution receipts and complete qualification evidence remain locally under
`logs/plan10_campaign_evidence/`, with the append-only ledger `logs/plan10_campaign.md`. The
committed monitor and sweep lines in section 4 preserve the campaign’s selection evidence even
though those working logs are ignored by Git.

## 1. The runs

Every run was fresh, used seed 1–10 once and the same 21-key settings below, and had a fixed
40-epoch budget. LoRA rank was 8, alpha 16 and dropout 0.1. Text curvature was fixed at 1. There
were 11039 eligible training queries and 2125 code anchors: 128 queries per step gave 87 steps per
epoch. No text validation loader or structural selection statistic was used.

```json
{
  "fusion": "masked_mean",
  "dimension": 16,
  "radius_bound": 8.0,
  "code_code_weight": 1.0,
  "radial_weight": 1.0,
  "target_temperature": 1.0,
  "radial_step": 1.0,
  "logit_scale_init": 1.0,
  "logit_scale_range": [
    0.01,
    100.0
  ],
  "learning_rate": 0.0001,
  "weight_decay": 0.01,
  "warmup_epochs": 1,
  "lr_plateau_factor": 0.5,
  "lr_plateau_patience": 2,
  "early_stopping_patience": 5,
  "max_epochs": 40,
  "queries_per_step": 128,
  "accumulate_grad_batches": 1,
  "gradient_clip_val": 1.0,
  "accelerator": "cuda",
  "precision": "bf16-mixed"
}
```

Epoch numbers are zero-based. Nine runs ended by early stopping; seed 9 exhausted the budget. The
tmux launcher recorded exit 0 for all ten finished runs, treating the training CLI’s early-stopping
exit 1 as finished. No budget was extended. The selected checkpoint is the earliest epoch with the
maximum recorded outcome MRR.

All timestamps below are UTC. Duration is segment creation before launch preflight through the
remote exit-file modification time. It includes launch preparation and is not a measurement of
training compute alone. Every row used instance C and NVIDIA A100-SXM4-40GB.

| Seed | Instance / GPU | Session | Segment | Epochs | Selected | Start UTC | End UTC | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | C / A100-SXM4-40GB | 20261007T143807Z | 20261007T144640Z | 9 | 3 | 2026-10-07T14:46:40.754853+00:00 | 2026-10-07T14:53:59.591359+00:00 | 438.836506 |
| 2 | C / A100-SXM4-40GB | 20261007T150906Z | 20261007T151149Z | 8 | 2 | 2026-10-07T15:11:49.109935+00:00 | 2026-10-07T15:18:28.536889+00:00 | 399.426954 |
| 3 | C / A100-SXM4-40GB | 20261007T152132Z | 20261007T152402Z | 28 | 22 | 2026-10-07T15:24:02.077575+00:00 | 2026-10-07T15:41:45.328658+00:00 | 1063.251083 |
| 4 | C / A100-SXM4-40GB | 20261007T154600Z | 20261007T154818Z | 8 | 2 | 2026-10-07T15:48:18.986965+00:00 | 2026-10-07T15:55:02.415930+00:00 | 403.428965 |
| 5 | C / A100-SXM4-40GB | 20261007T160145Z | 20261007T160423Z | 20 | 14 | 2026-10-07T16:04:23.275368+00:00 | 2026-10-07T16:17:41.137797+00:00 | 797.862429 |
| 6 | C / A100-SXM4-40GB | 20261007T162710Z | 20261007T163027Z | 30 | 24 | 2026-10-07T16:30:27.507247+00:00 | 2026-10-07T16:49:11.926653+00:00 | 1124.419406 |
| 7 | C / A100-SXM4-40GB | 20261007T165710Z | 20261007T165916Z | 34 | 28 | 2026-10-07T16:59:16.888304+00:00 | 2026-10-07T17:20:22.522062+00:00 | 1265.633758 |
| 8 | C / A100-SXM4-40GB | 20261007T172917Z | 20261007T173127Z | 32 | 26 | 2026-10-07T17:31:27.200578+00:00 | 2026-10-07T17:51:09.337895+00:00 | 1182.137317 |
| 9 | C / A100-SXM4-40GB | 20261007T180124Z | 20261007T180405Z | 40 | 39 | 2026-10-07T18:04:05.764944+00:00 | 2026-10-07T18:29:09.454296+00:00 | 1503.689352 |
| 10 | C / A100-SXM4-40GB | 20261007T183845Z | 20261007T184105Z | 17 | 11 | 2026-10-07T18:41:05.086292+00:00 | 2026-10-07T18:52:47.918720+00:00 | 702.832428 |

| Seed | Training run ID | Best monitor MRR | Selected checkpoint SHA-256 |
|---|---|---|---|
| 1 | fbd974a6725d41b88ea93266abccda79 | 0.3689121823545262 | 965103ed8a1f542de374c089d0b94c59685691865c23887a48247c4bfc5ed451 |
| 2 | f2cec082e2034b7295385a283e692bb7 | 0.3778286408066295 | 9df3806ff0863a298e661b7d95f057e34d51b3b4a5bb3d1bf3cf36a43c57afb7 |
| 3 | 399548edb02b42b99b3391f669d64569 | 0.44404386389812917 | 2fd4f46f3cf8f2452e483543affd0b17eaaf2d78ff1d705805b8bef3abeadec9 |
| 4 | d1d9e7f42947484fb5bf72ad232d2364 | 0.380653530478634 | 6a74ef670aa8516e3525cf91121e6ea83557453f7dc260f160777c7d8621a130 |
| 5 | db9cdfe7da154e1fb6659aff7a46b495 | 0.386360814763716 | 7431801944a3b696cb9c670d4462c5714eeb47e39eeaa6a6324687b941c61d05 |
| 6 | e0468acd9a414b9ea30f6d904aa9f1e6 | 0.40494631699639155 | 8cf0ad5de167aa26ce20fb575d79d4f9843380f5087a67f3a077fed432f182be |
| 7 | 81cff89acedb43c3a694380b2c6d315e | 0.4458322860948417 | ef7bcba22ded82eb55ca33a81da74aa01cb2d7c351e3ddb9032585d2f4f2d74e |
| 8 | be5b25e444664f45b54edfdbe6bfad74 | 0.4702177765932248 | 6d77b75c6a113df9c941fad22f62df06603bf53ea3c240898e7dee1fecac79f5 |
| 9 | 29c1d20278264fe68835ecb7cf7b2f57 | 0.46817039917573405 | 70f425e5af52b7a8e7b376bf4b7b4a41a3927b8faf2b60cf56ca06832cd305be |
| 10 | 2c222fa289154540a54aa24486e3bb71 | 0.38602594448256355 | 22625733b4f208c18d7865e90a5ed65cc4a20774a60a7ba9582ceb4f90e68019 |

Every run’s final pull retained its authoritative kept checkpoints, `last.ckpt`,
`monitor_reads.jsonl` and `epoch_summary.jsonl`. Remote and Mac hashes agreed; monitor epochs and
durable summary epochs/MRR agreed exactly. All 226 monitor records travel in the arm record with
training-run and epoch identity. Instance C was declared Safe to terminate after all ten final
backups and the operator confirmed termination.

## 2. The margins

These are the Mac sweep’s three decision statistics, copied at full stored precision from each arm
run. Both regressor statistics are level-6 `covariates+embedding` MSE. They are validation
statistics, including the held-out regime; held-out here does not mean a sealed outer/test split.
The monitor MRR in section 1 serves checkpoint selection and may differ slightly from the Mac’s
re-encoding result.

| Seed | Selected epoch | Outcome MRR | Regressor seen MSE | Regressor held-out MSE |
|---|---|---|---|---|
| 1 | 3 | 0.36891213639256426 | 0.09862303320076733 | 0.1108513238232455 |
| 2 | 2 | 0.37782863954881385 | 0.09818211124857824 | 0.1094407209902802 |
| 3 | 22 | 0.44404386389812917 | 0.0934997824396155 | 0.1051175041764843 |
| 4 | 2 | 0.3806500819793887 | 0.09738460637569481 | 0.10838905539633252 |
| 5 | 14 | 0.386360814763716 | 0.08380568857771621 | 0.09269696964326539 |
| 6 | 24 | 0.40494631699639155 | 0.07664155386016659 | 0.08720185392521118 |
| 7 | 28 | 0.4458322860948417 | 0.0750428496386214 | 0.08331550498931245 |
| 8 | 26 | 0.4702177765932248 | 0.07875383313119326 | 0.08823366590187243 |
| 9 | 39 | 0.46817039917573405 | 0.06888468333747536 | 0.0775760099792989 |
| 10 | 11 | 0.38602594448256355 | 0.07760419110831754 | 0.08741875702210121 |

Margins were fixed at `2026-10-07T21:46:57.683199+00:00`, with `multiple = 3.0`. SD is the sample
standard deviation (`ddof=1`) across ten stored-score panel statistics. The arm’s grouped summary
reduction and the margin input reduction can differ in the last floating-point bit; the margin
inputs, SD and δ were recomputed exactly from stored score objects using the production
`panel_statistic` path. No tolerance or saved value was changed.

| Panel | Statistic | Sample SD | δ = 3 × SD |
|---|---|---|---|
| outcome | mrr | 0.039564045981886904 | 0.11869213794566072 |
| regressor_seen | covariates+embedding | 0.011098457498591568 | 0.0332953724957747 |
| regressor_heldout | covariates+embedding | 0.012251621952623252 | 0.03675486585786976 |

The following ten-element lists are the actual inputs used by the margin record, in seed order 1–10.

```json
{
  "outcome": [
    0.3689121363925643,
    0.3778286395488139,
    0.4440438638981292,
    0.38065008197938877,
    0.386360814763716,
    0.40494631699639155,
    0.44583228609484166,
    0.4702177765932248,
    0.4681703991757341,
    0.3860259444825636
  ],
  "regressor_seen": [
    0.09862303320076733,
    0.09818211124857824,
    0.0934997824396155,
    0.09738460637569481,
    0.08380568857771621,
    0.07664155386016659,
    0.0750428496386214,
    0.07875383313119326,
    0.06888468333747536,
    0.07760419110831754
  ],
  "regressor_heldout": [
    0.1108513238232455,
    0.1094407209902802,
    0.1051175041764843,
    0.1083890553963325,
    0.0926969696432654,
    0.08720185392521118,
    0.08331550498931245,
    0.08823366590187241,
    0.07757600997929892,
    0.08741875702210121
  ]
}
```

| Record path | SHA-256 |
|---|---|
| `/Users/lowell/naics-artifacts/records/stage7/reference.json` | c885b9c5dc18b6be03670d0cb5a71db3974917da8f719e0dbb1f1ef5eec2d1a2 |
| `/Users/lowell/naics-artifacts/records/stage7/margins.json` | e619b3b30fcad07ab95b23c4f7cfcba52327db011a02a378874d7017ce10fd2f |

## 3. Radius and inert terms

All ten `tools radius-report` commands exited 0 and reported `passed: true`, no failed radius checks
and no inert terms. Each report reconstructed its seed’s real first batch (epoch 0, step 0, 25
anchors) with selected-checkpoint weights, and read the full 2125-row exported table without reading
a panel. The least absolute anchor-radius gradient was positive for every seed; every level 2–6
radius SD exceeded 10⁻³, and all 20 sector radii were positive and pairwise distinct.

At the largest radius the manifold residual satisfied |⟨x,x⟩_L + 1| ≤ 10⁻⁹ · x₀². Chunked checks
covered all 4515625 ordered pairs. The maximum relative float32 polar-distance error against the
float64 form was below 10⁻³. Coincident pairs were checked separately for zero distance, with zero
training-form error and no nonzero-distance pair read as zero. Every reported gradient norm for the
three terms and both learned scales was finite and positive.

| Seed | Least absolute anchor gradient | Minimum level SD | Least sector gap | Largest radius | Manifold residual | Max relative error | Task norm | Code-code norm | Radial norm | Task-scale norm | Code-scale norm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0005554857198148966 | 0.1767981547375595 | 0.0071217319250198585 | 5.564433953879321 | 3.637978807091713e-12 | 2.0275132994977523e-06 | 15.713854933674163 | 4.583101919462729 | 6.107683084069041 | 1.4744542837142944 | 0.6117522120475769 |
| 2 | 0.0009514465928077698 | 0.2079086215844606 | 0.009017916815884597 | 5.164210753795459 | 0.0 | 2.36052978337558e-06 | 11.983823202884299 | 3.3354695546499324 | 3.7339801309360903 | 1.274495005607605 | 0.5089486241340637 |
| 3 | 8.607366180513054e-05 | 0.22108256783438004 | 0.0017139745604928258 | 5.731183958055569 | 1.4551915228366852e-11 | 5.636087695180973e-07 | 5.766556388565121 | 3.6879992767284837 | 4.512792211050619 | 1.6004935503005981 | 0.038520004600286484 |
| 4 | 0.0004975929041393101 | 0.206956388779541 | 0.009113929868181714 | 5.302899446007069 | 1.8189894035458565e-12 | 1.3332588560899778e-06 | 11.835334342736227 | 3.1334855252767113 | 2.5095667746575105 | 1.423915982246399 | 0.45955994725227356 |
| 5 | 0.0007208710885606706 | 0.17681059086515902 | 0.003115936900460081 | 5.467837998368511 | 0.0 | 9.124335324532843e-07 | 8.944793674997817 | 2.8727643407440975 | 3.776754391619078 | 1.7728136777877808 | 0.21726582944393158 |
| 6 | 0.0004354982520453632 | 0.24942549125242292 | 0.010277686606351644 | 5.503422281943254 | 5.4569682106375694e-12 | 1.676902323422273e-06 | 8.129247707513509 | 2.620971177436539 | 2.5379809525867705 | 1.7279096841812134 | 0.027893122285604477 |
| 7 | 0.0008688904345035553 | 0.20999095371268658 | 0.004626832727220531 | 5.73798906586128 | 7.275957614183426e-12 | 1.1627882365981198e-06 | 6.183662175543753 | 3.1327812341892276 | 3.4334718840596588 | 1.6102516651153564 | 0.002673062961548567 |
| 8 | 0.003579352516680956 | 0.20505313935073372 | 0.002712042839085882 | 5.600299777695975 | 3.637978807091713e-12 | 4.7618851131326966e-07 | 5.584948128928235 | 3.5296525815161046 | 2.749010231915946 | 1.7001302242279053 | 0.03313326835632324 |
| 9 | 0.0016841490287333727 | 0.158538737570589 | 7.622796876538551e-05 | 5.694750535839993 | 3.637978807091713e-12 | 8.795446036074815e-07 | 7.386228567838613 | 1.844028968554835 | 3.406385676431592 | 1.4221583604812622 | 0.18590125441551208 |
| 10 | 0.00017628743080422282 | 0.15963294030226774 | 0.006019620486191091 | 5.514096285669262 | 3.637978807091713e-12 | 7.077622209433522e-07 | 8.753479685679936 | 3.003234676780823 | 2.471102685008653 | 1.6673376560211182 | 0.404790997505188 |

| Seed | Radius report SHA-256 |
|---|---|
| 1 | 85b65fc33898eb6b06d51060e3ce313f58a4d1b228d9c9ecbe49d797d93bc42e |
| 2 | dff570826a80e72d5cc9fe89eaf0b68ae5cbfc1de3ff77dac2b3ae15ae1f42b0 |
| 3 | 15a101634a065f9318bc1cdd456006abae5596e3f6c3021d27d4935aaf1e7c3c |
| 4 | 61c6bb0046cf85bb4a465196f51b52a733d44290643270b41fa4254cc5c30dfe |
| 5 | 22180e61a8b6f7a886e386969b2efcdde80d658a0ef3a62bab44e1a7a6c7f4a6 |
| 6 | fa26a0f0ae2511aabbaa6281d76e896ec1c56307683967a7a8785bf864591165 |
| 7 | 00d4a828f64ae4e58cb9eb1a953ecec611e2d0b3a1b17438ed2ae61099c04c11 |
| 8 | 081e946f40520a84d523169b843b9b2f0fa55e7a91f0a742b8b3e5a351a62204 |
| 9 | 98b2d2218d59e2daa80ba91eef7bcd3e0ea7aa0405f2c87d60649e385ff3eeda |
| 10 | bcc8ec17ca356e073cf95bce04294ca8f7355da01648493e2e04e778edab7028 |

## 4. The selection log

The exact Task 21 log check counted 1113 physical selection-log records and 226 monitor records, all
with `event: read` and `split: validation`. The physical count includes mirrored/cumulative remote
session logs, so it is not a count of unique reads. The Mac log has 37 records: seven protected
earlier records plus exactly 30 new sweep reads (three per seed). Its protected seven-line prefix
retains SHA-256 `dcbbf4fb690ae41841cea68a79f7239c233114c0adc92b956dbf21400eceb953`; the complete
37-line log has SHA-256 `654e2232f8a10eb406cdab580f771575a75a972842f02232d73637a8976a4a66`. The four
Phase 1 exit reads were not repeated.

| Selection log | Physical records | SHA-256 |
|---|---|---|
| logs/selection_log.jsonl | 37 | 654e2232f8a10eb406cdab580f771575a75a972842f02232d73637a8976a4a66 |
| logs/remote/20261007T012405Z/selection_log.jsonl | 3 | 42eb1c7eabdb4784b916744b63f9173de56db1bc24afa42d0a7d9767069c1632 |
| logs/remote/20261007T130007Z/selection_log.jsonl | 16 | 0b1e3b808db8083e581c2da0d3c4eb02edbc98080db7ece7d785f2c4a2e5ed62 |
| logs/remote/20261007T135403Z/selection_log.jsonl | 16 | 0b1e3b808db8083e581c2da0d3c4eb02edbc98080db7ece7d785f2c4a2e5ed62 |
| logs/remote/20261007T143807Z/selection_log.jsonl | 9 | f3df81a95259e997c885e917a90b7f8ecc05c0008b5c20b233071cdb77177dcb |
| logs/remote/20261007T150906Z/selection_log.jsonl | 17 | 0a002827512ae3b7da36e9cc13647956d49d9836c71ec3b97a445beb6cfe7533 |
| logs/remote/20261007T152132Z/selection_log.jsonl | 45 | 052fec7714295e210d16c94eaff6587aef9e3b67bd04e78090107a957f4d412d |
| logs/remote/20261007T154600Z/selection_log.jsonl | 53 | 95f795e200336930f5bc5c8865c1a1094b5302c22c54befc9600308524c7b26e |
| logs/remote/20261007T160145Z/selection_log.jsonl | 73 | f495804bb6fc46c3e8affaf17603b05f51d936f14711ce4dbe08ae1dc34a12df |
| logs/remote/20261007T162710Z/selection_log.jsonl | 103 | 103f07cfe000b89ecdf61fc449e199456724e436395b7ca27924bbeeb9c2fc33 |
| logs/remote/20261007T165710Z/selection_log.jsonl | 137 | 6384aad5d854365123e33efbb82fba99f674a9d8f4f376345dad2c72a180faf8 |
| logs/remote/20261007T172917Z/selection_log.jsonl | 169 | 0b768a755d96c1cb50241888d9f2cd20d23e74bd13facb50a6a6988683add9e3 |
| logs/remote/20261007T180124Z/selection_log.jsonl | 209 | 68f09668b27c0e9bfb27e0fc53f93b612ad63101dac5edb1476798b38040c96c |
| logs/remote/20261007T183845Z/selection_log.jsonl | 226 | f5a03a115fd29610d0e3a7687656b12eb7caa8e732a652df1e7a7ff710c66f6b |

The following 226 lines are copied verbatim from the ten final `monitor_reads.jsonl` files, in seed
and epoch order. They retain each MRR, training-run ID, seed, epoch, matrix fingerprint, purpose and
UTC read time. No monitor line is sampled or rewritten.

### Seed 1 monitor records

```json
{"mrr": 0.1836269680494462, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 0, "seed": 1, "table": "3ee1144f80e10233b53ee17240b5bfa2b1b6846b54e130670b8fa5bd3eff19b8", "training_run": "fbd974a6725d41b88ea93266abccda79"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s1 (seed 1)", "split": "validation", "time": "2026-10-07T14:49:34.901620+00:00"}}
{"mrr": 0.33612445849445904, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 1, "seed": 1, "table": "08a489348bcc98908af06f1cd079756db5b5f99113c3a27c40ad60b94b8d8221", "training_run": "fbd974a6725d41b88ea93266abccda79"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s1 (seed 1)", "split": "validation", "time": "2026-10-07T14:50:07.201747+00:00"}}
{"mrr": 0.3572821086625676, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 2, "seed": 1, "table": "a341037ecf80bd255b3c30d8c071c92f6d22963716a6cf641d0c6043a168c080", "training_run": "fbd974a6725d41b88ea93266abccda79"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s1 (seed 1)", "split": "validation", "time": "2026-10-07T14:50:39.825603+00:00"}}
{"mrr": 0.3689121823545262, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 3, "seed": 1, "table": "d71517e1ed13e4c3feeeb92606b559d86408555b82a41dd86047e96cea2c72ea", "training_run": "fbd974a6725d41b88ea93266abccda79"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s1 (seed 1)", "split": "validation", "time": "2026-10-07T14:51:12.735817+00:00"}}
{"mrr": 0.352435260140488, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 4, "seed": 1, "table": "dd0d068575f7e8dd9977d746f709275f085b1ac58a00aeed0f4ff4a3e66a8f3c", "training_run": "fbd974a6725d41b88ea93266abccda79"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s1 (seed 1)", "split": "validation", "time": "2026-10-07T14:51:45.158700+00:00"}}
{"mrr": 0.35895227866960866, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 5, "seed": 1, "table": "65e643b65854e959efaf0bcdaaead2a3bf75ed6dee0537ebd2b895e47157bd85", "training_run": "fbd974a6725d41b88ea93266abccda79"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s1 (seed 1)", "split": "validation", "time": "2026-10-07T14:52:17.109219+00:00"}}
{"mrr": 0.36575632358075716, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 6, "seed": 1, "table": "6907c84dd93b8074b1809b558bd87180d4ded925f7c7102159550a641f84ff6c", "training_run": "fbd974a6725d41b88ea93266abccda79"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s1 (seed 1)", "split": "validation", "time": "2026-10-07T14:52:49.707136+00:00"}}
{"mrr": 0.3647280042781121, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 7, "seed": 1, "table": "6ad406cb5cdcf6376d2071c3311144d620f38a987d0612564d9a758565150318", "training_run": "fbd974a6725d41b88ea93266abccda79"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s1 (seed 1)", "split": "validation", "time": "2026-10-07T14:53:21.772841+00:00"}}
{"mrr": 0.3632429611651035, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 8, "seed": 1, "table": "ea00a2d552797b064f003693ff91099105d948e96d7619fea9f0d0a80632f9da", "training_run": "fbd974a6725d41b88ea93266abccda79"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s1 (seed 1)", "split": "validation", "time": "2026-10-07T14:53:54.817372+00:00"}}
```

### Seed 2 monitor records

```json
{"mrr": 0.15674379275291683, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 0, "seed": 2, "table": "5e21cacea0ce58d6fcaedba44c56c9511d0a7de749c62af14a32f5d8b5d4e2ae", "training_run": "f2cec082e2034b7295385a283e692bb7"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s2 (seed 2)", "split": "validation", "time": "2026-10-07T15:14:40.276993+00:00"}}
{"mrr": 0.33789166664018006, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 1, "seed": 2, "table": "d87a89ed4d6bc6c2c40119235317ccf2d2e4efd9a70d47c0ee098e6e28617049", "training_run": "f2cec082e2034b7295385a283e692bb7"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s2 (seed 2)", "split": "validation", "time": "2026-10-07T15:15:12.334414+00:00"}}
{"mrr": 0.3778286408066295, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 2, "seed": 2, "table": "2cdee73bf2f9a8d1ae0cffcdf2fa6e56aa9925f4a355ff2314e2b4cc3fd2a01c", "training_run": "f2cec082e2034b7295385a283e692bb7"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s2 (seed 2)", "split": "validation", "time": "2026-10-07T15:15:44.375658+00:00"}}
{"mrr": 0.36767112652387185, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 3, "seed": 2, "table": "397234a75c072833b1c5659bbef3c6e5b0ac80181d9efe9fe03011edb3382ce6", "training_run": "f2cec082e2034b7295385a283e692bb7"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s2 (seed 2)", "split": "validation", "time": "2026-10-07T15:16:17.054917+00:00"}}
{"mrr": 0.3541417875211061, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 4, "seed": 2, "table": "3fdce4bc524288789125c71843af03cf0d825baa40abc3eb7496c5e454112e8c", "training_run": "f2cec082e2034b7295385a283e692bb7"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s2 (seed 2)", "split": "validation", "time": "2026-10-07T15:16:49.302064+00:00"}}
{"mrr": 0.3581898961454412, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 5, "seed": 2, "table": "877b8382c3f84bf9456a0803d3b210c944e4d804133f88b3d6492259bec55c35", "training_run": "f2cec082e2034b7295385a283e692bb7"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s2 (seed 2)", "split": "validation", "time": "2026-10-07T15:17:20.874734+00:00"}}
{"mrr": 0.36722906802949534, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 6, "seed": 2, "table": "f6b8f842eb144c5955f78abaac29028dfdced8985955e96ab0eb8b171fe0686c", "training_run": "f2cec082e2034b7295385a283e692bb7"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s2 (seed 2)", "split": "validation", "time": "2026-10-07T15:17:52.802057+00:00"}}
{"mrr": 0.3606507801991952, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 7, "seed": 2, "table": "d04d149c911239470176618596140b377cd0c73586bcb17826cff82ccb5061c4", "training_run": "f2cec082e2034b7295385a283e692bb7"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s2 (seed 2)", "split": "validation", "time": "2026-10-07T15:18:24.090148+00:00"}}
```

### Seed 3 monitor records

```json
{"mrr": 0.1311094977156624, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 0, "seed": 3, "table": "90f88fc043f04bb0073fe20b60681557fb0c677454a67e6b6e3995e6bfe8f19c", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:26:53.999859+00:00"}}
{"mrr": 0.2932494795341737, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 1, "seed": 3, "table": "1b8b11641830adfc9b7a50a6b2b17cccfaaab1c7a8ebbbb9af0acd9078e1b800", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:27:26.956156+00:00"}}
{"mrr": 0.32929744165511954, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 2, "seed": 3, "table": "6d0dcf27b6d451460c411a7d09ef741d4413b38591fa2efe0b5c2ccb7dcffec2", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:28:00.045096+00:00"}}
{"mrr": 0.3418561958452926, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 3, "seed": 3, "table": "974e5da924d0d2ad7ce0835aff6baeb427253ac1ae4ed5e5c4c9db11a99ff5e3", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:28:32.643927+00:00"}}
{"mrr": 0.35276366707505047, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 4, "seed": 3, "table": "2f08f20c793c02ec73eee67ffc33187b74d297ae6f2ae2af688f323d8422aac2", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:29:04.984865+00:00"}}
{"mrr": 0.37888627030765265, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 5, "seed": 3, "table": "8294ca201c4d3e724dbcc8443b7d9d76ec52611a52ef4e9543e198350ba06e2d", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:29:38.071429+00:00"}}
{"mrr": 0.3844905798473212, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 6, "seed": 3, "table": "8cb27b6d91a7c0ce6cd17702b46d720a6000db16bd797478668e1d96044bc1c8", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:30:10.388139+00:00"}}
{"mrr": 0.3976407565565001, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 7, "seed": 3, "table": "e9b28c71e2c5870bfb79028d5d7861aaedd700c26d7dfd452da7ec023f6a5bb6", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:30:43.303948+00:00"}}
{"mrr": 0.40094185113291914, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 8, "seed": 3, "table": "b325087886e9669e27435c946735b924adc0673a53c59458ae680c32e6b78e7e", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:31:16.852628+00:00"}}
{"mrr": 0.3945680990019125, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 9, "seed": 3, "table": "47856cc9459653a10ed4c9e05b31660a84ac054f553b5e572074bb11b467c5b1", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:31:50.140713+00:00"}}
{"mrr": 0.40517396740607736, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 10, "seed": 3, "table": "f94477e249472639d9b7e25638683555f4814b5f8f7cfd03e11d998b53f65cd0", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:32:23.118021+00:00"}}
{"mrr": 0.4097822783952131, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 11, "seed": 3, "table": "fb548db646d8c1eb64d0e4f10b80804838f57d5cb8c971f5e018f74ec6e140b1", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:32:55.949261+00:00"}}
{"mrr": 0.41520614389086574, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 12, "seed": 3, "table": "0cd1364a05eee0a8b095757f5cd9d967a8983f261f482e444a60d8efc825c6d2", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:33:28.927625+00:00"}}
{"mrr": 0.4187168912059476, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 13, "seed": 3, "table": "63204dffa6619956554825765ea2b3ce5ac7794b1446d5cee54eaa4336fe6809", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:34:01.876726+00:00"}}
{"mrr": 0.42185872608208763, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 14, "seed": 3, "table": "d944e8cbe3f3aefd8243f5c8ef5c1259065e2f9eb7689bd53318707241ca562c", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:34:35.051676+00:00"}}
{"mrr": 0.42710347879401, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 15, "seed": 3, "table": "86f142e6fa6686b9759db11a9e01a0cf39ac20b92bec49ea96ade483e62e7138", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:35:07.624213+00:00"}}
{"mrr": 0.43150571611225785, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 16, "seed": 3, "table": "71935bf53f543514f4c358eff051eedd63eac29c23dc3199d08f1380b83df16b", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:35:40.173848+00:00"}}
{"mrr": 0.43896487318355293, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 17, "seed": 3, "table": "8514f3c1e56c7c6ee6a271fddb4c1be42e90173581a4c87818344b190c2ddbde", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:36:12.306369+00:00"}}
{"mrr": 0.4411225162146126, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 18, "seed": 3, "table": "de50d0a624ebd6cb309d439cdbd77ecff7e094f0966bf9a418d04ee0041bc9de", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:36:45.025002+00:00"}}
{"mrr": 0.4425890274483532, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 19, "seed": 3, "table": "d599f32da07d57417ab0116f981be48f2f8ef0ca3463feb37707c855dda749a6", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:37:18.375768+00:00"}}
{"mrr": 0.443325127586993, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 20, "seed": 3, "table": "347bcf978641709f4fc660f3f0277e7e07c6d4770ad28eb9c64ff09f103d97c2", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:37:51.349172+00:00"}}
{"mrr": 0.4409702261437126, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 21, "seed": 3, "table": "7d27c8aabb23c400ddba6604272c269661b680ca3ffd7a4e25d086423d2d6f1e", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:38:24.468927+00:00"}}
{"mrr": 0.44404386389812917, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 22, "seed": 3, "table": "79f7ff1e26ad8b8c6710249f678b4cdab26f368446ce391045579936f323ef6f", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:38:57.169201+00:00"}}
{"mrr": 0.4351148893802674, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 23, "seed": 3, "table": "121cc8247208675121594ba08c45834c96aa7a8788cd8575228954ee460c61f6", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:39:29.983934+00:00"}}
{"mrr": 0.4396282030115033, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 24, "seed": 3, "table": "c51e895db62a61dccff5cbf2b6b742c5215d431546831db8a0610d74a80fd65d", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:40:02.707963+00:00"}}
{"mrr": 0.43817065274974115, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 25, "seed": 3, "table": "445bae0c3a4edbd8967376658029fc24e823f9fd8840a47a8f45a61ed399bccd", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:40:35.318277+00:00"}}
{"mrr": 0.4408294907276803, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 26, "seed": 3, "table": "21a93b9004ae621381b90e9cdb97c456c385541065700ea1edbee07f60b261e7", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:41:08.369607+00:00"}}
{"mrr": 0.44127940234435004, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 27, "seed": 3, "table": "1e36469ef7154cd66d43cddc435e8e4d7634e97c7dc2bd3ead131a501a71b442", "training_run": "399548edb02b42b99b3391f669d64569"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s3 (seed 3)", "split": "validation", "time": "2026-10-07T15:41:40.848628+00:00"}}
```

### Seed 4 monitor records

```json
{"mrr": 0.12486691896012729, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 0, "seed": 4, "table": "cdf174238ffe15be16120f44a65d27674f98631b8be0c4ccdf3c3ac9aef4e26c", "training_run": "d1d9e7f42947484fb5bf72ad232d2364"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s4 (seed 4)", "split": "validation", "time": "2026-10-07T15:51:09.680769+00:00"}}
{"mrr": 0.35606985803520075, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 1, "seed": 4, "table": "b5c8101f89c48de95393a7549828be8c65f127b62ae375dcec1a302ea8afce4a", "training_run": "d1d9e7f42947484fb5bf72ad232d2364"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s4 (seed 4)", "split": "validation", "time": "2026-10-07T15:51:42.371650+00:00"}}
{"mrr": 0.380653530478634, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 2, "seed": 4, "table": "8c40d7a02a168c54a5c77b1757da3420815e2ce3cd39489984a1f395889fbe8a", "training_run": "d1d9e7f42947484fb5bf72ad232d2364"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s4 (seed 4)", "split": "validation", "time": "2026-10-07T15:52:15.336051+00:00"}}
{"mrr": 0.35832199491166333, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 3, "seed": 4, "table": "a89af508c08f4d7e45ef99db34f43a66e19a1472fff30946ef6a495cb5d91003", "training_run": "d1d9e7f42947484fb5bf72ad232d2364"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s4 (seed 4)", "split": "validation", "time": "2026-10-07T15:52:48.525210+00:00"}}
{"mrr": 0.36426903713916153, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 4, "seed": 4, "table": "f30448df4508a0f42d28a368710f9d498c347c7a7ad30ddca7dd97d9c40c4212", "training_run": "d1d9e7f42947484fb5bf72ad232d2364"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s4 (seed 4)", "split": "validation", "time": "2026-10-07T15:53:20.977841+00:00"}}
{"mrr": 0.3584106890140943, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 5, "seed": 4, "table": "9e65f9383d2ab1793ba8e4a37a4ee3426102bd11ddc71f86491f5c93eb1beba1", "training_run": "d1d9e7f42947484fb5bf72ad232d2364"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s4 (seed 4)", "split": "validation", "time": "2026-10-07T15:53:53.054105+00:00"}}
{"mrr": 0.3698296527861345, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 6, "seed": 4, "table": "dc41dcc2b6b04cabbbaa816960aa1ec371a6ee99f7cd386bbbc67eb209bf5585", "training_run": "d1d9e7f42947484fb5bf72ad232d2364"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s4 (seed 4)", "split": "validation", "time": "2026-10-07T15:54:25.570301+00:00"}}
{"mrr": 0.3687294211747895, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 7, "seed": 4, "table": "5c2523f89f53c3add61cb4d0dbd6b490e5ba261b3ecd374f776bedbd699ad933", "training_run": "d1d9e7f42947484fb5bf72ad232d2364"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s4 (seed 4)", "split": "validation", "time": "2026-10-07T15:54:58.032274+00:00"}}
```

### Seed 5 monitor records

```json
{"mrr": 0.15942006841095338, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 0, "seed": 5, "table": "2f734834dd80717dd9e52455bb9d0214c35f8db5e441455c45685f5b39ba62c5", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:07:14.975995+00:00"}}
{"mrr": 0.3218413037640927, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 1, "seed": 5, "table": "687b3172f3b8be2a3f4c3002226bf85b7bdb80c892b5871b54779ea8d4f7b25b", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:07:47.514825+00:00"}}
{"mrr": 0.3479627436394903, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 2, "seed": 5, "table": "070ec9b72c6371640c3278ad62e7dd93ced38f423f8eefb549fdd1d1ab9b6627", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:08:20.274186+00:00"}}
{"mrr": 0.33152547266530963, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 3, "seed": 5, "table": "71e15c508b2fe6e2c81eeb3cd8383d6b23f19c69e86295253f41a2d1f7488ed0", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:08:53.402237+00:00"}}
{"mrr": 0.3113213854717686, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 4, "seed": 5, "table": "f8d126cc6370f70902fc4a02bfc6040484cfd63100684d9aebe0b0c4a8efa381", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:09:25.617755+00:00"}}
{"mrr": 0.3254876527781981, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 5, "seed": 5, "table": "e49342e606b91eb47f4a7f9123acf0c2ed45051643de233c6b91b2b9506f15e6", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:09:57.866099+00:00"}}
{"mrr": 0.35708044636533387, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 6, "seed": 5, "table": "f0cc53c277ca69ae30f6a90c823be27d467266862d99baa5f733d229841175d7", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:10:30.533253+00:00"}}
{"mrr": 0.36414798692058203, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 7, "seed": 5, "table": "6940112c748cb2ad6816d698ac205ce8a3b4ab1cc8a9d8e6dbabc5f8681599ad", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:11:03.537935+00:00"}}
{"mrr": 0.37102493201450915, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 8, "seed": 5, "table": "364a7926d0b4ea17a6bcda8c17bb0a5a87173ff923b2678620c4a7dbb070a0ef", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:11:36.812879+00:00"}}
{"mrr": 0.380994972411839, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 9, "seed": 5, "table": "a7a47373f28e822554da0e8d230134d6c07418cf47aea8207300baa32a3e0095", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:12:09.011491+00:00"}}
{"mrr": 0.3750561332581767, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 10, "seed": 5, "table": "c29227c6de9e5174682dc585e5f7c276f9e8e956f8a985046c362b9d952ea753", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:12:41.971117+00:00"}}
{"mrr": 0.37806605066862703, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 11, "seed": 5, "table": "5c171aa4911e35c729fe8c7d5e12f54883b434330f2a3754938f8288e1cde512", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:13:14.363798+00:00"}}
{"mrr": 0.37459751662637397, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 12, "seed": 5, "table": "d27c14edd17781faf6f73cef7524848e10bfe011a2219ac2bee2f2bb36b0fab0", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:13:47.320789+00:00"}}
{"mrr": 0.38390488472231304, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 13, "seed": 5, "table": "f2538fca5c0a16f971abbe77f073af28d4b7010498c10ce334ba1fb49ebc0061", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:14:20.162325+00:00"}}
{"mrr": 0.386360814763716, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 14, "seed": 5, "table": "9b14175a7f6fd0e2580327f7504de93ce8cb4be0206da347d42596424a596122", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:14:53.255012+00:00"}}
{"mrr": 0.3848175002318639, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 15, "seed": 5, "table": "ecb759470913e01e2dd99d9dd7e361e7e81ef1cc697cd38f896bf4588e6553df", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:15:26.109573+00:00"}}
{"mrr": 0.3827902720865483, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 16, "seed": 5, "table": "a73108cac4f3b4d6434383af9db722d74f31485ade0771256f88ddd2ad381061", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:15:58.303156+00:00"}}
{"mrr": 0.38424369386257534, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 17, "seed": 5, "table": "117f41b4d347dc414ec5dae4ad251d1cea68f2f82a67392f35a7f2f6459b21ca", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:16:30.816046+00:00"}}
{"mrr": 0.38311561603364436, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 18, "seed": 5, "table": "b2771d59266469bb680f23381d974d93f3fd28dff2e84a01889822ec3e3d97ae", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:17:03.444146+00:00"}}
{"mrr": 0.38108488693688675, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 19, "seed": 5, "table": "87851b325e968453c3522e73a168453e6261ed2359d5a9d5fe781a950cd365d0", "training_run": "db9cdfe7da154e1fb6659aff7a46b495"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s5 (seed 5)", "split": "validation", "time": "2026-10-07T16:17:36.393651+00:00"}}
```

### Seed 6 monitor records

```json
{"mrr": 0.14730696981954486, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 0, "seed": 6, "table": "275808569d3290c7f81725a0edcb078608e71bf48a1407514ef7411085b95bef", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:33:21.178965+00:00"}}
{"mrr": 0.3104481910296568, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 1, "seed": 6, "table": "749876079ff22a4b353b7d39f11eeefabd4453e13c4a7bc72c572bd1d5a33b07", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:33:53.206957+00:00"}}
{"mrr": 0.3476886603290373, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 2, "seed": 6, "table": "4719042393ff95b368d94b575bb2ae35bd564aea4c7ad675ea482cbd20e9e830", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:34:25.077976+00:00"}}
{"mrr": 0.33925475707765, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 3, "seed": 6, "table": "ff7b992bb868fc3ae381a816048831dd223a5d331062b7e7820a4eb99ee81aba", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:34:57.930170+00:00"}}
{"mrr": 0.32267180355885233, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 4, "seed": 6, "table": "77292f6993684627bcf43d3420052f7fabd4c01f447b4d6d061094ee9de4ba50", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:35:30.289666+00:00"}}
{"mrr": 0.34296133573373955, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 5, "seed": 6, "table": "1c7ed884578aa57a17b196d105da99c08682221a4b3b41dee08218a503c69746", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:36:02.811450+00:00"}}
{"mrr": 0.36310803932562896, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 6, "seed": 6, "table": "f5d1b59beb8bd2f3d6966a148a82ec64eda8e5b15b40d9fba0ae5ba42aca2de1", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:36:35.651602+00:00"}}
{"mrr": 0.3705667048749532, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 7, "seed": 6, "table": "57c1e797c564f3530b5f7cfc40a4a448151cd42381eb974fb5af2962e3ea1473", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:37:07.654783+00:00"}}
{"mrr": 0.37139657549134913, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 8, "seed": 6, "table": "c3e0690ff20987f5b493d481458ef6dac148339f61dc4bf6ace9f6bae64c3e79", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:37:40.549255+00:00"}}
{"mrr": 0.37525304147533833, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 9, "seed": 6, "table": "bd397e3afb1aa434365aef226bd15047627259135b83e6e8eff286acc3f4885e", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:38:13.095558+00:00"}}
{"mrr": 0.3798308938510962, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 10, "seed": 6, "table": "8a974f3ae8c64b0c5946ee39347ad8762a52aa1b61ab268b87d8abcc4447f13e", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:38:45.867387+00:00"}}
{"mrr": 0.37860168173300585, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 11, "seed": 6, "table": "ef9786a51d1271fd18f03e646cf33ec70ae4ac0f14bac3996f93ff6b2fce1a8e", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:39:18.381181+00:00"}}
{"mrr": 0.3775872290568427, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 12, "seed": 6, "table": "23a3110951ad676822c6b4550b572a1ae236aad07c278bb15684123bb00454bc", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:39:50.502421+00:00"}}
{"mrr": 0.3809233239499523, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 13, "seed": 6, "table": "38fd19728166fc9cb13b454121f767c705ddd8d109de5356215d643af60367ce", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:40:22.743151+00:00"}}
{"mrr": 0.37581639244424164, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 14, "seed": 6, "table": "23243d6249ff4d58ac24d16c3d96437579b8436d1367e3088d379ce9d9e46824", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:40:55.009742+00:00"}}
{"mrr": 0.3821090015717297, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 15, "seed": 6, "table": "a1b21686e8d515d9a4906cbc6318d93f95f866b0680030b13ab6a8d2704ad301", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:41:27.494752+00:00"}}
{"mrr": 0.384963568576188, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 16, "seed": 6, "table": "177f92cfa8f21708c70dfd8c0cae27fee8543faf0d7f657f60180f4b3d1a472a", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:41:59.791342+00:00"}}
{"mrr": 0.38867846355516344, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 17, "seed": 6, "table": "73957367cc3133a7e5416ff293deaaaa3ca0512cab4991e74a6b6e086ca8336a", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:42:32.640012+00:00"}}
{"mrr": 0.3924232751029384, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 18, "seed": 6, "table": "008a388be3756ceaa4d845f0eaf8130d44bc168290e23a9340ca8143ffd465a9", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:43:05.537529+00:00"}}
{"mrr": 0.3948209969418731, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 19, "seed": 6, "table": "148e163ba20eeeefe8f77ca136c752febdca99fd3b7f7abcae608237612ce9b6", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:43:38.295681+00:00"}}
{"mrr": 0.3961403818129417, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 20, "seed": 6, "table": "a7eb451a694f11b8c1f90efbac670f6e59c322287a83c4e3afde4a8240941a6a", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:44:10.027584+00:00"}}
{"mrr": 0.39857195555792896, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 21, "seed": 6, "table": "5d35668de32956e3930c7d5dd0c0cbac7e1e25b8bd377bd337c8a89a5a7e7c7c", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:44:41.931487+00:00"}}
{"mrr": 0.4027506320557939, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 22, "seed": 6, "table": "f65653caf2bb0b59d5e1ff64bfc615832b1f2164cb0997ba0f2a882a74f8cd50", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:45:15.456495+00:00"}}
{"mrr": 0.40274242999700177, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 23, "seed": 6, "table": "aae1d3a051dea26d5ab4525eb221b0c85a37029fe58c8a13723d1d3761a4fb1a", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:45:48.613437+00:00"}}
{"mrr": 0.40494631699639155, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 24, "seed": 6, "table": "71f77abdee23d238dc55d466f1c44f4cacd49897f56f614fcf474cbef48f4d6a", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:46:21.791685+00:00"}}
{"mrr": 0.3982621408674645, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 25, "seed": 6, "table": "3a8dbd2821263c5cb9ccbb6ccf19cccc16c03c7426913074c7d20f434c40d38a", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:46:54.770934+00:00"}}
{"mrr": 0.40338934444567726, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 26, "seed": 6, "table": "2aaa71d121e6207280ae8128c8761c58d41102a57a4ba890d84d76a767f8894c", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:47:27.663882+00:00"}}
{"mrr": 0.40381594260653964, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 27, "seed": 6, "table": "34b2e239b087c5a771b721ef1597dc941b89ebf9506450f06718f0b7a00c2663", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:48:00.308501+00:00"}}
{"mrr": 0.40405740718064265, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 28, "seed": 6, "table": "d8a8d01d341f767d1751701b9ca1a024b5b73bb0a78c8924f148465a4f91c6c4", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:48:33.907282+00:00"}}
{"mrr": 0.4022597128862942, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 29, "seed": 6, "table": "07ee659bd1bb24c3c0384d8f1fbdc6d5c534ee755ea3ad4e04e18fd28611044f", "training_run": "e0468acd9a414b9ea30f6d904aa9f1e6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s6 (seed 6)", "split": "validation", "time": "2026-10-07T16:49:07.126923+00:00"}}
```

### Seed 7 monitor records

```json
{"mrr": 0.17345594746680273, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 0, "seed": 7, "table": "e9c7a7a38190691e2b1a10411fa9d1bbfe41d716d329771beb98f99220c3f316", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:02:08.197461+00:00"}}
{"mrr": 0.31644810923891686, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 1, "seed": 7, "table": "25795ffbe594099dfdb7e6131a386c3460b0d8d482fcff9da1742476ce8f8096", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:02:40.964341+00:00"}}
{"mrr": 0.35909837770952074, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 2, "seed": 7, "table": "55c3b97737d51813ad752735434ddfb706f39abf216d5fc04b2e638dfe098ed0", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:03:13.362367+00:00"}}
{"mrr": 0.3535132392415631, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 3, "seed": 7, "table": "99a5c2f2cf5914fc3a87bc75a60fbeb94ab83c426b0747affa6214595fb56cbd", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:03:46.455742+00:00"}}
{"mrr": 0.3584656443857312, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 4, "seed": 7, "table": "c3e38756aad7d79374da7c62c8b9104086508b1dd3192b30dc7749f111aa1c9e", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:04:18.732802+00:00"}}
{"mrr": 0.36483958003951095, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 5, "seed": 7, "table": "77141dabfb1e40f2711fbf4575e6d4cf13af7e5ac876e2269cc40bf3f631a675", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:04:51.511047+00:00"}}
{"mrr": 0.3691206994241235, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 6, "seed": 7, "table": "1fd48ea71f16086e9495f5ef686d457d4bbb73da00edd8e454035bb3f3c0de72", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:05:24.255011+00:00"}}
{"mrr": 0.3476026568859908, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 7, "seed": 7, "table": "143ca142d225bed21c547e1fc04f5e3e305904933f6de2450683222a657172b0", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:05:56.864564+00:00"}}
{"mrr": 0.3751651059672912, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 8, "seed": 7, "table": "eefa2b2508ddced07ea697625302d60fdcaa364d47ffc6b1fb5d9148ab69fc5e", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:06:29.379599+00:00"}}
{"mrr": 0.3898306288235195, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 9, "seed": 7, "table": "71ac59d6e1d4f5249440b025a1a00cf92804b975288c819307104c8c00ab4705", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:07:02.166919+00:00"}}
{"mrr": 0.38513632246877166, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 10, "seed": 7, "table": "93622a43b4c5a69ab8865926f39882bec113ed4f94262a425cccb0481555c0c4", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:07:35.234970+00:00"}}
{"mrr": 0.39564704687883284, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 11, "seed": 7, "table": "0362008830cf2123a5e7cf2e9bddb1faeefc89b302b23b31b5d99d067b1d422d", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:08:07.952583+00:00"}}
{"mrr": 0.4098598524830452, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 12, "seed": 7, "table": "6d54f7de6f9464cd0306cb244ffe4a059ea66a73c980efa68e750e7285da4316", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:08:41.112823+00:00"}}
{"mrr": 0.4124546451277315, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 13, "seed": 7, "table": "2236201021a685ee063bf221a2f17d4e1fdd10977b959ff781a02bba85b51315", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:09:14.270441+00:00"}}
{"mrr": 0.417520440180189, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 14, "seed": 7, "table": "2eeab5c452378008ae0b15cc77fa3c90c7295ab569fe6383bf7ac62a7819d324", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:09:47.172575+00:00"}}
{"mrr": 0.4219707183895603, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 15, "seed": 7, "table": "ee048ffa901187d5800312d759cf06ee21c526729bcf4f147442cb54c19f82f9", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:10:20.010863+00:00"}}
{"mrr": 0.4246740717504353, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 16, "seed": 7, "table": "7cc3c49dfbe582767c701801fb2cf009fe113641597df50955d3451001f2eec4", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:10:52.699601+00:00"}}
{"mrr": 0.4279405002417078, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 17, "seed": 7, "table": "d5cffd4514a5cc442505d9432cdde4f7de32ed7849aff7bdd3024cce0c00265a", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:11:25.858571+00:00"}}
{"mrr": 0.4257418650039954, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 18, "seed": 7, "table": "05e699c7832064d5d8dece779b1d2bcfd192f48504e49ed14616105194721843", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:11:58.851263+00:00"}}
{"mrr": 0.4296802744310141, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 19, "seed": 7, "table": "26091b7c55684b3bd7dfcae0dfa050e1e9390375fcb9a334be227a691c04c723", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:12:32.663021+00:00"}}
{"mrr": 0.4319972929370203, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 20, "seed": 7, "table": "97c65c45706aa933c1b588a768e9704ffa83cc659c0579cbfba01acf936153a3", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:13:05.974950+00:00"}}
{"mrr": 0.435410933074229, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 21, "seed": 7, "table": "0274d3775710be88b320a4079165e8a265622ea35a896cf4fd2bf52b3bd88cc9", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:13:39.709990+00:00"}}
{"mrr": 0.43621802838564167, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 22, "seed": 7, "table": "afa3f2a54bf82dcabc0f61dc306294cecee36a21a6e4dc910b04005a1eca056c", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:14:14.379989+00:00"}}
{"mrr": 0.43707856480079466, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 23, "seed": 7, "table": "380aba6efd5fcea874d5dee984d178a8acdcea96af03223207098f40c14f4eac", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:14:47.785091+00:00"}}
{"mrr": 0.4369016238943531, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 24, "seed": 7, "table": "73c64a23f818c6f1df2fe46fe3aed151512d87f73fc884b5aaeea869b0cb8076", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:15:21.470721+00:00"}}
{"mrr": 0.4371172938029673, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 25, "seed": 7, "table": "e878756306f7229b8192eb78b392f64c7228555423307672b025340bf4091874", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:15:55.220848+00:00"}}
{"mrr": 0.4415677723306273, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 26, "seed": 7, "table": "e633737b1ae7e29eec87917b3256fd139453add87f535e81ab91d553ab803b50", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:16:27.951061+00:00"}}
{"mrr": 0.4440232510273726, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 27, "seed": 7, "table": "33a7c8133647cebe87ad9249147ead5016257e52d5a420f4e72aa7854aaba16a", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:17:00.713041+00:00"}}
{"mrr": 0.4458322860948417, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 28, "seed": 7, "table": "d39c5bf2dbfac5fdc0072218813a2ae2687f6d262a568f822781b41cc1bd7fe4", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:17:33.545514+00:00"}}
{"mrr": 0.44542132906881104, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 29, "seed": 7, "table": "d3e49b53100c05a9042ee9e33a05c65dd447d2e89d9fd16f91a2f3b89581e1b4", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:18:06.654680+00:00"}}
{"mrr": 0.4457342471470582, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 30, "seed": 7, "table": "7b2e05cd97f8e18b55e5d12385726f2354805ec5a79cfd9438796a773f37db9a", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:18:39.733595+00:00"}}
{"mrr": 0.44123884358070165, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 31, "seed": 7, "table": "876c61cb07d65ea3534402624e59b1c68b4abd2014052f1d29b355cad89aa258", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:19:12.279744+00:00"}}
{"mrr": 0.44507123742878274, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 32, "seed": 7, "table": "0b0e57ea94ba11bd2cdd5c927cc4aa5739e168a78d5b932927e71b2367377ea8", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:19:44.781401+00:00"}}
{"mrr": 0.44382517624948514, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 33, "seed": 7, "table": "b863f28f846e384c89dbb099435f7d18598495eafe23abcfaf967934d92c6c86", "training_run": "81cff89acedb43c3a694380b2c6d315e"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s7 (seed 7)", "split": "validation", "time": "2026-10-07T17:20:17.746624+00:00"}}
```

### Seed 8 monitor records

```json
{"mrr": 0.1697581749601293, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 0, "seed": 8, "table": "0c6796034862bcbfb0818ac2c17ddf04703d691ee03c3ccb8d07500eb036b291", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:34:18.295084+00:00"}}
{"mrr": 0.3310046732877733, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 1, "seed": 8, "table": "a6c1b3b64561c5ba41b38cf92f176364656305aa09be987f7e87a85754bbf641", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:34:50.757200+00:00"}}
{"mrr": 0.341386275330378, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 2, "seed": 8, "table": "2457612fae09bdbe695811bc170033f3708430b2e76b97f4d245bbd7f18fc17f", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:35:22.866570+00:00"}}
{"mrr": 0.3470932410262452, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 3, "seed": 8, "table": "4bd17a2183e557cdfcf0a20b98f40b25aa4e946680a218aebb10e6534c382747", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:35:55.212221+00:00"}}
{"mrr": 0.3542990329364044, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 4, "seed": 8, "table": "4a7cfc0be2448ebeba2896cdec8205554a325cef45d0412671b5984fa26361df", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:36:27.609170+00:00"}}
{"mrr": 0.3808054003618876, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 5, "seed": 8, "table": "8737e6f5bedf0d29b626ee5c9f75089c849920944d3b6d846ae5d780d1aa002b", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:37:00.238518+00:00"}}
{"mrr": 0.39065927528929967, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 6, "seed": 8, "table": "2925aae343cfd41adf254c627e8a29cfa3cc145e79eaa2e5264a88d2f65d0621", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:37:33.062151+00:00"}}
{"mrr": 0.4005322514198655, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 7, "seed": 8, "table": "b19552e9bab76bb430c85a6d422507c3733a0e19aba0709c32742ef24dc61f1f", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:38:05.481276+00:00"}}
{"mrr": 0.41346572975914253, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 8, "seed": 8, "table": "e2ccee76ced580732f570757a2cc5f234cca14f6c3d880978c3df7ae02a565ee", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:38:37.791866+00:00"}}
{"mrr": 0.41095225695047055, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 9, "seed": 8, "table": "637ef05e501ab3a42e084e43e7465629bb93e58b332525c5b6d3f336d88ab5e9", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:39:10.117252+00:00"}}
{"mrr": 0.4257096267842701, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 10, "seed": 8, "table": "cb9b321be065d5fe7bb3c8910a8039d5fde39227797052c72418be7f53702c45", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:39:42.688129+00:00"}}
{"mrr": 0.42880421066469365, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 11, "seed": 8, "table": "f4347c898b05971762baf936e3f8bedcbb0eecefecbc34263bc4dbbdcb84fe8f", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:40:15.230996+00:00"}}
{"mrr": 0.4246674882140754, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 12, "seed": 8, "table": "bca36467ddaec05b98c52ec31b3dcd4a9c34207606b02c27e2007b847a78044e", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:40:47.475657+00:00"}}
{"mrr": 0.4385032133667794, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 13, "seed": 8, "table": "80c443998fcc93035d695dc1d294e59a18917a492cf86f10f6e614c2c61a4703", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:41:19.378514+00:00"}}
{"mrr": 0.4435666126556225, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 14, "seed": 8, "table": "7a743b33c60ed01e1a5fac9f60f1f055b8a2d34d747a950fab7f6074bb779337", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:41:51.659213+00:00"}}
{"mrr": 0.4482628522507327, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 15, "seed": 8, "table": "4ac133bbb12a7de2227fe2b307b8e055f564cf9d79f42f448c9a49aaa0360fa2", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:42:24.298952+00:00"}}
{"mrr": 0.4507565899059476, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 16, "seed": 8, "table": "fb469bcd390aa46f5bef6537e283291766b8f654555aea045f9286489dfb8980", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:42:57.330105+00:00"}}
{"mrr": 0.45223220602482156, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 17, "seed": 8, "table": "98d8ff8eade0e457f080678c9fbeb3ef752a9bced2dc9809671b532a5a304358", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:43:29.931859+00:00"}}
{"mrr": 0.45641756835246194, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 18, "seed": 8, "table": "23bbb50421f7ec8f3534215148f28e99b501751486abfe2f3789aab6eee9f54b", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:44:02.206931+00:00"}}
{"mrr": 0.4564607550728455, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 19, "seed": 8, "table": "d4644ea21b90eee1fb680517241ba4a8534fef42ec99a234d06b078df6af99ad", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:44:34.583288+00:00"}}
{"mrr": 0.45886024254874547, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 20, "seed": 8, "table": "bc00fa5a5aae8a4d4f042d45934a5e465bcfd9b94c07ac6c197910b3065fd912", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:45:07.132292+00:00"}}
{"mrr": 0.46311955672412547, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 21, "seed": 8, "table": "8e56eb13293297035a88434b55a1b3bf4dbed84d188ad5364a36672fed782c0d", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:45:39.641315+00:00"}}
{"mrr": 0.46321869736803906, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 22, "seed": 8, "table": "888c53b9281a6031331586f2bffc9268d1030b8b9f534edde69ab399b58c940c", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:46:12.124283+00:00"}}
{"mrr": 0.46290478655419626, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 23, "seed": 8, "table": "bde62ae55cb1ca1e5202a662ac665709394c14b5befaa6242eb8df158c0ad802", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:46:44.588369+00:00"}}
{"mrr": 0.4648185530343196, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 24, "seed": 8, "table": "46aacd0d9d9a646500bb5d06ca9ee9e21d1d8541be969440178c26f975ae4dcc", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:47:17.375473+00:00"}}
{"mrr": 0.4646564041965078, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 25, "seed": 8, "table": "3684f51d3496dfb60f933dbc509ce56e06b2117b86b4bbb0d02b5eed4d385b53", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:47:49.908251+00:00"}}
{"mrr": 0.4702177765932248, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 26, "seed": 8, "table": "3c794ffc030af9eaa50894802621bd62672c6528b0337be30cf3308dd8315161", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:48:22.432147+00:00"}}
{"mrr": 0.4655816997136528, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 27, "seed": 8, "table": "bdb3a627a8ef99194286f61e31fbbf2a2ba2177a608977e27de3ad14ac1cb4e6", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:48:55.343856+00:00"}}
{"mrr": 0.4657541461791408, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 28, "seed": 8, "table": "bc143af7a6c0362ad8c3492cf80228462751c1d0aef5a3f51d9e8b84f495431c", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:49:27.789224+00:00"}}
{"mrr": 0.467577469647372, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 29, "seed": 8, "table": "4ad805b36be6cf58e68acdce1817f8038272d4640cda0925262b00d014ab055c", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:50:00.586131+00:00"}}
{"mrr": 0.46754287186420396, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 30, "seed": 8, "table": "fb4f467cb0eb3c4faeeed56cff5bd6bae1431dca6e67ee3f97165c99384ac999", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:50:32.999110+00:00"}}
{"mrr": 0.4665061359574752, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 31, "seed": 8, "table": "5248c1d9a4ffc8c5cae2798c5f75d13971945cc97821ba9bbfefeb0a9a33eee4", "training_run": "be5b25e444664f45b54edfdbe6bfad74"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s8 (seed 8)", "split": "validation", "time": "2026-10-07T17:51:04.733242+00:00"}}
```

### Seed 9 monitor records

```json
{"mrr": 0.1685049229011832, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 0, "seed": 9, "table": "cd9354be6a9955d607a03d5d395f66369b43aab0d27b1816b9d5480da3c16f93", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:06:56.826184+00:00"}}
{"mrr": 0.33403279519305507, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 1, "seed": 9, "table": "ce356abd344d906fa7d5ac73c7e8c7ed4c9f48977b5d816ae5c1de1dd22a9817", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:07:30.854019+00:00"}}
{"mrr": 0.3680873581492103, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 2, "seed": 9, "table": "7de79ad5fa9b409d37c571dcad415f9b65487453f96fa3f5429b6e18d39255fc", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:08:03.710984+00:00"}}
{"mrr": 0.3418732557153098, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 3, "seed": 9, "table": "26497819dd13900262814a7ef0360be0bdaf1fc65509cc473aac9e5eacb63517", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:08:36.621528+00:00"}}
{"mrr": 0.3670830512362055, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 4, "seed": 9, "table": "97e67e5a0de35b63c20785efb2944f14c32d8600030da5f1c20f79d477f9c555", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:09:09.427350+00:00"}}
{"mrr": 0.381110101496081, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 5, "seed": 9, "table": "c959c5cc651d8ed28eaad22e665082a6d7312815e745c8d0eeb43d7f0a993e05", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:09:42.774611+00:00"}}
{"mrr": 0.38295998410403, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 6, "seed": 9, "table": "9a321767d52049c4908a3fc9ad0090b09dbe78e15bf8dedf908d0c966d65c4b3", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:10:16.155593+00:00"}}
{"mrr": 0.4146101679002263, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 7, "seed": 9, "table": "7ed5f72ae99ad9785d91eb5e495250d45d5dce3ae394c226a31e864472523764", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:10:49.087679+00:00"}}
{"mrr": 0.41170128864190436, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 8, "seed": 9, "table": "4bd520b7242d910cf5cc2212ad644603d64c4aafb2d2b5938787951eabc7a12b", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:11:22.867982+00:00"}}
{"mrr": 0.42248220132969305, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 9, "seed": 9, "table": "84d8f7a3b12fafb35fcce8c1758be74d290ee25dddd9d5fcd6a4ae17a673ec49", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:11:56.154622+00:00"}}
{"mrr": 0.4292286402546435, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 10, "seed": 9, "table": "da639c5e9d51ddc5d1d3a7a9543c27ec31dd7950738b55c0fd9f403231837b14", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:12:29.859983+00:00"}}
{"mrr": 0.4259316462645275, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 11, "seed": 9, "table": "bf3f31cbbf2852ec8b39382fe8846b66f7b28ff106a9c8d9ef2fde9c543cac33", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:13:03.021705+00:00"}}
{"mrr": 0.42995657879224314, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 12, "seed": 9, "table": "0a041d0262bd81cf3680e6c11ce2602a9d7056013fd874aaa76a22e158fdd0db", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:13:38.660477+00:00"}}
{"mrr": 0.4345755483828913, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 13, "seed": 9, "table": "dfd803456a554b93785db63ad6139f1cccaf52727627eaef5efcfbc408bd0521", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:14:13.345070+00:00"}}
{"mrr": 0.43845181657512483, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 14, "seed": 9, "table": "49b29de58f7efe7db689c3d39c0999851a6235a554bf3c93881ca5cd0d7140db", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:14:48.892288+00:00"}}
{"mrr": 0.43929904213818183, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 15, "seed": 9, "table": "8cb15ff693c9ddc4007f6c6ca32bc2232430f63b0d4fcefb7a7c00c820cd90cc", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:15:22.568139+00:00"}}
{"mrr": 0.44387163207860947, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 16, "seed": 9, "table": "a7cebeac173d661e96398052a7b7529643495006695bc14e4d87f77d96b5c685", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:15:58.206673+00:00"}}
{"mrr": 0.44517385146095706, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 17, "seed": 9, "table": "36de3ef23a431ff3064955ba1f51bbff837a7c9a48afa23ab538202da2bf412b", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:16:34.205628+00:00"}}
{"mrr": 0.44628956430815503, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 18, "seed": 9, "table": "8c5ddf88d28525915ec7e6f1859499eaa29346d4b71256fc9686c9536a1ef1ca", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:17:09.803461+00:00"}}
{"mrr": 0.4507638700884413, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 19, "seed": 9, "table": "dec079a66edb185fe22b2db38192b6673ab9569184d69a98cf4289e89f3b0b13", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:17:46.487325+00:00"}}
{"mrr": 0.45201158650253226, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 20, "seed": 9, "table": "50a8bd57cf86e5dd749cc309cfd8f3655b49e33e8bcbc17a761777e30c784ee4", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:18:20.619193+00:00"}}
{"mrr": 0.45060021066057443, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 21, "seed": 9, "table": "9d2f305ca97a0214f3d4ca49af52bccccff8767d9026bb0a1944b0dbdb0000ff", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:18:55.793118+00:00"}}
{"mrr": 0.45456609685909105, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 22, "seed": 9, "table": "5ad0cacd09057ebb40c0d1a0c10957f6bb00d0995f2da3a65dd255440ff0dbcb", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:19:33.497880+00:00"}}
{"mrr": 0.45155385828606504, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 23, "seed": 9, "table": "74d261ceb2734dc597eb48f509a11c1d1876f7870e14e6b2cfacfce95056a98d", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:20:07.996368+00:00"}}
{"mrr": 0.4562966320776638, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 24, "seed": 9, "table": "42a2e467c6605e0555c9fbe8ae9057c0ebd6d43f6c13ec3c2141ed60f2bd44b0", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:20:42.215712+00:00"}}
{"mrr": 0.4573976978866254, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 25, "seed": 9, "table": "fc73e38dc39e06148ae98ddefd2bcf67ed32ab8517c1370648e208bec5f44202", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:21:16.927015+00:00"}}
{"mrr": 0.45618008083381784, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 26, "seed": 9, "table": "a3eea7165eb2f988b82500525c749b2a54dcd39398a46b24e9baf1f3c371c36a", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:21:51.841076+00:00"}}
{"mrr": 0.45586131919067424, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 27, "seed": 9, "table": "aa5ce77d3ffa02a05b9dc81538673810e908e23231c99a78b51dce655de71182", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:22:26.586726+00:00"}}
{"mrr": 0.457446239596181, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 28, "seed": 9, "table": "5d6fe7ca1c1b64ed7bc127cdcbc30d1ce8bb6e210588f929087b13695043de9c", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:23:00.502765+00:00"}}
{"mrr": 0.45794468662059945, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 29, "seed": 9, "table": "1803b1e1d5e6493405df50bbdc0323cfbdb15cd3cb44140b8b0be097df8eeb4d", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:23:33.718983+00:00"}}
{"mrr": 0.4552501473461259, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 30, "seed": 9, "table": "732a7175244a63e0c583db4bf5888a346c75e5848f8e35bc42f9d3032e6bea7e", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:24:07.311747+00:00"}}
{"mrr": 0.45463740308125744, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 31, "seed": 9, "table": "a7c9c19e77311c921dd7a5368778b8755a3d8016e19d18e060a13e12b68badc8", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:24:40.761616+00:00"}}
{"mrr": 0.4609962331555249, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 32, "seed": 9, "table": "828395e542ff0a79999426c306baa5ee93ee5179f3a29337d1655d8b5edc939c", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:25:14.317173+00:00"}}
{"mrr": 0.4629807053074219, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 33, "seed": 9, "table": "bfab922d7c5d29d724566f0c38d622d8efef43362ecbf4157b2df55c701fecdf", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:25:48.197287+00:00"}}
{"mrr": 0.4608819509278329, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 34, "seed": 9, "table": "922f6cedb725c8e19124d5d37443b9ba93f2cf4d008c038dba338faf419f0ace", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:26:21.415147+00:00"}}
{"mrr": 0.46084933873398015, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 35, "seed": 9, "table": "8b47a1d897aa1183145d659cf2dee3ad78901831db888d4e2567bc6a96fb85fd", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:26:54.235188+00:00"}}
{"mrr": 0.4622246900831858, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 36, "seed": 9, "table": "c4a371b46d1521719f4e361034adb9d5136b0f6583e87d5e342591a09d8fbc39", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:27:27.019547+00:00"}}
{"mrr": 0.4651964763072885, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 37, "seed": 9, "table": "ad714d398f6da98b81b4b98914185e50a16e1cec8ead66cb7ba9c0c8acbe40b9", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:27:59.060107+00:00"}}
{"mrr": 0.4658059932853687, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 38, "seed": 9, "table": "5094b091ed8636ac46eaee3966292da9fcc4a660b20871980ea1bccf088beedd", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:28:32.101761+00:00"}}
{"mrr": 0.46817039917573405, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 39, "seed": 9, "table": "594ac48f57655c238229cbb807623d5ff28ae0fc9da4e889960b5ba169ce4ae8", "training_run": "29c1d20278264fe68835ecb7cf7b2f57"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s9 (seed 9)", "split": "validation", "time": "2026-10-07T18:29:04.751084+00:00"}}
```

### Seed 10 monitor records

```json
{"mrr": 0.16549483701912396, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 0, "seed": 10, "table": "9bca69b1a999bac90fd3c435d1c7a1a3559e41fee4e4f8983c70415507ae5cb0", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:43:56.080992+00:00"}}
{"mrr": 0.31671061202950346, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 1, "seed": 10, "table": "ba5e09704f6e4ffdd16046ae96c3a9c1c3521566f6047e66781c39e79ec289cd", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:44:28.290229+00:00"}}
{"mrr": 0.35021467429499326, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 2, "seed": 10, "table": "da27c168de3d04cc4ee2a77c2df04f7a18de794af4768c5cd1f1f52fba9676d2", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:45:00.847930+00:00"}}
{"mrr": 0.36748683098801804, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 3, "seed": 10, "table": "e5198dff17d7c96d70d5da511258bab12d0de8d91165d63230af387127534ef6", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:45:33.731784+00:00"}}
{"mrr": 0.3492123508974394, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 4, "seed": 10, "table": "6cdf5d36802325542b071396d6256aca4e0d91da18e2c8b54726f97f08b0a4f3", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:46:06.335441+00:00"}}
{"mrr": 0.35412757190576016, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 5, "seed": 10, "table": "28d8ec8e322470d684748094a9179822ec014a8de8f3db1a8d1cd75cae6f0cfe", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:46:39.693806+00:00"}}
{"mrr": 0.3665582997888305, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 6, "seed": 10, "table": "153830f6e2b3fcbe3f5a29e1a5253364c81fa2a18014ce1ad06ac9a56f262915", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:47:15.993690+00:00"}}
{"mrr": 0.36386966266122306, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 7, "seed": 10, "table": "4695c79941f8d39cad79f0ff26183635d64b6b760a352f11d6533d96afba4488", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:47:49.147740+00:00"}}
{"mrr": 0.3746275249147644, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 8, "seed": 10, "table": "3a192edfacd8781e877f2b6905d1f767ecdd722e874b7578fbf7a9537812e75c", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:48:21.761744+00:00"}}
{"mrr": 0.378230566194336, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 9, "seed": 10, "table": "20cca3c47ab7dafb355158ec13bc7e85ac8016d4cabef19f814fe473a6a9bcae", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:48:54.968938+00:00"}}
{"mrr": 0.3800186722023806, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 10, "seed": 10, "table": "c068782f5e4bca8ab22952c6b3fcbe807fc66e8124e2408c64ea62be6e9054c0", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:49:28.195796+00:00"}}
{"mrr": 0.38602594448256355, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 11, "seed": 10, "table": "9166b1c8b8cbed7ac052a9b8d44e8daecb2ab40ed698e54ce2075875ee93e7c4", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:50:01.628567+00:00"}}
{"mrr": 0.38007495768900623, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 12, "seed": 10, "table": "7e12e8a910f294497c0a7b97506b7cd0a16429b66501be192e56e2079ca34da9", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:50:33.483741+00:00"}}
{"mrr": 0.38295110002082616, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 13, "seed": 10, "table": "223b52993c384f92213e48bd579d1292313c8a95fe7e4483a920a0d77f1ea62b", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:51:06.240979+00:00"}}
{"mrr": 0.37560677844822776, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 14, "seed": 10, "table": "8d1210604aef8c39dedbfd48d57b3f52a828a5d4a288531c8e3e51fede92ef61", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:51:39.042621+00:00"}}
{"mrr": 0.38345531255785054, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 15, "seed": 10, "table": "e33748c8351c685da6ad23ce10a0693346eb03533e7af3557eba435e2856224c", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:52:11.481399+00:00"}}
{"mrr": 0.3813890637387436, "read": {"detail": {"distance": "lorentz", "encoder": "LiveEncoder", "epoch": 16, "seed": 10, "table": "e4a1202e59789e0bf2c119d214dae530a761f2122f2bbde7402c45e639d8617a", "training_run": "2c222fa289154540a54aa24486e3bb71"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "D6 monitor: the validation MRR that selects an epoch of stage7-reference-s10 (seed 10)", "split": "validation", "time": "2026-10-07T18:52:43.451718+00:00"}}
```

### The 30 Mac sweep reads

These are the appended sweep lines, copied verbatim from `logs/selection_log.jsonl` after its
protected seven-line prefix. Their sequence exactly matches all ten arm runs’ three `log_records`
entries. The original seven records are retained in the local log and are not repeated here.

```json
{"detail": {"arm_name": "reference", "distance": "lorentz", "encoder": "ArmEncoder", "run": "reference/seed-1/2d18faf4010441d98d25add07a131ea9", "seed": 1, "table": "51f17bd3ece549b22347e3d78647440b3b44a581cdb9800c27c7259b0d43c142"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:40:10.185193+00:00"}
{"detail": {"arm": "51f17bd3ece549b22347e3d78647440b3b44a581cdb9800c27c7259b0d43c142", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "one_hot", "covariates+one_hot", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-1/2d18faf4010441d98d25add07a131ea9", "seed": 1, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_seen", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:40:13.117466+00:00"}
{"detail": {"arm": "51f17bd3ece549b22347e3d78647440b3b44a581cdb9800c27c7259b0d43c142", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-1/2d18faf4010441d98d25add07a131ea9", "seed": 1, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_heldout", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:40:20.558415+00:00"}
{"detail": {"arm_name": "reference", "distance": "lorentz", "encoder": "ArmEncoder", "run": "reference/seed-2/828775d800714f4f9cdc595bb77cef08", "seed": 2, "table": "fa327f8b6eab3f5a03e01bc27db618bbb78c0bbd495043e8dbb3c3a150de837c"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:40:41.920519+00:00"}
{"detail": {"arm": "fa327f8b6eab3f5a03e01bc27db618bbb78c0bbd495043e8dbb3c3a150de837c", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "one_hot", "covariates+one_hot", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-2/828775d800714f4f9cdc595bb77cef08", "seed": 2, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_seen", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:40:43.982576+00:00"}
{"detail": {"arm": "fa327f8b6eab3f5a03e01bc27db618bbb78c0bbd495043e8dbb3c3a150de837c", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-2/828775d800714f4f9cdc595bb77cef08", "seed": 2, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_heldout", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:40:51.499089+00:00"}
{"detail": {"arm_name": "reference", "distance": "lorentz", "encoder": "ArmEncoder", "run": "reference/seed-3/38d03590223c4f6c8e20e36d68884fa8", "seed": 3, "table": "7b4da75704c2606532dda66f5ff1427a3487c97435a22af519f6eb55eb6cd1c6"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:41:12.794059+00:00"}
{"detail": {"arm": "7b4da75704c2606532dda66f5ff1427a3487c97435a22af519f6eb55eb6cd1c6", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "one_hot", "covariates+one_hot", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-3/38d03590223c4f6c8e20e36d68884fa8", "seed": 3, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_seen", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:41:14.935531+00:00"}
{"detail": {"arm": "7b4da75704c2606532dda66f5ff1427a3487c97435a22af519f6eb55eb6cd1c6", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-3/38d03590223c4f6c8e20e36d68884fa8", "seed": 3, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_heldout", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:41:22.484612+00:00"}
{"detail": {"arm_name": "reference", "distance": "lorentz", "encoder": "ArmEncoder", "run": "reference/seed-4/8e12fa1fdbc2456ba13d1923acde0182", "seed": 4, "table": "ae2fd420c72c25f527d55b14e781a7c3accafbafffb3404127e943888b30aa84"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:41:43.858994+00:00"}
{"detail": {"arm": "ae2fd420c72c25f527d55b14e781a7c3accafbafffb3404127e943888b30aa84", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "one_hot", "covariates+one_hot", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-4/8e12fa1fdbc2456ba13d1923acde0182", "seed": 4, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_seen", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:41:46.315306+00:00"}
{"detail": {"arm": "ae2fd420c72c25f527d55b14e781a7c3accafbafffb3404127e943888b30aa84", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-4/8e12fa1fdbc2456ba13d1923acde0182", "seed": 4, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_heldout", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:41:53.829925+00:00"}
{"detail": {"arm_name": "reference", "distance": "lorentz", "encoder": "ArmEncoder", "run": "reference/seed-5/9593a9fd12d14685a258b56088f04f4e", "seed": 5, "table": "480d80c8daddae33768013ebed1837882cdd2cec376ccd3752c72e51d79f1956"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:42:15.234018+00:00"}
{"detail": {"arm": "480d80c8daddae33768013ebed1837882cdd2cec376ccd3752c72e51d79f1956", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "one_hot", "covariates+one_hot", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-5/9593a9fd12d14685a258b56088f04f4e", "seed": 5, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_seen", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:42:17.723199+00:00"}
{"detail": {"arm": "480d80c8daddae33768013ebed1837882cdd2cec376ccd3752c72e51d79f1956", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-5/9593a9fd12d14685a258b56088f04f4e", "seed": 5, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_heldout", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:42:25.246344+00:00"}
{"detail": {"arm_name": "reference", "distance": "lorentz", "encoder": "ArmEncoder", "run": "reference/seed-6/88970d81ec09424fa58b8949eb6b7bb6", "seed": 6, "table": "f0f52f562d50e927c5722c22982834a0c060d069afaac403e9566394c44b7974"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:42:46.605046+00:00"}
{"detail": {"arm": "f0f52f562d50e927c5722c22982834a0c060d069afaac403e9566394c44b7974", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "one_hot", "covariates+one_hot", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-6/88970d81ec09424fa58b8949eb6b7bb6", "seed": 6, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_seen", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:42:48.741365+00:00"}
{"detail": {"arm": "f0f52f562d50e927c5722c22982834a0c060d069afaac403e9566394c44b7974", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-6/88970d81ec09424fa58b8949eb6b7bb6", "seed": 6, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_heldout", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:42:56.384441+00:00"}
{"detail": {"arm_name": "reference", "distance": "lorentz", "encoder": "ArmEncoder", "run": "reference/seed-7/52cf4b1d0d904c2b94b8b32590235b4f", "seed": 7, "table": "d7efca0d96e11d9db88b8afa920331dc916f63239053443a69af0a55887cf979"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:43:17.719282+00:00"}
{"detail": {"arm": "d7efca0d96e11d9db88b8afa920331dc916f63239053443a69af0a55887cf979", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "one_hot", "covariates+one_hot", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-7/52cf4b1d0d904c2b94b8b32590235b4f", "seed": 7, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_seen", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:43:19.986456+00:00"}
{"detail": {"arm": "d7efca0d96e11d9db88b8afa920331dc916f63239053443a69af0a55887cf979", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-7/52cf4b1d0d904c2b94b8b32590235b4f", "seed": 7, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_heldout", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:43:27.485954+00:00"}
{"detail": {"arm_name": "reference", "distance": "lorentz", "encoder": "ArmEncoder", "run": "reference/seed-8/3729cb629b764f5b9b9ddd4af9cfac1e", "seed": 8, "table": "e600568d2e13452a4351a4c10876aacd9fd9e25623408536b0a225a4c3474c16"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:43:48.723331+00:00"}
{"detail": {"arm": "e600568d2e13452a4351a4c10876aacd9fd9e25623408536b0a225a4c3474c16", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "one_hot", "covariates+one_hot", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-8/3729cb629b764f5b9b9ddd4af9cfac1e", "seed": 8, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_seen", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:43:51.012994+00:00"}
{"detail": {"arm": "e600568d2e13452a4351a4c10876aacd9fd9e25623408536b0a225a4c3474c16", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-8/3729cb629b764f5b9b9ddd4af9cfac1e", "seed": 8, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_heldout", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:43:58.557860+00:00"}
{"detail": {"arm_name": "reference", "distance": "lorentz", "encoder": "ArmEncoder", "run": "reference/seed-9/b36a9c19d23e46328c553c85b4c22a95", "seed": 9, "table": "cfc09dfb6afefa0ec91b44c607062f75b007586be627ec328293ca75db4521c9"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:44:19.986757+00:00"}
{"detail": {"arm": "cfc09dfb6afefa0ec91b44c607062f75b007586be627ec328293ca75db4521c9", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "one_hot", "covariates+one_hot", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-9/b36a9c19d23e46328c553c85b4c22a95", "seed": 9, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_seen", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:44:22.211837+00:00"}
{"detail": {"arm": "cfc09dfb6afefa0ec91b44c607062f75b007586be627ec328293ca75db4521c9", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-9/b36a9c19d23e46328c553c85b4c22a95", "seed": 9, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_heldout", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:44:29.676379+00:00"}
{"detail": {"arm_name": "reference", "distance": "lorentz", "encoder": "ArmEncoder", "run": "reference/seed-10/57a1389f7712432e90a574b8ec299a62", "seed": 10, "table": "a39f0f9be6097e926dfd40b116db751ed0285c8c288134ca798dcb34199569d3"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:44:50.911064+00:00"}
{"detail": {"arm": "a39f0f9be6097e926dfd40b116db751ed0285c8c288134ca798dcb34199569d3", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "one_hot", "covariates+one_hot", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-10/57a1389f7712432e90a574b8ec299a62", "seed": 10, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_seen", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:44:53.430576+00:00"}
{"detail": {"arm": "a39f0f9be6097e926dfd40b116db751ed0285c8c288134ca798dcb34199569d3", "arm_name": "reference", "comparators": ["covariates", "embedding", "covariates+embedding", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "run": "reference/seed-10/57a1389f7712432e90a574b8ec299a62", "seed": 10, "text_only": "8941df995d7f1bfe39a79b0349a1771e73083e2cdda131b16344845210a1cbda"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_heldout", "purpose": "Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)", "split": "validation", "time": "2026-10-07T21:45:00.936213+00:00"}
```

## 5. Diagnostics, for the record (Req 6)

Nothing selects on these diagnostics, and they have no target values. The summaries below describe
the selected exports at curvature 1 across 2125 codes. Each report has 272103 same-sector and
1984647 cross-sector pairs; rank correlation has 2125 queries and zero undefined queries; ancestor
MAP has 2105 queries; each NDCG has 2125 queries; distance Pearson has 2256750 pairs; parent
retrieval has 1583 queries with 522 unary pairs excluded.

| Seed | Sector AUC | Rank corr/query | Rank corr/sector | Ancestor MAP | NDCG@5 | NDCG@10 | NDCG@20 | Distance Pearson | Parent@1 | Parent@5 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.7230007958110257 | 0.6110723889871414 | 0.6151781387536929 | 0.2592024418453515 | 0.584937843578135 | 0.5510307523605652 | 0.5279391900038194 | 0.7705901692750002 | 0.28679722046746686 | 0.5855969677826911 |
| 2 | 0.6619662913556899 | 0.5497348909301428 | 0.5391122860239361 | 0.22875432367343929 | 0.579571902873703 | 0.5467140298035442 | 0.5236664247480329 | 0.6564039138681299 | 0.2899557801642451 | 0.5855969677826911 |
| 3 | 0.7117401932551958 | 0.7098986989292956 | 0.691433885927583 | 0.4789890683938512 | 0.6022465623349439 | 0.5840354884599159 | 0.5711453453024771 | 0.8378703107203035 | 0.32090966519267217 | 0.6582438408085913 |
| 4 | 0.6406435497504582 | 0.5922768363185937 | 0.591393679071148 | 0.22192472324259538 | 0.5674993404304489 | 0.5353104999431951 | 0.5100078668926623 | 0.6584626589161144 | 0.28427037271004424 | 0.5628553379658876 |
| 5 | 0.7972168349322931 | 0.6430918205727704 | 0.6535728093706373 | 0.32049517717670023 | 0.5572340540550356 | 0.5367619177863995 | 0.5284536957444362 | 0.8109117148468127 | 0.27668982943777637 | 0.5824384080859129 |
| 6 | 0.8044892372870462 | 0.6928923455658572 | 0.6732528921683755 | 0.3957989904674286 | 0.5632297397243706 | 0.5490949488838746 | 0.5460676746985249 | 0.8298605638222557 | 0.3025900189513582 | 0.6140240050536955 |
| 7 | 0.7302916065530995 | 0.7417675258240893 | 0.7219834598899956 | 0.4740913262761605 | 0.6301296188170621 | 0.6110467985541119 | 0.6017022926983617 | 0.8632152061316637 | 0.3569172457359444 | 0.7049905243209097 |
| 8 | 0.7222112158964976 | 0.7404760262647796 | 0.7166669006528325 | 0.48216602184854346 | 0.6170874795498699 | 0.6008175980391166 | 0.5914506140857083 | 0.8719251158167785 | 0.34554643082754266 | 0.6835123183828175 |
| 9 | 0.6678361820679146 | 0.7702480587737794 | 0.7527900829971977 | 0.5177369285535656 | 0.6304782387308291 | 0.6117506524318774 | 0.5977098336115038 | 0.8701760834968626 | 0.36197094125078966 | 0.7233101705622236 |
| 10 | 0.7871641474579846 | 0.653113671183411 | 0.6541338904433027 | 0.3006401890521086 | 0.5614062521091765 | 0.5412306072688177 | 0.5301614040664263 | 0.8055777617189255 | 0.2722678458622868 | 0.5799115603284902 |

| Seed | Diagnostics SHA-256 |
|---|---|
| 1 | b3815ef955507c0b593832158ca1122318a726fefd8bdc4c6949709ad895d763 |
| 2 | c1bd9cc8f2aca0a01785f9b40107cafdc1b1be8dd7b30fbd11a855b70c091880 |
| 3 | f43aa28a0202bc3aa56a6877e56b66e07c924b8dd4a82fd290f38264ec04f604 |
| 4 | 09daf6fc4c5dea67d0e6b7708c0cf669c64ecc92604966d03ce4eb382872604d |
| 5 | 13f87b04f3c35f4358d5954c5ab5e911332a956b763317d81a424e19c6fce288 |
| 6 | 137d6df8baa1c2ec53fc1ea2f5e029f0ba157954abf5bff1490c32795631849b |
| 7 | 9f424a914c9b4a256aa68344d5af24a211d122a047d377738a8b7daf1a0b3297 |
| 8 | e49fdc6b2febd5ee462892c90dfecfa0d37cb620284f90e6af7e81b57c6a5465 |
| 9 | aa4d2ed643d4e3fd5cb68601e5e6c13728006100de36cdfc31d3d43774047a24 |
| 10 | c05ea03fecd38cbefc73138e3f1c416403676f32e7667f6dcf3894830e45f0fe |

The full saved summaries, including every sector correlation and level breakdown, follow. They are
descriptive receipts, not additional panel reads.

### Seed 1 diagnostic summary

```json
{
  "codes": 2125,
  "geometry": "hyperbolic",
  "curvature": 1.0,
  "sector_separation": {
    "auc": 0.7230007958110257,
    "same_sector_pairs": 272103,
    "cross_sector_pairs": 1984647
  },
  "within_sector_rank_correlation": {
    "mean_over_queries": 0.6110723889871414,
    "mean_over_sectors": 0.6151781387536929,
    "by_sector": {
      "11": 0.5147761672369726,
      "21": 0.6145640800864485,
      "22": 0.7169164483380942,
      "23": 0.5758347320967966,
      "31": 0.6297446975527963,
      "42": 0.45259853642836056,
      "44": 0.6155020445706798,
      "48": 0.6846603535947329,
      "51": 0.6779139721157066,
      "52": 0.6665193783189213,
      "53": 0.5935632777630778,
      "54": 0.6411390237389749,
      "55": 0.5363014148997203,
      "56": 0.6442175166511565,
      "61": 0.5598667093525446,
      "62": 0.5429689570525965,
      "71": 0.5436786062901562,
      "72": 0.7875429791398175,
      "81": 0.7169124015882408,
      "92": 0.5883414782580624
    },
    "queries": 2125,
    "undefined_queries": 0
  },
  "map_over_ancestors": {
    "value": 0.2592024418453515,
    "queries": 2105,
    "by_level": {
      "3": 0.3219496537065331,
      "4": 0.31191696950161724,
      "5": 0.24285562032110067,
      "6": 0.24833599241195786
    }
  },
  "ndcg": {
    "@5": {
      "value": 0.584937843578135,
      "queries": 2125,
      "by_level": {
        "2": 0.6177620218196256,
        "3": 0.42168174500510913,
        "4": 0.46388748269885005,
        "5": 0.5828176653560265,
        "6": 0.6380607841354804
      }
    },
    "@10": {
      "value": 0.5510307523605652,
      "queries": 2125,
      "by_level": {
        "2": 0.5312160915694711,
        "3": 0.36827762238807127,
        "4": 0.441399445573238,
        "5": 0.5520723025819051,
        "6": 0.6014155429546117
      }
    },
    "@20": {
      "value": 0.5279391900038194,
      "queries": 2125,
      "by_level": {
        "2": 0.4501474871798212,
        "3": 0.33170666391920395,
        "4": 0.4193228692127705,
        "5": 0.5320033342582334,
        "6": 0.5783816682379644
      }
    }
  },
  "distance_pearson": {
    "value": 0.7705901692750002,
    "pairs": 2256750
  },
  "parent_retrieval": {
    "at": {
      "1": 0.28679722046746686,
      "5": 0.5855969677826911
    },
    "queries": 1583,
    "unary_pairs_excluded": 522
  }
}
```

### Seed 2 diagnostic summary

```json
{
  "codes": 2125,
  "geometry": "hyperbolic",
  "curvature": 1.0,
  "sector_separation": {
    "auc": 0.6619662913556899,
    "same_sector_pairs": 272103,
    "cross_sector_pairs": 1984647
  },
  "within_sector_rank_correlation": {
    "mean_over_queries": 0.5497348909301428,
    "mean_over_sectors": 0.5391122860239361,
    "by_sector": {
      "11": 0.5004642915774116,
      "21": 0.555026511568419,
      "22": 0.71947101254927,
      "23": 0.3772361955988292,
      "31": 0.592312340755331,
      "42": 0.4922923915739948,
      "44": 0.5470692851903514,
      "48": 0.5196036148294549,
      "51": 0.6548823314833985,
      "52": 0.5927874023664744,
      "53": 0.42993080788354515,
      "54": 0.6145763388295401,
      "55": 0.5044651493695481,
      "56": 0.6182641006053138,
      "61": 0.3802357602482601,
      "62": 0.45238415370397034,
      "71": 0.5842542860284172,
      "72": 0.6237652021279798,
      "81": 0.6481050963708838,
      "92": 0.3751194478183311
    },
    "queries": 2125,
    "undefined_queries": 0
  },
  "map_over_ancestors": {
    "value": 0.22875432367343929,
    "queries": 2105,
    "by_level": {
      "3": 0.2150713253855078,
      "4": 0.29661153180095473,
      "5": 0.265066030949909,
      "6": 0.18467802072766762
    }
  },
  "ndcg": {
    "@5": {
      "value": 0.579571902873703,
      "queries": 2125,
      "by_level": {
        "2": 0.7350916081670669,
        "3": 0.6477498817165476,
        "4": 0.5744281030830982,
        "5": 0.5340870179282358,
        "6": 0.602563894956858
      }
    },
    "@10": {
      "value": 0.5467140298035442,
      "queries": 2125,
      "by_level": {
        "2": 0.6377475039238429,
        "3": 0.5784976800091303,
        "4": 0.5226582624445313,
        "5": 0.5070291799405365,
        "6": 0.576239857866831
      }
    },
    "@20": {
      "value": 0.5236664247480329,
      "queries": 2125,
      "by_level": {
        "2": 0.5311440257020108,
        "3": 0.5144946141375658,
        "4": 0.4871597246729319,
        "5": 0.4859507991518891,
        "6": 0.5611773649243169
      }
    }
  },
  "distance_pearson": {
    "value": 0.6564039138681299,
    "pairs": 2256750
  },
  "parent_retrieval": {
    "at": {
      "1": 0.2899557801642451,
      "5": 0.5855969677826911
    },
    "queries": 1583,
    "unary_pairs_excluded": 522
  }
}
```

### Seed 3 diagnostic summary

```json
{
  "codes": 2125,
  "geometry": "hyperbolic",
  "curvature": 1.0,
  "sector_separation": {
    "auc": 0.7117401932551958,
    "same_sector_pairs": 272103,
    "cross_sector_pairs": 1984647
  },
  "within_sector_rank_correlation": {
    "mean_over_queries": 0.7098986989292956,
    "mean_over_sectors": 0.691433885927583,
    "by_sector": {
      "11": 0.7227183873983679,
      "21": 0.8249025554747853,
      "22": 0.797348924118603,
      "23": 0.6801062503606785,
      "31": 0.7022940380817748,
      "42": 0.694502824276322,
      "44": 0.7304163330351033,
      "48": 0.737440543707648,
      "51": 0.763688423871546,
      "52": 0.712868385867205,
      "53": 0.7071799156207143,
      "54": 0.6841052973188673,
      "55": 0.08111921291160093,
      "56": 0.6614694934192851,
      "61": 0.6542702302017416,
      "62": 0.6128267068154265,
      "71": 0.7119842990619137,
      "72": 0.8340911992281871,
      "81": 0.7565764620305299,
      "92": 0.7587682357513618
    },
    "queries": 2125,
    "undefined_queries": 0
  },
  "map_over_ancestors": {
    "value": 0.4789890683938512,
    "queries": 2105,
    "by_level": {
      "3": 0.5217121855723247,
      "4": 0.39941847047263596,
      "5": 0.5021431644604007,
      "6": 0.4833894169321401
    }
  },
  "ndcg": {
    "@5": {
      "value": 0.6022465623349439,
      "queries": 2125,
      "by_level": {
        "2": 0.545730084913364,
        "3": 0.4646303893680505,
        "4": 0.6031399297625131,
        "5": 0.6047289038993425,
        "6": 0.6144560402476822
      }
    },
    "@10": {
      "value": 0.5840354884599159,
      "queries": 2125,
      "by_level": {
        "2": 0.4607619889915954,
        "3": 0.445237221700818,
        "4": 0.5782769245617674,
        "5": 0.5864282982860238,
        "6": 0.599761867223435
      }
    },
    "@20": {
      "value": 0.5711453453024771,
      "queries": 2125,
      "by_level": {
        "2": 0.37422730306780116,
        "3": 0.41798433086168596,
        "4": 0.5610193389047323,
        "5": 0.5776678904552796,
        "6": 0.5882071976653563
      }
    }
  },
  "distance_pearson": {
    "value": 0.8378703107203035,
    "pairs": 2256750
  },
  "parent_retrieval": {
    "at": {
      "1": 0.32090966519267217,
      "5": 0.6582438408085913
    },
    "queries": 1583,
    "unary_pairs_excluded": 522
  }
}
```

### Seed 4 diagnostic summary

```json
{
  "codes": 2125,
  "geometry": "hyperbolic",
  "curvature": 1.0,
  "sector_separation": {
    "auc": 0.6406435497504582,
    "same_sector_pairs": 272103,
    "cross_sector_pairs": 1984647
  },
  "within_sector_rank_correlation": {
    "mean_over_queries": 0.5922768363185937,
    "mean_over_sectors": 0.591393679071148,
    "by_sector": {
      "11": 0.6302745759774948,
      "21": 0.7087650381301861,
      "22": 0.7786521513965351,
      "23": 0.5432489193272036,
      "31": 0.6356685336864528,
      "42": 0.4623283555145722,
      "44": 0.5792770670974581,
      "48": 0.6258638921109928,
      "51": 0.7001678162091262,
      "52": 0.48563863751436653,
      "53": 0.47040832810904176,
      "54": 0.6499915566264266,
      "55": 0.6709480467528339,
      "56": 0.5331523006557871,
      "61": 0.4835328164755074,
      "62": 0.5396112711218756,
      "71": 0.5720550081342217,
      "72": 0.6901613580664036,
      "81": 0.6924975408815056,
      "92": 0.37563036763497104
    },
    "queries": 2125,
    "undefined_queries": 0
  },
  "map_over_ancestors": {
    "value": 0.22192472324259538,
    "queries": 2105,
    "by_level": {
      "3": 0.24091473912447303,
      "4": 0.24778950609822442,
      "5": 0.21903340912142202,
      "6": 0.21421990188419068
    }
  },
  "ndcg": {
    "@5": {
      "value": 0.5674993404304489,
      "queries": 2125,
      "by_level": {
        "2": 0.5763110461865637,
        "3": 0.38429454159003,
        "4": 0.5399791224660802,
        "5": 0.5318894578327008,
        "6": 0.6173243036877928
      }
    },
    "@10": {
      "value": 0.5353104999431951,
      "queries": 2125,
      "by_level": {
        "2": 0.5062809898225518,
        "3": 0.34699648825073287,
        "4": 0.49143269079409824,
        "5": 0.5037417821919227,
        "6": 0.5885950326244574
      }
    },
    "@20": {
      "value": 0.5100078668926623,
      "queries": 2125,
      "by_level": {
        "2": 0.42925017689156036,
        "3": 0.3124066168110374,
        "4": 0.45204735792518286,
        "5": 0.4798129816510631,
        "6": 0.5685463911034365
      }
    }
  },
  "distance_pearson": {
    "value": 0.6584626589161144,
    "pairs": 2256750
  },
  "parent_retrieval": {
    "at": {
      "1": 0.28427037271004424,
      "5": 0.5628553379658876
    },
    "queries": 1583,
    "unary_pairs_excluded": 522
  }
}
```

### Seed 5 diagnostic summary

```json
{
  "codes": 2125,
  "geometry": "hyperbolic",
  "curvature": 1.0,
  "sector_separation": {
    "auc": 0.7972168349322931,
    "same_sector_pairs": 272103,
    "cross_sector_pairs": 1984647
  },
  "within_sector_rank_correlation": {
    "mean_over_queries": 0.6430918205727704,
    "mean_over_sectors": 0.6535728093706373,
    "by_sector": {
      "11": 0.6213489358003738,
      "21": 0.762745668908244,
      "22": 0.6969375584462805,
      "23": 0.645104276582217,
      "31": 0.6438579672834209,
      "42": 0.4571665458734558,
      "44": 0.5873892839338115,
      "48": 0.7095813932736369,
      "51": 0.7069636700234581,
      "52": 0.7060085814259203,
      "53": 0.6374457270992978,
      "54": 0.6401663883221785,
      "55": 0.3847075832710591,
      "56": 0.6102501449261162,
      "61": 0.6733935970588023,
      "62": 0.586015740621094,
      "71": 0.7116438446616892,
      "72": 0.8512379498204835,
      "81": 0.7217267627068521,
      "92": 0.7177645673743562
    },
    "queries": 2125,
    "undefined_queries": 0
  },
  "map_over_ancestors": {
    "value": 0.32049517717670023,
    "queries": 2105,
    "by_level": {
      "3": 0.4859073767960685,
      "4": 0.35078550930473196,
      "5": 0.33692537344169843,
      "6": 0.28439893341634737
    }
  },
  "ndcg": {
    "@5": {
      "value": 0.5572340540550356,
      "queries": 2125,
      "by_level": {
        "2": 0.5891606482930783,
        "3": 0.3948044061391806,
        "4": 0.5119114328899086,
        "5": 0.5657364445827293,
        "6": 0.5800165980870902
      }
    },
    "@10": {
      "value": 0.5367619177863995,
      "queries": 2125,
      "by_level": {
        "2": 0.47400365388747845,
        "3": 0.36535121904976037,
        "4": 0.4964565103713733,
        "5": 0.5477966266780495,
        "6": 0.5590166049545584
      }
    },
    "@20": {
      "value": 0.5284536957444362,
      "queries": 2125,
      "by_level": {
        "2": 0.3719896657803261,
        "3": 0.3482750453201697,
        "4": 0.4876021361033291,
        "5": 0.5408921262955452,
        "6": 0.5526025423449883
      }
    }
  },
  "distance_pearson": {
    "value": 0.8109117148468127,
    "pairs": 2256750
  },
  "parent_retrieval": {
    "at": {
      "1": 0.27668982943777637,
      "5": 0.5824384080859129
    },
    "queries": 1583,
    "unary_pairs_excluded": 522
  }
}
```

### Seed 6 diagnostic summary

```json
{
  "codes": 2125,
  "geometry": "hyperbolic",
  "curvature": 1.0,
  "sector_separation": {
    "auc": 0.8044892372870462,
    "same_sector_pairs": 272103,
    "cross_sector_pairs": 1984647
  },
  "within_sector_rank_correlation": {
    "mean_over_queries": 0.6928923455658572,
    "mean_over_sectors": 0.6732528921683755,
    "by_sector": {
      "11": 0.6793562120943228,
      "21": 0.7920267097807252,
      "22": 0.8233722440171825,
      "23": 0.6323325363485819,
      "31": 0.709849231296557,
      "42": 0.628845973809585,
      "44": 0.6330398707230434,
      "48": 0.746529044759204,
      "51": 0.7618734005050425,
      "52": 0.6985880968104818,
      "53": 0.7266011914617124,
      "54": 0.6776852643951857,
      "55": 0.08111921291160093,
      "56": 0.6455373801300248,
      "61": 0.6415903903511806,
      "62": 0.5842260027993628,
      "71": 0.7807755224648047,
      "72": 0.7640912874416457,
      "81": 0.7371688556303188,
      "92": 0.7204494156369463
    },
    "queries": 2125,
    "undefined_queries": 0
  },
  "map_over_ancestors": {
    "value": 0.3957989904674286,
    "queries": 2105,
    "by_level": {
      "3": 0.5187902491027491,
      "4": 0.3811255079094925,
      "5": 0.4259608232703716,
      "6": 0.36806259619630793
    }
  },
  "ndcg": {
    "@5": {
      "value": 0.5632297397243706,
      "queries": 2125,
      "by_level": {
        "2": 0.6102270667139253,
        "3": 0.4631082292314173,
        "4": 0.5610212348444686,
        "5": 0.5611548431717428,
        "6": 0.5738834370517448
      }
    },
    "@10": {
      "value": 0.5490949488838746,
      "queries": 2125,
      "by_level": {
        "2": 0.522415068591823,
        "3": 0.4434753088350896,
        "4": 0.5412757178885603,
        "5": 0.5480828624830345,
        "6": 0.5627102984167403
      }
    },
    "@20": {
      "value": 0.5460676746985249,
      "queries": 2125,
      "by_level": {
        "2": 0.44154055423197064,
        "3": 0.4354433896369292,
        "4": 0.5282986999239685,
        "5": 0.5491468791779914,
        "6": 0.5619389653303976
      }
    }
  },
  "distance_pearson": {
    "value": 0.8298605638222557,
    "pairs": 2256750
  },
  "parent_retrieval": {
    "at": {
      "1": 0.3025900189513582,
      "5": 0.6140240050536955
    },
    "queries": 1583,
    "unary_pairs_excluded": 522
  }
}
```

### Seed 7 diagnostic summary

```json
{
  "codes": 2125,
  "geometry": "hyperbolic",
  "curvature": 1.0,
  "sector_separation": {
    "auc": 0.7302916065530995,
    "same_sector_pairs": 272103,
    "cross_sector_pairs": 1984647
  },
  "within_sector_rank_correlation": {
    "mean_over_queries": 0.7417675258240893,
    "mean_over_sectors": 0.7219834598899956,
    "by_sector": {
      "11": 0.7148351922117342,
      "21": 0.8497388597095306,
      "22": 0.8878708616609032,
      "23": 0.6953101230161663,
      "31": 0.7545589385057776,
      "42": 0.6837722398975672,
      "44": 0.7412860335469735,
      "48": 0.8052169596411343,
      "51": 0.817402000041524,
      "52": 0.7475401962429493,
      "53": 0.7421758282854861,
      "54": 0.6774514066442227,
      "55": 0.10714107322812591,
      "56": 0.6507641370532898,
      "61": 0.6804682193258353,
      "62": 0.6878113868858456,
      "71": 0.7950288804519923,
      "72": 0.8823493991030638,
      "81": 0.7714030654431826,
      "92": 0.7475443969046077
    },
    "queries": 2125,
    "undefined_queries": 0
  },
  "map_over_ancestors": {
    "value": 0.4740913262761605,
    "queries": 2105,
    "by_level": {
      "3": 0.48260989795418635,
      "4": 0.38033383084128736,
      "5": 0.4901416768215625,
      "6": 0.49089056954401467
    }
  },
  "ndcg": {
    "@5": {
      "value": 0.6301296188170621,
      "queries": 2125,
      "by_level": {
        "2": 0.6177620218196256,
        "3": 0.47104083833870103,
        "4": 0.6342952410334682,
        "5": 0.627571615788385,
        "6": 0.6459392307834424
      }
    },
    "@10": {
      "value": 0.6110467985541119,
      "queries": 2125,
      "by_level": {
        "2": 0.5005626650887022,
        "3": 0.45693329187004145,
        "4": 0.6113637357804946,
        "5": 0.612702314276359,
        "6": 0.6266261585468241
      }
    },
    "@20": {
      "value": 0.6017022926983617,
      "queries": 2125,
      "by_level": {
        "2": 0.4139622796161878,
        "3": 0.4371895568988897,
        "4": 0.5954489361112559,
        "5": 0.610164562091182,
        "6": 0.6171603491366703
      }
    }
  },
  "distance_pearson": {
    "value": 0.8632152061316637,
    "pairs": 2256750
  },
  "parent_retrieval": {
    "at": {
      "1": 0.3569172457359444,
      "5": 0.7049905243209097
    },
    "queries": 1583,
    "unary_pairs_excluded": 522
  }
}
```

### Seed 8 diagnostic summary

```json
{
  "codes": 2125,
  "geometry": "hyperbolic",
  "curvature": 1.0,
  "sector_separation": {
    "auc": 0.7222112158964976,
    "same_sector_pairs": 272103,
    "cross_sector_pairs": 1984647
  },
  "within_sector_rank_correlation": {
    "mean_over_queries": 0.7404760262647796,
    "mean_over_sectors": 0.7166669006528325,
    "by_sector": {
      "11": 0.7977127206740956,
      "21": 0.8745342166728154,
      "22": 0.8682738003448989,
      "23": 0.6556916283090762,
      "31": 0.7631724162722342,
      "42": 0.6200862862274489,
      "44": 0.7549651630581041,
      "48": 0.7597234345415715,
      "51": 0.787644039164089,
      "52": 0.80198696031074,
      "53": 0.6966258928833037,
      "54": 0.712669052247422,
      "55": 0.14952588809284997,
      "56": 0.6485967492233534,
      "61": 0.6586860292459342,
      "62": 0.6594383334470763,
      "71": 0.7838773534421375,
      "72": 0.8049576688323649,
      "81": 0.7915954936763133,
      "92": 0.743574886390819
    },
    "queries": 2125,
    "undefined_queries": 0
  },
  "map_over_ancestors": {
    "value": 0.48216602184854346,
    "queries": 2105,
    "by_level": {
      "3": 0.4763158700980392,
      "4": 0.38142727023572076,
      "5": 0.5099649633831123,
      "6": 0.4944542425476343
    }
  },
  "ndcg": {
    "@5": {
      "value": 0.6170874795498699,
      "queries": 2125,
      "by_level": {
        "2": 0.6243222756952427,
        "3": 0.5109323217126672,
        "4": 0.6208025969631995,
        "5": 0.6162672395233214,
        "6": 0.6264423100285758
      }
    },
    "@10": {
      "value": 0.6008175980391166,
      "queries": 2125,
      "by_level": {
        "2": 0.5288292077028482,
        "3": 0.49121875807252086,
        "4": 0.599712892785902,
        "5": 0.601691117600483,
        "6": 0.6123785176870684
      }
    },
    "@20": {
      "value": 0.5914506140857083,
      "queries": 2125,
      "by_level": {
        "2": 0.4384529197187416,
        "3": 0.4697633606102533,
        "4": 0.5827623485579707,
        "5": 0.5970462863890609,
        "6": 0.6048522917403688
      }
    }
  },
  "distance_pearson": {
    "value": 0.8719251158167785,
    "pairs": 2256750
  },
  "parent_retrieval": {
    "at": {
      "1": 0.34554643082754266,
      "5": 0.6835123183828175
    },
    "queries": 1583,
    "unary_pairs_excluded": 522
  }
}
```

### Seed 9 diagnostic summary

```json
{
  "codes": 2125,
  "geometry": "hyperbolic",
  "curvature": 1.0,
  "sector_separation": {
    "auc": 0.6678361820679146,
    "same_sector_pairs": 272103,
    "cross_sector_pairs": 1984647
  },
  "within_sector_rank_correlation": {
    "mean_over_queries": 0.7702480587737794,
    "mean_over_sectors": 0.7527900829971977,
    "by_sector": {
      "11": 0.7950288271827484,
      "21": 0.8627587460005007,
      "22": 0.8671717248800178,
      "23": 0.7569166943250939,
      "31": 0.7832707497996928,
      "42": 0.6650621488361128,
      "44": 0.777349646874764,
      "48": 0.8043419810924787,
      "51": 0.8018749490146883,
      "52": 0.8373632420858605,
      "53": 0.8175801419062286,
      "54": 0.7304966924157693,
      "55": 0.2537086381222081,
      "56": 0.6717410672793385,
      "61": 0.6485220612239994,
      "62": 0.7074128854596189,
      "71": 0.8632443782162484,
      "72": 0.8473524897341441,
      "81": 0.8086515131158252,
      "92": 0.755953082378615
    },
    "queries": 2125,
    "undefined_queries": 0
  },
  "map_over_ancestors": {
    "value": 0.5177369285535656,
    "queries": 2105,
    "by_level": {
      "3": 0.4900424464363609,
      "4": 0.43675971708593564,
      "5": 0.5338664905451946,
      "6": 0.5340278210466969
    }
  },
  "ndcg": {
    "@5": {
      "value": 0.6304782387308291,
      "queries": 2125,
      "by_level": {
        "2": 0.5438021509124721,
        "3": 0.5394905833469494,
        "4": 0.6296928593360265,
        "5": 0.6238964883942012,
        "6": 0.645542526782959
      }
    },
    "@10": {
      "value": 0.6117506524318774,
      "queries": 2125,
      "by_level": {
        "2": 0.47776438267272925,
        "3": 0.5037939528277567,
        "4": 0.6043914641155579,
        "5": 0.6127818084267529,
        "6": 0.626177265157308
      }
    },
    "@20": {
      "value": 0.5977098336115038,
      "queries": 2125,
      "by_level": {
        "2": 0.3828513480518077,
        "3": 0.46067155588680636,
        "4": 0.5881736771990492,
        "5": 0.6054521299870865,
        "6": 0.612586847786429
      }
    }
  },
  "distance_pearson": {
    "value": 0.8701760834968626,
    "pairs": 2256750
  },
  "parent_retrieval": {
    "at": {
      "1": 0.36197094125078966,
      "5": 0.7233101705622236
    },
    "queries": 1583,
    "unary_pairs_excluded": 522
  }
}
```

### Seed 10 diagnostic summary

```json
{
  "codes": 2125,
  "geometry": "hyperbolic",
  "curvature": 1.0,
  "sector_separation": {
    "auc": 0.7871641474579846,
    "same_sector_pairs": 272103,
    "cross_sector_pairs": 1984647
  },
  "within_sector_rank_correlation": {
    "mean_over_queries": 0.653113671183411,
    "mean_over_sectors": 0.6541338904433027,
    "by_sector": {
      "11": 0.6953767493165828,
      "21": 0.709939191765549,
      "22": 0.8561327311859285,
      "23": 0.6412233026343859,
      "31": 0.6981887017742052,
      "42": 0.3092109267134079,
      "44": 0.5738949470158079,
      "48": 0.7087867530969644,
      "51": 0.7151587132376169,
      "52": 0.6818083618165103,
      "53": 0.67160065070448,
      "54": 0.6443649470937108,
      "55": 0.3673596763933758,
      "56": 0.6534595823990166,
      "61": 0.6051074526784622,
      "62": 0.586228786456861,
      "71": 0.6927279605098051,
      "72": 0.805965305014257,
      "81": 0.7439146498687038,
      "92": 0.7222284191904235
    },
    "queries": 2125,
    "undefined_queries": 0
  },
  "map_over_ancestors": {
    "value": 0.3006401890521086,
    "queries": 2105,
    "by_level": {
      "3": 0.4927780697311947,
      "4": 0.3241180335021276,
      "5": 0.31066944286940307,
      "6": 0.268440022534407
    }
  },
  "ndcg": {
    "@5": {
      "value": 0.5614062521091765,
      "queries": 2125,
      "by_level": {
        "2": 0.568867590224233,
        "3": 0.3823165892301958,
        "4": 0.5173193488241068,
        "5": 0.5668186995696799,
        "6": 0.5879803339131248
      }
    },
    "@10": {
      "value": 0.5412306072688177,
      "queries": 2125,
      "by_level": {
        "2": 0.48893927584459124,
        "3": 0.35842207170630686,
        "4": 0.4954968528919045,
        "5": 0.5503575412437128,
        "6": 0.5673106318556479
      }
    },
    "@20": {
      "value": 0.5301614040664263,
      "queries": 2125,
      "by_level": {
        "2": 0.3744140125419171,
        "3": 0.3417541689187867,
        "4": 0.48224100577449286,
        "5": 0.541788602076473,
        "6": 0.5577803622182611
      }
    }
  },
  "distance_pearson": {
    "value": 0.8055777617189255,
    "pairs": 2256750
  },
  "parent_retrieval": {
    "at": {
      "1": 0.2722678458622868,
      "5": 0.5799115603284902
    },
    "queries": 1583,
    "unary_pairs_excluded": 522
  }
}
```

## 6. What later stages read

Stage 8 and every Stages 8–10 decision read
`/Users/lowell/naics-artifacts/records/stage7/margins.json`, with outcome δ = 0.11869213794566072,
regressor-seen δ = 0.0332953724957747 and regressor-held-out δ = 0.03675486585786976. They keep
`uv.lock` frozen at `4167042e8a5a8caa9af62973151f681fffb50afaa1a7f6d1f801bd9e58bdac21` until their
last decision. These margins do not authorize candidate selection before its required validation
reads.

Stage 9 makes R7’s term-weight decision through one-factor ablations from the selected cell under
Req 5. The reference starts with code-code and radial weights equal to 1, jointly with the task
objective and independent learned positive logit scales. R10 keeps the training-pairs member in
contract v2, unread by text training. That member and D5’s margin axis in its generation leave at
the next contract bump. HGCN’s graph objective remains separate.

Stage 12 consumes the within-run monitor history through `SeedRun.monitor_records`, retaining
training-run ID, seed, epoch, matrix fingerprint and read purpose. The checkpoint is selected by
outcome validation MRR, with earliest epoch breaking a maximum tie; structural diagnostics do not
become a second selector. Later releases must deliberately reconcile the local source corrections
before claiming the public base has this native qualification.

## Reproduction

Use the pinned canonical inputs, cached backbone revision and frozen lock, plus the recorded local
source corrections and configuration. These commands describe the approved sequential workflow;
completed directories must not be relaunched fresh. For a new reproduction, choose new
experiment/output names. The public finding branch does not contain the six local commits or the
machine-specific configuration. QCEW directory resolution must point to the four pinned files before
any panel read; on this Mac it was `/Users/lowell/Projects/naics-embedder/data/QCEW`. Lifecycle
launch and termination were operator-controlled.

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder remote up --host ubuntu@<NEW_INSTANCE_IP>
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder remote train seed=<s> experiment_name=stage7-reference-s<s>
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder remote sync --once
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder remote status
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder remote finish
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder tools sweep --runs 'checkpoints/stage7-reference-s{seed}' --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 6 --seed 7 --seed 8 --seed 9 --seed 10 --text-only checkpoints/plan9_exit/text_only.parquet --store ~/naics-artifacts --output ~/naics-artifacts/records/stage7/reference.json --purpose 'Stage 7 reference sweep: 10 seeds, the margins fixed from them (R8)' supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
uv run --locked --no-sync naics-embedder tools margins --reference ~/naics-artifacts/records/stage7/reference.json --multiple 3 --name stage7-reference --store ~/naics-artifacts --output ~/naics-artifacts/records/stage7/margins.json
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder tools radius-report --checkpoint 'checkpoints/stage7-reference-s<s>/epoch=<k>.ckpt' --table 'checkpoints/stage7-reference-s<s>/arm_table_epoch=<k>.parquet' --output checkpoints/stage7-reference-s<s>/radius_report.json supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
uv run --locked --no-sync naics-embedder tools diagnostics --table 'checkpoints/stage7-reference-s<s>/arm_table_epoch=<k>.parquet' --geometry hyperbolic --codebook data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/naics_codebook.parquet --output checkpoints/stage7-reference-s<s>/diagnostics.json
shasum -a 256 uv.lock ~/naics-artifacts/records/stage7/reference.json ~/naics-artifacts/records/stage7/margins.json
```

Run seeds 1–10 one at a time, finishing and verifying each pull before the next seed. The manifest
for remote transport is the same explicit local canonical bundle persisted in the approved private
configuration; remote operations use their complete persisted session configuration. On seed 1,
validate the pulled epoch-0 monitor identity and saved CUDA/BF16 settings before launching anything
else. If an unfinished run needs another instance, restore the entire coherent checkpoint/history
set under the same absolute directory and user, then continue only with `remote train --resume` and
`last.ckpt`, keeping seed, settings and budget exact. Never resume a finished run to extend it.

After all ten coherent final backups, sweep once (all-seed preflight precedes the first panel read),
require exactly 30 new validation reads, fix the margins and audit all monitor/decision identities.
Then run radius reports and descriptive diagnostics using each selected epoch formatted as three
digits. The line counts and every saved hash must agree before closeout. CUDA training is not
bitwise reproducible: a rerun’s checkpoints, selected epochs and δ can differ, even with the same
locked inputs and settings.
