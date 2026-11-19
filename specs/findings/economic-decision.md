# Economic decision: finding

**Status: FINAL (2026-11-18).** Redesign roadmap Stage 3
(`specs/naics-artifact-redesign-roadmap.md`), implemented by plan 19
(`specs/plans/completed/19-economic-decision-read.md`).

**No candidate is adopted. The reference `covariates+ancestors` stands, and there is no economic
artifact.** The decision (`economic`, SHA-256
3d510fa319865e84b0321435fe607081d0cbd5987768f68233650237bb69515d) read all 18 candidates of the
frozen list on `growth_seen` and `growth_heldout`, against the 2026-10-17 margins. None is
non-inferior to the reference on both panels, so none is eligible. On `growth_seen`, 0 of 18 are
non-inferior. On `growth_heldout`, 16 of 18 are, and 0 are superior. Propagation does not help at
dimension 8, 16 or 32: `smoothed-dN` is not adopted over `text-only-dN`.

The reads appended exactly 118 validation reads to the selection log. No outer set opened, and
`tools growth-margins` did not run again.

## Sources

| Input | SHA-256 |
|---|---|
| Bundle 301cce28 `naics_codebook.parquet` | 0f9e61343e673540e1e18ecc5581618be099c5d79531b68153496bf982d6e96d |
| MiniLM text-only table `checkpoints/plan9_exit/text_only.parquet` | f4fb4f574f880c940ee8cf44c7f86615638e59f113b527d0b86973cd45ba32ff |
| Its provenance `text_only_provenance.json` (the list's `text_only_provenance_sha256`) | bd21c628d06b96c8a0ffe1318c18f81199cb00ca8862560dbdb8dc57887ea05e |
| The frozen list `conf/data/economic_candidates.yaml` | f5a9223b780b4cd2329f9e4b2a0fcd58bb46a3cadd5881dc0e65e5b2c693662a |
| The margin record `growth_margins.json` | 4f86a0b2f3bde358bd08a1f3f17f1b4c59ee261ee4e5409ac83615deaa1f37d0 |
| `uv.lock` | 6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948 |
| The decision record `decision.json` (DECISION_SHA) | 3d510fa319865e84b0321435fe607081d0cbd5987768f68233650237bb69515d |

The 18 candidate records and their hashes are in §1. The reads ran at `origin/main`
37b47c4108275b86e8c9408b5c16a66e47578636, from the main checkout's root, on `main` with a clean
tree. One read, `text-only-d8`, ran at 2f153107372f74f6f8bbc16dae7554f3ab7cada7. §1 explains why
that is sound. The record store is `~/naics-artifacts`.

## 1. The reads

Each read used `tools economic-read`, one candidate at a time, in the list's order:

```bash
cd /Users/lowell/Projects/naics-embedder && HF_HUB_OFFLINE=1 COLUMNS=200 uv run --locked --no-sync naics-embedder tools economic-read --candidate NAME \
  --codebook /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/naics_codebook.parquet \
  --margins /Users/lowell/naics-artifacts/records/economic/growth_margins.json \
  --store /Users/lowell/naics-artifacts \
  --text-only /Users/lowell/Projects/naics-embedder/checkpoints/plan9_exit/text_only.parquet \
  --runs-root /Users/lowell/Projects/naics-embedder/checkpoints \
  --purpose 'economic candidate read: redesign roadmap Stage 3, plan 19 (Req 29)' \
  --log /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl \
  --lock /Users/lowell/Projects/naics-embedder/uv.lock \
  --output /Users/lowell/naics-artifacts/records/economic/NAME.json
```

A script checked each record with `check_economic_arm`, `check_reference_pairing` and
`check_economic_margins_first`, and checked the log, before the next read began.

| Environment | Value |
|---|---|
| Platform | macOS-27.2.1-arm64-arm-64bit |
| Python | 3.12.12 |
| numpy | 2.3.4 |
| polars | 1.35.1 |
| BLAS | accelerate (found True) |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, `POLARS_MAX_THREADS` | all unset |

The reads took 78 to 112 seconds for a deterministic candidate (one run), 361 to 400 seconds for a
trained candidate of 5 seeds, and 793 seconds for `hyperbolic-d16` (10 seeds).

| Candidate | Kind | Runs | `growth_seen` MSE | `growth_heldout` MSE | Record SHA-256 |
|---|---|---|---|---|---|
| text-only-d8 | text-only | 1 | 0.00367451 | 0.00319862 | 7f51c93f969cc0defcd86d079590ec5b065d51e8bb64172f1aa35c5f18add455 |
| text-only-d16 | text-only | 1 | 0.00366432 | 0.00321648 | ffe7a98386308c1b71d06299642ed1ba1089584372353cb2456bd66c399c0f36 |
| text-only-d32 | text-only | 1 | 0.00355201 | 0.00312177 | a4cadf38255a74068054411d9beb4f91be132fe78cb1f1194bc4810aa8e65f66 |
| tree-d8 | tree | 1 | 0.00369591 | 0.00331969 | dcedebad9e29f5d0ccfd89c022d6c7f8bf29ea97d648ccab6aec0d0bc8aaadc8 |
| tree-d16 | tree | 1 | 0.00364045 | 0.00331019 | 7941ea8acaf60e1dde2044b37093714ceae8376915e2902615e3c8441faa71ee |
| tree-d32 | tree | 1 | 0.00360892 | 0.00333938 | 5684dba1059d759e420ed53dcec3fb67398c1226ac939aae0b6e8476f3804eb8 |
| smoothed-d8 | smoothed | 1 | 0.00363453 | 0.00318084 | b23259389b7793e61e62caf501791154b789b19cb85f011f21e5afa7c4bb45c6 |
| smoothed-d16 | smoothed | 1 | 0.00365557 | 0.00323545 | 18331aff9e245f81d7c9cd4d680a9f19cd85ca32e1bdc69ed11290a2f3ede974 |
| smoothed-d32 | smoothed | 1 | 0.00351518 | 0.00314257 | 0dcebc97fc7396f51cc33657b68ff015bc8d32d2a95954d912c82dd0a5f1ef19 |
| hyperbolic-d16 | trained | 10 | 0.00362666 | 0.00318123 | 009f81446efa97ec57f0a13f997efe26257f65f7bd8a1886d4a28a957830f138 |
| euclidean-d8 | trained | 5 | 0.00368968 | 0.00329541 | 56cb5849f206dfb317c8295bdbdf6f6e2c98394baf9c6ff995bead927ad33f27 |
| euclidean-d16 | trained | 5 | 0.00364302 | 0.0032263 | 2fb7480d88ede1865848e7f407f182496c4b5c55038a06025837e4dfd81f27da |
| euclidean-d32 | trained | 5 | 0.00360295 | 0.00316308 | 532f06775142169d75191158782323ba052cf21c99a9c8039445fb54e4bbac26 |
| hyperbolic-d8 | trained | 5 | 0.00365956 | 0.00320071 | 879100284d4f403fd8d278de9972cd00335e7878801dddf16bb3ad4125fc55d2 |
| hyperbolic-d32 | trained | 5 | 0.00359603 | 0.00316261 | 103f864ab41dc07c39fb1f530dfb95cbb8d6b75b5e01ac815f2495fa42b8cebb |
| spherical-d8 | trained | 5 | 0.00366581 | 0.00324632 | 7331cbf5980f557950bdfecd7e31438f8ab2770f745d8f2fc09ff43804b0732c |
| spherical-d16 | trained | 5 | 0.00365077 | 0.00323873 | b3d9b7185fbffadf01ae1edea463e7efefdada708b682e25782d4466edd47f54 |
| spherical-d32 | trained | 5 | 0.00361954 | 0.00317971 | 120a7c70f7c87175c5f63b7504f23d9682c88a460d6e9614ec8587fd348237b0 |

MSE is the mean over a candidate's runs of the `covariates+embedding` statistic.

### The canary stop and the pairing tolerance

The first read, `text-only-d8`, ran at 2f153107372f74f6f8bbc16dae7554f3ab7cada7 and failed
`check_reference_pairing`. The margin read had run on macOS 26.6.2. This read ran on macOS 27.2.1,
with the same lock, Python, numpy and Accelerate. 11,406 of the 11,655 reference predictions
differed in the last bits, by at most 6.9e-16. The reference's validation MSE changed by 2.7e-16
relative on `growth_seen` and not at all on `growth_heldout`, so δ is unchanged for any practical
purpose. The plan stopped after that one read, with two validation reads logged.

The owner chose to pair within a tolerance. PR #172 (merged as
37b47c4108275b86e8c9408b5c16a66e47578636) changed `check_reference_pairing` to require the
reference's rows to match exactly and each prediction to match to within 1e-13, absolute. It amended
`specs/completed/economic-candidates.md` §4.8 with the evidence and the bound. The owner kept
`text-only-d8.json` as read, with an exception to the plan's rule of one read commit: `git diff
2f15310..37b47c4 -- src/` touches only the pairing check, its constant, docstrings and the refusal
text, and `uv.lock` and `conf/` did not change. `text-only-d8` passes the amended check, and the
other 17 reads ran at 37b47c4.

One more pin fails on this Mac for the same kind of reason. The level-mode regressor test
`test_level_predictions_are_the_captured_ones_bit_for_bit` compares predictions to literals captured
in an earlier environment, and it fails on macOS 27.2.1. The test's own note says that such a
failure means the environment changed. CI on Linux skips it. This plan did not re-capture the
literals. `specs/deferred_items.md` lists it under `19-economic-decision-read`.

## 2. The decision

The rule is Req 5's with the 2026-10-17 margins. A candidate is adopted over the reference when it
is non-inferior on both panels (the 95 % interval's lower bound is above −δ) and superior on at
least one (the 97.5 % interval's lower bound is above 0). Δ is the reference's MSE minus the
candidate's, so a positive Δ favors the candidate. The decision drew 10,000 replicates with
bootstrap seed 20260924.

δ = 0.00015934 on `growth_seen` (reference MSE 0.00318679).

| Candidate | Runs | MSE | Δ | 95 % non-inferiority interval | 97.5 % superiority interval | Non-inferior | Superior |
|---|---|---|---|---|---|---|---|
| text-only-d8 | 1 | 0.00367451 | -0.0004877 | [-0.001214, +6.507e-06] | [-0.001337, +5.568e-05] | no | no |
| text-only-d16 | 1 | 0.00366432 | -0.0004775 | [-0.001211, +2.528e-05] | [-0.001342, +6.884e-05] | no | no |
| text-only-d32 | 1 | 0.00355201 | -0.0003652 | [-0.0009823, +7.209e-05] | [-0.00108, +0.0001152] | no | no |
| tree-d8 | 1 | 0.00369591 | -0.0005091 | [-0.001238, -3.994e-06] | [-0.001379, +4.6e-05] | no | no |
| tree-d16 | 1 | 0.00364045 | -0.0004537 | [-0.00119, +5.901e-05] | [-0.001319, +0.0001016] | no | no |
| tree-d32 | 1 | 0.00360892 | -0.0004221 | [-0.001165, +8.167e-05] | [-0.001298, +0.0001291] | no | no |
| smoothed-d8 | 1 | 0.00363453 | -0.0004477 | [-0.001124, +2.353e-05] | [-0.001229, +6.836e-05] | no | no |
| smoothed-d16 | 1 | 0.00365557 | -0.0004688 | [-0.001226, +4.374e-05] | [-0.001357, +9.178e-05] | no | no |
| smoothed-d32 | 1 | 0.00351518 | -0.0003284 | [-0.0009455, +0.0001132] | [-0.001052, +0.0001579] | no | no |
| hyperbolic-d16 | 10 | 0.00362666 | -0.0004399 | [-0.001141, +3.805e-05] | [-0.001246, +8.487e-05] | no | no |
| euclidean-d8 | 5 | 0.00368968 | -0.0005029 | [-0.001252, +1.577e-05] | [-0.001379, +6.71e-05] | no | no |
| euclidean-d16 | 5 | 0.00364302 | -0.0004562 | [-0.001176, +4.797e-05] | [-0.001281, +9.601e-05] | no | no |
| euclidean-d32 | 5 | 0.00360295 | -0.0004162 | [-0.001069, +3.63e-05] | [-0.001169, +8.029e-05] | no | no |
| hyperbolic-d8 | 5 | 0.00365956 | -0.0004728 | [-0.001185, +2.071e-05] | [-0.001316, +7.158e-05] | no | no |
| hyperbolic-d32 | 5 | 0.00359603 | -0.0004092 | [-0.00108, +5.272e-05] | [-0.001188, +9.728e-05] | no | no |
| spherical-d8 | 5 | 0.00366581 | -0.000479 | [-0.001215, +2.282e-05] | [-0.001346, +7.193e-05] | no | no |
| spherical-d16 | 5 | 0.00365077 | -0.000464 | [-0.001191, +3.706e-05] | [-0.001315, +8.676e-05] | no | no |
| spherical-d32 | 5 | 0.00361954 | -0.0004328 | [-0.001111, +3.326e-05] | [-0.001231, +8.086e-05] | no | no |

δ = 0.000161958 on `growth_heldout` (reference MSE 0.00323916).

| Candidate | Runs | MSE | Δ | 95 % non-inferiority interval | 97.5 % superiority interval | Non-inferior | Superior |
|---|---|---|---|---|---|---|---|
| text-only-d8 | 1 | 0.00319862 | +4.053e-05 | [-5.466e-05, +0.0001419] | [-6.984e-05, +0.0001592] | yes | no |
| text-only-d16 | 1 | 0.00321648 | +2.268e-05 | [-8.15e-05, +0.0001246] | [-9.782e-05, +0.0001382] | yes | no |
| text-only-d32 | 1 | 0.00312177 | +0.0001174 | [-2.346e-05, +0.0002695] | [-4.409e-05, +0.0002948] | yes | no |
| tree-d8 | 1 | 0.00331969 | -8.054e-05 | [-0.0001634, -2.378e-06] | [-0.0001753, +8.674e-06] | no | no |
| tree-d16 | 1 | 0.00331019 | -7.104e-05 | [-0.0001495, +3.453e-06] | [-0.0001593, +1.415e-05] | yes | no |
| tree-d32 | 1 | 0.00333938 | -0.0001002 | [-0.0001915, -2.465e-05] | [-0.000207, -1.556e-05] | no | no |
| smoothed-d8 | 1 | 0.00318084 | +5.831e-05 | [-4.121e-05, +0.0001687] | [-5.43e-05, +0.0001868] | yes | no |
| smoothed-d16 | 1 | 0.00323545 | +3.71e-06 | [-0.0001161, +0.0001081] | [-0.0001344, +0.0001231] | yes | no |
| smoothed-d32 | 1 | 0.00314257 | +9.658e-05 | [-1.141e-05, +0.0002159] | [-2.673e-05, +0.0002335] | yes | no |
| hyperbolic-d16 | 10 | 0.00318123 | +5.792e-05 | [-4.1e-05, +0.0001654] | [-5.551e-05, +0.000185] | yes | no |
| euclidean-d8 | 5 | 0.00329541 | -5.625e-05 | [-0.0001534, +3.756e-05] | [-0.0001682, +5.099e-05] | yes | no |
| euclidean-d16 | 5 | 0.0032263 | +1.286e-05 | [-9.802e-05, +0.0001206] | [-0.0001143, +0.0001387] | yes | no |
| euclidean-d32 | 5 | 0.00316308 | +7.608e-05 | [-3.735e-05, +0.0001988] | [-5.382e-05, +0.0002184] | yes | no |
| hyperbolic-d8 | 5 | 0.00320071 | +3.844e-05 | [-7.53e-05, +0.0001543] | [-9.064e-05, +0.0001731] | yes | no |
| hyperbolic-d32 | 5 | 0.00316261 | +7.655e-05 | [-2.935e-05, +0.0001892] | [-4.361e-05, +0.0002068] | yes | no |
| spherical-d8 | 5 | 0.00324632 | -7.169e-06 | [-0.0001089, +9.036e-05] | [-0.0001242, +0.0001051] | yes | no |
| spherical-d16 | 5 | 0.00323873 | +4.273e-07 | [-0.0001007, +0.0001061] | [-0.0001147, +0.0001195] | yes | no |
| spherical-d32 | 5 | 0.00317971 | +5.944e-05 | [-4.674e-05, +0.0001711] | [-6.334e-05, +0.000188] | yes | no |

- **Adopted over the reference:** none. **Eligible:** none. **Non-dominated set and tie order:**
  empty. **Chosen:** none; the reference `covariates+ancestors` stands.
- **`growth_seen`.** Every candidate is worse than the reference, by 10.3 to 16.0 % of its MSE. 0 of
  18 are non-inferior, and 0 are superior.
- **`growth_heldout`.** Δ runs from -3.1 % to +3.6 % of the reference's MSE. 16 of 18 are
  non-inferior, and 0 are superior.
- **The largest held-out gain** is `text-only-d32`'s, +0.0001174. On `growth_seen` it is -0.0003652
  with a non-inferiority interval of [-0.0009823, +7.209e-05] against −δ = -0.0001593, so it fails
  there.
- **What the decision says.** No candidate passes Req 5's test against the reference. It does not
  say that the embeddings carry no signal.

## 3. Propagation (split Req 19)

Spec §4.7's rule: propagation helps at d if `smoothed-dN` is adopted over its twin `text-only-dN`.
The comparison is a report and chooses nothing.

`smoothed-d8` against `text-only-d8`: **not adopted**.

| Panel | Δ | 95 % non-inferiority interval | 97.5 % superiority interval | δ | Non-inferior | Superior |
|---|---|---|---|---|---|---|
| growth_seen | +3.998e-05 | [-1.165e-05, +0.000105] | [-1.782e-05, +0.000117] | 0.00015934 | yes | no |
| growth_heldout | +1.778e-05 | [-1.741e-05, +5.421e-05] | [-2.239e-05, +5.863e-05] | 0.000161958 | yes | no |

`smoothed-d16` against `text-only-d16`: **not adopted**.

| Panel | Δ | 95 % non-inferiority interval | 97.5 % superiority interval | δ | Non-inferior | Superior |
|---|---|---|---|---|---|---|
| growth_seen | +8.744e-06 | [-3.776e-05, +5.421e-05] | [-4.408e-05, +6.04e-05] | 0.00015934 | yes | no |
| growth_heldout | -1.897e-05 | [-7.211e-05, +3.029e-05] | [-7.948e-05, +3.742e-05] | 0.000161958 | yes | no |

`smoothed-d32` against `text-only-d32`: **not adopted**.

| Panel | Δ | 95 % non-inferiority interval | 97.5 % superiority interval | δ | Non-inferior | Superior |
|---|---|---|---|---|---|---|
| growth_seen | +3.683e-05 | [-1.481e-05, +9.213e-05] | [-2.114e-05, +0.0001012] | 0.00015934 | yes | no |
| growth_heldout | -2.08e-05 | [-8.022e-05, +3.622e-05] | [-8.772e-05, +4.482e-05] | 0.000161958 | yes | no |

**Answer.** Propagation helps at none of d = 8, 16, 32. At each d the smoothed candidate is
non-inferior to its twin on both panels and superior on neither. This finding records no evidence
that smoothing over lineage helps, and the parked graph stage stays parked for this artifact.

## 4. Provenance

Each candidate's provenance names text and tree inputs only (Req 29). `check_candidate_provenance`
passed for every run of every arm. The table shows the first run's provenance for each candidate.

| Candidate | Kind | Runs | Builder | Provenance file (first run) | SHA-256 |
|---|---|---|---|---|---|
| text-only-d8 | text-only | 1 | text-only | `objects/9c/9c0475375231046d8a6184284d15d9d443f84c3cc62815ebbd0f86b01baa5ed4/text-only-d8_provenance.json` | 9c0475375231046d8a6184284d15d9d443f84c3cc62815ebbd0f86b01baa5ed4 |
| text-only-d16 | text-only | 1 | text-only | `objects/9e/9e8c18ad532b2afff0aa0d09ea3408e02c2ef431bd95a69a3c845f3db77e9a7f/text-only-d16_provenance.json` | 9e8c18ad532b2afff0aa0d09ea3408e02c2ef431bd95a69a3c845f3db77e9a7f |
| text-only-d32 | text-only | 1 | text-only | `objects/8c/8c6100bbf9f3cfa6020da5ac63490d4254664986529ab191171d87819ef7d6ed/text-only-d32_provenance.json` | 8c6100bbf9f3cfa6020da5ac63490d4254664986529ab191171d87819ef7d6ed |
| tree-d8 | tree | 1 | tree | `objects/54/54f9567d338f32b08162784bbe469b4ebe35a4c858e89e8ee54fef79175acd77/tree-d8_provenance.json` | 54f9567d338f32b08162784bbe469b4ebe35a4c858e89e8ee54fef79175acd77 |
| tree-d16 | tree | 1 | tree | `objects/64/649efee26e31832628d965940f335d9e9cf5f0a978fddb35ef296c70d70624ba/tree-d16_provenance.json` | 649efee26e31832628d965940f335d9e9cf5f0a978fddb35ef296c70d70624ba |
| tree-d32 | tree | 1 | tree | `objects/08/08422527123d48a28d2bd7eaeb9e63666b8c0d7e2746389038c46c2714261a2c/tree-d32_provenance.json` | 08422527123d48a28d2bd7eaeb9e63666b8c0d7e2746389038c46c2714261a2c |
| smoothed-d8 | smoothed | 1 | smoothed | `objects/17/175617229073d06407110cbbe045c2c5dd9795d5867863437ba68f8f4376ae1e/smoothed-d8_provenance.json` | 175617229073d06407110cbbe045c2c5dd9795d5867863437ba68f8f4376ae1e |
| smoothed-d16 | smoothed | 1 | smoothed | `objects/c6/c699a800ae79b78cbc28044c6fc06cad1eb0c9eb682838eafd7a69080594ca8c/smoothed-d16_provenance.json` | c699a800ae79b78cbc28044c6fc06cad1eb0c9eb682838eafd7a69080594ca8c |
| smoothed-d32 | smoothed | 1 | smoothed | `objects/0d/0dd4885cb3b35a83f7e53ced6665a53fa6fc6361f1cdbb4e0d8a4fa4c74944c7/smoothed-d32_provenance.json` | 0dd4885cb3b35a83f7e53ced6665a53fa6fc6361f1cdbb4e0d8a4fa4c74944c7 |
| hyperbolic-d16 | trained | 10 | export | `objects/97/976ce5adeba20e7585d4184683c6dd87f0cb37a698ae0e50115df05c9be2bd3d/arm_table_epoch=003_provenance.json` | 976ce5adeba20e7585d4184683c6dd87f0cb37a698ae0e50115df05c9be2bd3d |
| euclidean-d8 | trained | 5 | export | `objects/98/98f1babe93d8ba25a09abd133b5d6add1b8c63883b9a743ea98355569de49c9b/arm_table_epoch=038_provenance.json` | 98f1babe93d8ba25a09abd133b5d6add1b8c63883b9a743ea98355569de49c9b |
| euclidean-d16 | trained | 5 | export | `objects/c1/c1bc0d146a3ef12113f3dca2d3ba7e4b667a4a7e4d25c101b20215202be9fb28/arm_table_epoch=039_provenance.json` | c1bc0d146a3ef12113f3dca2d3ba7e4b667a4a7e4d25c101b20215202be9fb28 |
| euclidean-d32 | trained | 5 | export | `objects/a5/a5d4a7ac6c04c1fa81043a9b55470d6c6b75a630a0b0c22bcb15624eb0c4499b/arm_table_epoch=001_provenance.json` | a5d4a7ac6c04c1fa81043a9b55470d6c6b75a630a0b0c22bcb15624eb0c4499b |
| hyperbolic-d8 | trained | 5 | export | `objects/04/04b859217cbde892329a9cb453b8fd9e0f3afe6dfd37922478484f139f19c472/arm_table_epoch=039_provenance.json` | 04b859217cbde892329a9cb453b8fd9e0f3afe6dfd37922478484f139f19c472 |
| hyperbolic-d32 | trained | 5 | export | `objects/c8/c8596272674685bd4385823923b453276e77f89728d3eee06dc6005e8dcfc775/arm_table_epoch=002_provenance.json` | c8596272674685bd4385823923b453276e77f89728d3eee06dc6005e8dcfc775 |
| spherical-d8 | trained | 5 | export | `objects/40/4007cbfe05776524e0dde71fca869330b4a4bb402dad8c476ea776712348e8c7/arm_table_epoch=003_provenance.json` | 4007cbfe05776524e0dde71fca869330b4a4bb402dad8c476ea776712348e8c7 |
| spherical-d16 | trained | 5 | export | `objects/83/83aa9738d650828e8ce0e5aaf6a6cb594e0e89447acb65aaa7899fd6df8489e0/arm_table_epoch=001_provenance.json` | 83aa9738d650828e8ce0e5aaf6a6cb594e0e89447acb65aaa7899fd6df8489e0 |
| spherical-d32 | trained | 5 | export | `objects/6b/6b53e0d55b262f8648e44a86dcc3bdaecb5089c12e246fefdd1f3122d35767d8/arm_table_epoch=001_provenance.json` | 6b53e0d55b262f8648e44a86dcc3bdaecb5089c12e246fefdd1f3122d35767d8 |

## 5. The selection log

| | Lines | SHA-256 |
|---|---|---|
| Before the reads (the margin read's two growth records, no candidate read) | 159 | 51aef24099f0303c218271fc9ab110c3f1a8e7a56198d97eab3c84a0731a2a41 |
| After the 18 reads and the decision | 277 | 85013962623b5751a1d01dd199a2d2de876817e47f8f2b92713ef33cab18cad0 |

- The first 159 lines are unchanged, and the log gained 118 records, all validation reads of a
  growth panel with the plan's purpose (59 of `growth_seen`, 59 of `growth_heldout`).
- No record is anything other than a validation read. Neither regime has an opening under the
  held-out draw `deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4`.
- Only two growth records carry the margin read's purpose, so `tools growth-margins` did not run
  again. The decision reads the stored predictions and wrote no log record.

## 6. What embedding roadmap Stage 12 reads

- **The decision record:** `~/naics-artifacts/records/economic/decision.json`, SHA-256
  3d510fa319865e84b0321435fe607081d0cbd5987768f68233650237bb69515d.
- **The chosen candidate's table:** none. The reference stands, so the economic opening has no
  candidate table to read.
- **The margin record:** `growth_margins.json`, SHA-256
  4f86a0b2f3bde358bd08a1f3f17f1b4c59ee261ee4e5409ac83615deaa1f37d0.
- **The environment an opening must pair under:** macOS arm64, Python 3.12.12, numpy 2.3.4 on
  Accelerate, `uv.lock` 6de4fbdc, with the reference's predictions equal to the margin read's to
  within 1e-13 (`economic-candidates` §4.8, amended 2026-11-18).
- **The outer sets** of both growth regimes stay sealed until Stage 12 opens them once.
