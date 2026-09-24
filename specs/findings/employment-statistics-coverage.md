# Employment-statistics coverage: finding

**Status: FINAL (2026-09-24).** Roadmap Stage 1 (`specs/naics-embedding-roadmap.md`). This
finding discharges Verification "Employment-statistics coverage" and the Req 2 (open) item of
`specs/naics-embedding.md` (d9126ce). It is produced by plan 3 with
`scripts/employment_statistics_coverage.py`. Stage 3 reads the decision block verbatim.

## Decision for Stage 3

<!-- decision:begin -->
- **Branch:** A. The verified window supports a time-respecting outcome (the outcome dated after the features, with splits by time), so the panel includes one.
- **Source:** QCEW annual averages, reference years 2022, 2023, 2024, 2025, private ownership (own_code 5).
- **Row grain:** a six-digit code in a reference year (national, private ownership).
- **Population:** 980 codes for the seen-code regime and 980 for the held-out-code regime, of the 1,012 six-digit codes in the codebook.
- **Time-respecting outcome:** yes.
- **Seen-code regime:** yes.
- **Rule:** plan 3, survival floor 506 codes, ask band 405 to 607; user review not required.
- **Reasons:**
  - national: 980 codes can run the seen-code regime (floor 506); mean suppressed share 0.0000
  - state: 978 codes can run the seen-code regime (floor 506); mean suppressed share 0.2379
  - county: 948 codes can run the seen-code regime (floor 506); mean suppressed share 0.7384
  - msa: not a candidate (not a candidate grain, or no six-digit rows in a window year)
  - national: 980 codes are time-eligible (floor 506)
<!-- decision:end -->

## 1. Reference years published on NAICS 2022 at six digits

The window is 2022, 2023, 2024 and 2025: every reference year coded on NAICS 2022 that has an
annual-average single file. All four years are final on the read date.

- **Finality.** BLS's release calendar says "Final quarterly and annual averages data for each
  year will be available with the release of first quarter data (preliminary) for the subsequent
  year." First-quarter 2026 data came out on "Friday, Aug. 28, 2026", which made 2025 final; the
  first-quarter releases of 2023, 2024 and 2025 had made 2022, 2023 and 2024 final (Sources,
  Pages).
- **The two 2025 dates.** The 2025 files carry Last-Modified dates of Fri, 21 Aug 2026 (12:51:16
  GMT for the national slice, 13:12:50 GMT for the single file), a week before that release. Both
  dates are recorded here as read. The same order holds for every window year: each year's files
  carry a Last-Modified date before the first-quarter release that made the year final (Sources,
  Files and Pages).
- **The NAICS 2022 boundary.** BLS introduced NAICS 2022 "on September 7, 2022, with the full data
  release of first quarter 2022 Quarterly Census of Employment and Wages (QCEW) data", and "Data
  from 2022 forward are classified under the NAICS 2022 system." while "Data from 2017-2021 are
  classified under the NAICS 2017 system." (Sources, Pages).
- **The files agree** ("Vintage check (national six-digit codes)"). The check sets aside the 38
  BLS 238 codes and 999999, which every slice publishes outside the codebook by design
  (section 6). Besides those, the 2021 national slice publishes 139 six-digit codes outside the
  NAICS 2022 codebook among its 1,075 (212111, 212112, 212113 and others), and leaves 96 codebook
  codes unpublished besides the split codes. Each of 2022–2025 publishes 1,029 codes, 0 of them
  outside the codebook besides the 38 BLS 238 codes and 999999, and leaves the same 3 codebook
  codes unpublished: 112130, 517122 and 541120, which BLS lists as not used in the United States
  (the first and last) or not used by BLS (517122).
- **No 2026 annual file.** Annual averages are "published with the 4th quarter data for that
  reference year", and the calendar dates fourth-quarter 2026 data "To be determined, 2027". The
  window therefore ends at 2025.
- **No earlier years.** Years before 2022 are coded on NAICS 2017, and Req 2 excludes them:
  "Training on changes coded on an earlier NAICS vintage (Claude C28) is not used without a
  concordance (rejected here; cross-vintage work is Out of scope)."

## 2. Grains at which six-digit series are published

QCEW publishes NAICS 2022 six-digit cells at four grains, each at an aggregation level titled
"-- by ownership sector" in `agglevel_titles.csv` ("Lookup files"). Area counts are the `areas`
column of "Private cells by grain and year".

- **By year: national** (six-digit level 18, five-digit level 17). One area, `US000`, in every
  year.
- **By area: state** (58 and 57). 53 areas in every year: the 51 codes up to FIPS 56 (the 50
  states and DC) plus Puerto Rico (`72000`) and the Virgin Islands (`78000`).
- **By area: county** (78 and 77). 3,223 areas in 2022 and 2023 and 3,224 in 2024 and 2025, not
  counting the "Unknown Or Undefined" codes ending in 999. Connecticut's 8 legacy counties give
  way to its 9 planning regions in 2024 ("Connecticut county-equivalents"). BLS suspended
  Colorado's industry and substate data on November 20, 2024 and resumed them on February 19,
  2025; Colorado has 64 counties with private six-digit rows in every year ("Single-file scan").
- **By area: MSA** (48 and 47). 388 areas in 2022 and 2023, 393 in 2024 and none in 2025. The 2024
  change follows the new OMB delineations, which "QCEW data will reflect" from first-quarter 2024
  data, and "Historical data will not be re-tabulated". BLS dropped MSA industry detail from
  third-quarter 2025 data, so the 2025 annual file has no MSA six-digit rows ("MSA six-digit rows
  per year"). MSA rows carry private ownership only ("Single-file scan").
- **CSA and MicroSA** carry no industry detail: `agglevel_titles.csv` gives them only level 30,
  "CMSA or CSA, Total Covered", and level 80, "MicroSA, Total Covered" ("Lookup files").

Six-digit cells are published by ownership, with no total-covered row: no six-digit row in any
year has `own_code` 0 ("File conventions (six-digit rows)"). Summing ownerships is invalid wherever
one of them is suppressed, so the panel reads private ownership (`own_code` 5).

## 3. Suppressed share per year, grain and series

From "Private cells by grain and year". Cell shares divide by the published private cells. The
last column divides by the codebook's 1,012 six-digit codes, so an absent code counts as a code
with no usable cell.

| Grain | Year | Areas | Published cells | Suppressed share of cells (employment and wages) | Suppressed share of cells (establishments) | Codes with no usable cell (of 1,012) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| national | 2022 | 1 | 980 | 0.0000 | 0.0000 | 32 (0.0316) |
| national | 2023 | 1 | 980 | 0.0000 | 0.0000 | 32 (0.0316) |
| national | 2024 | 1 | 980 | 0.0000 | 0.0000 | 32 (0.0316) |
| national | 2025 | 1 | 980 | 0.0000 | 0.0000 | 32 (0.0316) |
| state | 2022 | 53 | 45,737 | 0.2225 | 0.0017 | 35 (0.0346) |
| state | 2023 | 53 | 45,819 | 0.2184 | 0.0012 | 34 (0.0336) |
| state | 2024 | 53 | 45,848 | 0.2495 | 0.0031 | 34 (0.0336) |
| state | 2025 | 53 | 45,824 | 0.2614 | 0.0031 | 34 (0.0336) |
| county | 2022 | 3,223 | 866,433 | 0.7153 | 0.0136 | 56 (0.0553) |
| county | 2023 | 3,223 | 878,426 | 0.7157 | 0.0147 | 59 (0.0583) |
| county | 2024 | 3,224 | 880,769 | 0.7568 | 0.0255 | 62 (0.0613) |
| county | 2025 | 3,224 | 880,614 | 0.7660 | 0.0242 | 64 (0.0632) |
| MSA | 2022 | 388 | 232,153 | 0.6402 | 0.0077 | 51 (0.0504) |
| MSA | 2023 | 388 | 234,131 | 0.6381 | 0.0076 | 53 (0.0524) |
| MSA | 2024 | 393 | 238,195 | 0.8074 | 0.0139 | 120 (0.1186) |
| MSA | 2025 | 0 | 0 | - | - | 1,012 (1.0000) |

At the national grain, private ownership ("National grain: codebook codes by status",
`own_code` 5, the `employment` rows):

- 2022: 980 disclosed, 0 suppressed, 0 other, 32 absent.
- 2023: 980 disclosed, 0 suppressed, 0 other, 32 absent.
- 2024: 980 disclosed, 0 suppressed, 0 other, 32 absent.
- 2025: 980 disclosed, 0 suppressed, 0 other, 32 absent.

Employment and wages share one disclosure flag, so their rows in that table are identical in every
year and ownership. Establishment counts survive suppression when positive: at the state grain in
2022, 10,176 private cells are suppressed for employment and wages but only 80 for
establishments. The government ownerships, reported but not read by the rule, are suppressed
nationally: state government (`own_code` 2) suppresses 80 to 95 codes a year and local government
(3) 163 to 179, while federal government (1) suppresses none.

Suppression removes small cells ("Establishments of disclosed and suppressed private cells"). The
median suppressed cell has 3 to 4 establishments at the state grain against 56 to 65 for a
disclosed cell, 1 against 9 to 12 at the county grain, and 2 to 3 against 14 to 15 at the MSA
grain. The national grain has no suppressed private cell. At any area grain, dropping suppressed
cells would therefore truncate the outcome from below.

## 4. Panel population, row grain, time-respecting outcome and seen-code regime

The pre-registered rule names **branch A** at the **national grain**. A row is a six-digit code in
a reference year, in private ownership, from QCEW annual averages for 2022–2025. Of the codebook's
1,012 six-digit codes, 980 can run the seen-code regime, and the same 980 form the held-out-code
population (decision block). No deciding count fell in the ask band of 405 to 607, so the rule
needed no user review.

- **Time-respecting outcome: yes.** The window is four consecutive final years, 2022–2025, above
  the minimum of 3. At the national grain 980 codes are time-eligible against the floor of 506:
  each has a usable pair for 2024→2025 and a usable earlier pair.
- **Seen-code regime: yes.** All three candidate grains survive. 980 codes can run the regime at
  the national grain, 978 at the state grain and 948 at the county grain, each above the floor of
  506. The rule takes the grain with the lowest mean suppressed share: national at 0.0000, against
  0.2379 for state and 0.7384 for county. MSA is not a candidate: it has no six-digit rows in
  2025.
- **Excluded codes: 32**, all for "no private cell" ("Codes excluded at the chosen grain"). Three
  have no national cell in any ownership: 112130, 517122 and 541120, the codes BLS does not use
  (section 1). The other 29 are public-administration codes in NAICS 92, 921110 to 928120, with
  cells in government ownerships only ("Codes with no private national cell (last year)").

Neither answer is "no". The panel therefore follows Req 2's first branch: "If the verified window
supports a time-respecting outcome (the outcome dated after the features, with splits by time),
the panel includes one (ChatGPT C15; Claude C28)."

## 5. Other public employment series screened

**CES.** The CES handbook (`https://www.bls.gov/opub/hom/ces/presentation.htm`, read 2026-09-24)
says "Using data from the CES sample, the CES-National program produces and publishes thousands of
data series, including national estimates of employment, hours, and earnings by detailed
industry." The CES Published Series page (`https://www.bls.gov/web/empsit/cesseriespub.htm`, read
2026-09-24) adds that "CES industry codes are based on NAICS codes, but are not always a
one-to-one mapping with the NAICS structure due to sample size limitations." CES cannot serve: its
estimates come from a sample, not a census count; its industry detail does not reach all 1,012
six-digit codes; and it publishes no establishment counts beside employment and earnings.

**OEWS.** The OEWS handbook (`https://www.bls.gov/opub/hom/oews/presentation.htm`, read
2026-09-24) says the program publishes "U.S. industry-specific estimates by 2-, 3-, most 4-, and
some 5- and 6-digit NAICS levels", and that "Modeled estimates developed from a sample will differ
from the results of a census." OEWS cannot serve: its industry detail stops short of all six-digit
codes, its estimates are modeled from a sample, and its rows are occupational employment and wage
estimates with no establishment counts.

**BED.** The BED handbook (`https://www.bls.gov/opub/hom/bdm/presentation.htm`, read 2026-09-24)
says "Data on the private sector are available for the nation as a whole and by NAICS sector and
subsector. In addition, BED state data are available by NAICS sector." BED cannot serve: its
industry detail ends at the three-digit subsector.

Roadmap D1 settles the source: "covariates are log establishment counts and log wages from the same
QCEW rows". Among these programs only QCEW publishes employment, establishment counts and wages on
one row for each six-digit code, so QCEW is the panel's source.

## 6. File conventions verified

- **Disclosure codes and columns** ("File conventions (six-digit rows)", over every six-digit level
  and ownership). The disclosure code is blank or `N`, with no other code in any year: 467,112
  blank and 852,716 `N` rows in 2022, 472,235 and 862,595 in 2023, 392,184 and 949,625 in 2024, and
  336,210 and 761,027 in 2025. The establishment column is `annual_avg_estabs` in every year. No
  six-digit row has `own_code` 0.
- **Suppressed rows.** Every `N` row has zero employment and wages
  (`suppressed_rows_with_emp_or_wages` is 0 in every year), and most keep a positive establishment
  count: 837,846 of 852,716 in 2022, 846,712 of 862,595 in 2023, 921,691 of 949,625 in 2024 and
  737,563 of 761,027 in 2025. The rule reads an `N` row's zero establishment count as suppressed.
- **Split codes.** QCEW publishes none of the codebook's 19 six-digit codes under NAICS 238. BLS
  "uses six-digit NAICS codes ending in “1” for residential construction units and “2” for
  nonresidential construction units, instead of “0.”" (Sources, Pages), and `industry_titles.csv`
  lists the 38 such codes and no 238 code ending in 0 ("Lookup files"). Each of the 19 is its
  five-digit parent's only child, so the script reads each from the parent's row ("Split codes
  recovered from their five-digit parent"; `recovered_via_parent` is 19 for private ownership in
  every year). There are no direct 238 codes. The plan expected 17 split codes, two direct 238
  codes and 34 BLS codes, inferred from BLS's count of "1,030 industries"; that count also leaves
  out the two codes not used in the United States (1,012 − 19 + 38 + 1 − 2 = 1,030). The run uses
  19 under the project owner's ruling of 2026-09-24.
- **999999.** `industry_titles.csv` has one 999999 row ("Lookup files"), and the vintage check
  treats the code as expected outside the codebook.
- **Unknown and pseudo-counties.** `area_titles.csv` has 53 "Unknown Or Undefined" codes ending in
  999 ("Lookup files"). 51 of them carry county-level rows in each year's file, and the county
  grain drops them all. No overseas, multi-county or out-of-state code (ending in 996, 997 or 998)
  and no legacy Alaska division carries county-level rows ("Single-file scan").
- **Connecticut and MSA.** Section 2 gives the Connecticut recode ("Connecticut county-equivalents")
  and the MSA break ("MSA six-digit rows per year", "Single-file scan").
- **Reconciliation.**
  - The single files and the Open Data Access `US000` slices agree on every national row in every
    year: "Invariant failures" lists no disagreement between them.
  - Disclosed state detail never exceeds its national cell, and disclosed county detail never
    exceeds its state cell, for any code and year: "Invariant failures" has no such row.
  - Disclosed national six-digit employment exceeds the national private total by 46 in 2022
    (128,718,106 against 128,718,060) and by 27 in 2023 (131,289,708 against 131,289,681). These
    are the two rows of "Invariant failures", and the run exits 2 on them. "National six-digit
    reconciliation" shows that they are rounding, not a data fault: all 1,000 national private
    six-digit rows are disclosed, wages, which are exact dollar sums, reconcile to the total with a
    difference of 0 in every year, and the employment residual changes sign (+46, +27, -5, -9), as
    the sector-level residual does (+0, +2, -1, -1). Each annual-average employment figure is
    rounded separately, and the national check is the one comparison in the script with no
    rounding allowance. Under the project owner's ruling of 2026-09-24 the run stands as it is,
    with no re-run and no change to the check.
- **Independent recount.** The stdlib-`csv` recount of plan 3 Task 6 Step 3 gives 980 disclosed, 0
  suppressed, 0 other and 32 absent codes in every year ("Independent recount"). That matches the
  `employment` rows for `own_code` 5 in "National grain: codebook codes by status".

## 7. Consequences for later stages

- Stage 3 takes the decision block: branch A, one row per six-digit code and reference year
  (national, private ownership) for 2022–2025, with 980 codes for the seen-code regime and 980 for
  the held-out-code regime.
- D1's covariates: a usable row carries employment and wages under one disclosure flag.
  Establishment counts stay published on most suppressed rows, but a suppressed row is not usable.
  At the national grain no private cell is suppressed in 2022–2025, so every row carries all three
  series.
- `metrics/qcew.py` reads `tot_wages` where the files say `total_annual_wages`, and keeps one row
  per code (2022, private): the rejected definition. It also selects no `agglvl_code`,
  `area_fips` or `disclosure_code`, so its per-code mean pools the rows of every grain and reads
  zero-filled suppressed cells as zeros.
- An area design must handle Connecticut's 2024 recode. An MSA design has only 2022–2024 and a
  2023/2024 break.
- At any area grain, suppression removes small cells (section 3), so a design that drops them
  truncates the outcome from below.
- The 2026 annual file does not exist yet, so the window ends at 2025.
- derive-roadmap's resume step re-validates later stages against this finding; no other stage
  entry is edited here.

## Sources

### Files

Read from `~/Downloads/Data/QCEW/`. Download times are the header dumps' modification times
(UTC).

| File | URL | Bytes | sha256 | Last-Modified | Downloaded (UTC) |
| --- | --- | ---: | --- | --- | --- |
| `2022_annual_singlefile.zip` | `https://data.bls.gov/cew/data/files/2022/csv/2022_annual_singlefile.zip` | 77,024,919 | `29f852b0f6405615b45e41d40e930d2e93dd8cecab505f8395b0ceb50061551d` | Thu, 31 Aug 2023 13:49:04 GMT | 2026-09-24T15:11:43Z |
| `2023_annual_singlefile.zip` | `https://data.bls.gov/cew/data/files/2023/csv/2023_annual_singlefile.zip` | 82,932,544 | `6284aff110a81196803c4f2ab8613a511838878d1763b8a1a8a21231200b8f61` | Thu, 29 Aug 2024 14:43:53 GMT | 2026-09-24T15:11:47Z |
| `2024_annual_singlefile.zip` | `https://data.bls.gov/cew/data/files/2024/csv/2024_annual_singlefile.zip` | 74,697,761 | `334ac1ec9101f516ed9698859a165666c6f6d08a51b35cb5fc1e1bf62e06b0da` | Tue, 02 Sep 2025 11:20:46 GMT | 2026-09-24T15:11:54Z |
| `2025_annual_singlefile.zip` | `https://data.bls.gov/cew/data/files/2025/csv/2025_annual_singlefile.zip` | 62,799,396 | `94ebbc589b1b500671c1fd507e2589a9eeb24dcd30eb07f1ce37b0018643f865` | Fri, 21 Aug 2026 13:12:50 GMT | 2026-09-24T15:11:57Z |
| `2021_US000_annual.csv` | `https://data.bls.gov/cew/data/api/2021/a/area/US000.csv` | 846,843 | `2a6714f05b9fec6a4487310fe03f97439655e7f6206906e1db99e5d2eda3a3e7` | Wed, 31 Aug 2022 19:18:44 GMT | 2026-09-24T15:11:15Z |
| `2022_US000_annual.csv` | `https://data.bls.gov/cew/data/api/2022/a/area/US000.csv` | 823,163 | `c45cbb64a1b1eef16bfd743510d9d02792ccad82f60e9df202c5daa3e8c5cc18` | Thu, 31 Aug 2023 14:13:44 GMT | 2026-09-24T15:11:23Z |
| `2023_US000_annual.csv` | `https://data.bls.gov/cew/data/api/2023/a/area/US000.csv` | 849,504 | `fe9ffe874f6e657f6bb1558971965ce6acc015ace45d831ed32c90d97097aee9` | Wed, 28 Aug 2024 21:18:54 GMT | 2026-09-24T15:11:25Z |
| `2024_US000_annual.csv` | `https://data.bls.gov/cew/data/api/2024/a/area/US000.csv` | 847,197 | `48db086828a01798731242c6d3d4957f80f941afe75463a1ff7d43de774bea46` | Tue, 02 Sep 2025 11:07:10 GMT | 2026-09-24T15:11:26Z |
| `2025_US000_annual.csv` | `https://data.bls.gov/cew/data/api/2025/a/area/US000.csv` | 841,756 | `0b5528f70d66a84ff9729691f365c667a09f854f0af3d841bdd660ef3cb01811` | Fri, 21 Aug 2026 12:51:16 GMT | 2026-09-24T15:11:28Z |
| `industry_titles.csv` | `https://data.bls.gov/cew/doc/titles/industry/industry_titles.csv` | 164,248 | `facc122ef582f3efdbb5294f103331894d8fdab694de38e4ee7bff1b941ab8e1` | Wed, 31 Aug 2022 17:09:54 GMT | 2026-09-24T15:11:32Z |
| `agglevel_titles.csv` | `https://data.bls.gov/cew/doc/titles/agglevel/agglevel_titles.csv` | 2,750 | `7de7143e739476ff4ef605a4f043550d376b4d5695260d0674f6e0c7f9f95250` | Fri, 15 Oct 2010 05:00:00 GMT | 2026-09-24T15:11:34Z |
| `area_titles.csv` | `https://data.bls.gov/cew/doc/titles/area/area_titles.csv` | 343,559 | `92ec00f591bd986cd46e027a05e35b33b0d2d724f3b76c96969f0e73145698d6` | Fri, 06 Sep 2024 17:10:14 GMT | 2026-09-24T15:11:36Z |
| `ownership_titles.csv` | `https://data.bls.gov/cew/doc/titles/ownership/ownership_titles.csv` | 230 | `9586b30199a494e1add8615998638801b58cc7d52dacacd5e62b41feab5b11b6` | Fri, 15 Oct 2010 05:00:00 GMT | 2026-09-24T15:11:37Z |

### Pages

Each page was opened in the built-in browser and read with `get_page_text` on 2026-09-24,
between 15:15 and 15:19 UTC. Quotes are verbatim; a page with more than one fact has one row per
quote.

| Page | URL | Read (UTC) | Quote |
| --- | --- | --- | --- |
| QCEW Introduces NAICS 2022 Industry Coding | `https://www.bls.gov/cew/classifications/industry/naics-2022.htm` | 2026-09-24 | "This revision will be introduced by the Bureau of Labor Statistics (BLS) on September 7, 2022, with the full data release of first quarter 2022 Quarterly Census of Employment and Wages (QCEW) data." |
| QCEW Introduces NAICS 2022 Industry Coding | `https://www.bls.gov/cew/classifications/industry/naics-2022.htm` | 2026-09-24 | "Overall, in QCEW, there are 21 sectors and 1,030 industries in NAICS 2022, including the BLS-specific residential and non-residential industry codes within NAICS 238 (Specialty Trade Contractors), as well as the inclusion of an "Unclassified" (NAICS 999999) industry designation, for worksites where an economic activity has not been identified." |
| Industry Classification Systems Used By QCEW | `https://www.bls.gov/cew/classifications/industry/` | 2026-09-24 | "Data from 2022 forward are classified under the NAICS 2022 system." and "Data from 2017-2021 are classified under the NAICS 2017 system." |
| BLS and QCEW NAICS Differences | `https://www.bls.gov/cew/additional-resources/bls-and-qcew-naics-differences.htm` | 2026-09-24 | "For the specialty trade contractor industries (NAICS 238), BLS uses six-digit NAICS codes ending in “1” for residential construction units and “2” for nonresidential construction units, instead of “0.”" The page then lists 38 codes, 238111 through 238992. |
| BLS and QCEW NAICS Differences | `https://www.bls.gov/cew/additional-resources/bls-and-qcew-naics-differences.htm` | 2026-09-24 | "There are two NAICS codes included in the NAICS 2022 manual that are not used in the United States." (112130 and 541120) and "There is one NAICS code included in the NAICS 2022 manual that is not used by BLS." (517122) |
| Schedule of News Releases and Full Data Availability for County Employment and Wages | `https://www.bls.gov/cew/release-calendar.htm` | 2026-09-24 | "Final quarterly and annual averages data for each year will be available with the release of first quarter data (preliminary) for the subsequent year." |
| Schedule of News Releases and Full Data Availability for County Employment and Wages | `https://www.bls.gov/cew/release-calendar.htm` | 2026-09-24 | First-quarter release rows: 1st Quarter 2026, "Friday, Aug. 28, 2026"; 1st Quarter 2025, "Tuesday, Sep. 9, 2025"; 1st Quarter 2024 full data, "Wednesday, Sep. 4, 2024"; 1st Quarter 2023 full data, "Wednesday, Sep. 6, 2023". |
| Schedule of News Releases and Full Data Availability for County Employment and Wages | `https://www.bls.gov/cew/release-calendar.htm` | 2026-09-24 | "Concerning the QCEW full data update, for each reference year, preliminary quarterly data are published for each quarter on the date of the news release, including annual average data (preliminary) published with the 4th quarter data for that reference year." The 4th Quarter 2026 row reads "To be determined, 2027". |
| Change in the presentation of Metropolitan Statistical Area (MSA) data in QCEW | `https://www.bls.gov/cew/notices/2025/change-in-the-presentation-of-metropolitan-statistical-area-data-in-qcew.htm` | 2026-09-24 | "Beginning with third quarter 2025 data, to be released on March 10, 2026, the Quarterly Census of Employment and Wages (QCEW) will publish Metropolitan Statistical Area (MSA) employment and wages at only the total covered employment for the area. Data for detailed industry within the area will no longer be available." |
| QCEW News Release Notes (August 21, 2024, 2024/1) | `https://www.bls.gov/cew/about-data/news-release-notes.htm` | 2026-09-24 | "The replacement of Connecticut's eight counties with the state's nine planning regions was announced in June 2022. Effective with this news release, the QCEW program will tabulate data using the new planning regions as county-equivalents." |
| QCEW News Release Notes (August 21, 2024, 2024/1) | `https://www.bls.gov/cew/about-data/news-release-notes.htm` | 2026-09-24 | "With the full data update for first quarter 2024 on September 4, 2024, QCEW data will reflect the new definitions." and "Historical data will not be re-tabulated to reflect the new definitions." (the entry links OMB Bulletin 23-01) |
| QCEW News Release Notes (March 10, 2026, 2025/3) | `https://www.bls.gov/cew/about-data/news-release-notes.htm` | 2026-09-24 | "With this release, the Quarterly Census of Employment and Wages (QCEW) will publish Metropolitan Statistical Area (MSA) employment and wages only at the total covered employment aggregation level. Data for detailed industry within MSAs will no longer be available." |
| QCEW News Release Notes (August 24, 2022, 2022/1) | `https://www.bls.gov/cew/about-data/news-release-notes.htm` | 2026-09-24 | "Beginning with the full release of first quarter 2022 data on September 7, 2022, the QCEW program will use the 2022 version of the North American Industry Classification System (NAICS) as the basis for the publication of economic data by industry." |
| QCEW News Release Notes (November 20, 2024, 2024/2, and February 19, 2025, 2024/3) | `https://www.bls.gov/cew/about-data/news-release-notes.htm` | 2026-09-24 | "Effective with this release, the Quarterly Census of Employment and Wages (QCEW) is suspending publication of industry and substate employment and wage data for Colorado because of data quality issues." and "Effective with this release, QCEW is resuming publication of substate and industry employment and wage data for Colorado." |
| Current Employment Statistics - National: Presentation | `https://www.bls.gov/opub/hom/ces/presentation.htm` | 2026-09-24 | "Using data from the CES sample, the CES-National program produces and publishes thousands of data series, including national estimates of employment, hours, and earnings by detailed industry." |
| CES Published Series | `https://www.bls.gov/web/empsit/cesseriespub.htm` | 2026-09-24 | "CES industry codes are based on NAICS codes, but are not always a one-to-one mapping with the NAICS structure due to sample size limitations." |
| Occupational Employment and Wage Statistics: Presentation | `https://www.bls.gov/opub/hom/oews/presentation.htm` | 2026-09-24 | "The Occupational Employment and Wage Statistics (OEWS) program publishes cross-industry occupational data for the United States as a whole, for individual states, and for metropolitan and nonmetropolitan areas, along with U.S. industry-specific estimates by 2-, 3-, most 4-, and some 5- and 6-digit NAICS levels." and "Modeled estimates developed from a sample will differ from the results of a census." |
| Business Employment Dynamics: Presentation | `https://www.bls.gov/opub/hom/bdm/presentation.htm` | 2026-09-24 | "Data on the private sector are available for the nation as a whole and by NAICS sector and subsector. In addition, BED state data are available by NAICS sector." |

## Reproduction

The script is `scripts/employment_statistics_coverage.py` at commit 74714f6. Its inputs are the
files and hashes under Sources, plus the supervision bundle's codebook (sha256
`5c485aa96fc9d016c8aa7f95e269f4222b85e8ee395e529facc7a9f8adcaab7b`). From the repository root:

```bash
uv run python scripts/employment_statistics_coverage.py manifest --qcew-dir ~/Downloads/Data/QCEW
uv run python scripts/employment_statistics_coverage.py run --qcew-dir ~/Downloads/Data/QCEW --codebook /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet --final-years 2022 2023 2024 2025 --out-dir ~/Downloads/Data/QCEW/coverage
```

The run exits 2 on the two invariant failures of section 6. Its `coverage/decision.md` and
`coverage/tables.md` are pasted verbatim above.

The checks outside the script feed "Appendix: checks outside the script":

- **Independent recount:** the `/tmp/esc-recount.py` listing of plan 3
  (`specs/plans/completed/3-employment-statistics-coverage.md`, Task 6 Step 3), which takes the
  QCEW directory and the codebook as positional arguments:

```bash
uv run python /tmp/esc-recount.py ~/Downloads/Data/QCEW /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet
```

- **Lookup files:** the `grep` and `cat` commands of that plan's Task 5 Step 4.
- **National six-digit reconciliation:**

```bash
uv run python -c 'import csv, pathlib; d = pathlib.Path("~/Downloads/Data/QCEW").expanduser(); rows = lambda y: list(csv.DictReader((d / f"{y}_US000_annual.csv").open())); pick = lambda rs, lvl: [r for r in rs if r["own_code"] == "5" and r["agglvl_code"] == lvl]; [print("| {} | {:,} | {} | {:+,} | {:+,} | {:+,} |".format(y, len(six), sum(r["disclosure_code"] == "N" for r in six), sum(int(r["annual_avg_emplvl"]) for r in six if r["disclosure_code"] == "") - int(tot["annual_avg_emplvl"]), sum(int(r["total_annual_wages"]) for r in six if r["disclosure_code"] == "") - int(tot["total_annual_wages"]), sum(int(r["annual_avg_emplvl"]) for r in pick(rs, "14")) - int(tot["annual_avg_emplvl"]))) for y in (2022, 2023, 2024, 2025) for rs in [rows(y)] for six in [pick(rs, "18")] for tot in [pick(rs, "11")[0]]]'
```

- **Single-file scan:** this script, run with `uv run python`:

```python
import zipfile
from pathlib import Path

import polars as pl

QCEW = Path('~/Downloads/Data/QCEW').expanduser()
LEGACY = [
    '02030', '02040', '02080', '02120', '02160', '02190', '02200', '02210', '02250', '02260',
    '55901'
]

def owners(frame: pl.DataFrame, levels: list[str]) -> str:
    codes = frame.filter(pl.col('agglvl_code').is_in(levels))['own_code'].unique().sort()
    return ', '.join(codes.to_list()) or 'none'

for year in (2022, 2023, 2024, 2025):
    with zipfile.ZipFile(QCEW / f'{year}_annual_singlefile.zip') as archive:
        member = next(name for name in archive.namelist() if name.endswith('.csv'))
        data = archive.read(member)
    frame = pl.read_csv(
        data,
        columns=['area_fips', 'own_code', 'agglvl_code'],
        infer_schema=False,
    )
    county = frame.filter(pl.col('agglvl_code').is_in(['77', '78']))
    suffix = pl.col('area_fips').str.slice(2)
    unknown = county.filter(suffix == '999')['area_fips'].n_unique()
    pseudo = county.filter(suffix.is_in(['996', '997', '998'])).height
    legacy = county.filter(pl.col('area_fips').is_in(LEGACY)).height
    colorado = county.filter(
        (pl.col('agglvl_code') == '78') & (pl.col('own_code') == '5')
        & pl.col('area_fips').str.starts_with('08') & (suffix != '999')
    )['area_fips'].n_unique()
    msa_rows = frame.filter(pl.col('agglvl_code').is_in(['47', '48'])).height
    print(
        f'| {year} | {county["area_fips"].n_unique():,} | {unknown} | {pseudo} | {legacy} | '
        f'{colorado} | {msa_rows:,} | {owners(frame, ["47", "48"])} | '
        f'{owners(frame, ["18", "58", "78"])} |'
    )
```

## Appendix: generated tables

### Invariant failures

| failure |
| --- |
| 2022: disclosed six-digit emp 128718106 exceeds total 128718060 |
| 2023: disclosed six-digit emp 131289708 exceeds total 131289681 |

### File conventions (six-digit rows)

| year | disclosure_codes | own_code_0_rows | suppressed_rows | suppressed_rows_with_emp_or_wages | suppressed_rows_with_estabs | estabs_column |
| --- | --- | --- | --- | --- | --- | --- |
| 2022 | {'blank': 467112, 'N': 852716} | 0 | 852716 | 0 | 837846 | annual_avg_estabs |
| 2023 | {'blank': 472235, 'N': 862595} | 0 | 862595 | 0 | 846712 | annual_avg_estabs |
| 2024 | {'blank': 392184, 'N': 949625} | 0 | 949625 | 0 | 921691 | annual_avg_estabs |
| 2025 | {'blank': 336210, 'N': 761027} | 0 | 761027 | 0 | 737563 | annual_avg_estabs |

### Vintage check (national six-digit codes)

| year | published_six_digit | outside_codebook | outside_examples | codebook_unpublished | unpublished_examples |
| --- | --- | --- | --- | --- | --- |
| 2021 | 1075 | 139 | 212111, 212112, 212113, 212221, 212222, 212291, 212299, 212324, 212325, 212391, 212392, 212393 | 96 | 112130, 212114, 212115, 212220, 212290, 212323, 212390, 315120, 315250, 316990, 321215, 322120 |
| 2022 | 1029 | 0 | - | 3 | 112130, 517122, 541120 |
| 2023 | 1029 | 0 | - | 3 | 112130, 517122, 541120 |
| 2024 | 1029 | 0 | - | 3 | 112130, 517122, 541120 |
| 2025 | 1029 | 0 | - | 3 | 112130, 517122, 541120 |

### Split codes recovered from their five-digit parent

| code |
| --- |
| 238110 |
| 238120 |
| 238130 |
| 238140 |
| 238150 |
| 238160 |
| 238170 |
| 238190 |
| 238210 |
| 238220 |
| 238290 |
| 238310 |
| 238320 |
| 238330 |
| 238340 |
| 238350 |
| 238390 |
| 238910 |
| 238990 |

### National grain: codebook codes by status

| year | own_code | series | disclosed | suppressed | other | absent | recovered_via_parent | suppressed_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | 1 | employment | 207 | 0 | 0 | 805 | 0 | 0.0000 |
| 2022 | 1 | wages | 207 | 0 | 0 | 805 | 0 | 0.0000 |
| 2022 | 1 | establishments | 207 | 0 | 0 | 805 | 0 | 0.0000 |
| 2022 | 2 | employment | 152 | 80 | 0 | 780 | 1 | 0.0791 |
| 2022 | 2 | wages | 152 | 80 | 0 | 780 | 1 | 0.0791 |
| 2022 | 2 | establishments | 231 | 1 | 0 | 780 | 1 | 0.0010 |
| 2022 | 3 | employment | 271 | 163 | 0 | 578 | 11 | 0.1611 |
| 2022 | 3 | wages | 271 | 163 | 0 | 578 | 11 | 0.1611 |
| 2022 | 3 | establishments | 434 | 0 | 0 | 578 | 11 | 0.0000 |
| 2022 | 5 | employment | 980 | 0 | 0 | 32 | 19 | 0.0000 |
| 2022 | 5 | wages | 980 | 0 | 0 | 32 | 19 | 0.0000 |
| 2022 | 5 | establishments | 980 | 0 | 0 | 32 | 19 | 0.0000 |
| 2023 | 1 | employment | 207 | 0 | 0 | 805 | 0 | 0.0000 |
| 2023 | 1 | wages | 207 | 0 | 0 | 805 | 0 | 0.0000 |
| 2023 | 1 | establishments | 207 | 0 | 0 | 805 | 0 | 0.0000 |
| 2023 | 2 | employment | 153 | 94 | 0 | 765 | 8 | 0.0929 |
| 2023 | 2 | wages | 153 | 94 | 0 | 765 | 8 | 0.0929 |
| 2023 | 2 | establishments | 246 | 1 | 0 | 765 | 8 | 0.0010 |
| 2023 | 3 | employment | 282 | 166 | 0 | 564 | 11 | 0.1640 |
| 2023 | 3 | wages | 282 | 166 | 0 | 564 | 11 | 0.1640 |
| 2023 | 3 | establishments | 448 | 0 | 0 | 564 | 11 | 0.0000 |
| 2023 | 5 | employment | 980 | 0 | 0 | 32 | 19 | 0.0000 |
| 2023 | 5 | wages | 980 | 0 | 0 | 32 | 19 | 0.0000 |
| 2023 | 5 | establishments | 980 | 0 | 0 | 32 | 19 | 0.0000 |
| 2024 | 1 | employment | 210 | 0 | 0 | 802 | 0 | 0.0000 |
| 2024 | 1 | wages | 210 | 0 | 0 | 802 | 0 | 0.0000 |
| 2024 | 1 | establishments | 210 | 0 | 0 | 802 | 0 | 0.0000 |
| 2024 | 2 | employment | 151 | 85 | 0 | 776 | 2 | 0.0840 |
| 2024 | 2 | wages | 151 | 85 | 0 | 776 | 2 | 0.0840 |
| 2024 | 2 | establishments | 234 | 2 | 0 | 776 | 2 | 0.0020 |
| 2024 | 3 | employment | 277 | 175 | 0 | 560 | 11 | 0.1729 |
| 2024 | 3 | wages | 277 | 175 | 0 | 560 | 11 | 0.1729 |
| 2024 | 3 | establishments | 446 | 6 | 0 | 560 | 11 | 0.0059 |
| 2024 | 5 | employment | 980 | 0 | 0 | 32 | 19 | 0.0000 |
| 2024 | 5 | wages | 980 | 0 | 0 | 32 | 19 | 0.0000 |
| 2024 | 5 | establishments | 980 | 0 | 0 | 32 | 19 | 0.0000 |
| 2025 | 1 | employment | 204 | 0 | 0 | 808 | 0 | 0.0000 |
| 2025 | 1 | wages | 204 | 0 | 0 | 808 | 0 | 0.0000 |
| 2025 | 1 | establishments | 204 | 0 | 0 | 808 | 0 | 0.0000 |
| 2025 | 2 | employment | 142 | 95 | 0 | 775 | 2 | 0.0939 |
| 2025 | 2 | wages | 142 | 95 | 0 | 775 | 2 | 0.0939 |
| 2025 | 2 | establishments | 237 | 0 | 0 | 775 | 2 | 0.0000 |
| 2025 | 3 | employment | 270 | 179 | 0 | 563 | 11 | 0.1769 |
| 2025 | 3 | wages | 270 | 179 | 0 | 563 | 11 | 0.1769 |
| 2025 | 3 | establishments | 446 | 3 | 0 | 563 | 11 | 0.0030 |
| 2025 | 5 | employment | 980 | 0 | 0 | 32 | 19 | 0.0000 |
| 2025 | 5 | wages | 980 | 0 | 0 | 32 | 19 | 0.0000 |
| 2025 | 5 | establishments | 980 | 0 | 0 | 32 | 19 | 0.0000 |

### Private cells by grain and year

| grain | year | areas | published_cells | suppressed_cells | other_cells | suppressed_share | estabs_suppressed_cells | estabs_suppressed_share | codes_usable | codes_usable_2plus_areas | codes_published_never_usable | codes_absent | codes_without_usable | share_without_usable | median_usable_areas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| national | 2022 | 1 | 980 | 0 | 0 | 0.0000 | 0 | 0.0000 | 980 | 0 | 0 | 32 | 32 | 0.0316 | 1.0000 |
| national | 2023 | 1 | 980 | 0 | 0 | 0.0000 | 0 | 0.0000 | 980 | 0 | 0 | 32 | 32 | 0.0316 | 1.0000 |
| national | 2024 | 1 | 980 | 0 | 0 | 0.0000 | 0 | 0.0000 | 980 | 0 | 0 | 32 | 32 | 0.0316 | 1.0000 |
| national | 2025 | 1 | 980 | 0 | 0 | 0.0000 | 0 | 0.0000 | 980 | 0 | 0 | 32 | 32 | 0.0316 | 1.0000 |
| state | 2022 | 53 | 45737 | 10176 | 0 | 0.2225 | 80 | 0.0017 | 977 | 977 | 3 | 32 | 35 | 0.0346 | 40.0000 |
| state | 2023 | 53 | 45819 | 10006 | 0 | 0.2184 | 57 | 0.0012 | 978 | 978 | 2 | 32 | 34 | 0.0336 | 40.0000 |
| state | 2024 | 53 | 45848 | 11440 | 0 | 0.2495 | 140 | 0.0031 | 978 | 977 | 2 | 32 | 34 | 0.0336 | 38.5000 |
| state | 2025 | 53 | 45824 | 11978 | 0 | 0.2614 | 143 | 0.0031 | 978 | 973 | 2 | 32 | 34 | 0.0336 | 39.0000 |
| county | 2022 | 3223 | 866433 | 619758 | 0 | 0.7153 | 11826 | 0.0136 | 956 | 933 | 24 | 32 | 56 | 0.0553 | 98.0000 |
| county | 2023 | 3223 | 878426 | 628649 | 0 | 0.7157 | 12869 | 0.0147 | 953 | 939 | 27 | 32 | 59 | 0.0583 | 100.0000 |
| county | 2024 | 3224 | 880769 | 666537 | 0 | 0.7568 | 22480 | 0.0255 | 950 | 921 | 30 | 32 | 62 | 0.0613 | 78.0000 |
| county | 2025 | 3224 | 880614 | 674557 | 0 | 0.7660 | 21298 | 0.0242 | 948 | 920 | 32 | 32 | 64 | 0.0632 | 77.0000 |
| msa | 2022 | 388 | 232153 | 148614 | 0 | 0.6402 | 1791 | 0.0077 | 961 | 943 | 19 | 32 | 51 | 0.0504 | 55.0000 |
| msa | 2023 | 388 | 234131 | 149406 | 0 | 0.6381 | 1775 | 0.0076 | 959 | 945 | 21 | 32 | 53 | 0.0524 | 56.0000 |
| msa | 2024 | 393 | 238195 | 192319 | 0 | 0.8074 | 3319 | 0.0139 | 892 | 835 | 88 | 32 | 120 | 0.1186 | 23.0000 |
| msa | 2025 | 0 | 0 | 0 | 0 | - | 0 | - | 0 | 0 | 0 | 1012 | 1012 | 1.0000 | - |

### Establishments of disclosed and suppressed private cells

| grain | year | status | cells | median_estabs | p90_estabs |
| --- | --- | --- | --- | --- | --- |
| national | 2022 | disclosed | 980 | 2100.5000 | 22920.0000 |
| national | 2023 | disclosed | 980 | 2127.0000 | 23315.0000 |
| national | 2024 | disclosed | 980 | 2172.0000 | 23356.0000 |
| national | 2025 | disclosed | 980 | 2171.5000 | 23867.0000 |
| state | 2022 | disclosed | 35561 | 56.0000 | 586.0000 |
| state | 2022 | suppressed | 10176 | 3.0000 | 18.0000 |
| state | 2023 | disclosed | 35813 | 58.0000 | 607.0000 |
| state | 2023 | suppressed | 10006 | 3.0000 | 19.0000 |
| state | 2024 | disclosed | 34408 | 63.0000 | 643.0000 |
| state | 2024 | suppressed | 11440 | 4.0000 | 22.0000 |
| state | 2025 | disclosed | 33846 | 65.0000 | 653.0000 |
| state | 2025 | suppressed | 11978 | 4.0000 | 25.0000 |
| county | 2022 | disclosed | 246675 | 9.0000 | 59.0000 |
| county | 2022 | suppressed | 619758 | 1.0000 | 4.0000 |
| county | 2023 | disclosed | 249777 | 10.0000 | 60.0000 |
| county | 2023 | suppressed | 628649 | 1.0000 | 4.0000 |
| county | 2024 | disclosed | 214232 | 11.0000 | 70.0000 |
| county | 2024 | suppressed | 666537 | 1.0000 | 5.0000 |
| county | 2025 | disclosed | 206057 | 12.0000 | 72.0000 |
| county | 2025 | suppressed | 674557 | 1.0000 | 5.0000 |
| msa | 2022 | disclosed | 83539 | 15.0000 | 130.0000 |
| msa | 2022 | suppressed | 148614 | 2.0000 | 21.0000 |
| msa | 2023 | disclosed | 84725 | 15.0000 | 135.0000 |
| msa | 2023 | suppressed | 149406 | 2.0000 | 21.0000 |
| msa | 2024 | disclosed | 45876 | 14.0000 | 128.0000 |
| msa | 2024 | suppressed | 192319 | 3.0000 | 42.0000 |

### Codes with no private national cell (last year)

| code | ownerships_with_cells |
| --- | --- |
| 112130 | - |
| 517122 | - |
| 541120 | - |
| 921110 | 1, 2, 3 |
| 921120 | 1, 2, 3 |
| 921130 | 1, 2, 3 |
| 921140 | 1, 2, 3 |
| 921150 | 3 |
| 921190 | 1, 2, 3 |
| 922110 | 1, 2, 3 |
| 922120 | 1, 2, 3 |
| 922130 | 1, 2, 3 |
| 922140 | 1, 2, 3 |
| 922150 | 1, 2, 3 |
| 922160 | 2, 3 |
| 922190 | 1, 2, 3 |
| 923110 | 1, 2, 3 |
| 923120 | 1, 2, 3 |
| 923130 | 1, 2, 3 |
| 923140 | 1, 2, 3 |
| 924110 | 1, 2, 3 |
| 924120 | 1, 2, 3 |
| 925110 | 1, 2, 3 |
| 925120 | 1, 2, 3 |
| 926110 | 1, 2, 3 |
| 926120 | 1, 2, 3 |
| 926130 | 1, 2, 3 |
| 926140 | 1, 2, 3 |
| 926150 | 1, 2, 3 |
| 927110 | 1, 2, 3 |
| 928110 | 1, 2, 3 |
| 928120 | 1, 2, 3 |

### Connecticut county-equivalents

| year | legacy_counties | planning_regions |
| --- | --- | --- |
| 2022 | 8 | 0 |
| 2023 | 8 | 0 |
| 2024 | 0 | 9 |
| 2025 | 0 | 9 |

### MSA six-digit rows per year

| year | rows |
| --- | --- |
| 2022 | 232153 |
| 2023 | 234131 |
| 2024 | 238195 |
| 2025 | 0 |

### Decision inputs

| grain | complete | mean_suppressed_share | seen_by_year | seen_by_area | time_eligible | heldout_population | seen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| national | True | 0.0000 | 980 | 0 | 980 | 980 | 980 |
| state | True | 0.2379 | 978 | 973 | 978 | 978 | 978 |
| county | True | 0.7384 | 948 | 920 | 935 | 962 | 948 |
| msa | False | 0.6952 | 0 | 0 | 0 | 967 | 0 |

### Codes excluded at the chosen grain

| code | reason |
| --- | --- |
| 112130 | no private cell |
| 517122 | no private cell |
| 541120 | no private cell |
| 921110 | no private cell |
| 921120 | no private cell |
| 921130 | no private cell |
| 921140 | no private cell |
| 921150 | no private cell |
| 921190 | no private cell |
| 922110 | no private cell |
| 922120 | no private cell |
| 922130 | no private cell |
| 922140 | no private cell |
| 922150 | no private cell |
| 922160 | no private cell |
| 922190 | no private cell |
| 923110 | no private cell |
| 923120 | no private cell |
| 923130 | no private cell |
| 923140 | no private cell |
| 924110 | no private cell |
| 924120 | no private cell |
| 925110 | no private cell |
| 925120 | no private cell |
| 926110 | no private cell |
| 926120 | no private cell |
| 926130 | no private cell |
| 926140 | no private cell |
| 926150 | no private cell |
| 927110 | no private cell |
| 928110 | no private cell |
| 928120 | no private cell |

## Appendix: checks outside the script

These tables come from the lookup files and from checks run on the downloaded files outside the
script. Reproduction gives each command.

### Lookup files

| Check | File | Result |
| --- | --- | --- |
| Levels of the four grains | `agglevel_titles.csv` | 11 "National, Total -- by ownership sector"; 17 "National, NAICS 5-digit -- by ownership sector"; 18 "National, NAICS 6-digit -- by ownership sector"; 47 "MSA, NAICS 5-digit -- by ownership sector"; 48 "MSA, NAICS 6-digit -- by ownership sector"; 57 "State, NAICS 5-digit -- by ownership sector"; 58 "State, NAICS 6-digit -- by ownership sector"; 77 "County, NAICS 5-digit -- by ownership sector"; 78 "County, NAICS 6-digit -- by ownership sector" |
| CSA and MicroSA levels | `agglevel_titles.csv` | 30 "CMSA or CSA, Total Covered" and 80 "MicroSA, Total Covered"; no industry-detail level for either |
| NAICS 238 six-digit codes | `industry_titles.csv` | 38 codes ending in 1 or 2 (238111 to 238992); 0 codes ending in 0 |
| 999999 | `industry_titles.csv` | 1 row |
| Unknown or undefined areas | `area_titles.csv` | 53 codes ending in 999, one each for the 50 states, DC, Puerto Rico and the Virgin Islands; `11000` and `11001` match the search only through their title, "District of Columbia, not unknown" |
| Connecticut | `area_titles.csv` | 8 legacy counties (`09001` to `09015`) and 9 planning regions (`09110` to `09190`) |
| Ownership | `ownership_titles.csv` | 1 "Federal Government", 2 "State Government", 3 "Local Government", 5 "Private" |
| Duplicate rows | `area_titles.csv` | every area row appears twice (`"09001"` is on 2 lines); the script does not read this file |

### Single-file scan

Rows of each year's single file. County-level rows are aggregation levels 77 and 78; MSA rows are
47 and 48.

| Year | County-level areas, 999 codes included | Areas ending in 999 with rows | Rows ending in 996, 997 or 998 | Legacy Alaska division or 55901 rows | Colorado counties with private six-digit rows | MSA rows | MSA `own_code` values | Six-digit (18, 58, 78) `own_code` values |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2022 | 3,274 | 51 | 0 | 0 | 64 | 419,768 | 5 | 1, 2, 3, 5 |
| 2023 | 3,274 | 51 | 0 | 0 | 64 | 422,930 | 5 | 1, 2, 3, 5 |
| 2024 | 3,275 | 51 | 0 | 0 | 64 | 430,234 | 5 | 1, 2, 3, 5 |
| 2025 | 3,275 | 51 | 0 | 0 | 64 | 0 | none | 1, 2, 3, 5 |

### National six-digit reconciliation

From the Open Data Access `US000` slices, private ownership. Each difference is disclosed detail
minus the national private total (level 11): six-digit detail (level 18) for employment and wages,
and sector detail (level 14) in the last column.

| Year | Six-digit private rows | `N` rows | Employment: detail − total | Wages: detail − total | Sector employment: detail − total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | 1,000 | 0 | +46 | +0 | +0 |
| 2023 | 1,000 | 0 | +27 | +0 | +2 |
| 2024 | 1,000 | 0 | -5 | +0 | -1 |
| 2025 | 1,000 | 0 | -9 | +0 | -1 |

### Independent recount

The stdlib-`csv` recount of national private statuses (plan 3 Task 6 Step 3).

| Year | disclosed | suppressed | other | absent |
| --- | ---: | ---: | ---: | ---: |
| 2022 | 980 | 0 | 0 | 32 |
| 2023 | 980 | 0 | 0 | 32 |
| 2024 | 980 | 0 | 0 | 32 |
| 2025 | 980 | 0 | 0 | 32 |
