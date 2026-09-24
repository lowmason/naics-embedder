# Employment-statistics coverage: finding

**Status: DRAFT.** Roadmap Stage 1 (`specs/naics-embedding-roadmap.md`). This finding discharges
Verification "Employment-statistics coverage" and the Req 2 (open) item of
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

<!-- Task 7 -->

## 2. Grains at which six-digit series are published

<!-- Task 7 -->

## 3. Suppressed share per year, grain and series

<!-- Task 7 -->

## 4. Panel population, row grain, time-respecting outcome and seen-code regime

<!-- Task 7 -->

## 5. Other public employment series screened

<!-- Task 7 -->

## 6. File conventions verified

<!-- Task 7 -->

## 7. Consequences for later stages

<!-- Task 7 -->

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

<!-- Task 7 -->

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
