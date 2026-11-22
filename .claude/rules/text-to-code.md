---
paths:
  - "src/naics_embedder/panels/ita*.py"
  - "src/naics_embedder/decision/text_to_code.py"
  - "src/naics_embedder/decision/ita_reads.py"
  - "src/naics_embedder/decision/ita_gap.py"
  - "tests/unit/test_ita_*.py"
  - "tests/unit/test_decision_text_to_code.py"
  - "tests/fixtures/ita.py"
  - "conf/data/ita_panel.yaml"
  - "conf/data/ita_evidence.yaml"
---

# Text→Code Artifact

The root `CLAUDE.md` keeps the ITA panel's guardrails: the sealed part opens only through
`ItaPanel.open_sealed`, once, no ITA pair enters training, and no code or test reads a real pair.
This file holds the reads, the tools and the traps.

## Reads and Tools

Stage 4 measured the gap between Census phrases and owner text on the validation part on
2026-11-19 (`specs/findings/text-to-code-gap.md`), as a development measurement.
`tools ita-read` reads a reference arm on the ITA validation part after checking the pinned
run (`conf/data/ita_evidence.yaml`), and `tools ita-gap` measures the gap from those reads
without reading a panel.

## Traps

These come from the plan 17 and plan 20 Handoffs and `specs/deferred_items.md`. Check the
deferred items before relying on one: a later stage may have fixed it.

- `panel_pairs` refuses a pair table that disagrees with spec §4.1, a null in a read column and a
  repeated `pair_id`. A build that fails these refuses at construction and logs nothing.
- `fix_ita_margin` refuses repeated seeds and seedless arms before any read.
  `check_ita_margin_first` calls `check_ita_margin`, which checks the record's own consistency.
- `check_ita_margin` does not require reads of both reference arms, and it does not refuse a
  repeated seed among `log_records`. A hand-edited record can pass it.
- `tools ita-read` takes `--outcome` on a d16 read and leaves it off by default. It does not check
  a d16 seed's checkpoint and table against the Stage 8 record before it logs.

## Tests

- `test_ita_split.py`, `test_ita_inputs.py`, `test_ita_scoring.py`, `test_ita_panel.py` and
  `test_decision_text_to_code.py`: the ITA panel's salted split, input refusals, shortlist ranks
  and summaries, logged reads and sealed opening, and text→code's margin and decision. All read
  `tests/fixtures/ita.py` tables only.
- `test_ita_evidence.py`, `test_ita_reads.py` and `test_ita_gap.py`: the ITA run pin and its C1
  check, Stage 4's reference reads and their refusals before logging, and the gap's strata,
  matched codes, refusals of reads that do not pair and D2. All read fixture tables only.
