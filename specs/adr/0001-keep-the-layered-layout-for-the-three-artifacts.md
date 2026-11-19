# 0001. Keep the layered package layout for the three artifacts

- **Status:** Accepted (2026-11-14)
- **Date:** 2026-11-14
- **Deciders:** the project owner ("Accept both", 2026-11-14)
- **Blast radius:** naics-embedder only: `src/naics_embedder/`, `tests/` and `docs/api/`

## Context

`specs/naics-artifact-redesign.md` (user adjudication Q4, approved on 2026-11-14 as U31) defines
three artifacts over the 2,125 NAICS 2022 codes: relatedness (Req 27), text→code (Req 28) and
economic (Req 29). The tree is their shared codebook (Req 25). Each artifact's design belongs to
its own stage spec. This record decides where the code of all three lives, before the roadmap
stages that work. Evidence is read at origin/main 7fdc170.

The package is layered by function. Paths are under `src/naics_embedder/`:

- `supervision/` loads and validates the bundle and its codebook.
- `text_model/` trains the shared text encoder. `graph_model/` holds the parked HGCN stage.
- `panels/` holds the outcome panel, the regressor panel and the selection log.
- `decision/` holds resampling, the decision rule, the records and the economic decision.

These facts drive the choice:

1. **Two artifacts share the text model.** The trained text model is the text→code artifact. The
   outcome panel still selects the checkpoints of trained economic candidates (Req 28, split
   Req 23).
2. **All three artifacts share the tooling.** The codebook, the selection log, resampling and the
   decision rule serve every artifact.
3. **The economic path is pinned.** The growth margins were read once, on 2026-10-17, on the Mac
   with Python 3.12.12 and `uv.lock` 6de4fbdc (`specs/findings/growth-margins.md`). `decide`
   pairs a candidate only where the reference predictions match that read's to within 1e-13
   (`decision/economic.py`; amended 2026-11-18 from "bit for bit": economic-candidates §4.8).
   The level-mode fixture is pinned at d61f49b. Sixteen test files
   import `panels/regressor.py`, `panels/qcew_rows.py` or `decision/economic.py`. Amended
   2026-11-19: Stage 3 adopted no candidate, and no economic read is planned, so the pairing no
   longer binds the other artifacts' `uv.lock` (`specs/naics-artifact-redesign-roadmap.md`,
   Decisions, 2026-11-19). The level-mode fixture pin is a test pin and is unchanged.
4. **Evidence roles are not import rules.** Req 25 says that no artifact's adoption decision reads
   another artifact's panel, and Req 26 gives each table or part one role. Data and records
   enforce this today, not packages. The selection log keys each opening on its panel and
   fingerprint (`panels/selection_log.py`), and `decide` takes validation reads only
   (`decision/decide.py`).
5. **Relatedness has no code yet.** The spec lets it reuse code from `graph_model/` (Req 27).

## Decision

We will keep the layered layout. Each artifact is a set of modules across the existing layers.
Each stage spec puts its new code in the layer that owns that function: a model in the model layer
that fits its method, a panel under `panels/`, and a decision under `decision/`. The model layers
today are `text_model/` and `graph_model/`. A method that fits neither gets a new model layer
beside them.

## Consequences

- **Positive:**
  - No pinned economic module moves. The economic reads keep their paths, tests and fixtures.
  - Shared tooling stays in one place. Each new panel and decision reuses resampling, the rule,
    the records and the selection log.
  - Stage specs follow the rule CLAUDE.md already gives: prefer the module that owns the
    behavior.
- **Negative:**
  - No directory shows one artifact's code. A reader needs a map from each artifact to its
    modules.
  - `graph_model/` can come to hold two unrelated models, the parked HGCN and the relatedness
    model, with different geometry and inputs. The relatedness stage spec can give its model a
    module of its own inside that layer.
  - Packages do not stop one artifact's code from reading another artifact's panel. Reqs 25 and
    26 rest on the selection log, the records' checks and review.
- **Neutral / follow-on:**
  - The Q4 record (the spec's Rollout note, item 1) can add the map from artifact to modules to
    CLAUDE.md.
  - Each stage spec names the modules it adds and the layer of each.
  - The check that this holds: every stage plan's new modules sit in an existing layer, and the
    economic tests and the level-mode fixture pin pass unchanged.

## Alternatives considered

- **One subpackage per artifact over a shared core** (`relatedness/`, `retrieval/` and
  `economic/`). Rejected: the text model and the outcome panel serve two artifacts, and the
  codebook, the selection log and the decision tooling serve all three. All of that would go to
  the core, so the artifact packages would be thin. The move would also touch the pinned economic
  path, its 16 test files and the 20 mkdocstrings entries for `panels` and `decision` in `docs/`,
  and it would change no behavior.
- **A hybrid: new subpackages for new code only.** The ITA scoring would go in `retrieval/` and
  the relatedness model in `relatedness/`, and existing code would stay. Rejected: the tree would
  follow two rules, so a reader could not predict where code lives. The ITA panel, for example,
  would sit apart from the outcome panel that it is reported beside (Req 28).
- **A uv workspace: a core package plus one package per artifact.** Rejected now: it changes
  `uv.lock` and the install path, and every economic candidate read must pair, to within 1e-13,
  with the margin read made under lock 6de4fbdc (economic-candidates §4.8). Each package would
  also need its own CI, coverage and docs
  configuration.
- **A separate repository for the relatedness artifact.** Rejected: it would copy the bundle
  loading and validation, the decision tooling and the remote workflow, and two repositories
  would have to pin one codebook.

## Trade-offs & reversibility

This trades a directory per artifact for shared tooling and an untouched economic path. It is a
two-way door: a later move into artifact subpackages is mechanical (imports, tests and docs
pages). The move gets cheaper once the economic decision and its sealed opening are done, because
the pinned reads stop binding. Revisit it with a superseding ADR if:

- a layer fills with modules that serve one artifact only;
- the economic decision and its sealed opening are done, so the pin no longer blocks a move or a
  workspace;
- a fourth artifact is added.
