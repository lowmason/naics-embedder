'''
The text-and-tree-only rule for a candidate's provenance (spec economic-candidates, 4.5; Req 29).

A candidate's provenance must come from one of four writers: the export (``tools export-table``)
for a trained candidate, and the three builders of ``tree_model`` for a deterministic one. Each
writes a fixed set of entries. The rule refuses an entry its writer does not write, so a
provenance that names a QCEW file, a panel, the selection log or any unknown entry never passes. It
also refuses an input file other than the bundle's codebook and the shared text-only table, and a
checkpoint contract other than the frozen list's.

The export has two known versions: Stage 7's ten tables have no ``geometry`` entry, and Stage 8's
forty have the same entries plus ``geometry``. The rule accepts both.

The rule describes the provenance and checks no file. It reads ``decision.candidates``, which
never reads this module.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Any, FrozenSet, Mapping, NoReturn

from naics_embedder.decision.candidates import CandidateEntry, CandidateList
from naics_embedder.tree_model.builders import BUILDER_ENTRIES, BUILDER_INPUTS, KINDS

# What ``tools export-table`` writes, as Stage 7's ten files hold it (verified 2026-11-14 on the
# main checkout's 50 files: two key sets, which differ by ``geometry``)
EXPORT_ENTRIES: FrozenSet[str] = frozenset(
    {
        'backbone',
        'checkpoint',
        'codes',
        'contract',
        'coordinates',
        'descriptions',
        'dimension',
        'generated_at',
        'library_versions',
        'matrix_fingerprint',
        'max_length',
        'revision',
        'summaries',
        'table_sha256',
        'tokenizer',
    }
)
# Stage 8's forty files
EXPORT_ENTRIES_WITH_GEOMETRY: FrozenSet[str] = EXPORT_ENTRIES | {'geometry'}

_RULE = '(Req 29: text and tree only)'
_FILE_ENTRIES = frozenset({'path', 'sha256'})
_CONTRACT_FIELDS = ('bundle_id', 'contract_version')
# The export's input files: the contract names the checkpoint's supervision
_EXPORT_FILES = ('checkpoint', 'descriptions')

# -------------------------------------------------------------------------------------------------
# The rule
# -------------------------------------------------------------------------------------------------

def check_candidate_provenance(
    content: Mapping[str, Any], entry: CandidateEntry, candidates: CandidateList
) -> None:
    '''
    Require a candidate's provenance to come from text and tree only (Req 29).

    A provenance with a ``builder`` entry is a deterministic candidate's. The builder is one of
    ``KINDS`` and the entry's own kind. The entries are exactly ``BUILDER_ENTRIES[builder]``.
    ``inputs`` is an object with exactly ``BUILDER_INPUTS[builder]``, each a ``path`` and a
    ``sha256``. The codebook's sha256 is the list's, and so is the text-only table's where the
    builder reads one.

    A provenance with no ``builder`` entry is an export's, and the entry is a trained candidate.
    Its entries are exactly ``EXPORT_ENTRIES`` or ``EXPORT_ENTRIES_WITH_GEOMETRY``. Its contract
    names the list's ``bundle_id`` and ``contract_version``, and its checkpoint and descriptions
    are each a ``path`` and a ``sha256``.

    The checks run in that order, and each message names the one that refused (``builder``,
    ``entries``, ``inputs``, ``codebook``, ``text_only`` or ``contract``).

    Args:
        content: The provenance's JSON content.
        entry: The candidate's list entry.
        candidates: The frozen list, which holds the sha256 of the codebook and of the text-only
            table, and the checkpoint contract.

    Raises:
        ValueError: With the entry's name first and ``(Req 29: text and tree only)`` last, if the
            provenance breaks the rule.
    '''

    if 'builder' in content:
        _check_builder_provenance(content, entry, candidates)
    else:
        _check_export_provenance(content, entry, candidates)

# -------------------------------------------------------------------------------------------------
# A deterministic candidate's builder
# -------------------------------------------------------------------------------------------------

def _check_builder_provenance(
    content: Mapping[str, Any], entry: CandidateEntry, candidates: CandidateList
) -> None:
    builder = content['builder']
    if builder not in KINDS or builder != entry.kind:
        _refuse(
            entry, 'builder', f'its provenance names the builder {builder!r}, but this is a '
            f'{entry.kind} candidate and the builders are {", ".join(KINDS)}'
        )
    _check_entries(content, entry, BUILDER_ENTRIES[builder], BUILDER_ENTRIES[builder], builder)
    inputs = content['inputs']
    names = BUILDER_INPUTS[builder]
    if not isinstance(inputs, Mapping) or set(inputs) != set(names):
        _refuse(
            entry, 'inputs', f'the {builder} builder reads exactly {list(names)}, not {inputs!r}'
        )
    for name in names:
        _check_file(inputs[name], entry, f'the input {name!r}')
    if inputs['codebook']['sha256'] != candidates.codebook_sha256:
        _refuse(entry, 'codebook', "its codebook is not the frozen list's")
    if 'text_only' in inputs and inputs['text_only']['sha256'] != candidates.text_only_sha256:
        _refuse(entry, 'text_only', "its text-only table is not the frozen list's")

# -------------------------------------------------------------------------------------------------
# A trained candidate's export
# -------------------------------------------------------------------------------------------------

def _check_export_provenance(
    content: Mapping[str, Any], entry: CandidateEntry, candidates: CandidateList
) -> None:
    if entry.kind != 'trained':
        _refuse(
            entry, 'builder', f"its provenance names no builder, so it is an export's, but this is "
            f'a {entry.kind} candidate and only a trained candidate has an export'
        )
    _check_entries(content, entry, EXPORT_ENTRIES, EXPORT_ENTRIES_WITH_GEOMETRY, 'export')
    named = candidates.checkpoint_contract
    contract = content['contract']
    if not isinstance(contract, Mapping) or any(
        contract.get(field) != getattr(named, field) for field in _CONTRACT_FIELDS
    ):
        _refuse(
            entry, 'contract', f'its contract is not bundle {named.bundle_id} at version '
            f'{named.contract_version}: {contract!r}'
        )
    for field in _EXPORT_FILES:
        _check_file(content[field], entry, f'its {field}')

# -------------------------------------------------------------------------------------------------
# Shared checks
# -------------------------------------------------------------------------------------------------

def _check_entries(
    content: Mapping[str, Any],
    entry: CandidateEntry,
    required: FrozenSet[str],
    allowed: FrozenSet[str],
    writer: str,
) -> None:
    '''Require every ``required`` entry and no entry outside ``allowed``.'''

    extra = sorted(set(content) - allowed)
    missing = sorted(required - set(content))
    if extra or missing:
        problems = []
        if extra:
            problems.append(f'the entries {extra}, which the {writer} does not write')
        if missing:
            problems.append(f'no entry {missing}, which the {writer} writes')
        _refuse(entry, 'entries', f'its provenance has {" and has ".join(problems)}')

def _check_file(value: Any, entry: CandidateEntry, what: str) -> None:
    '''Require a file reference: an object of a ``path`` and a ``sha256``, both strings.'''

    if (
        not isinstance(value, Mapping) or set(value) != _FILE_ENTRIES or not all(
            isinstance(value[field], str) for field in _FILE_ENTRIES
        )
    ):
        _refuse(entry, 'inputs', f"{what} is not exactly a string 'path' and a string 'sha256'")

def _refuse(entry: CandidateEntry, check: str, detail: str) -> NoReturn:
    raise ValueError(f'{entry.name}: {check}: {detail} {_RULE}')
