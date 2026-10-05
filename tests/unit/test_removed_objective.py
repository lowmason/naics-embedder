'''
The old objective is gone (spec 4.5; Req 10, 11; D5).

The six-term objective and everything that served it are deleted: its terms, the mining, the
curriculum, the false-negative strategies, the candidate pools and selection, the pre-sampled rows,
the config models that tuned them, and ``tools investigate`` (spec 4.4). No training path reads the
relation margin axis (D5): the bundle's training-pairs member stays, unread (R10), so the build
keeps writing its margin columns until the next contract bump retires the member (section 10).
'''

import ast
import importlib
import importlib.util
from pathlib import Path
from typing import Iterator, List

import polars as pl
import pytest

import naics_embedder
from naics_embedder.data.create_triplets import _structural_margins

# -------------------------------------------------------------------------------------------------
# The deleted modules and names (spec 4.5's table)
# -------------------------------------------------------------------------------------------------

DELETIONS = [
    # Machinery
    'naics_embedder.text_model.curriculum',
    'naics_embedder.text_model.hard_negative_mining',
    'naics_embedder.text_model.hyperbolic_clustering',
    'naics_embedder.text_model.false_negative_strategies',
    'naics_embedder.text_model.mixins.curriculum',
    'naics_embedder.text_model.mixins.distributed',
    'naics_embedder.text_model.mixins.validation',
    # Supervision
    'naics_embedder.supervision.selection',
    'naics_embedder.supervision.candidates',
    'naics_embedder.supervision.margins',
    'naics_embedder.supervision.index',
    'naics_embedder.supervision.schema:SelectionReason',
    'naics_embedder.supervision.schema:SamplingProvenance',
    'naics_embedder.supervision.schema:SAMPLING_ROLE_TO_ID',
    # The checkpoint contract's versions of the old terms and mining; it names the objective now
    'naics_embedder.supervision.schema:STRUCTURAL_PREFERENCE_LOSS_VERSION',
    'naics_embedder.supervision.schema:MINING_CONTRACT_VERSION',
    # Data
    'naics_embedder.text_model.dataloader.streaming_dataset',
    'naics_embedder.text_model.dataloader.difficulty_sampler',
    # Terms
    'naics_embedder.text_model.loss:HyperbolicInfoNCELoss',
    'naics_embedder.text_model.loss:HierarchyPreservationLoss',
    'naics_embedder.text_model.loss:StructuralPreferenceLoss',
    'naics_embedder.text_model.loss:structural_preference_from_distances',
    'naics_embedder.text_model.loss:effective_false_negative_mask',
    # Geometry that only the old terms, mining and diagnostics read
    'naics_embedder.text_model.hyperbolic:LorentzDistance',
    'naics_embedder.text_model.hyperbolic:log_hyperbolic_diagnostics',
    'naics_embedder.text_model.hyperbolic:_exp_map_zero_compiled',
    'naics_embedder.text_model.hyperbolic:_lorentz_dot_compiled',
    'naics_embedder.text_model.hyperbolic:_lorentz_distance_compiled',
    'naics_embedder.text_model.hyperbolic:_batched_lorentz_dot_compiled',
    'naics_embedder.text_model.hyperbolic:_mark_cudagraph_step',
    # The curvature guard: no model takes a curvature, and the contract's objective refuses every
    # older checkpoint instead (spec 4.2)
    'naics_embedder.text_model.export:require_unit_curvature',
    # The config models and checks of the removed keys (P22)
    'naics_embedder.utils.config:CurriculumConfig',
    'naics_embedder.utils.config:AnnealConfig',
    'naics_embedder.utils.config:SamplingConfig',
    'naics_embedder.utils.config:SansStaticConfig',
    'naics_embedder.utils.config:FalseNegativeConfig',
    'naics_embedder.utils.config:StructuralPreferenceConfig',
    'naics_embedder.utils.config:LEGACY_STREAMING_PATHS',
    'naics_embedder.utils.config:Config.validate_supervision_contract',
    'naics_embedder.utils.validation:validate_distances_schema',
    # tools investigate (spec 4.4)
    'naics_embedder.tools._investigate_hierarchy',
    'naics_embedder.tools.metrics_tools:investigate_hierarchy',
    'naics_embedder.tools.metrics_tools:HAS_INVESTIGATE',
]

@pytest.mark.unit
@pytest.mark.parametrize('path', DELETIONS)
def test_the_old_objective_and_its_machinery_are_deleted(path):
    '''Spec 4.5: nothing of the six-term objective, its mining, curriculum and sampling remains.'''

    module, _, attribute = path.partition(':')
    if not attribute:
        assert importlib.util.find_spec(module) is None
        return
    *owners, name = attribute.split('.')
    owner = importlib.import_module(module)
    for part in owners:
        owner = getattr(owner, part)
    assert not hasattr(owner, name)

# -------------------------------------------------------------------------------------------------
# D5: no training path reads the relation margin axis
# -------------------------------------------------------------------------------------------------

SOURCE = Path(naics_embedder.__file__).parent

# The columns the build's _structural_margins adds to the training-pairs member: D5's axis
MARGIN_COLUMNS = ('relation_margin', 'distance_margin', 'margin')
# Read as names too (an attribute, a keyword); a bare 'margin' counts only as a column name
MARGIN_FIELDS = ('relation_margin', 'distance_margin')
# Where the axis is computed, and its two constants
MARGIN_MODULES = ('naics_embedder.data.create_triplets', 'naics_embedder.supervision.margins')
MARGIN_CONSTANTS = ('CROSS_SECTOR_RELATION_MARGIN', 'EQUAL_DISTANCE_MARGIN')
# The bundle member that carries the margin columns (R10: it stays, unread)
TRAINING_PAIRS = 'training_pairs'

def _training_path() -> List[Path]:
    '''The text stage's modules, and the train command's.'''

    return sorted((SOURCE / 'text_model').rglob('*.py')) + [
        SOURCE / 'cli' / 'commands' / 'training.py',
        SOURCE / 'utils' / 'training.py',
    ]

def _margin_reads(path: Path) -> Iterator[str]:
    '''Each place in a module that reads the margin axis or its member, as "file:line what".'''

    where = path.relative_to(SOURCE.parent)
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Constant) and node.value in MARGIN_COLUMNS + (TRAINING_PAIRS, ):
            yield f'{where}:{node.lineno} the string {node.value!r}'
        elif isinstance(node, ast.Attribute) and node.attr in MARGIN_FIELDS:
            yield f'{where}:{node.lineno} the attribute .{node.attr}'
        elif isinstance(node, ast.keyword) and node.arg in MARGIN_FIELDS:
            yield f'{where}:{node.lineno} the keyword {node.arg}='
        elif isinstance(node, ast.Name) and node.id in MARGIN_FIELDS + MARGIN_CONSTANTS:
            yield f'{where}:{node.lineno} the name {node.id}'
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            if node.module in MARGIN_MODULES or set(names) & set(MARGIN_CONSTANTS):
                yield f'{where}:{node.lineno} from {node.module} import {", ".join(names)}'
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in MARGIN_MODULES:
                    yield f'{where}:{node.lineno} import {alias.name}'

@pytest.mark.unit
def test_no_training_path_reads_the_relation_margin_axis():
    '''D5: the text stage and the train command read no margin column, compute no margin, and do not
    read the training-pairs member that carries them (R10).'''

    reads = [read for path in _training_path() for read in _margin_reads(path)]

    assert reads == []

@pytest.mark.unit
def test_the_margin_columns_are_those_the_bundle_build_writes():
    '''The scan above looks for exactly the columns _structural_margins adds to the training pairs,
    so a margin column the build adds later cannot pass it unseen.'''

    frame = pl.DataFrame(
        {
            'positive_structural_relation_id': [1],
            'positive_structural_distance': [1.0],
            'negative_structural_relation_id': [2],
            'negative_structural_distance': [3.0],
        }
    )

    added = set(_structural_margins(frame).columns) - set(frame.columns)

    assert added == set(MARGIN_COLUMNS)
