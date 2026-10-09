'''
A tiny backbone for the shared encoder's tests, so they download nothing (spec §6), and two arms
built on it.

``tiny_bert`` builds a one-layer BERT whose vocabulary is MiniLM's, so token rows from the real
tokenizer fit it. ``tiny_backbone`` makes every ``SharedEncoder`` a test builds load it in place of
MiniLM.

The arm fixtures train nothing. ``shared_model`` is a d = 16 model of the five-code supervision
bundle (``tests/fixtures/supervision.py``) on the tiny backbone, and ``shared_checkpoint`` saves it
as Lightning would. ``truncated_checkpoint`` and ``pre_stage7_checkpoint`` save it as checkpoints
trained before Stage 6b and before Stage 7, which every load refuses. ``pre_stage8_checkpoint``
saves it as Stage 7 did, naming no geometry, which every load reads as hyperbolic (P7).
``text_only_comparator_table``
is a table a read can be pointed at by mistake: the text-only comparator's, written by its own
builder.

``reference_arm_model`` is a d = 16 model of the reference bundle, whose 17 codes span levels 2-6
and whose 11 task queries train Req 11's three terms. ``reference_arm_steps`` holds one epoch of
its two-stream steps, in the layout ``StepDataset`` hands the training step.
'''

from pathlib import Path
from typing import Any, Dict, List, Mapping

import polars as pl
import pytest
import pytorch_lightning as pyl
import torch
from transformers import AutoTokenizer, BertConfig, BertModel

from naics_embedder.panels.text_only import build_text_only_table
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.supervision.code_targets import CodeTargets
from naics_embedder.supervision.queries import build_task_queries
from naics_embedder.text_model.dataloader.datamodule import StepDataset, tokenize_task_queries
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.export import export_code_table
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import TokenizationConfig

TINY_HIDDEN = 8

def tiny_bert(name: str = 'tiny-bert') -> BertModel:
    '''A seeded one-layer BERT of width 8 over MiniLM's 30,522-token vocabulary, in eval mode.'''

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=30522,
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=512,
    )
    # from_pretrained returns the real backbone in eval mode; a train-mode stand-in would hide it
    return BertModel(config).eval()

@pytest.fixture
def tiny_backbone(monkeypatch):
    '''Every ``SharedEncoder`` built in the test loads ``tiny_bert`` instead of MiniLM.'''

    monkeypatch.setattr('naics_embedder.text_model.shared_encoder.load_base_model', tiny_bert)
    return tiny_bert

# -------------------------------------------------------------------------------------------------
# A shared-encoder arm of the five-code bundle
# -------------------------------------------------------------------------------------------------

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
# The five-code bundle's codebook, in code_id order
FIVE_CODES = ('111111', '111112', '111113', '222222', '333333')
ARM_DIMENSION = 16
TOKEN_WINDOW = 32

def lightning_checkpoint(model: NAICSContrastiveModel) -> Dict[str, Any]:
    '''The dict Lightning saves for ``model``, its checkpoint contract included.'''

    checkpoint = {
        'state_dict': model.state_dict(),
        'hyper_parameters': dict(model.hparams),
        'pytorch-lightning_version': pyl.__version__,
    }
    model.on_save_checkpoint(checkpoint)
    return checkpoint

def five_code_token_rows(token_config: TokenizationConfig,
                         bundle: ValidatedSupervisionBundle) -> List[Dict[str, Any]]:
    '''The five codes' cached token rows, in codebook order (the descriptions' ``index``).'''

    cache = tokenization_cache(
        token_config,
        description_fingerprint=bundle.manifest.description_fingerprint,
        codebook_fingerprint=bundle.manifest.codebook_fingerprint,
    )
    return [cache[index] for index in range(len(FIVE_CODES))]

@pytest.fixture
def five_code_descriptions_parquet(tmp_path, text_descriptions_fixture) -> Path:
    '''The five-code descriptions with their text channels and a ``level``, under ``tmp_path``.'''

    path = tmp_path / 'naics_descriptions.parquet'
    text_descriptions_fixture.with_columns(level=pl.lit(6)).write_parquet(path)
    return path

@pytest.fixture
def five_code_token_config(tmp_path, five_code_descriptions_parquet) -> TokenizationConfig:
    '''The five codes' token cache: MiniLM's tokenizer, a 32-token window, under ``tmp_path``.'''

    return TokenizationConfig(
        descriptions_parquet=str(five_code_descriptions_parquet),
        tokenizer_name=MINILM,
        max_length=TOKEN_WINDOW,
        output_path=str(tmp_path / 'token_cache' / 'token_cache.pt'),
    )

@pytest.fixture
def shared_model(tiny_backbone, generated_bundle) -> NAICSContrastiveModel:
    '''
    A d = 16 masked-mean model of the five-code bundle on the tiny backbone, in eval mode.

    It records MiniLM's summaries, as training does: under the test seam, the dummy pin's sha256.
    '''

    model = NAICSContrastiveModel(
        base_model_name=MINILM,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        fusion='masked_mean',
        dimension=ARM_DIMENSION,
        supervision_manifest_path=str(generated_bundle),
        summaries=summaries_identity(MINILM),
    )
    return model.eval()

@pytest.fixture
def shared_checkpoint(tmp_path, shared_model) -> Path:
    '''``shared_model`` saved as a Lightning checkpoint.'''

    path = tmp_path / 'arm.ckpt'
    torch.save(lightning_checkpoint(shared_model), path)
    return path

@pytest.fixture
def truncated_checkpoint(tmp_path, shared_model) -> Path:
    '''``shared_model`` saved as a checkpoint trained before Stage 6b: it records no summaries.'''

    checkpoint = lightning_checkpoint(shared_model)
    del checkpoint['stage3_supervision']['summaries']
    del checkpoint['hyper_parameters']['summaries']
    path = tmp_path / 'truncated.ckpt'
    torch.save(checkpoint, path)
    return path

# The fields every contract saved before Stage 7 carried, at their last values, which the contract
# no longer has (spec 4.5): legacy containment's supervision mode, and the six-term objective's
# loss and mining versions. None of those contracts names an objective
PRE_STAGE_7_FIELDS = {
    'supervision_mode': 'repaired',
    'structural_preference_loss_version': 'structural-preference-v1',
    'mining_contract_version': 'negative-selection-v2',
}
# P23's refusal of a checkpoint trained before Stage 7, as every load raises it
PRE_STAGE_7_REFUSAL = r'objective pre-req11, not req11-v1 .*nothing migrates \(D2\)'

def pre_stage_7_contract(contract: Mapping[str, Any]) -> Dict[str, Any]:
    '''
    ``contract`` as a model saved it before Stage 7: its supervision identity, encoder record and
    summaries, with the three fields Stage 7 dropped, and no objective.
    '''

    kept = ('contract_version', 'bundle_id', 'codebook_fingerprint', 'encoder', 'summaries')
    return {**{name: contract[name] for name in kept}, **PRE_STAGE_7_FIELDS}

@pytest.fixture
def pre_stage7_checkpoint(tmp_path, shared_model) -> Path:
    '''
    ``shared_model`` saved as a checkpoint trained before Stage 7 (spec 4.5): its contract carries
    the three fields Stage 7 dropped and names no objective.

    Its hyperparameters hold the old head's curvature, which no model takes now, and no radius
    bound, so a load that let it through would rebuild the head at the default R (spec 4.2).
    '''

    checkpoint = lightning_checkpoint(shared_model)
    checkpoint['stage3_supervision'] = pre_stage_7_contract(checkpoint['stage3_supervision'])
    checkpoint['hyper_parameters']['curvature'] = 1.0
    del checkpoint['hyper_parameters']['radius_bound']
    path = tmp_path / 'pre_stage7.ckpt'
    torch.save(checkpoint, path)
    return path

@pytest.fixture
def pre_stage8_checkpoint(tmp_path, shared_model) -> Path:
    '''
    ``shared_model`` saved as Stage 7 saved its checkpoints (P7): neither its encoder record nor
    its hyperparameters name a geometry, so every load reads it as hyperbolic.
    '''

    checkpoint = lightning_checkpoint(shared_model)
    del checkpoint['stage3_supervision']['encoder']['geometry']
    del checkpoint['hyper_parameters']['geometry']
    path = tmp_path / 'pre_stage8.ckpt'
    torch.save(checkpoint, path)
    return path

def forbid_model_loads(monkeypatch) -> None:
    '''
    From here on, fail the test if a checkpoint's model loads (``load_from_checkpoint``): a refusal
    of its contract must come first.

    Call it in the test's body, after its fixtures: some of them export a table, which loads one.
    '''

    def never(*_args, **_kwargs):
        # AssertionError, so a test's pytest.raises(ValueError) cannot swallow an unwanted load
        raise AssertionError('the model loaded before its contract was refused')

    monkeypatch.setattr(NAICSContrastiveModel, 'load_from_checkpoint', never)

@pytest.fixture
def exported_table(tmp_path, shared_checkpoint, validated_bundle, five_code_token_config) -> Path:
    '''``shared_checkpoint``'s code table, exported on the CPU, with its provenance beside it.'''

    return export_code_table(
        shared_checkpoint, validated_bundle, five_code_token_config, tmp_path / 'arm_table.parquet'
    )

@pytest.fixture
def text_only_comparator_table(tmp_path, five_code_descriptions_parquet) -> Path:
    '''
    The five codes' text-only comparator table, with its provenance beside it.

    ``build_text_only_table`` writes both, on the tiny backbone and MiniLM's tokenizer. The
    provenance has the table's hash and window, as an export's does, but names no checkpoint.
    '''

    return build_text_only_table(
        five_code_descriptions_parquet,
        tmp_path / 'text_only.parquet',
        backbone=MINILM,
        max_length=TOKEN_WINDOW,
        model=tiny_bert(),
        tokenizer=AutoTokenizer.from_pretrained(MINILM),
    )

# -------------------------------------------------------------------------------------------------
# A shared-encoder arm of the reference bundle (Stage 7)
#
# The reference bundle (tests/fixtures/supervision.py) has 17 codes at levels 2-6, four unary pairs
# and 11 task queries. Its longest marked channel text has 44 tokens, so its token cache uses the
# shipped 128-token window, which the dummy summaries pin fits. At 4 queries a step an epoch has 3
# steps: 6, 6 and 5 of the codes as anchors, and 4, 4 and 3 queries.
# -------------------------------------------------------------------------------------------------

REFERENCE_WINDOW = 128
REFERENCE_QUERIES_PER_STEP = 4

@pytest.fixture(scope='session')
def minilm_tokenizer():
    '''MiniLM's tokenizer: its vocabulary is the only download.'''

    return AutoTokenizer.from_pretrained(MINILM)

@pytest.fixture
def reference_arm_token_config(tmp_path, reference_bundle) -> TokenizationConfig:
    '''The reference codes' token cache: MiniLM's tokenizer, a 128-token window, under tmp_path.'''

    parameters = reference_bundle.manifest.generation_parameters
    return TokenizationConfig(
        descriptions_parquet=parameters['descriptions_parquet'],
        tokenizer_name=MINILM,
        max_length=REFERENCE_WINDOW,
        output_path=str(tmp_path / 'reference_token_cache' / 'token_cache.pt'),
    )

def reference_token_rows(token_config: TokenizationConfig,
                         bundle: ValidatedSupervisionBundle) -> List[Dict[str, Any]]:
    '''Every reference code's cached token rows, in codebook order.'''

    cache = tokenization_cache(
        token_config,
        description_fingerprint=bundle.manifest.description_fingerprint,
        codebook_fingerprint=bundle.manifest.codebook_fingerprint,
    )
    return [cache[code_id] for code_id in range(len(cache))]

@pytest.fixture
def reference_arm_code_rows(reference_arm_token_config, reference_bundle) -> List[Dict[str, Any]]:
    return reference_token_rows(reference_arm_token_config, reference_bundle)

def build_reference_model(
    manifest: Path, bundle: ValidatedSupervisionBundle, **overrides: Any
) -> NAICSContrastiveModel:
    '''
    A d = 16 masked-mean model of the reference bundle, as constructed (in train mode).

    Call it with the tiny backbone in place (``tiny_backbone``). ``overrides`` replace or add
    constructor arguments.
    '''

    arguments: Dict[str, Any] = {
        'base_model_name': MINILM,
        'lora_r': 2,
        'lora_alpha': 4,
        'lora_dropout': 0.0,
        'fusion': 'masked_mean',
        'dimension': ARM_DIMENSION,
        'supervision_manifest_path': str(manifest),
        'summaries': summaries_identity(MINILM),
        'supervision_bundle': bundle,
    }
    arguments.update(overrides)
    return NAICSContrastiveModel(**arguments)

@pytest.fixture
def reference_arm_model(
    tiny_backbone, reference_manifest, reference_bundle
) -> NAICSContrastiveModel:
    '''``build_reference_model`` with its defaults, on the tiny backbone.'''

    return build_reference_model(reference_manifest, reference_bundle)

def reference_step_dataset(
    bundle: ValidatedSupervisionBundle,
    code_rows: List[Dict[str, Any]],
    tokenizer: Any,
    *,
    queries_per_step: int = REFERENCE_QUERIES_PER_STEP,
    seed: int = 0,
) -> StepDataset:
    '''The reference bundle's two-stream steps, as training builds them: no epoch set yet.'''

    targets = CodeTargets.from_bundle(bundle)
    code_ids = {code: code_id for code_id, code in enumerate(targets.codes)}
    queries = tokenize_task_queries(
        build_task_queries(bundle), tokenizer, REFERENCE_WINDOW, code_ids
    )
    return StepDataset(
        code_rows=code_rows,
        code_levels=targets.levels,
        queries=queries,
        n_codes=len(targets.codes),
        seed=seed,
        queries_per_step=queries_per_step,
    )

@pytest.fixture
def reference_arm_steps(reference_bundle, reference_arm_code_rows, minilm_tokenizer) -> StepDataset:
    '''The reference bundle's steps at epoch 0, under seed 0.'''

    dataset = reference_step_dataset(reference_bundle, reference_arm_code_rows, minilm_tokenizer)
    dataset.set_epoch(0)
    return dataset
