'''
Unit tests for NAICSContrastiveModel, the text stage's Lightning module (spec 4.1-4.4, section 6).

The five-code arm on MiniLM carries the construction, forward, checkpoint and contract tests. The
training step runs on the reference arm (``tests/fixtures/shared_encoder.py``): a d = 16 model of
the reference bundle on the tiny backbone, whose 17 codes span levels 2-6 and whose 11 task
queries fill an epoch of three two-stream steps. Its tests cover Req 11's three terms on a live
code cache (No inert terms, Coverage, Precision, Cache), the optimizer and its schedule (P16), the
training hooks (P15, P18) and the health logs (P20).

The Trainer is a stand-in (``_attach_stub_trainer``) and the monitor a scripted one, which writes
nothing, so no test here reads a split or touches a selection log outside ``tmp_path`` (P28).
'''

import inspect
import logging
import math
import re
import statistics
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence
from unittest.mock import Mock

import pytest
import pytorch_lightning as pyl
import torch
from pytorch_lightning.utilities.model_helpers import is_overridden
from transformers import PreTrainedModel

from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.text_only import matrix_fingerprint
from naics_embedder.supervision.checkpoints import contract_for_bundle, shared_encoder_architecture
from naics_embedder.supervision.code_targets import CodeTargets
from naics_embedder.supervision.schema import CONTRACT_VERSION
from naics_embedder.text_model import naics_model as model_module
from naics_embedder.text_model.export import encode_token_rows
from naics_embedder.text_model.fields import CHANNELS, QUERY
from naics_embedder.text_model.loss import LogitScale
from naics_embedder.text_model.monitor import (
    MONITOR_RECORDS,
    MonitorRead,
    OutcomeMonitor,
    read_monitor_records,
)
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.text_model.shared_encoder import SharedEncoder
from tests.fixtures.shared_encoder import (
    MINILM,
    PRE_STAGE_7_REFUSAL,
    REFERENCE_WINDOW,
    build_reference_model,
    lightning_checkpoint,
    pre_stage_7_contract,
)

logger = logging.getLogger(__name__)

# The key the monitor's MRR is logged, stepped and checkpointed under (P17, P18)
OUTCOME_MRR = 'val/outcome_mrr'
# The reference bundle's four unary pairs, both ways (tests/fixtures/supervision.py)
REFERENCE_PARTNERS = {
    '31111': '311111',
    '31121': '311211',
    '32111': '321111',
    '44111': '441111',
}
REFERENCE_PARTNERS.update({child: parent for parent, child in list(REFERENCE_PARTNERS.items())})

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def model_config(generated_bundle):
    '''Minimal model configuration for fast testing, backed by the five-code bundle fixture.'''

    return {
        'base_model_name': MINILM,
        'lora_r': 4,
        'lora_alpha': 8,
        'lora_dropout': 0.1,
        'num_experts': 4,
        'top_k': 2,
        'moe_hidden_dim': 512,
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'load_balancing_coef': 0.01,
        'supervision_manifest_path': str(generated_bundle),
    }

@pytest.fixture
def naics_model(model_config, test_device):
    '''Create NAICSContrastiveModel instance for testing.'''

    model = NAICSContrastiveModel(**model_config)
    model.to(test_device)
    model.eval()
    return model

@pytest.fixture
def sample_training_batch(test_device, batch_size=4):
    '''A batch of four codes' channel inputs, for the forward pass.'''

    seq_length = 32

    def create_channel_inputs(batch_size):
        return {
            channel: {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_length), device=test_device),
                'attention_mask': torch.ones(batch_size, seq_length, device=test_device),
                'present': torch.ones(batch_size, dtype=torch.bool, device=test_device),
            }
            for channel in CHANNELS
        }

    return {'anchor': create_channel_inputs(batch_size), 'batch_size': batch_size}

@pytest.fixture
def reference_model(tiny_backbone, reference_manifest, reference_bundle) -> Callable[..., Any]:
    '''Build a model of the reference bundle on the tiny backbone, with constructor overrides.'''

    def build(**overrides: Any) -> NAICSContrastiveModel:
        return build_reference_model(reference_manifest, reference_bundle, **overrides)

    return build

@pytest.fixture
def epoch_steps(reference_arm_steps) -> List[Dict[str, Any]]:
    '''The three steps of the reference bundle's epoch 0.'''

    return [reference_arm_steps[step] for step in range(len(reference_arm_steps))]

@pytest.fixture
def code_targets(reference_bundle) -> CodeTargets:
    return CodeTargets.from_bundle(reference_bundle)

# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------

class ScriptedMonitor:
    '''
    A stand-in for ``OutcomeMonitor``: it records each call in ``events`` and returns scripted
    MRRs. It reads no split and writes nothing, so no selection log is touched (P28).
    '''

    def __init__(self, mrrs: Sequence[float] = (0.5, ), events: Optional[List[Any]] = None):
        self.mrrs = list(mrrs)
        self.events: List[Any] = [] if events is None else events
        self.reads: List[Dict[str, Any]] = []

    def start(self, *, resumed_epoch: Optional[int]) -> None:
        self.events.append(('start', resumed_epoch))

    def read(self, model, cache, *, training_run: str, seed: int, epoch: int) -> MonitorRead:
        self.reads.append(
            {
                'model': model,
                'cache': cache,
                'training_run': training_run,
                'seed': seed,
                'epoch': epoch
            }
        )
        self.events.append(('read', epoch))
        mrr = self.mrrs[len(self.reads) - 1]
        return MonitorRead(epoch=epoch, mrr=mrr, record={'epoch': epoch})

    def append(self, read: MonitorRead) -> None:
        self.events.append(('append', read.epoch))

def _attach_stub_trainer(
    model: NAICSContrastiveModel,
    code_rows: Sequence[Dict[str, Any]],
    *,
    epoch: int = 0,
    global_step: int = 0,
    num_training_batches: int = 3,
) -> SimpleNamespace:
    '''
    Attach a stand-in for the Trainer: what the hooks and ``optimizer_step`` read of it.

    The optimizer and plateau come from ``configure_optimizers``, as Lightning's would. A real
    ``self.log`` writes into a Trainer's results, so it becomes a Mock. Returns the trainer, the
    optimizer, the plateau and the log.
    '''

    config = model.configure_optimizers()
    plateau = config['lr_scheduler']['scheduler']
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(code_rows=code_rows),
        current_epoch=epoch,
        global_step=global_step,
        num_training_batches=num_training_batches,
        lr_scheduler_configs=[SimpleNamespace(scheduler=plateau)],
        logger=None,
    )
    model.trainer = trainer
    model.log = Mock()
    return SimpleNamespace(
        trainer=trainer, optimizer=config['optimizer'], plateau=plateau, log=model.log
    )

def _logged(log: Mock) -> Dict[str, Any]:
    '''Each key the mocked ``self.log`` received, with its last call.'''

    return {call.args[0]: call for call in log.call_args_list}

def _first_rows(batch: Dict[str, Any], anchors: int = 1, queries: int = 1) -> Dict[str, Any]:
    '''A step cut down to its first anchors and queries, in the same layout.'''

    def cut(inputs: Dict[str, Dict[str, torch.Tensor]], rows: int):
        return {
            field: {
                name: value[:rows]
                for name, value in tensors.items()
            }
            for field, tensors in inputs.items()
        }

    codes, step_queries = batch['codes'], batch['queries']
    return {
        'codes': {
            'inputs': cut(codes['inputs'], anchors),
            'ids': codes['ids'][:anchors],
            'levels': codes['levels'][:anchors],
        },
        'queries': {
            'inputs': cut(step_queries['inputs'], queries),
            'levels': step_queries['levels'][:queries],
            'targets': step_queries['targets'][:queries],
            'negatives': step_queries['negatives'][:queries],
        },
    }

def _scale_closure(model: NAICSContrastiveModel, optimizer, loss: Callable[[], torch.Tensor]):
    '''A closure as Lightning's runs one: zero the gradients, then the loss and its backward.'''

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        value = loss()
        value.backward()
        return value

    return closure

def _float32(value: float) -> float:
    '''``value`` as a float32 parameter stores it.'''

    return torch.tensor(value, dtype=torch.float32).item()

def _flags(model: torch.nn.Module) -> Dict[str, bool]:
    return {name: module.training for name, module in model.named_modules()}

# -------------------------------------------------------------------------------------------------
# Test: Initialization
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestModelInitialization:
    '''Test NAICSContrastiveModel initialization and configuration.'''

    def test_the_constructor_takes_exactly_its_arguments_with_their_defaults(self):
        '''P15: Req 11's settings, the schedule's and the run's, in this order.'''

        expected = [
            ('base_model_name', MINILM),
            ('lora_r', 8),
            ('lora_alpha', 16),
            ('lora_dropout', 0.1),
            ('fusion', 'masked_mean'),
            ('dimension', 16),
            ('num_experts', 4),
            ('top_k', 2),
            ('moe_hidden_dim', 1024),
            ('radius_bound', 8.0),
            ('code_code_weight', 1.0),
            ('radial_weight', 1.0),
            ('target_temperature', 1.0),
            ('radial_step', 1.0),
            ('logit_scale_init', 1.0),
            ('logit_scale_range', (0.01, 100.0)),
            ('learning_rate', 1e-4),
            ('weight_decay', 0.01),
            ('warmup_epochs', 1),
            ('lr_plateau_factor', 0.5),
            ('lr_plateau_patience', 2),
            ('load_balancing_coef', 0.01),
            ('seed', 0),
            ('run_settings', None),
            ('supervision_manifest_path', None),
            ('supervision_contract_version', CONTRACT_VERSION),
            ('summaries', None),
            ('checkpoint_contract', None),
            ('supervision_bundle', None),
            ('monitor', None),
        ]

        parameters = inspect.signature(NAICSContrastiveModel.__init__).parameters
        assert [
            (name, parameter.default) for name, parameter in parameters.items() if name != 'self'
        ] == expected

    def test_model_creation(self, naics_model):
        '''The encoder and two logit scales; none of the old objective's machinery (Req 10, 11).'''

        assert isinstance(naics_model, pyl.LightningModule)
        assert isinstance(naics_model.encoder, SharedEncoder)
        assert isinstance(naics_model.logit_scale_task, LogitScale)
        assert isinstance(naics_model.logit_scale_code, LogitScale)
        assert naics_model.code_cache is None
        for name in (
            'loss_fn',
            'hard_negative_miner',
            'router_guided_miner',
            'norm_adaptive_margin',
            'hierarchy_loss_fn',
            'structural_preference_loss_fn',
            'supervision_index',
            'selection_coordinator',
            'embedding_eval',
            'embedding_stats',
            'hierarchy_metrics',
            'naics_hierarchy',
        ):
            assert not hasattr(naics_model, name), name

    def test_the_text_stage_has_no_validation_loop_and_none_of_the_old_mixins(
        self, reference_arm_model
    ):
        '''The monitor is the validation (spec 4.3, 4.4); no structural statistic is computed.'''

        model = reference_arm_model
        assert not is_overridden('validation_step', model)
        assert not is_overridden('on_validation_epoch_end', model)
        bases = {base.__name__ for base in type(model).__mro__}
        assert bases.isdisjoint({'ValidationMixin', 'CurriculumMixin', 'DistributedMixin'})
        assert {'LossMixin', 'LoggingMixin', 'OptimizerMixin'} <= bases

    def test_hyperparameters_saved(self, naics_model, model_config):
        '''The settings are saved with their defaults; the old objective's are gone.'''

        hparams = naics_model.hparams
        assert hparams['learning_rate'] == model_config['learning_rate']
        assert hparams['weight_decay'] == model_config['weight_decay']
        assert {
            name: hparams[name]
            for name in (
                'code_code_weight',
                'radial_weight',
                'target_temperature',
                'radial_step',
                'logit_scale_init',
                'logit_scale_range',
                'warmup_epochs',
                'lr_plateau_factor',
                'lr_plateau_patience',
                'seed',
            )
        } == {
            'code_code_weight': 1.0,
            'radial_weight': 1.0,
            'target_temperature': 1.0,
            'radial_step': 1.0,
            'logit_scale_init': 1.0,
            'logit_scale_range': (0.01, 100.0),
            'warmup_epochs': 1,
            'lr_plateau_factor': 0.5,
            'lr_plateau_patience': 2,
            'seed': 0,
        }
        for name in ('temperature', 'curvature', 'hierarchy_weight', 'warmup_steps'):
            assert name not in hparams, name

    def test_the_seed_and_run_settings_are_saved_but_the_monitor_is_not(self, reference_model):
        monitor = ScriptedMonitor()
        settings = {'fusion': 'masked_mean', 'max_epochs': 40}
        model = reference_model(seed=7, run_settings=settings, monitor=monitor)

        assert model.monitor is monitor
        assert model.hparams['seed'] == 7
        assert model.hparams['run_settings'] == settings
        for name in ('monitor', 'supervision_bundle', 'checkpoint_contract'):
            assert name not in model.hparams, name

    def test_encoder_configuration(self, naics_model, model_config):
        '''One MiniLM backbone, masked-mean fusion and one Linear(384 -> 16) to the head.'''

        encoder = naics_model.encoder
        assert isinstance(encoder, SharedEncoder)
        # The head takes the bound, not a curvature (spec 4.2)
        assert encoder.head.radius_bound == 8.0
        assert sum(isinstance(module, PreTrainedModel) for module in naics_model.modules()) == 1
        assert encoder.fusion_name == 'masked_mean'
        assert (encoder.projection.in_features, encoder.projection.out_features) == (384, 16)
        assert list(encoder.head.parameters()) == []

    def test_the_radius_bound_is_saved_and_reaches_the_head(self, model_config, tiny_backbone):
        model = NAICSContrastiveModel(**model_config, radius_bound=5.0)

        assert model.hparams['radius_bound'] == 5.0
        assert model.encoder.head.radius_bound == 5.0

    def test_an_unknown_dimension_is_refused(self, model_config):
        with pytest.raises(ValueError, match='unknown dimension'):
            NAICSContrastiveModel(**model_config, dimension=12)

    def test_the_code_targets_are_buffers_that_checkpoints_leave_out(
        self, reference_arm_model, code_targets
    ):
        '''P15: D*, the unary partners and the levels, in codebook order, never saved.'''

        model = reference_arm_model
        assert model.codes == code_targets.codes
        expected = {
            'structural_distance': torch.from_numpy(code_targets.structural_distance),
            'unary_partner': torch.from_numpy(code_targets.unary_partner),
            'code_levels': torch.from_numpy(code_targets.levels),
        }
        buffers = dict(model.named_buffers())
        state = model.state_dict()
        for name, value in expected.items():
            assert buffers[name] is getattr(model, name)
            assert buffers[name].dtype == value.dtype
            assert torch.equal(buffers[name], value), name
            assert name not in state, name

    def test_the_two_logit_scales_start_at_their_init_inside_their_range(self, reference_model):
        model = reference_model(logit_scale_init=2.0, logit_scale_range=(0.5, 10.0))

        for scale in (model.logit_scale_task, model.logit_scale_code):
            assert (scale.low, scale.high) == (0.5, 10.0)
            assert scale().item() == pytest.approx(2.0)
        assert model.logit_scale_task is not model.logit_scale_code
        assert model.logit_scale_task.log_scale is not model.logit_scale_code.log_scale

    @pytest.mark.parametrize(
        ('setting', 'value', 'message'),
        [
            ('code_code_weight', -0.5, 'code_code_weight'),
            ('radial_weight', -1.0, 'radial_weight'),
            ('target_temperature', 0.0, 'target_temperature'),
            ('radial_step', 0.0, 'radial_step'),
            ('logit_scale_range', (1.0, 1.0), 'logit-scale range'),
            ('logit_scale_init', 200.0, 'inside its range'),
            ('warmup_epochs', -1, 'warmup_epochs'),
            ('lr_plateau_factor', 1.0, 'lr_plateau_factor'),
            ('lr_plateau_patience', -1, 'lr_plateau_patience'),
        ],
        ids=[
            'code_code_weight',
            'radial_weight',
            'target_temperature',
            'radial_step',
            'logit_scale_range',
            'logit_scale_init',
            'warmup_epochs',
            'lr_plateau_factor',
            'lr_plateau_patience',
        ],
    )
    def test_a_setting_outside_its_range_is_refused(self, reference_model, setting, value, message):
        '''The refusals the config's validators mirror (spec 5, P22), before any step runs.'''

        with pytest.raises(ValueError, match=message):
            reference_model(**{setting: value})

    def test_repaired_mode_requires_manifest(self, model_config):
        '''Repaired training fails closed without a supervision manifest.'''

        model_config.pop('supervision_manifest_path')
        with pytest.raises(ValueError, match='supervision_manifest_path'):
            NAICSContrastiveModel(**model_config)

    @pytest.mark.parametrize(
        ('argument', 'value'),
        [
            ('supervision_mode', 'repaired'),
            ('distance_matrix_path', 'naics_distance_matrix.parquet'),
            ('relations_parquet_path', 'naics_relations.parquet'),
        ],
    )
    def test_the_containment_arguments_are_gone(self, model_config, argument, value):
        '''D2: the model is always repaired, so neither a mode nor a legacy input is an argument.'''

        with pytest.raises(TypeError, match=argument):
            NAICSContrastiveModel(**model_config, **{argument: value})

    def test_rejects_mismatched_contract_version(self, model_config):
        '''The bundle must match the expected supervision contract.'''

        with pytest.raises(ValueError, match='contract'):
            NAICSContrastiveModel(
                **model_config,
                supervision_contract_version='stage3-supervision-v0',
            )

    def test_fusion_defaults_to_masked_mean_and_an_unknown_one_is_refused(
        self, naics_model, model_config
    ):
        assert naics_model.fusion == 'masked_mean'
        assert naics_model.hparams['fusion'] == 'masked_mean'
        with pytest.raises(ValueError, match='unknown fusion'):
            NAICSContrastiveModel(**model_config, fusion='concatenate')

# -------------------------------------------------------------------------------------------------
# Test: Forward Pass
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestForwardPass:
    '''Test model forward pass.'''

    def test_forward_basic(self, naics_model, sample_training_batch):
        '''The default fusion returns the point, its tangent, radius and direction, and no gates.'''

        with torch.no_grad():
            output = naics_model(sample_training_batch['anchor'])

        assert set(output) == {'embedding', 'tangent', 'radius', 'direction'}

    def test_forward_output_shapes(self, naics_model, sample_training_batch):
        '''The Lorentz point is (batch, 17), its bounded tangent and direction (batch, 16) and
        its radius (batch,).'''

        batch_size = sample_training_batch['batch_size']

        with torch.no_grad():
            output = naics_model(sample_training_batch['anchor'])

        assert naics_model.encoder.dimension == 16
        assert output['embedding'].shape == (batch_size, 17)
        assert output['tangent'].shape == (batch_size, 16)
        assert output['radius'].shape == (batch_size, )
        assert output['direction'].shape == (batch_size, 16)

# -------------------------------------------------------------------------------------------------
# Test: The step's losses (Req 11 on the live cache)
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeLosses:
    '''``compute_losses``: Req 11's three terms on a step of two streams and the code cache.'''

    def test_a_step_before_any_refresh_is_refused(self, reference_arm_model, epoch_steps):
        '''The candidates come from the cache, so a step needs one (on_train_start builds it).'''

        with pytest.raises(RuntimeError, match='cache'):
            reference_arm_model.compute_losses(epoch_steps[0])

    def test_every_term_and_both_scales_get_gradient(
        self, reference_arm_model, reference_arm_code_rows, epoch_steps
    ):
        '''Spec 6, No inert terms; and dL/dr_a is nonzero for every anchor (Verification
        "Radius").'''

        model = reference_arm_model
        model.refresh_code_cache(reference_arm_code_rows)

        losses = model.compute_losses(epoch_steps[0])

        assert isinstance(losses, model_module.StepLosses)
        assert losses.load_balancing is None
        weight = model.encoder.projection.weight
        for name in ('task', 'code_code', 'radial'):
            (gradient, ) = torch.autograd.grad(getattr(losses, name), weight, retain_graph=True)
            assert gradient.abs().sum() > 0, name
        task_scale = model.logit_scale_task.log_scale
        code_scale = model.logit_scale_code.log_scale
        # Each term has its own scale, and reads only that one
        gradients = {
            (term, scale_name): torch.autograd.grad(
                getattr(losses, term), scale, retain_graph=True, allow_unused=True
            )[0]
            for term in ('task', 'code_code')
            for scale_name, scale in (('task', task_scale), ('code', code_scale))
        }
        for pair in (('task', 'task'), ('code_code', 'code')):
            assert gradients[pair] is not None and gradients[pair].item() != 0, pair
        assert gradients['task', 'code'] is None
        assert gradients['code_code', 'task'] is None
        (radius_gradient, ) = torch.autograd.grad(losses.total, losses.anchor_radius)
        assert losses.anchor_radius.shape == epoch_steps[0]['codes']['ids'].shape
        assert (radius_gradient != 0).all()

    def test_the_total_weights_the_terms_and_the_settings_reach_them(
        self, reference_model, reference_arm_code_rows, epoch_steps, monkeypatch
    ):
        '''L = L_task + w_c L_cc + w_r L_rad (spec 4.1), with tau_t and rho as configured.'''

        model = reference_model(
            code_code_weight=0.25, radial_weight=2.0, target_temperature=0.5, radial_step=0.75
        )
        model.refresh_code_cache(reference_arm_code_rows)
        settings = {}
        code_code, radial = model_module.code_code_loss, model_module.radial_loss

        def code_code_spy(distances, scale, structural, keep, target_temperature):
            settings['target_temperature'] = target_temperature
            return code_code(distances, scale, structural, keep, target_temperature)

        def radial_spy(radius, levels, radial_step):
            settings['radial_step'] = radial_step
            return radial(radius, levels, radial_step)

        monkeypatch.setattr(model_module, 'code_code_loss', code_code_spy)
        monkeypatch.setattr(model_module, 'radial_loss', radial_spy)

        losses = model.compute_losses(epoch_steps[0])

        assert settings == {'target_temperature': 0.5, 'radial_step': 0.75}
        expected = losses.task + 0.25 * losses.code_code + 2.0 * losses.radial
        torch.testing.assert_close(losses.total, expected, rtol=1e-6, atol=0.0)

    def test_load_balancing_reads_both_streams_gets_gradient_and_enters_the_total_under_moe(
        self, reference_model, reference_arm_code_rows, epoch_steps, monkeypatch
    ):
        '''Spec 6, No inert terms under moe; the term is added with its coefficient (spec 4.1).'''

        model = reference_model(fusion='moe', moe_hidden_dim=16, load_balancing_coef=0.25)
        model.log = Mock()
        model.refresh_code_cache(reference_arm_code_rows)
        rows = []
        balance = model._compute_load_balancing_loss

        def spy(gate_probs_list, topk_indices_list, batch_size):
            rows.append(sum(len(gate_probs) for gate_probs in gate_probs_list))
            return balance(gate_probs_list, topk_indices_list, batch_size)

        monkeypatch.setattr(model, '_compute_load_balancing_loss', spy)
        step = epoch_steps[0]

        losses = model.compute_losses(step)

        assert rows == [len(step['codes']['ids']) + len(step['queries']['levels'])]
        gate = model.encoder.fusion.moe.gate.weight
        (gradient, ) = torch.autograd.grad(losses.load_balancing, gate, retain_graph=True)
        assert gradient.abs().sum() > 0
        # The terms' weights are 1, and the coefficient 0.25
        expected = losses.task + losses.code_code + losses.radial + 0.25 * losses.load_balancing
        torch.testing.assert_close(losses.total, expected, rtol=1e-6, atol=0.0)

    def test_every_step_scores_each_anchor_against_every_code_but_itself_and_its_partner(
        self, reference_arm_model, reference_arm_code_rows, epoch_steps, code_targets, monkeypatch
    ):
        '''Spec 6, Coverage: J_a over all N codes, and each code an anchor once over the epoch.'''

        model = reference_arm_model
        model.refresh_code_cache(reference_arm_code_rows)
        calls = []
        code_code = model_module.code_code_loss

        def spy(distances, scale, structural, keep, target_temperature):
            calls.append((tuple(distances.shape), structural.detach().clone(), keep.clone()))
            return code_code(distances, scale, structural, keep, target_temperature)

        monkeypatch.setattr(model_module, 'code_code_loss', spy)

        for step in epoch_steps:
            model.compute_losses(step)

        codes = code_targets.codes
        anchors = []
        assert len(calls) == len(epoch_steps)
        for step, (shape, structural, keep) in zip(epoch_steps, calls):
            ids = step['codes']['ids'].tolist()
            assert shape == (len(ids), len(codes))
            for row, code_id in enumerate(ids):
                code = codes[code_id]
                masked = {codes[other] for other in range(len(codes)) if not keep[row, other]}
                partner = {REFERENCE_PARTNERS[code]} if code in REFERENCE_PARTNERS else set()
                assert masked == {code} | partner, code
                assert torch.equal(
                    structural[row], torch.from_numpy(code_targets.structural_distance[code_id])
                )
                anchors.append(code)
        assert sorted(anchors) == sorted(codes)

    def test_the_task_candidates_are_the_levels_codes_and_the_forced_negatives(
        self, reference_arm_model, reference_arm_code_rows, epoch_steps, code_targets, monkeypatch
    ):
        '''Spec 4.1(i): C = the codes at the query's level, plus N, over all codes (Req 8(b)).'''

        model = reference_arm_model
        model.refresh_code_cache(reference_arm_code_rows)
        calls = []
        task = model_module.task_loss

        def spy(distances, scale, candidates, targets):
            calls.append((tuple(distances.shape), candidates.clone(), targets.clone()))
            return task(distances, scale, candidates, targets)

        monkeypatch.setattr(model_module, 'task_loss', spy)

        for step in epoch_steps:
            model.compute_losses(step)

        codes = code_targets.codes

        def named(mask: torch.Tensor) -> frozenset:
            return frozenset(codes[index] for index in mask.nonzero().flatten().tolist())

        by_query = {}
        for step, (shape, candidates, targets) in zip(epoch_steps, calls):
            levels = step['queries']['levels'].tolist()
            assert shape == (len(levels), len(codes))
            assert torch.equal(targets, step['queries']['targets'])
            for row, level in enumerate(levels):
                by_query[level, named(targets[row])] = named(candidates[row])
        assert sum(len(step['queries']['levels']) for step in epoch_steps) == 11
        # 'Retailing new cars' (level 2): both sectors, and its referencing code
        assert by_query[2, frozenset({'44'})] == {'31', '44', '311211'}
        # 'Dealing in new cars' (level 5): the five-digit codes, and its referencing code
        assert by_query[5, frozenset({'44111'})] == {'31111', '31121', '32111', '44111', '311111'}

    @pytest.mark.parametrize('fusion', ['masked_mean', 'moe'])
    def test_the_distances_and_terms_run_in_float32_with_autocast_off(
        self, reference_model, reference_arm_code_rows, epoch_steps, monkeypatch, fusion
    ):
        '''Spec 6, Precision: under CPU bf16 autocast, every distance and term is float32.'''

        model = reference_model(fusion=fusion, moe_hidden_dim=16)
        model.log = Mock()
        model.refresh_code_cache(reference_arm_code_rows)
        seen = []

        def floating(values: Sequence[Any]) -> List[torch.Tensor]:
            found = []
            for value in values:
                if isinstance(value, (list, tuple)):
                    found.extend(floating(value))
                elif isinstance(value, torch.Tensor) and value.is_floating_point():
                    found.append(value)
            return found

        def spy(name: str, function: Callable[..., torch.Tensor]):

            def wrapped(*args, **kwargs):
                result = function(*args, **kwargs)
                dtypes = {tensor.dtype for tensor in floating([*args, *kwargs.values()])}
                seen.append((name, torch.is_autocast_enabled('cpu'), dtypes, result.dtype))
                return result

            return wrapped

        for name in ('polar_distance', 'task_loss', 'code_code_loss', 'radial_loss'):
            monkeypatch.setattr(model_module, name, spy(name, getattr(model_module, name)))
        if fusion == 'moe':
            monkeypatch.setattr(
                model,
                '_compute_load_balancing_loss',
                spy('load_balancing', model._compute_load_balancing_loss),
            )

        with torch.autocast('cpu', dtype=torch.bfloat16):
            losses = model.compute_losses(epoch_steps[0])

        names = [entry[0] for entry in seen]
        expected = [
            'polar_distance', 'polar_distance', 'task_loss', 'code_code_loss', 'radial_loss'
        ]
        if fusion == 'moe':
            expected.append('load_balancing')
        assert sorted(names) == sorted(expected)
        for name, autocast, inputs, output in seen:
            assert not autocast, name
            assert inputs == {torch.float32}, name
            assert output == torch.float32, name
        for name in ('total', 'task', 'code_code', 'radial', 'anchor_radius'):
            assert getattr(losses, name).dtype == torch.float32, name

    def test_the_candidates_are_the_cache_with_the_anchors_live(
        self, reference_arm_model, reference_arm_code_rows, epoch_steps, monkeypatch
    ):
        '''Spec 4.3, Cache: every other row is a constant of the last refresh, and only the
        anchors' rows carry gradient. The step encodes its anchors and queries only.'''

        model = reference_arm_model
        cache = model.refresh_code_cache(reference_arm_code_rows)
        # Dropout off, so the anchors' live points can be recomputed exactly
        model.eval()
        # Move the weights after the refresh: the cache is now stale
        generator = torch.Generator().manual_seed(5)
        with torch.no_grad():
            weight = model.encoder.projection.weight
            weight.add_(0.5 * torch.randn(weight.shape, generator=generator))
        step = epoch_steps[0]
        ids = step['codes']['ids']
        others = torch.tensor([code_id not in set(ids.tolist()) for code_id in range(17)])
        encoded = []
        forward = model.encoder.forward

        def forward_spy(inputs):
            fields = tuple(sorted(inputs))
            encoded.append((fields, len(inputs[fields[0]]['input_ids'])))
            return forward(inputs)

        monkeypatch.setattr(model.encoder, 'forward', forward_spy)
        candidates = []
        distance = model_module.polar_distance

        def distance_spy(radius_a, direction_a, radius_b, direction_b):
            candidates.append((radius_b, direction_b))
            return distance(radius_a, direction_a, radius_b, direction_b)

        monkeypatch.setattr(model_module, 'polar_distance', distance_spy)

        model.compute_losses(step)

        queries = len(step['queries']['levels'])
        assert sorted(encoded) == sorted(
            [(tuple(sorted(CHANNELS)), len(ids)), ((QUERY, ), queries)]
        )
        with torch.no_grad():
            live = forward(step['codes']['inputs'])
            fresh = encode_token_rows(model, reference_arm_code_rows)
        # The weights moved, so the cache is stale at the anchors' rows and at every other row
        assert not torch.allclose(live['radius'], cache.radius[ids])
        assert not torch.allclose(fresh['radius'][others].float(), cache.radius[others])
        assert len(candidates) == 2
        projection = model.encoder.projection.weight
        for radius, direction in candidates:
            assert torch.equal(radius[others], cache.radius[others])
            assert torch.equal(direction[others], cache.direction[others])
            assert torch.equal(radius[ids].detach(), live['radius'])
            assert torch.equal(direction[ids].detach(), live['direction'])
            (anchor_gradient,
             ) = torch.autograd.grad(radius[ids].sum(), projection, retain_graph=True)
            (other_gradient,
             ) = torch.autograd.grad(radius[others].sum(), projection, retain_graph=True)
            assert anchor_gradient.abs().sum() > 0
            assert other_gradient.abs().sum() == 0
        assert not cache.radius.requires_grad

# -------------------------------------------------------------------------------------------------
# Test: Training Step
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestTrainingStep:
    '''The training step: ``compute_losses`` on one step of the two streams.'''

    def test_training_step_basic(
        self, reference_arm_model, reference_arm_code_rows, epoch_steps, monkeypatch
    ):
        '''It returns the total ``compute_losses`` computed, and logs it for the progress bar.'''

        model = reference_arm_model
        model.log = Mock()
        model.refresh_code_cache(reference_arm_code_rows)
        computed = []
        compute = model.compute_losses

        def spy(batch):
            computed.append(compute(batch))
            return computed[-1]

        monkeypatch.setattr(model, 'compute_losses', spy)

        loss = model.training_step(epoch_steps[0], batch_idx=0)

        assert loss is computed[0].total
        assert loss.ndim == 0
        assert loss.item() > 0
        assert torch.isfinite(loss)
        call = _logged(model.log)['loss/step']
        assert call.args[1].item() == loss.item()
        assert call.kwargs['batch_size'] == 1
        assert (call.kwargs['on_step'], call.kwargs['on_epoch']) == (True, False)
        assert call.kwargs['prog_bar'] is True

    def test_training_step_gradient_flow(
        self, reference_arm_model, reference_arm_code_rows, epoch_steps
    ):
        model = reference_arm_model
        model.log = Mock()
        model.refresh_code_cache(reference_arm_code_rows)

        model.training_step(epoch_steps[1], batch_idx=1).backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(grad is not None for grad in grads)
        assert all(grad is None or torch.isfinite(grad).all() for grad in grads)

    def test_a_step_at_dimension_16_trains_the_adapter_and_the_projection(
        self, reference_arm_model, reference_arm_code_rows, epoch_steps
    ):
        '''Spec §6: the step reaches LoRA and the projection, and logs no load-balancing term.'''

        model = reference_arm_model
        model.log = Mock()
        model.refresh_code_cache(reference_arm_code_rows)

        model.training_step(epoch_steps[0], batch_idx=0).backward()

        encoder = model.encoder
        assert encoder.dimension == 16
        assert encoder.projection.weight.grad.abs().sum() > 0
        # PEFT starts lora_B at zero, so lora_A's first gradient is exactly zero (P9); the
        # pooler's adapter never gets one, since mean pooling never reads it (P8)
        adapters = {
            name: parameter
            for name, parameter in encoder.backbone.named_parameters()
            if 'lora_B' in name and '.pooler.' not in name
        }
        assert adapters
        for name, parameter in adapters.items():
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
        keys = set(_logged(model.log))
        assert 'loss/load_balancing' not in keys
        assert not any(key.startswith('train/moe/') for key in keys)

    @pytest.mark.parametrize(('fusion', 'logged'), [('masked_mean', False), ('moe', True)])
    def test_load_balancing_is_computed_and_logged_only_under_moe(
        self, reference_model, reference_arm_code_rows, epoch_steps, fusion, logged
    ):
        '''R11: only the MoE fusion has experts, so only it has a load-balancing term.'''

        model = reference_model(fusion=fusion, moe_hidden_dim=16)
        model.log = Mock()
        model.refresh_code_cache(reference_arm_code_rows)

        losses = model.compute_losses(epoch_steps[0])

        keys = set(_logged(model.log))
        assert (losses.load_balancing is not None) is logged
        assert any(key.startswith('train/moe/') for key in keys) is logged
        assert torch.isfinite(losses.total)

# -------------------------------------------------------------------------------------------------
# Test: Numerical Stability
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestNumericalStability:
    '''Finite losses and gradients, the logit scales at either end of their range included.'''

    def test_no_nan_in_training(self, reference_arm_model, reference_arm_code_rows, epoch_steps):
        model = reference_arm_model
        model.log = Mock()
        model.refresh_code_cache(reference_arm_code_rows)

        loss = model.training_step(epoch_steps[2], batch_idx=2)
        loss.backward()

        assert torch.isfinite(loss)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all(), f'non-finite gradient in {name}'

    @pytest.mark.parametrize('init', [0.01, 100.0])
    def test_a_step_at_either_end_of_the_logit_scale_range_is_finite(
        self, reference_model, reference_arm_code_rows, epoch_steps, init
    ):
        model = reference_model(logit_scale_init=init)
        model.log = Mock()
        model.refresh_code_cache(reference_arm_code_rows)

        loss = model.training_step(epoch_steps[0], batch_idx=0)
        loss.backward()

        assert torch.isfinite(loss)
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), name

# -------------------------------------------------------------------------------------------------
# Test: Edge Cases
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEdgeCases:
    '''Test edge cases.'''

    def test_batch_size_one(self, reference_arm_model, reference_arm_code_rows, epoch_steps):
        '''One anchor and one query make a step.'''

        model = reference_arm_model
        model.log = Mock()
        model.refresh_code_cache(reference_arm_code_rows)

        loss = model.training_step(_first_rows(epoch_steps[0]), batch_idx=0)

        assert torch.isfinite(loss)

# -------------------------------------------------------------------------------------------------
# Test: The code cache
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCodeCache:
    '''The model's cache of every code's (r, û) (spec 4.3, P14).'''

    def test_a_refresh_encodes_every_code_in_eval_mode_without_gradient(
        self, reference_arm_model, reference_arm_code_rows, code_targets, monkeypatch
    ):
        '''Spec 6, Cache: in eval mode, without gradient, in codebook order; the training flags,
        mixed ones included, come back.'''

        model = reference_arm_model
        model.train()
        model.encoder.projection.eval()
        flags = _flags(model)
        seen = []
        forward = model.encoder.forward

        def spy(inputs):
            seen.append(
                (torch.is_grad_enabled(), any(module.training for module in model.modules()))
            )
            return forward(inputs)

        monkeypatch.setattr(model.encoder, 'forward', spy)

        cache = model.refresh_code_cache(reference_arm_code_rows)

        assert seen and set(seen) == {(False, False)}
        assert _flags(model) == flags
        assert model.code_cache is cache
        assert cache.codes == code_targets.codes
        assert cache.radius.shape == (17, ) and cache.direction.shape == (17, 16)
        assert cache.radius.dtype == cache.direction.dtype == torch.float32
        assert not cache.radius.requires_grad and not cache.direction.requires_grad
        expected = encode_token_rows(model, reference_arm_code_rows)
        assert torch.equal(cache.radius, expected['radius'].float())
        assert torch.equal(cache.tangent, expected['tangent'])

    def test_a_restored_model_refreshes_to_the_same_cache(
        self,
        tmp_path,
        reference_model,
        reference_bundle,
        reference_arm_code_rows,
        epoch_steps,
    ):
        '''Spec 6, Cache: the cache is never saved, and the restored weights rebuild it.'''

        model = reference_model()
        model.log = Mock()
        model.refresh_code_cache(reference_arm_code_rows)
        before = model.code_cache.radius.clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        model.training_step(epoch_steps[0], batch_idx=0).backward()
        optimizer.step()
        model.refresh_code_cache(reference_arm_code_rows)
        assert not torch.equal(model.code_cache.radius, before)
        path = tmp_path / 'arm.ckpt'
        torch.save(lightning_checkpoint(model), path)

        restored = NAICSContrastiveModel.load_from_checkpoint(
            path,
            map_location='cpu',
            supervision_manifest_path=str(reference_bundle.manifest_path),
            supervision_bundle=reference_bundle,
        )
        assert restored.code_cache is None
        restored.refresh_code_cache(reference_arm_code_rows)

        for name in ('radius', 'direction', 'tangent'):
            assert torch.equal(getattr(restored.code_cache, name), getattr(model.code_cache, name))

# -------------------------------------------------------------------------------------------------
# Test: Optimizer Configuration
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestOptimizerConfiguration:
    '''The optimizer, the warmup, the plateau and the logit-scale clamp (P16, P30).'''

    def test_configure_optimizers_basic(self, reference_arm_model):
        config = reference_arm_model.configure_optimizers()

        assert set(config) == {'optimizer', 'lr_scheduler'}

    def test_adamw_optimizer(self, reference_arm_model):
        '''Two groups: the logit scales take no weight decay (spec 4.4).'''

        model = reference_arm_model
        optimizer = model.configure_optimizers()['optimizer']

        assert isinstance(optimizer, torch.optim.AdamW)
        first, second = optimizer.param_groups
        scales = [model.logit_scale_task.log_scale, model.logit_scale_code.log_scale]
        trainable = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad and all(parameter is not scale for scale in scales)
        ]
        assert [id(parameter) for parameter in second['params']] == [id(scale) for scale in scales]
        assert [id(parameter) for parameter in first['params']] == [id(p) for p in trainable]
        assert (first['weight_decay'], second['weight_decay']) == (0.01, 0.0)
        assert first['lr'] == second['lr'] == model.hparams['learning_rate'] == 1e-4

    def test_the_plateau_is_registered_non_strict_on_the_outcome_mrr(self, reference_model):
        '''P16, P30: mode max on the monitor's MRR, so Lightning checkpoints its state.'''

        model = reference_model(lr_plateau_factor=0.25, lr_plateau_patience=3)
        config = model.configure_optimizers()

        scheduler = dict(config['lr_scheduler'])
        plateau = scheduler.pop('scheduler')
        assert isinstance(plateau, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert plateau.optimizer is config['optimizer']
        assert (plateau.mode, plateau.factor, plateau.patience) == ('max', 0.25, 3)
        assert plateau.threshold == 0.0
        assert scheduler == {'monitor': OUTCOME_MRR, 'interval': 'epoch', 'strict': False}

    def test_lightning_never_steps_the_plateau(self, reference_arm_model):
        '''P16: on_train_epoch_end steps it by hand, before ModelCheckpoint saves.'''

        model = reference_arm_model
        plateau = model.configure_optimizers()['lr_scheduler']['scheduler']
        before = plateau.state_dict()

        model.lr_scheduler_step(plateau, torch.tensor(0.5, dtype=torch.float64))

        assert plateau.state_dict() == before

    def test_the_warmup_ramps_the_rate_then_leaves_it_to_the_plateau(
        self, reference_model, reference_arm_code_rows
    ):
        '''P16: lr = base (t + 1) / (W S) for the first W epochs of S steps, then untouched.'''

        base = 1e-3
        model = reference_model(learning_rate=base, warmup_epochs=2)
        stub = _attach_stub_trainer(model, reference_arm_code_rows, num_training_batches=3)
        optimizer = stub.optimizer
        closure = _scale_closure(model, optimizer, lambda: model.logit_scale_task().square())
        rates = []
        for step in range(8):
            stub.trainer.global_step = step
            model.optimizer_step(step // 3, step % 3, optimizer, closure)
            rates.append([group['lr'] for group in optimizer.param_groups])

        expected = [base * (step + 1) / 6 for step in range(6)] + [base, base]
        assert rates == [[pytest.approx(rate, rel=1e-12)] * 2 for rate in expected]
        # The plateau halves the rate after three epochs without a better MRR (patience 2)
        for mrr in (0.5, 0.4, 0.4, 0.4):
            stub.plateau.step(mrr)
        halved = pytest.approx([base / 2] * 2, rel=1e-12)
        assert [group['lr'] for group in optimizer.param_groups] == halved
        stub.trainer.global_step = 8
        model.optimizer_step(2, 2, optimizer, closure)
        assert [group['lr'] for group in optimizer.param_groups] == halved

    def test_no_warmup_at_zero_warmup_epochs(self, reference_model, reference_arm_code_rows):
        model = reference_model(learning_rate=1e-3, warmup_epochs=0)
        stub = _attach_stub_trainer(model, reference_arm_code_rows)
        closure = _scale_closure(model, stub.optimizer, lambda: model.logit_scale_task().square())

        model.optimizer_step(0, 0, stub.optimizer, closure)

        assert [group['lr'] for group in stub.optimizer.param_groups] == [1e-3, 1e-3]

    def test_a_scale_carried_past_its_bound_returns_to_it_and_can_come_back(
        self, reference_model, reference_arm_code_rows
    ):
        '''P16: the forward clamp passes no gradient beyond the range, so the step's in-place
        clamp brings theta back to the bound, where the gradient reaches it again.'''

        model = reference_model(learning_rate=0.01, warmup_epochs=0)
        stub = _attach_stub_trainer(model, reference_arm_code_rows)
        task, code = model.logit_scale_task, model.logit_scale_code
        high, low = math.log(task.high), math.log(code.low)
        with torch.no_grad():
            task.log_scale.fill_(high + 1.0)
            code.log_scale.fill_(low - 1.0)
        held = _scale_closure(model, stub.optimizer, lambda: task() + code())

        model.optimizer_step(0, 0, stub.optimizer, held)

        assert task.log_scale.grad == 0 and code.log_scale.grad == 0
        assert task.log_scale.item() == _float32(high)
        assert code.log_scale.item() == _float32(low)
        # At the bound the clamp passes gradient again, so a step can move theta back inside
        inward = _scale_closure(model, stub.optimizer, lambda: task() - code())
        stub.trainer.global_step = 1
        model.optimizer_step(0, 1, stub.optimizer, inward)
        assert task.log_scale.item() < _float32(high)
        assert code.log_scale.item() > _float32(low)

    def test_a_step_that_carries_a_scale_out_of_its_range_ends_on_the_bound(
        self, reference_model, reference_arm_code_rows
    ):
        model = reference_model(learning_rate=1.0, warmup_epochs=0)
        stub = _attach_stub_trainer(model, reference_arm_code_rows)
        task = model.logit_scale_task
        high = math.log(task.high)
        with torch.no_grad():
            task.log_scale.fill_(high - 0.1)
        outward = _scale_closure(model, stub.optimizer, lambda: -task())

        model.optimizer_step(0, 0, stub.optimizer, outward)

        assert task.log_scale.item() == _float32(high)

# -------------------------------------------------------------------------------------------------
# Test: The training hooks (P15, P18)
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestTrainingHooks:
    '''The training run, the monitor's reads, the plateau and the checkpoint hooks.'''

    def test_on_save_checkpoint_writes_the_contract_and_the_training_run(self, reference_arm_model):
        model = reference_arm_model
        model.training_run = 'run-a'
        checkpoint = {}

        model.on_save_checkpoint(checkpoint)

        assert checkpoint == {
            'stage3_supervision': model.checkpoint_contract.model_dump(),
            'training_run': 'run-a',
        }

    def test_a_fresh_fit_mints_a_training_run_starts_the_monitor_and_refreshes(
        self, reference_model, reference_arm_code_rows, code_targets
    ):
        monitor = ScriptedMonitor()
        model = reference_model(monitor=monitor)
        _attach_stub_trainer(model, reference_arm_code_rows)

        model.on_train_start()

        assert re.fullmatch('[0-9a-f]{32}', model.training_run)
        assert monitor.events == [('start', None)]
        assert model.code_cache is not None and model.code_cache.codes == code_targets.codes

    def test_a_resumed_fit_keeps_its_training_run_and_resumes_the_monitor(
        self, reference_model, reference_arm_code_rows
    ):
        '''on_load_checkpoint only stashes (it also runs under load_from_checkpoint); the fit
        start resumes the monitor from the restored epoch.'''

        monitor = ScriptedMonitor()
        model = reference_model(monitor=monitor)
        contract = model.checkpoint_contract.model_dump()

        model.on_load_checkpoint(
            {
                'stage3_supervision': contract,
                'training_run': 'run-a',
                'epoch': 3
            }
        )

        assert monitor.events == []
        assert model.code_cache is None
        _attach_stub_trainer(model, reference_arm_code_rows, epoch=4)
        model.on_train_start()
        assert model.training_run == 'run-a'
        assert monitor.events == [('start', 3)]
        assert model.code_cache is not None

    def test_a_refused_contract_stashes_nothing(self, reference_model, reference_arm_code_rows):
        monitor = ScriptedMonitor()
        model = reference_model(monitor=monitor)
        other = {**model.checkpoint_contract.model_dump(), 'bundle_id': 'other-bundle'}

        with pytest.raises(ValueError, match='exact resume'):
            model.on_load_checkpoint(
                {
                    'stage3_supervision': other,
                    'training_run': 'run-b',
                    'epoch': 5
                }
            )

        _attach_stub_trainer(model, reference_arm_code_rows)
        model.on_train_start()
        assert model.training_run != 'run-b'
        assert monitor.events == [('start', None)]

    def test_the_epoch_end_refreshes_reads_logs_steps_appends_then_logs_health(
        self, reference_model, reference_arm_code_rows, monkeypatch
    ):
        '''P15, P18: one float64 MRR goes to the log and the plateau, before the record is
        appended; the module's hook runs before ModelCheckpoint's, so the epoch's checkpoint
        sees both.'''

        events: List[Any] = []
        monitor = ScriptedMonitor([0.25], events)
        model = reference_model(monitor=monitor, seed=11)
        stub = _attach_stub_trainer(model, reference_arm_code_rows, epoch=2)
        model.on_train_start()
        first_cache = model.code_cache
        events.clear()
        refresh = model.refresh_code_cache

        def refresh_spy(code_rows):
            events.append('refresh')
            return refresh(code_rows)

        monkeypatch.setattr(model, 'refresh_code_cache', refresh_spy)
        stub.log.side_effect = lambda name, *args, **kwargs: events.append(('log', name))
        plateau_metrics = []
        step = stub.plateau.step

        def plateau_spy(metrics):
            events.append('plateau')
            plateau_metrics.append(metrics)
            return step(metrics)

        stub.plateau.step = plateau_spy

        model.on_train_epoch_end()

        assert events[:5] == [
            'refresh',
            ('read', 2),
            ('log', OUTCOME_MRR),
            'plateau',
            ('append', 2),
        ]
        health = events[5:]
        assert health and all(event[0] == 'log' and event[1] != OUTCOME_MRR for event in health)
        (read, ) = monitor.reads
        assert read['cache'] is model.code_cache and read['cache'] is not first_cache
        assert read['model'] is model
        assert (read['training_run'], read['seed'], read['epoch']) == (model.training_run, 11, 2)
        call = _logged(stub.log)[OUTCOME_MRR]
        mrr = call.args[1]
        assert plateau_metrics == [mrr] and plateau_metrics[0] is mrr
        assert mrr.dtype == torch.float64 and mrr.device.type == 'cpu' and mrr.item() == 0.25
        assert call.kwargs['batch_size'] == 1
        assert stub.plateau.best == 0.25

    def test_without_a_monitor_the_epoch_end_refreshes_and_logs_health_only(
        self, reference_arm_model, reference_arm_code_rows
    ):
        model = reference_arm_model
        stub = _attach_stub_trainer(model, reference_arm_code_rows)
        model.on_train_start()
        first_cache = model.code_cache
        plateau = stub.plateau.state_dict()

        model.on_train_epoch_end()

        assert model.code_cache is not first_cache
        assert OUTCOME_MRR not in _logged(stub.log)
        assert _logged(stub.log)
        assert stub.plateau.state_dict() == plateau

    def test_an_epoch_end_with_the_outcome_monitor_logs_steps_and_records_one_mrr(
        self, tmp_path, reference_model, reference_bundle, reference_arm_code_rows, minilm_tokenizer
    ):
        '''P18 end to end on the real monitor: the MRR logged, the plateau's and the record's are
        one value. The selection log is under tmp_path (P28).'''

        panel = OutcomePanel.from_bundle(
            reference_bundle, tmp_path / 'logs' / 'selection_log.jsonl'
        )
        records = tmp_path / 'checkpoints' / MONITOR_RECORDS
        monitor = OutcomeMonitor(
            panel, minilm_tokenizer, REFERENCE_WINDOW, records, 'model hook unit test'
        )
        model = reference_model(monitor=monitor, seed=3)
        stub = _attach_stub_trainer(model, reference_arm_code_rows)
        plateau_metrics = []
        step = stub.plateau.step
        stub.plateau.step = lambda metrics: (plateau_metrics.append(metrics), step(metrics))[1]
        model.on_train_start()

        model.on_train_epoch_end()

        (record, ) = read_monitor_records(records)
        mrr = _logged(stub.log)[OUTCOME_MRR].args[1]
        assert plateau_metrics[0] is mrr
        assert mrr.dtype == torch.float64 and mrr.item() == record['mrr']
        detail = record['read']['detail']
        cache = model.code_cache
        assert {
            name: detail[name]
            for name in ('training_run', 'seed', 'epoch', 'table')
        } == {
            'training_run': model.training_run,
            'seed': 3,
            'epoch': 0,
            'table': matrix_fingerprint(cache.codes, cache.tangent.numpy()),
        }

# -------------------------------------------------------------------------------------------------
# Test: The health logs (P20)
# -------------------------------------------------------------------------------------------------

def _run_epoch(model: NAICSContrastiveModel, steps: Sequence[Dict[str, Any]], monkeypatch) -> List:
    '''Run ``steps`` through ``training_step``, returning each step's ``StepLosses``.'''

    computed = []
    compute = model.compute_losses

    def spy(batch):
        computed.append(compute(batch))
        return computed[-1]

    monkeypatch.setattr(model, 'compute_losses', spy)
    for index, step in enumerate(steps):
        model.training_step(step, batch_idx=index)
    monkeypatch.setattr(model, 'compute_losses', compute)
    return computed

@pytest.mark.unit
class TestHealthLogs:
    '''Each epoch logs its terms' means, the two scales and r per level; nothing selects on them.'''

    def test_the_health_logs_are_epoch_means_the_scales_and_the_radii_per_level(
        self, reference_arm_model, reference_arm_code_rows, epoch_steps, code_targets, monkeypatch
    ):
        model = reference_arm_model
        stub = _attach_stub_trainer(model, reference_arm_code_rows)
        model.on_train_start()
        computed = _run_epoch(model, epoch_steps, monkeypatch)
        stub.log.reset_mock()

        model.on_train_epoch_end()

        logged = _logged(stub.log)
        levels = range(2, 7)
        assert set(logged) == {
            'loss/task',
            'loss/code_code',
            'loss/radial',
            'loss/total',
            'logit_scale/task',
            'logit_scale/code_code',
            *(
                f'radius/{statistic}/level_{level}' for statistic in ('mean', 'sd')
                for level in levels
            ),
        }
        for call in logged.values():
            assert call.kwargs['batch_size'] == 1
            assert (call.kwargs['on_step'], call.kwargs['on_epoch']) == (False, True)
        # Each term's epoch mean is the plain mean over the steps, whatever their sizes
        for name in ('task', 'code_code', 'radial', 'total'):
            mean = statistics.fmean(getattr(losses, name).item() for losses in computed)
            assert logged[f'loss/{name}'].args[1] == pytest.approx(mean, rel=1e-12), name
        assert logged['logit_scale/task'].args[1] == pytest.approx(model.logit_scale_task().item())
        assert logged['logit_scale/code_code'].args[1] == pytest.approx(
            model.logit_scale_code().item()
        )
        radius = model.code_cache.radius.to(torch.float64)
        code_levels = torch.from_numpy(code_targets.levels)
        for level in levels:
            at_level = radius[code_levels == level]
            mean = logged[f'radius/mean/level_{level}'].args[1]
            sd = logged[f'radius/sd/level_{level}'].args[1]
            assert mean == pytest.approx(at_level.mean().item(), rel=1e-12)
            assert sd == pytest.approx(at_level.std(correction=0).item(), rel=1e-9, abs=1e-15)

    def test_a_new_epoch_starts_new_means(
        self, reference_arm_model, reference_arm_code_rows, epoch_steps, monkeypatch
    ):
        model = reference_arm_model
        stub = _attach_stub_trainer(model, reference_arm_code_rows)
        model.on_train_start()
        _run_epoch(model, epoch_steps, monkeypatch)
        model.on_train_epoch_end()
        stub.log.reset_mock()

        (losses, ) = _run_epoch(model, epoch_steps[:1], monkeypatch)
        model.on_train_epoch_end()

        logged = _logged(stub.log)
        assert logged['loss/task'].args[1] == pytest.approx(losses.task.item(), rel=1e-12)
        assert logged['loss/total'].args[1] == pytest.approx(losses.total.item(), rel=1e-12)

    def test_moe_adds_the_load_balancing_mean(
        self, reference_model, reference_arm_code_rows, epoch_steps, monkeypatch
    ):
        model = reference_model(fusion='moe', moe_hidden_dim=16)
        stub = _attach_stub_trainer(model, reference_arm_code_rows)
        model.on_train_start()
        computed = _run_epoch(model, epoch_steps[:2], monkeypatch)
        stub.log.reset_mock()

        model.on_train_epoch_end()

        mean = statistics.fmean(losses.load_balancing.item() for losses in computed)
        logged = _logged(stub.log)
        assert logged['loss/load_balancing'].args[1] == pytest.approx(mean, rel=1e-12)

# -------------------------------------------------------------------------------------------------
# Test: Checkpoint Loading
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckpointLoading:
    '''Test model checkpoint saving and loading.'''

    def test_state_dict_save_load(self, naics_model, test_device):
        '''Test that model state dict can be saved and loaded.'''

        # Save state dict
        state_dict = naics_model.state_dict()

        # Create new model
        new_model = NAICSContrastiveModel(**naics_model.hparams).to(test_device)

        # Load state dict
        new_model.load_state_dict(state_dict)

        # Check that parameters match
        for (name1, param1), (name2, param2) in zip(
            naics_model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            torch.testing.assert_close(param1, param2)

    def test_hparams_save_load(self, naics_model, tmp_path):
        '''A checkpoint's hyperparameters rebuild the same settings.'''

        path = tmp_path / 'arm.ckpt'
        torch.save(lightning_checkpoint(naics_model), path)

        restored = NAICSContrastiveModel.load_from_checkpoint(path, map_location='cpu')

        assert dict(restored.hparams) == dict(naics_model.hparams)
        assert restored.hparams['logit_scale_range'] == (0.01, 100.0)

# -------------------------------------------------------------------------------------------------
# Test: Supervision checkpoint contract
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckpointContract:
    '''The supervision contract travels with checkpoints and gates every restore.'''

    def test_model_contract_matches_its_bundle(self, naics_model, validated_bundle):
        contract = naics_model.checkpoint_contract

        assert contract.objective == 'req11-v1'
        assert contract.bundle_id == validated_bundle.manifest.bundle_id
        assert contract.codebook_fingerprint == validated_bundle.manifest.codebook_fingerprint
        assert contract.encoder == shared_encoder_architecture(
            fusion='masked_mean', dimension=16, backbone='sentence-transformers/all-MiniLM-L6-v2'
        )

    def test_a_runtime_contract_of_another_encoder_is_refused(self, model_config, validated_bundle):
        other = contract_for_bundle(
            validated_bundle.manifest,
            encoder=shared_encoder_architecture(
                fusion='masked_mean', dimension=8, backbone=model_config['base_model_name']
            ),
            summaries=None,
        )

        with pytest.raises(ValueError, match='does not match'):
            NAICSContrastiveModel(**model_config, checkpoint_contract=other)

    def test_the_model_records_its_summaries_in_its_contract(self, model_config):
        model = NAICSContrastiveModel(**model_config, summaries='e' * 64)

        assert model.checkpoint_contract.summaries == 'e' * 64
        # Saved with the hyperparameters, so load_from_checkpoint rebuilds the same contract
        assert model.hparams['summaries'] == 'e' * 64

    def test_on_save_checkpoint_writes_contract(self, naics_model):
        checkpoint = {}
        naics_model.on_save_checkpoint(checkpoint)

        assert checkpoint['stage3_supervision'] == naics_model.checkpoint_contract.model_dump()

    def test_a_saved_checkpoint_records_the_req11_objective(self, shared_model):
        '''Spec 4.5: a new checkpoint names its objective, and no field of the old one's.'''

        saved = lightning_checkpoint(shared_model)['stage3_supervision']

        assert saved['objective'] == 'req11-v1'
        assert set(saved) == {
            'contract_version',
            'bundle_id',
            'codebook_fingerprint',
            'objective',
            'encoder',
            'summaries',
        }

    def test_on_load_checkpoint_rejects_legacy_and_mismatched_contracts(self, naics_model):
        contract = naics_model.checkpoint_contract.model_dump()

        with pytest.raises(ValueError, match='exact resume'):
            naics_model.on_load_checkpoint({})
        with pytest.raises(ValueError, match='exact resume'):
            naics_model.on_load_checkpoint(
                {'stage3_supervision': {
                    **contract, 'bundle_id': 'other-bundle'
                }}
            )
        naics_model.on_load_checkpoint({'stage3_supervision': contract})

    def test_runtime_contract_must_match_the_loaded_bundle(self, model_config):
        from naics_embedder.supervision.checkpoints import CheckpointContract

        other = CheckpointContract(
            objective='req11-v1',
            bundle_id='other-bundle',
            codebook_fingerprint='f' * 64,
        )
        with pytest.raises(ValueError, match='bundle'):
            NAICSContrastiveModel(**model_config, checkpoint_contract=other)

    def test_prevalidated_bundle_is_reused_and_kept_out_of_hparams(
        self, model_config, validated_bundle, monkeypatch
    ):
        monkeypatch.setattr(
            model_module,
            'load_validated_bundle',
            Mock(side_effect=AssertionError('bundle validated twice')),
        )
        model = NAICSContrastiveModel(**model_config, supervision_bundle=validated_bundle)

        assert model.supervision_bundle_id == validated_bundle.manifest.bundle_id
        assert 'supervision_bundle' not in model.hparams
        assert 'checkpoint_contract' not in model.hparams

    def test_prevalidated_bundle_must_be_the_configured_manifest(
        self, model_config, validated_bundle, tmp_path
    ):
        model_config['supervision_manifest_path'] = str(tmp_path / 'other' / 'manifest.json')

        with pytest.raises(ValueError, match='manifest'):
            NAICSContrastiveModel(**model_config, supervision_bundle=validated_bundle)

    def test_load_from_checkpoint_round_trips_the_contract(self, naics_model, tmp_path):
        path = tmp_path / 'repaired.ckpt'
        torch.save(lightning_checkpoint(naics_model), path)

        restored = NAICSContrastiveModel.load_from_checkpoint(path, map_location='cpu')

        assert restored.checkpoint_contract == naics_model.checkpoint_contract
        assert restored.checkpoint_contract.encoder.layout == 'shared'
        assert restored.encoder.dimension == 16

    def test_load_from_checkpoint_rejects_a_legacy_checkpoint(self, naics_model, tmp_path):
        checkpoint = lightning_checkpoint(naics_model)
        del checkpoint['stage3_supervision']
        path = tmp_path / 'legacy.ckpt'
        torch.save(checkpoint, path)

        with pytest.raises(ValueError, match='exact resume'):
            NAICSContrastiveModel.load_from_checkpoint(path, map_location='cpu')

    def test_load_from_checkpoint_refuses_a_four_copy_checkpoint_before_its_weights(
        self, naics_model, tmp_path
    ):
        '''Spec 4.4: a pre-Stage-6 checkpoint meets the D2 refusal, never a state-dict key error.'''

        checkpoint = lightning_checkpoint(naics_model)
        # Contracts saved before Stage 6 carry no encoder record, and their hyperparameters
        # predate fusion and dimension
        del checkpoint['stage3_supervision']['encoder']
        for name in ('fusion', 'dimension'):
            del checkpoint['hyper_parameters'][name]
        # The four-copy layout's keys, which a strict load_state_dict would reject
        checkpoint['state_dict'] = {
            'encoder.encoders.title.base_model.model.embeddings.word_embeddings.weight': torch
            .zeros(1)
        }
        path = tmp_path / 'four-copy.ckpt'
        torch.save(checkpoint, path)

        with pytest.raises(ValueError, match='D2'):
            NAICSContrastiveModel.load_from_checkpoint(path, map_location='cpu')

    def test_load_from_checkpoint_refuses_another_dimension(self, naics_model, tmp_path):
        path = tmp_path / 'shared.ckpt'
        torch.save(lightning_checkpoint(naics_model), path)

        with pytest.raises(ValueError, match='D2'):
            NAICSContrastiveModel.load_from_checkpoint(path, map_location='cpu', dimension=8)

    def test_load_from_checkpoint_refuses_a_pre_stage_7_checkpoint(self, pre_stage7_checkpoint):
        '''
        Spec 4.5: the model's own hook refuses it on its objective before its state dict loads,
        for any caller of ``load_from_checkpoint``, and nothing migrates it (D2).
        '''

        with pytest.raises(ValueError, match=PRE_STAGE_7_REFUSAL):
            NAICSContrastiveModel.load_from_checkpoint(pre_stage7_checkpoint, map_location='cpu')

# -------------------------------------------------------------------------------------------------
# Test: Legacy containment is deleted (roadmap D2)
# -------------------------------------------------------------------------------------------------

def test_a_saved_containment_checkpoint_never_restores(naics_model):
    '''
    A checkpoint saved under legacy containment, before D2 deleted it, is refused: like every
    checkpoint saved before Stage 7, on its objective (spec 4.5).
    '''

    containment = {
        **pre_stage_7_contract(naics_model.checkpoint_contract.model_dump()),
        'supervision_mode': 'legacy_containment',
        'bundle_id': 'legacy-containment',
        'codebook_fingerprint': 'unversioned',
    }

    with pytest.raises(ValueError, match=PRE_STAGE_7_REFUSAL):
        naics_model.on_load_checkpoint({'stage3_supervision': containment})
