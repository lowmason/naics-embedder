# -------------------------------------------------------------------------------------------------
# NAICS Contrastive Learning Model
# -------------------------------------------------------------------------------------------------
'''
The text stage's Lightning module: the shared encoder, trained on Req 11's three terms over a
live code cache and selected on the outcome panel's validation MRR (spec 4.1-4.4; D6).

- SharedEncoder: one LoRA-tuned backbone over field-marked channels, masked fusion, one affine map
  to dimension d and the geometry head (Req 12), which gives each text its point, its radius r and
  its direction û.
- A step reads two streams (spec 4.3): a chunk of the codes, as anchors, and a chunk of the task
  queries. Every candidate comes from the code cache, each code's (r, û) at its last refresh, with
  the step's anchors replaced by their live points.
- Req 11's terms (spec 4.1): the task term over each query's candidates and the code-code
  listwise term over each anchor's J_a, both under the arm's own distance, and, in the hyperbolic
  arm only, the radial term (Req 12); under ``moe`` only, the experts' load balancing.
- The cache is refreshed at fit start and at each epoch's end. The end-of-epoch refresh feeds the
  outcome monitor's read, whose MRR is logged as ``val/outcome_mrr`` and steps the plateau.

The model is decomposed into functional mixins:
- LossMixin: the experts' load-balancing term (``moe`` only)
- LoggingMixin: the epoch's health logs (P20)
- OptimizerMixin: AdamW, the warmup, the plateau and the logit scales' clamp (P16)
'''

import logging
import math
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import pytorch_lightning as pyl
import torch

from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, load_validated_bundle
from naics_embedder.supervision.checkpoints import (
    CHECKPOINT_KEY,
    CheckpointContract,
    contract_for_bundle,
    shared_encoder_architecture,
    validate_checkpoint_contract,
)
from naics_embedder.supervision.code_targets import NO_PARTNER, CodeTargets
from naics_embedder.supervision.schema import CONTRACT_VERSION
from naics_embedder.text_model.epoch_summary import EPOCH_SUMMARY, EpochSummary
from naics_embedder.text_model.fusion import FUSIONS
from naics_embedder.text_model.heads import GEOMETRIES
from naics_embedder.text_model.loss import LogitScale, code_code_loss, radial_loss, task_loss
from naics_embedder.text_model.mixins import OUTCOME_MRR, LoggingMixin, LossMixin, OptimizerMixin
from naics_embedder.text_model.shared_encoder import DIMENSIONS, SharedEncoder

if TYPE_CHECKING:
    # The monitor imports the export, which imports this module, so the methods that use the
    # monitor import it when they run
    from naics_embedder.text_model.monitor import CodeCache, OutcomeMonitor

__all__ = ['NAICSContrastiveModel', 'StepLosses', 'TRAINING_RUN_KEY']

logger = logging.getLogger(__name__)

# The checkpoint key of the training run's id, which every monitor read names (spec 4.4)
TRAINING_RUN_KEY = 'training_run'

# -------------------------------------------------------------------------------------------------
# A step's losses
# -------------------------------------------------------------------------------------------------

class StepLosses(NamedTuple):
    '''
    One step's losses (spec 4.1).

    Attributes:
        total: What the step optimizes: L = L_task + w_c · L_cc, plus w_r · L_rad in the
            hyperbolic arm, plus the load-balancing term times its coefficient under ``moe``.
        task: L_task, the task term.
        code_code: L_cc, the code-code listwise term.
        radial: L_rad, the radial term; None unless the arm is hyperbolic (Req 12).
        load_balancing: The experts' load-balancing term, before its coefficient; None unless the
            fusion is ``moe``.
        anchor_radius: The anchors' live radii r_a, (A,): in the hyperbolic arm, every term's
            gradient reaches the radius through them (Verification "Radius").
    '''

    total: torch.Tensor
    task: torch.Tensor
    code_code: torch.Tensor
    radial: Optional[torch.Tensor]
    load_balancing: Optional[torch.Tensor]
    anchor_radius: torch.Tensor

def _refuse_settings(
    *,
    code_code_weight: float,
    radial_weight: float,
    target_temperature: float,
    radial_step: float,
    warmup_epochs: int,
    lr_plateau_factor: float,
    lr_plateau_patience: int,
) -> None:
    '''
    Refuse a setting the config's validators refuse (spec 5, P22), before anything loads.

    The logit scales refuse their own range and start (``LogitScale``).

    Raises:
        ValueError: If a weight is below 0, the target temperature or radial step is not
            positive, a value is not finite, an epoch count is not an integer at or above 0, or
            the plateau's factor is not in (0, 1).
    '''

    for name, value in (('code_code_weight', code_code_weight), ('radial_weight', radial_weight)):
        if not (math.isfinite(value) and value >= 0):
            raise ValueError(f'{name} must be a finite number at or above 0, not {value!r}')
    for name, value in (('target_temperature', target_temperature), ('radial_step', radial_step)):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f'{name} must be a positive finite number, not {value!r}')
    for name, count in (
        ('warmup_epochs', warmup_epochs), ('lr_plateau_patience', lr_plateau_patience)
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f'{name} must be an integer at or above 0, not {count!r}')
    if not 0 < lr_plateau_factor < 1:
        raise ValueError(f'lr_plateau_factor must lie in (0, 1), not {lr_plateau_factor!r}')

# -------------------------------------------------------------------------------------------------
# Main NAICS Contrastive Learning Model
# -------------------------------------------------------------------------------------------------

class NAICSContrastiveModel(LossMixin, LoggingMixin, OptimizerMixin, pyl.LightningModule):
    '''
    The text stage's model: the shared encoder, trained on Req 11's three terms over a live code
    cache and selected on the outcome panel's validation MRR (spec 4.1-4.4; D6).

    Supervision comes only from one validated supervision bundle: its codebook, its tree metric D*
    and its unary pairs (``CodeTargets``), held as buffers in codebook order and never saved.

    Args:
        base_model_name: HuggingFace model name for the backbone
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
        fusion: Channel fusion: ``masked_mean`` (default), ``attention`` or ``moe``. The
            load-balancing term exists only under ``moe`` (R11)
        dimension: Embedding dimension, one of 8, 16 or 32: the width of the one
            ``Linear(hidden → d)`` before the geometry head
        geometry: The geometry arm (Req 12): ``euclidean``, ``spherical`` or ``hyperbolic``
            (the default), which picks the head; the radial term applies under hyperbolic only
        num_experts: Number of MoE experts (``moe`` only)
        top_k: Number of experts each row is routed to (``moe`` only)
        moe_hidden_dim: Hidden dimension of the experts (``moe`` only)
        radius_bound: R, the hyperbolic head's bound on every radius: r = R · tanh(‖v‖ / R)
            (spec 4.2); the flat heads do not read it
        code_code_weight: w_c, the code-code term's weight in the total (spec 4.1)
        radial_weight: w_r, the radial term's weight in the total
        target_temperature: τ_t, the temperature of the code-code target softmax(−D* / τ_t)
        radial_step: ρ, the radius from one level to the next: the radial target is ρ · (λ − 1)
        logit_scale_init: Where both logit scales s = exp(θ) start
        logit_scale_range: (low, high), the range both logit scales are clamped to
        learning_rate: AdamW's base learning rate
        weight_decay: AdamW's weight decay, on every parameter but the logit scales
        warmup_epochs: W, the epochs of the linear warmup; 0 for none
        lr_plateau_factor: The factor the plateau cuts the learning rate by
        lr_plateau_patience: The epochs without a better ``val/outcome_mrr`` before a cut
        load_balancing_coef: The load-balancing term's coefficient (``moe`` only)
        seed: The run's seed, which every monitor read names
        run_settings: The run's free settings (``utils/training.run_settings``), saved so an
            exact resume under other settings can be refused; None outside ``train``
        supervision_manifest_path: Manifest of the validated supervision bundle (required: it is
            the one authority for the codes, D* and the unary pairs)
        supervision_contract_version: Expected supervision contract version
        summaries: The sha256 of the window-fitting summaries the token cache applied, or None
            for a backbone with no pin; recorded in the checkpoint contract
        checkpoint_contract: Optional runtime contract; must match the loaded bundle
        supervision_bundle: Optional already-validated bundle for ``supervision_manifest_path``
        monitor: The run's ``OutcomeMonitor``, which reads the validation split at each epoch's
            end; with None, no epoch reads it, logs ``val/outcome_mrr`` or steps the plateau

    ``checkpoint_contract``, ``supervision_bundle`` and ``monitor`` are not saved in the
    hyperparameters.

    Raises:
        ValueError: If the fusion, dimension or geometry is unknown, a setting is out of its range,
            the manifest is missing, a pre-validated bundle is not the configured one, or the
            runtime contract is not the bundle's.
    '''

    def __init__(
        self,
        base_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        fusion: str = 'masked_mean',
        dimension: int = 16,
        geometry: str = 'hyperbolic',
        num_experts: int = 4,
        top_k: int = 2,
        moe_hidden_dim: int = 1024,
        radius_bound: float = 8.0,
        code_code_weight: float = 1.0,
        radial_weight: float = 1.0,
        target_temperature: float = 1.0,
        radial_step: float = 1.0,
        logit_scale_init: float = 1.0,
        logit_scale_range: Tuple[float, float] = (0.01, 100.0),
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 1,
        lr_plateau_factor: float = 0.5,
        lr_plateau_patience: int = 2,
        load_balancing_coef: float = 0.01,
        seed: int = 0,
        run_settings: Optional[Dict[str, Any]] = None,
        supervision_manifest_path: Optional[str] = None,
        supervision_contract_version: str = CONTRACT_VERSION,
        summaries: Optional[str] = None,
        checkpoint_contract: Optional[CheckpointContract] = None,
        supervision_bundle: Optional[ValidatedSupervisionBundle] = None,
        monitor: Optional['OutcomeMonitor'] = None,
    ):
        super().__init__()

        if fusion not in FUSIONS:
            raise ValueError(f'unknown fusion {fusion!r}; expected one of {list(FUSIONS)}')
        if dimension not in DIMENSIONS:
            raise ValueError(f'unknown dimension {dimension!r}; expected one of {list(DIMENSIONS)}')
        if geometry not in GEOMETRIES:
            raise ValueError(f'unknown geometry {geometry!r}; expected one of {list(GEOMETRIES)}')
        _refuse_settings(
            code_code_weight=code_code_weight,
            radial_weight=radial_weight,
            target_temperature=target_temperature,
            radial_step=radial_step,
            warmup_epochs=warmup_epochs,
            lr_plateau_factor=lr_plateau_factor,
            lr_plateau_patience=lr_plateau_patience,
        )
        # The one switch for the MoE-only load-balancing term (R11)
        self.fusion = fusion

        # Training is always repaired: legacy containment is deleted (roadmap D2)
        if supervision_manifest_path is None:
            raise ValueError(
                'repaired supervision requires supervision_manifest_path; generate a bundle '
                'with `naics-embedder data supervision` and set the printed manifest path'
            )

        # Bundle, contract and monitor objects stay out of hyperparameters: checkpoints record
        # paths and identifiers, and a restored model re-validates its bundle from the manifest
        # path.
        self.save_hyperparameters(ignore=['checkpoint_contract', 'supervision_bundle', 'monitor'])
        # The architecture this model's weights belong to; a checkpoint of any other is refused
        # (spec 4.4, roadmap D2)
        encoder_record = shared_encoder_architecture(
            fusion=fusion, dimension=dimension, backbone=base_model_name, geometry=geometry
        )

        # Load the validated supervision bundle before any model construction: the single
        # authority for code identity and the structural facts. A caller that already validated
        # it may pass it in.
        if supervision_bundle is None:
            bundle = load_validated_bundle(
                supervision_manifest_path,
                expected_contract=supervision_contract_version,
            )
        else:
            if supervision_bundle.manifest_path != Path(supervision_manifest_path).resolve():
                raise ValueError(
                    f'pre-validated supervision bundle manifest '
                    f'{supervision_bundle.manifest_path} is not the configured manifest '
                    f'{supervision_manifest_path}'
                )
            if supervision_bundle.manifest.contract_version != supervision_contract_version:
                raise ValueError(
                    f'pre-validated supervision bundle has contract '
                    f'{supervision_bundle.manifest.contract_version}, expected '
                    f'{supervision_contract_version}'
                )
            bundle = supervision_bundle
        runtime_contract = contract_for_bundle(
            bundle.manifest, encoder=encoder_record, summaries=summaries
        )
        if checkpoint_contract is not None and checkpoint_contract != runtime_contract:
            raise ValueError(
                f'runtime checkpoint contract {checkpoint_contract.model_dump()} does not match '
                f'the model supervision contract {runtime_contract.model_dump()}'
            )
        self.checkpoint_contract = runtime_contract
        self.supervision_bundle_id = runtime_contract.bundle_id

        # Each code's level, D* to every code and unary partner, in codebook order (P15). Buffers,
        # so they move with the model; not persistent, since the bundle holds them, not the
        # checkpoint
        targets = CodeTargets.from_bundle(bundle)
        self.codes: Tuple[str, ...] = targets.codes
        self.register_buffer(
            'structural_distance', torch.from_numpy(targets.structural_distance), persistent=False
        )
        self.register_buffer(
            'unary_partner', torch.from_numpy(targets.unary_partner), persistent=False
        )
        self.register_buffer('code_levels', torch.from_numpy(targets.levels), persistent=False)

        # The task term's and the code-code term's learned logit scales (spec 4.1). Each refuses
        # a range that is not 0 < low < high and a start outside it
        low, high = logit_scale_range
        self.logit_scale_task = LogitScale(logit_scale_init, low, high)
        self.logit_scale_code = LogitScale(logit_scale_init, low, high)

        # The shared encoder: one backbone, fusion, one affine map, the head
        self.encoder = SharedEncoder(
            base_model_name=base_model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fusion=fusion,
            dimension=dimension,
            geometry=geometry,
            num_experts=num_experts,
            top_k=top_k,
            moe_hidden_dim=moe_hidden_dim,
            radius_bound=radius_bound,
        )

        self.monitor = monitor
        # Every code's (r, û) at the last refresh: plain state, never saved, so an exact resume
        # rebuilds it from the restored weights (spec 4.3)
        self.code_cache: Optional['CodeCache'] = None
        # The run's id: minted at a fresh fit's start, kept from the checkpoint on exact resume
        self.training_run: Optional[str] = None
        self._resumed_run: Optional[str] = None
        self._resumed_epoch: Optional[int] = None
        self.epoch_summary: Optional[EpochSummary] = None
        self._reset_health()

    def forward(self, channel_inputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        '''
        Forward pass through the shared encoder.

        Args:
            channel_inputs: Per field, tokenized text and a boolean ``present``, as
                ``stack_text_inputs`` builds them: a code batch's four channels, or ``query``

        Returns:
            Dictionary containing:
            - embedding: The arm's points, Lorentz (batch_size, dimension + 1) under hyperbolic
            - tangent: The coordinates the export writes (batch_size, dimension)
            - radius, direction: Each point's r (batch_size,) and û (batch_size, dimension)
            - gate_probs, top_k_indices: The experts' gates, under ``moe`` fusion only
        '''
        return self.encoder(channel_inputs)

    # ---------------------------------------------------------------------------------------------
    # The code cache and a step's losses
    # ---------------------------------------------------------------------------------------------

    def refresh_code_cache(self, code_rows: Sequence[Mapping[str, Any]]) -> 'CodeCache':
        '''
        Re-encode every code into ``code_cache`` (spec 4.3, P14): in eval mode, without gradient,
        in float32 with autocast off, in codebook order. The training flags are put back.

        Args:
            code_rows: Every code's cached token rows, in codebook order.

        Returns:
            The new cache.
        '''

        from naics_embedder.text_model.monitor import refresh_code_cache

        self.code_cache = refresh_code_cache(self, code_rows, self.codes)
        return self.code_cache

    def compute_losses(self, batch: Dict[str, Dict[str, Any]]) -> StepLosses:
        '''
        One step's losses over its two streams (spec 4.1, 4.3).

        The step's anchors and queries go through the encoder, and nothing else does. Every
        candidate is the code cache's point, with the anchors' rows replaced by their live points,
        so gradient reaches the codes through the anchors alone, as anchors and as candidates.
        The distances and the terms run in float32 with autocast off: only the backbone runs in
        reduced precision (spec 4.2). Every distance is the arm's own (``head.pair_distance``),
        and the radial term exists only in the hyperbolic arm (Req 12).

        Args:
            batch: One step, as ``StepDataset`` builds it: ``codes`` (``inputs``, ``ids`` and
                ``levels``) and ``queries`` (``inputs``, ``levels``, ``targets`` and
                ``negatives``).

        Returns:
            The step's losses.

        Raises:
            RuntimeError: If no refresh has built the code cache.
        '''

        cache = self.code_cache
        if cache is None:
            raise RuntimeError(
                'compute_losses reads the code cache, but none was built: refresh_code_cache '
                'builds it, as on_train_start does'
            )
        codes, queries = batch['codes'], batch['queries']
        code_output = self(codes['inputs'])
        query_output = self(queries['inputs'])
        ids = codes['ids']
        settings = self.hparams
        head = self.encoder.head
        with torch.autocast(device_type=ids.device.type, enabled=False):
            anchor_radius = code_output['radius'].float()
            anchor_direction = code_output['direction'].float()
            radius, direction = cache.with_live(ids, anchor_radius, anchor_direction)
            # The task term: each query against the codes at its level and its forced negatives,
            # over all N codes (spec 4.1(i))
            query_distances = head.pair_distance(
                query_output['radius'].float(),
                query_output['direction'].float(),
                radius,
                direction,
            )
            at_level = self.code_levels[None, :] == queries['levels'][:, None]
            task = task_loss(
                query_distances,
                self.logit_scale_task(),
                at_level | queries['negatives'],
                queries['targets'],
            )
            # The code-code term: each anchor against J_a (spec 4.1(ii))
            code_code = code_code_loss(
                head.pair_distance(anchor_radius, anchor_direction, radius, direction),
                self.logit_scale_code(),
                self.structural_distance[ids],
                self._keep(ids),
                settings.target_temperature,
            )
            # The radial term (spec 4.1(iii)), in the hyperbolic arm only (Req 12). It and the
            # total are computed in Stage 7's order, so the hyperbolic arm's values and gradients
            # are Stage 7's bit for bit (P6)
            radial = None
            if head.radial:
                radial = radial_loss(anchor_radius, codes['levels'], settings.radial_step)
            total = task + settings.code_code_weight * code_code
            if radial is not None:
                total = total + settings.radial_weight * radial
            load_balancing = None
            if self.fusion == 'moe':
                # Over both streams' gates (R11)
                gate_probs, top_k_indices = self._collect_gate_outputs([code_output, query_output])
                load_balancing = self._compute_load_balancing_loss(
                    gate_probs, top_k_indices, sum(len(rows) for rows in gate_probs)
                )
                total = total + settings.load_balancing_coef * load_balancing
        return StepLosses(total, task, code_code, radial, load_balancing, anchor_radius)

    def _keep(self, ids: torch.Tensor) -> torch.Tensor:
        '''
        J_a for each anchor, as ``CodeTargets.keep`` gives it: every code but the anchor and, for
        a unary pair, its partner (spec 4.1(ii)). Built from the buffers, on the anchors' device.
        '''

        rows = torch.arange(len(ids), device=ids.device)
        keep = torch.ones((len(ids), len(self.codes)), dtype=torch.bool, device=ids.device)
        keep[rows, ids] = False
        partners = self.unary_partner[ids]
        paired = partners != NO_PARTNER
        keep[rows[paired], partners[paired]] = False
        return keep

    def training_step(self, batch: Dict[str, Dict[str, Any]], batch_idx: int) -> torch.Tensor:
        '''
        One step over the two streams: its losses, kept for the epoch's health logs.

        Args:
            batch: One step, as ``StepDataset`` builds it
            batch_idx: The step's index in the epoch

        Returns:
            The step's total loss.
        '''

        losses = self.compute_losses(batch)
        self._record_health(losses)
        # The progress bar's value; the epoch means are the health logs (P20)
        self.log(
            'loss/step',
            losses.total.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=1,
        )
        return losses.total

    # ---------------------------------------------------------------------------------------------
    # The training run: its start, each epoch's end and its checkpoints
    # ---------------------------------------------------------------------------------------------

    def on_train_start(self) -> None:
        '''
        Start the training run: keep the restored run's id or mint one, ready the monitor's
        records and epoch summary, and build the code cache (spec 4.3, 4.4).

        On exact resume, ``on_load_checkpoint`` has stashed the checkpoint's training run and
        epoch: the run keeps its id, and the monitor keeps its records through that epoch. The
        cache is never saved, so a resume rebuilds it from the restored weights.
        '''

        self.training_run = self._resumed_run or uuid.uuid4().hex
        if self.monitor is not None:
            self.monitor.start(resumed_epoch=self._resumed_epoch)
        callback = getattr(self.trainer, 'checkpoint_callback', None)
        directory = getattr(callback, 'dirpath', None)
        if directory is None and self.monitor is not None:
            records = getattr(self.monitor, 'records_path', None)
            directory = Path(records).parent if records is not None else None
        if directory is not None:
            self.epoch_summary = EpochSummary(Path(directory) / EPOCH_SUMMARY)
            self.epoch_summary.start(resumed_epoch=self._resumed_epoch)
        self._reset_health()
        self.refresh_code_cache(self.trainer.datamodule.code_rows)

    def on_train_epoch_end(self) -> None:
        '''
        End the epoch (P15, P18): refresh the cache; then, with a monitor, read the validation
        split on it, log the MRR as ``val/outcome_mrr``, step the plateau on it and append the
        read to the run's records; then log the health values and write the epoch summary (P20).

        The MRR is one float64 value: the one logged, the one the plateau steps on and the one the
        record holds. This hook runs before ModelCheckpoint's, so the epoch's checkpoint holds
        the stepped plateau, and the run's records hold the epoch's read.
        '''

        self.refresh_code_cache(self.trainer.datamodule.code_rows)
        outcome_mrr = None
        if self.monitor is not None:
            read = self.monitor.read(
                self,
                self.code_cache,
                training_run=self.training_run,
                seed=self.hparams.seed,
                epoch=self.current_epoch,
            )
            outcome_mrr = read.mrr
            mrr = torch.tensor(read.mrr, dtype=torch.float64)
            self.log(OUTCOME_MRR, mrr, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
            self._step_plateau(mrr)
            self.monitor.append(read)
        health = self._log_health()
        if self.epoch_summary is not None:
            self.epoch_summary.append(epoch=self.current_epoch, mrr=outcome_mrr, health=health)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        '''
        Record the supervision contract this checkpoint belongs to, Req 11's objective and the
        encoder architecture included (spec 4.5), and the training run's id.
        '''

        checkpoint[CHECKPOINT_KEY] = self.checkpoint_contract.model_dump()
        checkpoint[TRAINING_RUN_KEY] = self.training_run

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        '''
        Refuse a checkpoint of any other objective, supervision contract or encoder architecture;
        then stash its training run and epoch for ``on_train_start``.

        Runs for Lightning exact resume and ``load_from_checkpoint`` before the state dict loads,
        so a checkpoint of another objective, such as every one saved before Stage 7, or of the
        four-copy encoder meets a refusal that cites D2, never a key mismatch. It reads and writes
        no file, since ``load_from_checkpoint`` runs it too.
        '''

        validate_checkpoint_contract(checkpoint.get(CHECKPOINT_KEY), self.checkpoint_contract)
        self._resumed_run = checkpoint.get(TRAINING_RUN_KEY)
        self._resumed_epoch = checkpoint.get('epoch')
