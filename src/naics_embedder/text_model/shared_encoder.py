'''
The shared encoder (Req 14; spec 4.1): one backbone with one LoRA adapter, for every field.

Per code the path is:

1. field-marked channel texts;
2. one backbone with one LoRA adapter;
3. an attention-masked mean over each text's tokens;
4. fusion over the present channels;
5. one ``Linear(hidden → d)``;
6. the geometry head, which gives the point.

A query is a one-field batch, ``{'query': …}``, and takes the same path.

Each field's present texts go to the backbone in chunks of at most ``max_texts_per_call``, each
trimmed to its own longest text. So no absent text, and no padding column beyond a chunk's longest
text, enters the backbone. Presence comes from each field's ``present`` flag, never from the
attention mask (Req 9).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict, List, Mapping

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, PreTrainedModel

from naics_embedder.text_model.fields import FIELDS
from naics_embedder.text_model.fusion import FUSIONS, build_fusion
from naics_embedder.text_model.hyperbolic import HyperbolicHead

logger = logging.getLogger(__name__)

DIMENSIONS = (8, 16, 32)
# The most texts in one backbone call; its backward peak is ~13 MiB per 128-token text, ~3 GiB here
MAX_TEXTS_PER_CALL = 256

# -------------------------------------------------------------------------------------------------
# Backbone
# -------------------------------------------------------------------------------------------------

def load_base_model(name: str) -> PreTrainedModel:
    '''The backbone's pretrained weights. Tests replace this with a one-layer BERT.'''

    return AutoModel.from_pretrained(name)

# -------------------------------------------------------------------------------------------------
# Shared encoder
# -------------------------------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    '''
    One LoRA-adapted backbone for every field, then fusion, one affine map and the head.

    Args:
        base_model_name: The backbone's Hugging Face name.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout rate.
        fusion: One of ``FUSIONS``: ``masked_mean`` (the default), ``attention`` or ``moe``.
        dimension: The embedding dimension, one of ``DIMENSIONS``.
        num_experts: The number of experts, under ``moe`` only.
        top_k: The experts each code is routed to, under ``moe`` only.
        moe_hidden_dim: The experts' hidden width, under ``moe`` only.
        curvature: The head's curvature.
        use_gradient_checkpointing: Recompute the backbone's activations in the backward pass.
        max_texts_per_call: The most texts one backbone call carries. It bounds a call's memory and
            leaves every output unchanged.

    Raises:
        ValueError: If the fusion or the dimension is outside its set, or ``max_texts_per_call``
            is below 1.
    '''

    def __init__(
        self,
        base_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        fusion: str = 'masked_mean',
        dimension: int = 16,
        num_experts: int = 4,
        top_k: int = 2,
        moe_hidden_dim: int = 1024,
        curvature: float = 1.0,
        use_gradient_checkpointing: bool = True,
        max_texts_per_call: int = MAX_TEXTS_PER_CALL,
    ):
        super().__init__()

        if fusion not in FUSIONS:
            raise ValueError(f'unknown fusion {fusion!r}; expected one of {list(FUSIONS)}')
        if dimension not in DIMENSIONS:
            raise ValueError(f'unknown dimension {dimension!r}; expected one of {list(DIMENSIONS)}')
        if max_texts_per_call < 1:
            raise ValueError(f'max_texts_per_call must be at least 1, got {max_texts_per_call!r}')

        base_model = load_base_model(base_model_name)
        self.hidden_size = int(base_model.config.hidden_size)
        # The resolved snapshot, read as panels/text_only.load_backbone reads it
        self.backbone_revision = getattr(base_model.config, '_commit_hash', None)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules='all-linear',
            lora_dropout=lora_dropout,
            bias='none',
            task_type='FEATURE_EXTRACTION',
        )
        self.backbone = get_peft_model(base_model, lora_config)
        if use_gradient_checkpointing:
            # Both calls are needed: checkpointed blocks reach the adapter only through inputs
            # that require grad
            self.backbone.enable_input_require_grads()
            self.backbone.base_model.gradient_checkpointing_enable()

        self.fusion_name = fusion
        self.fusion = build_fusion(
            fusion,
            self.hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            moe_hidden_dim=moe_hidden_dim,
        )
        self.projection = nn.Linear(self.hidden_size, dimension)
        self.head = HyperbolicHead(curvature=curvature)
        self.dimension = dimension
        self.curvature = curvature
        self.max_texts_per_call = max_texts_per_call

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            'Shared encoder initialized:\n'
            f'  • backbone: {base_model_name} (hidden size {self.hidden_size})\n'
            f'  • fusion: {fusion}; dimension: {dimension}\n'
            f'  • trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)\n'
        )

    def forward(self, channel_inputs: Mapping[str, Mapping[str, torch.Tensor]]
                ) -> Dict[str, torch.Tensor]:
        '''
        Encode a batch of codes (the four channels) or of queries (the field ``query``).

        Args:
            channel_inputs: Per field, ``input_ids`` and ``attention_mask`` (B, L) and a boolean
                ``present`` (B,), as ``stack_text_inputs`` builds them.

        Returns:
            ``embedding`` (B, d + 1), the Lorentz point; ``tangent`` (B, d), the capped tangent
            vector at the origin; and ``gate_probs`` and ``top_k_indices`` under ``moe`` only.

        Raises:
            ValueError: If the batch has no field, a field outside the marker set, or a field
                without its ``present`` flag.
        '''

        if not channel_inputs:
            raise ValueError('an encoder batch needs at least one field')
        for field, inputs in channel_inputs.items():
            if field not in FIELDS:
                raise ValueError(f'unknown field {field!r}; the marker set is {list(FIELDS)}')
            if 'present' not in inputs:
                raise ValueError(
                    f'the {field!r} batch has no present flag; build it with stack_text_inputs'
                )
        # A fixed field order, so the output never depends on the mapping's key order
        fields = [field for field in FIELDS if field in channel_inputs]
        device = channel_inputs[fields[0]]['input_ids'].device
        present = torch.stack(
            [
                channel_inputs[field]['present'].to(device=device, dtype=torch.bool)
                for field in fields
            ],
            dim=1,
        )

        pooled = self._pool_present(channel_inputs, fields, present)
        fused = self.fusion(pooled, present)
        tangent, embedding = self.head(self.projection(fused.vector))
        output = {'embedding': embedding, 'tangent': tangent}
        if fused.gate_probs is not None:
            output['gate_probs'] = fused.gate_probs
            output['top_k_indices'] = fused.top_k_indices
        return output

    def _pool_present(
        self,
        channel_inputs: Mapping[str, Mapping[str, torch.Tensor]],
        fields: List[str],
        present: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Mean-pool every present (row, field) text, to (B, F, H).

        Each field's present texts, in row order, go to the backbone in chunks of at most
        ``max_texts_per_call``, each trimmed to its own longest text. So no absent text, and no
        padding column beyond a chunk's longest text, enters the backbone (P4). A field with no
        present text makes no call, and a batch with none makes no call at all. Absent slots stay
        zeros, and fusion masks them.
        '''

        batch_size = present.shape[0]
        pooled = self.projection.weight.new_zeros((batch_size, len(fields), self.hidden_size))
        for column, field in enumerate(fields):
            rows = present[:, column].nonzero().squeeze(1)
            if rows.numel() == 0:
                # split() of no rows still yields one empty chunk, and a call needs a text
                continue
            for chunk in rows.split(self.max_texts_per_call):
                pooled[chunk, column] = self._pool_chunk(
                    channel_inputs[field]['input_ids'][chunk],
                    channel_inputs[field]['attention_mask'][chunk],
                )
        return pooled

    def _pool_chunk(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        One backbone call over a chunk of present texts, mean-pooled to (n, H).

        The chunk is trimmed to its longest text: the last column where any row's mask is nonzero,
        plus one, or one column if none is.
        '''

        used = torch.nonzero(attention_mask.ne(0).any(dim=0))
        length = int(used[-1]) + 1 if used.numel() else 1
        ids, mask = input_ids[:, :length], attention_mask[:, :length]
        hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        weights = mask.unsqueeze(-1).float()
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
