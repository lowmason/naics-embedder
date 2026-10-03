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

A batch's present (row, field) texts are gathered into one backbone call, trimmed to the longest
of them, so neither an absent text nor a padding column enters the backbone. Presence comes from
each field's ``present`` flag, never from the attention mask (Req 9).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict, List, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, PreTrainedModel

from naics_embedder.text_model.fields import FIELDS
from naics_embedder.text_model.fusion import FUSIONS, build_fusion
from naics_embedder.text_model.hyperbolic import HyperbolicHead

logger = logging.getLogger(__name__)

DIMENSIONS = (8, 16, 32)

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

    Raises:
        ValueError: If the fusion or the dimension is outside its set.
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
    ):
        super().__init__()

        if fusion not in FUSIONS:
            raise ValueError(f'unknown fusion {fusion!r}; expected one of {list(FUSIONS)}')
        if dimension not in DIMENSIONS:
            raise ValueError(f'unknown dimension {dimension!r}; expected one of {list(DIMENSIONS)}')

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
        Mean-pool every present (row, field) text in one backbone call, to (B, F, H).

        Present texts are gathered field by field and right-padded to one width, which is then
        trimmed to the longest text, so neither an absent text nor a padding column enters the
        backbone (P4). Absent slots stay zeros, and fusion masks them.
        '''

        batch_size = present.shape[0]
        width = max(int(channel_inputs[field]['input_ids'].shape[1]) for field in fields)
        input_ids: List[torch.Tensor] = []
        attention_mask: List[torch.Tensor] = []
        for column, field in enumerate(fields):
            rows = present[:, column]
            ids = channel_inputs[field]['input_ids'][rows]
            mask = channel_inputs[field]['attention_mask'][rows]
            input_ids.append(F.pad(ids, (0, width - ids.shape[1])))
            attention_mask.append(F.pad(mask, (0, width - mask.shape[1])))
        ids = torch.cat(input_ids)
        mask = torch.cat(attention_mask)
        if ids.shape[0] == 0:
            # No present text in the batch: every slot is absent, and fusion masks them all
            return self.projection.weight.new_zeros((batch_size, len(fields), self.hidden_size))

        used = torch.nonzero(mask.ne(0).any(dim=0))
        length = int(used[-1]) + 1 if used.numel() else 1
        ids, mask = ids[:, :length], mask[:, :length]
        hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        weights = mask.unsqueeze(-1).float()
        vectors = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)

        # Field-major slots, in the order the texts were gathered
        flat = vectors.new_zeros((len(fields) * batch_size, vectors.shape[1]))
        flat[present.t().reshape(-1)] = vectors
        return flat.reshape(len(fields), batch_size, -1).transpose(0, 1)
