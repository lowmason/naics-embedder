# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Multi-Channel Encoder with LoRA
# -------------------------------------------------------------------------------------------------

class MultiChannelEncoder(nn.Module):
    
    def __init__(
        self,
        base_model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        use_moe: bool = True,
        num_experts: int = 4,
        top_k: int = 2,
        moe_hidden_dim: int = 1024
    ):
        super().__init__()
        
        self.channels = ['title', 'description', 'excluded', 'examples']
        self.use_moe = use_moe
        
        # Get base model to determine embedding dimension
        base_model = AutoModel.from_pretrained(base_model_name)
        self.embedding_dim = base_model.config.hidden_size
        
        # Configure LoRA with universal target (works with any model)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules='all-linear',  # Universal: targets all linear layers
            lora_dropout=lora_dropout,
            bias='none',
            task_type='FEATURE_EXTRACTION'
        )
        
        # Create separate LoRA-adapted encoder for each channel
        logger.info(f'Creating {len(self.channels)} channel encoders with LoRA (r={lora_r})...')
        self.encoders = nn.ModuleDict({
            channel: get_peft_model(
                AutoModel.from_pretrained(base_model_name),
                lora_config
            )
            for channel in self.channels
        })
        
        # Optional: Mixture of Experts
        if use_moe:
            from naics_gemini.model.moe import MixtureOfExperts
            logger.info(f'Initializing MoE with {num_experts} experts (top-k={top_k})...')
            self.moe = MixtureOfExperts(
                input_dim=self.embedding_dim,
                hidden_dim=moe_hidden_dim,
                num_experts=num_experts,
                top_k=top_k
            )
        else:
            self.moe = None
        
        logger.info(f'Encoder initialized: embedding_dim={self.embedding_dim}, use_moe={use_moe}')
    
    
    def forward(
        self,
        channel_inputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        '''
        Forward pass through multi-channel encoder.
        
        Args:
            channel_inputs: Dict mapping channel names to tokenized inputs
                           Each channel has 'input_ids' and 'attention_mask'
        
        Returns:
            Dict with 'embedding' and optional 'load_balancing_loss'
        '''
        channel_embeddings = []
        
        # Encode each channel separately
        for channel in self.channels:
            inputs = channel_inputs[channel]
            
            # Forward through LoRA-adapted encoder
            outputs = self.encoders[channel](
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            # Mean pooling over sequence dimension
            embeddings = outputs.last_hidden_state
            mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.shape).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            
            channel_embeddings.append(pooled)
        
        # Concatenate channel embeddings
        combined = torch.cat(channel_embeddings, dim=1)  # (batch, embedding_dim * 4)
        
        # Optional: Pass through MoE
        if self.moe is not None:
            output, load_balancing_loss = self.moe(combined)
            return {
                'embedding': output,
                'load_balancing_loss': load_balancing_loss
            }
        else:
            # Simple projection to reduce dimension
            projection = nn.Linear(
                self.embedding_dim * len(self.channels),
                self.embedding_dim
            ).to(combined.device)
            output = projection(combined)
            
            return {
                'embedding': output,
                'load_balancing_loss': torch.tensor(0.0, device=combined.device)
            }