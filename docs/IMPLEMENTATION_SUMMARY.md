# NAICS Contrastive Learning Training System - Implementation Summary

## File Structure

```
naics_training_system/
├── README.md                           # Main documentation
├── cli.py                              # Updated CLI with training integration
│
├── src/naics_gemini/
│   ├── train.py                        # Standalone training script
│   │
│   ├── data/
│   │   ├── tokenization_cache.py       # Pre-tokenization and caching
│   │   ├── streaming_dataset.py        # Curriculum-filtered streaming dataset
│   │   └── datamodule.py               # PyTorch Lightning DataModule
│   │
│   └── model/
│       ├── moe.py                      # Mixture of Experts layer
│       ├── encoder.py                  # Multi-channel encoder with LoRA
│       ├── loss.py                     # Hyperbolic InfoNCE loss
│       └── naics_model.py              # Main Lightning module
│
└── conf/
    ├── config.yaml                     # Base configuration
    ├── data/default.yaml               # Data loading config
    ├── model/default.yaml              # Model architecture config
    ├── loss/default.yaml               # Loss function config
    ├── training/default.yaml           # Training hyperparameters
    └── curriculum/
        ├── 01_stage_easy.yaml          # Easy curriculum
        ├── 02_stage_medium.yaml        # Medium curriculum
        └── 03_stage_hard.yaml          # Hard curriculum
```

## Core Components

### 1. Data Pipeline

**tokenization_cache.py**
- Pre-tokenizes all NAICS codes (title, description, excluded, examples)
- Caches to disk as `./data/token_cache/token_cache.pt`
- Loads pretrained tokenizer once
- Handles empty/null text fields

**streaming_dataset.py**
- Implements IterableDataset for memory-efficient streaming
- Filters triplets by curriculum constraints:
  - Positive levels (2-6 digit codes)
  - Distance ranges
  - Maximum positives per anchor
  - Difficulty buckets (1-8)
- Samples K negatives with configurable percentages
- Implements fallback sampling for sparse buckets
- Maps hardness levels based on excluded/unrelated/distance_diff

**datamodule.py**
- PyTorch Lightning DataModule wrapper
- Handles train/val split
- Custom collate function for batching
- Reshapes negatives for (batch_size * K) format

### 2. Model Architecture

**encoder.py (MultiChannelEncoder)**
- Loads 4 instances of all-mpnet-base-v2
- Applies LoRA to query/value matrices independently per channel
- Concatenates channel embeddings (768 * 4 = 3072 dim)
- Optional MoE fusion or simple linear fusion
- Returns fused embedding + load balancing loss

**moe.py (MixtureOfExpertsLayer)**
- Gating network produces expert probabilities
- Top-2 selection per input
- 4 expert networks (FFN: input_dim → hidden_dim → input_dim)
- Computes auxiliary load balancing loss
- Formula: N * Σ(importance_i * load_i)

**loss.py (HyperbolicInfoNCELoss)**
- Projects Euclidean embeddings to Lorentz hyperboloid
- Exponential map from tangent space at origin
- Computes pairwise Lorentzian distances
- InfoNCE with negative distances as logits
- Temperature scaling (default 0.07)

**naics_model.py (NAICSContrastiveModel)**
- Main Lightning module tying everything together
- Forward pass through encoder
- Computes contrastive + load balancing losses
- Logging for monitoring
- AdamW optimizer with cosine annealing

### 3. Training & Configuration

**train.py**
- Standalone training script
- Hydra configuration composition
- Sets up callbacks (checkpointing, early stopping)
- TensorBoard logging
- Handles full training loop

**cli.py**
- Updated CLI with real training implementation
- Integrates with data generation commands
- Hydra override support
- Rich console output

**Configuration Files**
- Modular configs for data, model, loss, training
- Curriculum stages define difficulty progression
- Easy to experiment with hyperparameters

## Key Implementation Details

### Curriculum Learning
Each curriculum stage defines:
- `positive_levels`: Which code levels to use (2-6)
- `positive_distance_min/max`: Filter by hierarchy distance
- `max_positives`: Cap positives per anchor
- `difficulty_buckets`: Which hardness levels (1-8)
- `bucket_percentages`: Allocation of K negatives
- `k_negatives`: Total negatives per positive (16)

### Hardness Mapping
```python
8: excluded=True, unrelated=True       # Semantic exclusions
7: excluded=False, distance_diff≤0.5   # Very hard siblings
6: excluded=False, distance_diff≤1.0   # Hard siblings
5: excluded=False, distance_diff≤2.0   # Moderate
4: excluded=False, distance_diff≤3.0   # Easier
3: excluded=False, distance_diff≤4.0   # Easier
2: excluded=False, distance_diff≤6.5   # Easy
1: excluded=False, unrelated=True      # Different sectors
```

### Negative Sampling with Fallback
1. Calculate target count per bucket from percentages
2. Sample from highest hardness first
3. If bucket insufficient, use fallback to lower hardness
4. Ensures K negatives always available

### Hyperbolic Projection
- Linear projection: R^768 → R^769
- Exponential map: tangent space → hyperboloid
- Lorentz inner product: ⟨u,v⟩ = Σu_i·v_i - u_0·v_0
- Distance: √c · arcosh(-⟨u,v⟩)

### MoE Load Balancing
- Global batch statistics (not micro-batch)
- importance = softmax(gate_logits).sum(0) / sum
- load = expert_assignments.sum() / (batch * top_k)
- loss = num_experts * Σ(importance * load)
- Enables domain specialization

## Usage

### Training
```bash
# Stage 1: Easy negatives
uv run naics-gemini train -c 01_stage_easy

# Stage 2: Medium difficulty
uv run naics-gemini train -c 02_stage_medium

# Stage 3: Full difficulty
uv run naics-gemini train -c 03_stage_hard

# Custom overrides
uv run naics-gemini train -c 01_stage_easy \
    training.trainer.max_epochs=20 \
    model.lora.r=16 \
    data.batch_size=64
```

### Monitoring
```bash
# TensorBoard
tensorboard --logdir ./outputs

# Checkpoints
./checkpoints/{experiment_name}/
```

## Dependencies
- torch==2.5.1
- pytorch-lightning>=2.4
- transformers[torch]>=4.46
- peft>=0.13 (LoRA)
- hydra-core>=1.3
- polars>=1.9 (streaming Parquet)

## Next Steps

1. **Run Training**: Execute curriculum stages sequentially
2. **Monitor**: Check TensorBoard for loss curves
3. **Inference**: Use best checkpoint for embedding generation
4. **Evaluation**: Compute retrieval metrics on held-out codes
5. **Fine-tuning**: Adjust curriculum based on validation performance

## Implementation Highlights

✅ Streaming dataset (handles 260M+ triplets)
✅ Pre-tokenized cache (fast startup)
✅ Curriculum filtering (progressive difficulty)
✅ K-negative sampling with fallbacks
✅ Multi-channel encoder (4 text fields)
✅ LoRA fine-tuning (parameter efficient)
✅ MoE fusion (specialized experts)
✅ Hyperbolic loss (hierarchical geometry)
✅ Global-batch load balancing
✅ Mixed precision training
✅ Checkpointing & early stopping
✅ Hydra configuration system
✅ TensorBoard logging
