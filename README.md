# NAICS Gemini Training System

Minimal production training system for hierarchical contrastive learning on NAICS codes.

## Architecture

### Components

1. **Multi-Channel Encoder** (`src/naics_gemini/model/encoder.py`)
   - 4 separate channels: title, description, excluded, examples
   - LoRA fine-tuning on pretrained transformer (all-mpnet-base-v2)
   - LoRA applied to query/value matrices
   - Parameters: r=8, alpha=16

2. **Mixture of Experts** (`src/naics_gemini/model/moe.py`)
   - 4 experts with Top-2 gating
   - Global-batch auxiliary loss for load balancing
   - Enables specialized fusion strategies

3. **Hyperbolic Loss** (`src/naics_gemini/model/loss.py`)
   - Projects embeddings to Lorentz hyperboloid
   - InfoNCE contrastive loss using Lorentzian distance
   - Preserves hierarchical structure naturally

4. **Streaming Dataset** (`src/naics_gemini/data/streaming_dataset.py`)
   - Filters by positive levels, distances, difficulty buckets
   - Samples K negatives with configurable percentages per bucket
   - Fallback sampling for sparse hardness levels

5. **Tokenization Cache** (`src/naics_gemini/data/tokenization_cache.py`)
   - Pre-tokenizes all code descriptions once
   - Caches to disk for fast training startup

## Training

### Basic Usage

```bash
# Train with easy curriculum
uv run naics-gemini train -c 01_stage_easy

# Train with custom overrides
uv run naics-gemini train -c 02_stage_medium training.trainer.max_epochs=20
```

### Curriculum Stages

1. **Stage 1 (Easy)**: Level 6 codes, unrelated negatives
2. **Stage 2 (Medium)**: Levels 5-6, mixed difficulty
3. **Stage 3 (Hard)**: All levels, full difficulty spectrum including exclusions

### Configuration

All configs in `conf/`:
- `config.yaml`: Base configuration
- `data/`: Data paths and loading
- `model/`: Architecture hyperparameters
- `loss/`: Loss function settings
- `training/`: Optimizer and trainer settings
- `curriculum/`: Difficulty progression

## Key Features

### 1. Curriculum Learning
Progressive difficulty via filtering:
- Positive pair selection (level, distance)
- Negative sampling (hardness buckets)
- Configurable bucket percentages

### 2. Efficient Training
- Streaming from Parquet (handles 260M+ triplets)
- Pre-tokenized cache
- Mixed precision (16-bit)
- Conditional computation (MoE)

### 3. Hierarchical Awareness
- Hyperbolic geometry (exponential volume growth)
- Lorentz model (numerical stability)
- Distance-based contrastive learning

## Hardness Levels

```
8: Exclusions (semantically close but different sectors)
7: Distance diff ≤ 0.5 (very hard siblings)
6: Distance diff ≤ 1.0 (hard siblings)
5: Distance diff ≤ 2.0 (moderate)
4: Distance diff 2.5-3.0
3: Distance diff 3.5-4.0
2: Distance diff 4.5-6.5
1: Unrelated (different sectors)
```

## Output

Training produces:
- Checkpoints: `./checkpoints/{experiment_name}/`
- Logs: `./outputs/{experiment_name}/`
- Best model selection via validation loss

## Implementation Notes

### Load Balancing (MoE)
- Auxiliary loss coefficient: 0.01
- Calculated on global batch (not micro-batch)
- Enables domain specialization

### Negative Sampling
- Target K=16 negatives per positive
- Percentage allocation across buckets
- Fallback to easier buckets if sparse
- Ensures K negatives always available

### Validation Split
- 5% of data reserved
- Separate curriculum instance
- Different random seed
