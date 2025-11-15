# GPU Memory Optimization Guide

## Overview

This guide helps you optimize training configurations for your available GPU memory. The scripts automatically calculate optimal values for:

- **batch_size**: Samples per training step
- **accumulate_grad_batches**: Steps before optimizer update
- **n_positives**: Positive samples per anchor
- **n_negatives**: Negative samples per anchor

## Quick Start

### 1. Check Your Current GPU

```bash
nvidia-smi
```

Look for:
- GPU model (e.g., "Quadro RTX 6000", "A100-SXM4-80GB")
- Memory total (e.g., "24576MiB", "81920MiB")

### 2. Get Optimal Configuration

```bash
# Auto-detect GPU and show recommendations
uv run python scripts/optimize_gpu_config.py --auto

# Or specify GPU memory manually
uv run python scripts/optimize_gpu_config.py --gpu-memory 24
```

### 3. Apply Configuration

```bash
# Apply recommended settings to config files
uv run python scripts/optimize_gpu_config.py --auto --apply
```

This will:
- ✓ Backup current configs (`.backup` files)
- ✓ Update `conf/config.yaml`
- ✓ Update curriculum stage files

### 4. Verify Training

```bash
# Start training and monitor memory
watch -n 1 nvidia-smi

# In another terminal
uv run naics-embedder train --stage 01
```

## Common GPU Configurations

### Quadro RTX 6000 (24 GB) - Current Setup

**Recommended Settings:**
```yaml
# conf/config.yaml
data_loader:
  batch_size: 45

training:
  trainer:
    accumulate_grad_batches: 5  # Effective batch = 225

# conf/text_curriculum/01_stage.yaml
n_positives: 32
n_negatives: 24

# conf/text_curriculum/02-05_stage.yaml  
n_positives: 16
n_negatives: 8
```

**Expected Memory Usage:** ~19.8 GB (83% utilization)

### Tesla V100 (32 GB)

**Recommended Settings:**
```yaml
data_loader:
  batch_size: 64

training:
  trainer:
    accumulate_grad_batches: 4  # Effective batch = 256
```

**Expected Memory Usage:** ~26.5 GB (83% utilization)

### A100 (40 GB)

**Recommended Settings:**
```yaml
data_loader:
  batch_size: 82

training:
  trainer:
    accumulate_grad_batches: 3  # Effective batch = 246
```

**Expected Memory Usage:** ~33.9 GB (85% utilization)

### A100 (80 GB)

**Recommended Settings:**
```yaml
data_loader:
  batch_size: 128

training:
  trainer:
    accumulate_grad_batches: 2  # Effective batch = 256
```

**Expected Memory Usage:** ~51.5 GB for stage 01, ~21.4 GB for later stages

## Memory Breakdown

Understanding what uses GPU memory:

| Component | Size | Notes |
|-----------|------|-------|
| **Model Parameters** | ~200 MB | 4 channel encoders + MoE + projection |
| **Optimizer State** | ~800 MB | AdamW momentum + variance (FP32) |
| **Gradients** | ~200 MB | Same size as trainable parameters |
| **Activations** | ~17.5 GB | Encoder + MoE intermediate tensors |
| **Batch Data** | ~37 MB | Input tokens + attention masks |
| **Overhead** | ~1.5 GB | CUDA context + PyTorch framework |

**Total:** ~20 GB for typical configuration

### Memory Optimization Techniques

The model uses several techniques to reduce memory:

1. **LoRA (Low-Rank Adaptation)**: Only ~1% of base model parameters are trainable
2. **Gradient Checkpointing**: Saves ~75% of activation memory
3. **Mixed Precision (FP16)**: Halves parameter and activation memory
4. **Gradient Accumulation**: Achieves large effective batch without memory increase

## Troubleshooting

### OOM (Out of Memory) Error

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

1. **Reduce batch size by 25-50%**
   ```bash
   # Manually edit conf/config.yaml
   data_loader:
     batch_size: 22  # Was 45, now half
   
   training:
     trainer:
       accumulate_grad_batches: 10  # Increase to maintain effective batch
   ```

2. **Reduce negatives in early stages**
   ```bash
   # Edit conf/text_curriculum/01_stage.yaml
   n_negatives: 16  # Was 24
   ```

3. **Clear GPU cache before training**
   ```bash
   python -c "import torch; torch.cuda.empty_cache()"
   ```

4. **Re-run optimizer with more conservative margin**
   ```python
   # In scripts/optimize_gpu_config.py, change:
   safety_margin=0.75  # Instead of 0.85
   ```

### Memory Underutilized

**Symptoms:**
- `nvidia-smi` shows only 30-50% memory usage
- Training seems slower than expected

**Solutions:**

1. **Increase batch size gradually**
   ```bash
   # Try increments of 25%
   batch_size: 56  # Was 45
   ```

2. **Profile actual usage**
   ```bash
   uv run python scripts/optimize_gpu_config.py --auto --profile
   ```

3. **Adjust accumulation**
   ```bash
   # Keep effective batch constant
   accumulate_grad_batches: 4  # Was 5
   ```

### Inconsistent Memory Usage

**Symptoms:**
- Memory usage varies significantly between steps
- Occasional OOM errors

**Causes:**
- Variable sequence lengths in batch
- Different numbers of negatives per anchor
- MoE routing variations

**Solutions:**

1. **Use fixed padding**
   ```yaml
   # conf/config.yaml
   data_loader:
     tokenization:
       padding: max_length  # Not 'longest'
   ```

2. **Add safety margin**
   ```bash
   # Reduce batch_size by 10-20%
   ```

## Advanced Usage

### Profile Actual Memory

Run a test training step to measure real memory usage:

```bash
uv run python scripts/optimize_gpu_config.py --auto --profile
```

This will:
- Load the actual model
- Run a forward + backward pass
- Measure peak memory usage
- Compare to estimates

### Custom Target Batch Size

```bash
# Target larger effective batch for better gradient stability
uv run python scripts/optimize_gpu_config.py --auto --target-effective-batch 512

# Or smaller for faster iterations
uv run python scripts/optimize_gpu_config.py --auto --target-effective-batch 128
```

### Multiple GPU Setup

For multi-GPU training (not currently configured):

```yaml
training:
  trainer:
    devices: 2  # Use 2 GPUs
    strategy: ddp  # Distributed Data Parallel
    
data_loader:
  batch_size: 45  # Per GPU
  # Effective batch = 45 * 2 * accumulate_grad_batches
```

### Manual Configuration

Edit configuration files directly:

```bash
# Main config
vim conf/config.yaml

# Early stage curriculum
vim conf/text_curriculum/01_stage.yaml

# Later stages
vim conf/text_curriculum/{02,03,04,05}_stage.yaml
```

Key parameters:
- `data_loader.batch_size`: Samples per step
- `training.trainer.accumulate_grad_batches`: Accumulation steps
- `n_positives`: Positive samples per anchor
- `n_negatives`: Negative samples per anchor

## Monitoring During Training

### Real-time GPU Memory

```bash
# Update every second
watch -n 1 nvidia-smi

# Or with more detail
watch -n 1 'nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv'
```

### Training Logs

```bash
# Watch training progress
tail -f logs/train_*.log

# Check TensorBoard
tensorboard --logdir outputs/
```

### Expected Behavior

**Normal:**
- Memory stable after first few steps
- Utilization 80-90% of available memory
- No OOM errors
- Smooth loss curves

**Warning Signs:**
- Memory creeping up over time (memory leak)
- Frequent OOM errors
- Very low utilization (<50%)
- Erratic loss values

## Performance Tips

### Maximize Throughput

1. **Use full GPU memory (80-90%)**
   - Better hardware utilization
   - Larger batches = more stable gradients

2. **Balance batch size vs accumulation**
   - Larger batch_size = faster training (less overhead)
   - But use accumulation to avoid OOM

3. **Optimize num_workers**
   ```yaml
   data_loader:
     num_workers: 4  # Try 0, 2, 4, 8
   ```

### Maximize Training Quality

1. **Larger effective batch size**
   - More stable gradient estimates
   - Better convergence
   - Target 256-512

2. **More negatives in early stages**
   - Better contrastive learning
   - Harder examples
   - If memory allows: n_negatives=32 for stage 01

## Script Reference

### optimize_gpu_config.py

Main optimization script.

**Options:**
- `--auto`: Auto-detect GPU
- `--gpu-memory N`: Specify GPU memory (GB)
- `--profile`: Run actual memory profiling
- `--apply`: Update config files
- `--target-effective-batch N`: Target effective batch size
- `--config-path PATH`: Path to config file

**Examples:**
```bash
# Basic usage
uv run python scripts/optimize_gpu_config.py --auto

# Profile and apply
uv run python scripts/optimize_gpu_config.py --auto --profile --apply

# Custom target
uv run python scripts/optimize_gpu_config.py --gpu-memory 80 --target-effective-batch 512
```

### example_optimizer_usage.py

Example showing programmatic usage.

```bash
uv run python scripts/example_optimizer_usage.py
```

## FAQ

**Q: Why is my memory estimate different from actual usage?**

A: Estimates are conservative and don't account for:
- PyTorch's memory allocator overhead
- Variable sequence lengths
- MoE routing patterns
- GPU memory fragmentation

Actual usage may be ±20% from estimate.

**Q: Should I use gradient checkpointing?**

A: Yes, it's enabled by default and essential for memory efficiency. It trades compute (re-computing activations) for memory savings (~75%).

**Q: What's the optimal effective batch size?**

A: 256-512 is a good range for contrastive learning. Larger is generally better for training stability, but has diminishing returns beyond 512.

**Q: Can I train on multiple GPUs?**

A: The code supports it (PyTorch Lightning handles this), but configs need adjustment:
- Keep per-GPU batch_size reasonable
- Reduce accumulate_grad_batches proportionally
- Use strategy: 'ddp' or 'ddp_spawn'

**Q: What if I change the model architecture?**

A: Update `ModelConfig` in the script:
```python
@dataclass
class ModelConfig:
    embedding_dim: int = 768  # For larger base model
    lora_r: int = 16          # Higher rank
    num_experts: int = 8      # More experts
    # ...
```

**Q: How often should I re-run the optimizer?**

A: Re-run when:
- Switching to a different GPU
- Changing model architecture
- Updating number of channels
- Experiencing OOM or underutilization

## Additional Resources

- PyTorch Lightning Memory Tips: https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html
- CUDA Memory Management: https://pytorch.org/docs/stable/notes/cuda.html
- Gradient Accumulation: https://lightning.ai/docs/pytorch/stable/common/trainer.html#accumulate-grad-batches

## Getting Help

If you encounter issues:

1. Check logs: `logs/train_*.log`
2. Run profiling: `--profile` flag
3. Reduce batch size by 50% as emergency fix
4. Check GPU with `nvidia-smi` for other processes

```bash
# Kill other GPU processes if needed
fuser -v /dev/nvidia*
```

