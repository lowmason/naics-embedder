# Training Configuration Scripts

## GPU Memory Optimization

### optimize_gpu_config.py

This script helps you find optimal training configurations for your GPU by analyzing available memory and suggesting parameters that maximize utilization while avoiding OOM (Out of Memory) errors.

#### Features

- **Auto-detection**: Automatically detects your GPU and available memory
- **Memory estimation**: Calculates memory usage for different configurations
- **Multiple suggestions**: Provides optimized configs for different training stages
- **Memory profiling**: Can run actual GPU tests to validate estimates
- **Config updates**: Can automatically update your YAML configuration files

#### Usage

##### 1. Auto-detect GPU and get recommendations

```bash
python scripts/optimize_gpu_config.py --auto
```

##### 2. Specify GPU memory manually

```bash
# For RTX 6000 (24 GB)
python scripts/optimize_gpu_config.py --gpu-memory 24

# For A100 (80 GB)
python scripts/optimize_gpu_config.py --gpu-memory 80
```

##### 3. Run memory profiling

Test actual memory usage on your GPU:

```bash
python scripts/optimize_gpu_config.py --auto --profile
```

##### 4. Apply recommended configuration

Automatically update your config files:

```bash
python scripts/optimize_gpu_config.py --auto --apply
```

This will:
- Backup your current config files (`.backup` extension)
- Update `conf/config.yaml` with optimal `batch_size` and `accumulate_grad_batches`
- Update curriculum stage files with optimal `n_positives` and `n_negatives`

##### 5. Custom target batch size

Specify a different effective batch size target:

```bash
python scripts/optimize_gpu_config.py --auto --target-effective-batch 512
```

#### Parameters Optimized

The script optimizes the following parameters:

1. **batch_size** (`conf/config.yaml`)
   - Number of anchor samples per training batch
   - Directly affects per-step memory usage

2. **accumulate_grad_batches** (`conf/config.yaml`)
   - Number of batches to accumulate before optimizer step
   - Achieves larger effective batch size without increasing memory

3. **n_positives** (`conf/curriculum/XX_stage.yaml`)
   - Number of positive samples per anchor
   - Increases batch memory linearly

4. **n_negatives** (`conf/curriculum/XX_stage.yaml`)
   - Number of negative samples per anchor
   - Increases batch memory linearly

#### Memory Estimation

The script estimates memory usage based on:

- **Model Parameters**: LoRA-adapted encoders, MoE experts, projection layers
- **Optimizer State**: AdamW momentum and variance (2x parameters in FP32)
- **Gradients**: Same size as parameters
- **Activations**: Intermediate tensors (reduced with gradient checkpointing)
- **Batch Data**: Input tokens and attention masks
- **Overhead**: CUDA context and PyTorch framework overhead (~1.5 GB)

#### Example Output

```
================================================================================
GPU Memory Configuration Optimizer
================================================================================
Detected GPU: Quadro RTX 6000
Total Memory: 24.00 GB

Model Configuration:
  Base Model: sentence-transformers/all-MiniLM-L6-v2
  Embedding Dim: 384
  Channels: 4
  LoRA Rank: 8
  MoE Experts: 4
  FP16: True

Optimizing for 24.0 GB GPU memory...
Target effective batch size: 256

================================================================================
RECOMMENDED CONFIGURATIONS
================================================================================

────────────────────────────────────────────────────────────────────────────────
Configuration 1: 01_stage (Early)
────────────────────────────────────────────────────────────────────────────────
  batch_size:              4
  n_positives:             32
  n_negatives:             24
  accumulate_grad_batches: 64
  Effective batch size:    256
  Memory utilization:      82.3%

Memory Breakdown:
  Model Parameters:         156.8 MB
  Optimizer State:          628.5 MB
  Gradients:                156.8 MB
  Activations:            3,842.1 MB
  Batch Data:            15,728.6 MB
  Overhead:               1,500.0 MB
  ───────────────────────────────────
  Total Estimated:       20,012.8 MB (19.54 GB)

────────────────────────────────────────────────────────────────────────────────
Configuration 2: 02-05_stage (Later)
────────────────────────────────────────────────────────────────────────────────
  batch_size:              12
  n_positives:             16
  n_negatives:             8
  accumulate_grad_batches: 21
  Effective batch size:    252
  Memory utilization:      81.7%

Memory Breakdown:
  Model Parameters:         156.8 MB
  Optimizer State:          628.5 MB
  Gradients:                156.8 MB
  Activations:            3,458.4 MB
  Activations:            3,458.4 MB
  Batch Data:            15,932.2 MB
  Overhead:               1,500.0 MB
  ───────────────────────────────────
  Total Estimated:       19,832.7 MB (19.37 GB)
```

#### Tips

1. **Safety Margin**: The script uses 85% of available memory by default to prevent OOM
2. **Start Conservative**: Begin with the suggested config and adjust if needed
3. **Monitor Training**: Watch GPU memory usage with `nvidia-smi` during training
4. **Stage-Specific**: Early stages (01) have more samples and need smaller batches
5. **Effective Batch**: Larger effective batch sizes generally improve training stability

#### Troubleshooting

**OOM Error Despite Recommendations**:
- Reduce `batch_size` by 25-50%
- Increase `accumulate_grad_batches` proportionally
- Ensure gradient checkpointing is enabled

**Memory Underutilized**:
- Increase `batch_size` gradually
- Reduce `accumulate_grad_batches` to maintain effective batch size
- Run with `--profile` to see actual usage

**Script Errors**:
- Ensure CUDA and PyTorch are properly installed
- Check that you're in the correct conda/virtual environment
- Verify config file paths are correct

#### Advanced Usage

##### Custom Configuration

Edit the script's `ModelConfig` class to match your model architecture:

```python
@dataclass
class ModelConfig:
    embedding_dim: int = 768  # Change for larger models
    lora_r: int = 16          # Higher rank = more parameters
    num_experts: int = 8      # More experts = more memory
    # ... etc
```

##### Different Safety Margins

Modify the `safety_margin` parameter in `find_optimal_batch_size()`:

```python
# Use 90% of memory (more aggressive)
batch_size, estimate = find_optimal_batch_size(
    gpu_memory_gb, model_cfg, n_pos, n_neg, safety_margin=0.90
)
```

#### Related Files

- `conf/config.yaml` - Main training configuration
- `conf/curriculum/01_stage.yaml` - Early training stage
- `conf/curriculum/02-05_stage.yaml` - Later training stages
- `src/naics_embedder/model/encoder.py` - Model architecture
- `src/naics_embedder/data_loader/datamodule.py` - Data loading

#### References

- PyTorch Lightning Trainer: https://lightning.ai/docs/pytorch/stable/common/trainer.html
- Gradient Accumulation: https://lightning.ai/docs/pytorch/stable/common/trainer.html#accumulate-grad-batches
- Mixed Precision Training: https://pytorch.org/docs/stable/amp.html
