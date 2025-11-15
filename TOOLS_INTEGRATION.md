# Tools Integration Summary

All utility scripts have been successfully integrated into the NAICS embedder package and are now accessible through the CLI interface.

## What Was Integrated

1. **GPU Configuration Tool** (`scripts/gpu_config.py` → `src/naics_embedder/tools/gpu_tools.py`)
   - Optimizes training configuration based on available GPU memory
   - Suggests optimal batch sizes and gradient accumulation

2. **Configuration Display Tool** (`scripts/current_config.py` → `src/naics_embedder/tools/config_tools.py`)
   - Displays current training and curriculum configuration

3. **Metrics Visualization Tool** (`scripts/visualize_metrics.py` → `src/naics_embedder/tools/_visualize_metrics.py` + `metrics_tools.py`)
   - Visualizes training metrics from log files
   - Creates comprehensive plots and analysis

4. **Hierarchy Investigation Tool** (`scripts/investigate_hierarchy.py` → `src/naics_embedder/tools/_investigate_hierarchy.py` + `metrics_tools.py`)
   - Investigates why hierarchy preservation correlations might be low
   - Analyzes ground truth distances and evaluation configuration

## New CLI Commands

All tools are now accessible through the `tools` subcommand:

```bash
# Display current configuration
naics-embedder tools config [--config PATH]

# Optimize GPU configuration
naics-embedder tools gpu [--auto] [--gpu-memory GB] [--target-effective-batch N] [--apply] [--config PATH]

# Visualize training metrics
naics-embedder tools visualize [--stage STAGE] [--log-file PATH] [--output-dir PATH]

# Investigate hierarchy correlations
naics-embedder tools investigate [--distance-matrix PATH] [--config PATH]
```

## Examples

### View Current Configuration
```bash
naics-embedder tools config
```

### Auto-detect GPU and Optimize Configuration
```bash
naics-embedder tools gpu --auto --apply
```

### Visualize Metrics for Stage 02
```bash
naics-embedder tools visualize --stage 02_stage
```

### Investigate Hierarchy Correlations
```bash
naics-embedder tools investigate
```

## File Structure

```
src/naics_embedder/
├── tools/
│   ├── __init__.py              # Module exports
│   ├── config_tools.py          # Configuration display
│   ├── gpu_tools.py             # GPU optimization
│   ├── metrics_tools.py         # Metrics wrapper functions
│   ├── _visualize_metrics.py   # Visualization implementation
│   └── _investigate_hierarchy.py # Investigation implementation
└── cli.py                       # CLI commands added here
```

## Migration Notes

- Original scripts in `scripts/` directory are preserved for reference
- All functionality has been moved into the package structure
- Tools are now importable as Python modules: `from naics_embedder.tools import ...`
- CLI integration uses Typer sub-commands for clean organization

## Benefits

1. **Unified Interface**: All tools accessible through single CLI
2. **Better Organization**: Tools are part of the package structure
3. **Importable**: Can be used as Python modules in other scripts
4. **Consistent**: Follows same patterns as other CLI commands
5. **Maintainable**: Centralized location for all utility tools

## Next Steps

The original scripts in `scripts/` can be removed if desired, or kept for reference. The integrated versions in `src/naics_embedder/tools/` are the canonical implementations.

