# Training Metrics Visualization and Analysis

This directory contains scripts to visualize and analyze training metrics from your NAICS embedder model.

## Scripts

### 1. `visualize_metrics.py`

Creates comprehensive visualizations of training metrics from log files.

**Usage:**
```bash
python scripts/visualize_metrics.py --stage 02_stage
```

**Options:**
- `--stage`: Stage name to filter (default: `02_stage`)
- `--log-file`: Path to log file (default: `logs/train_sequential.log`)
- `--output-dir`: Output directory for plots (default: `outputs/visualizations/`)

**Output:**
- Creates a PNG visualization with 6 subplots:
  1. Hyperbolic Radius Over Time
  2. Hierarchy Preservation Correlations (Cophenetic & Spearman)
  3. Embedding Diversity Metrics (Norm CV & Distance CV)
  4. Radius vs Hierarchy Preservation Scatter
  5. Hyperbolic Radius Spread (Std Dev)
  6. Summary Statistics Table

- Prints detailed analysis and recommendations
- Displays a metrics summary table

### 2. `investigate_hierarchy.py`

Investigates why hierarchy preservation correlations might be low.

**Usage:**
```bash
python scripts/investigate_hierarchy.py
```

**What it does:**
- Analyzes ground truth distance matrix (if polars is available)
- Checks evaluation configuration
- Provides analysis of potential correlation issues
- Gives recommendations for improvement

## Understanding the Metrics

### Hyperbolic Radius

**What it is:**
- The time coordinate (x₀) in the Lorentz model of hyperbolic space
- Represents distance from the origin in hyperbolic space

**What to look for:**
- **Increasing radius**: Model is using hyperbolic space to separate embeddings
- **Large radius (>20)**: May indicate instability or over-separation
- **Increasing std**: More variation in distances (can indicate hierarchy levels at different distances)

**Your current values:**
- Epoch 0: 2.23 ± 0.52
- Epoch 5: 16.29 ± 11.06
- **Status**: Growing rapidly but still in reasonable range

### Hierarchy Preservation Metrics

**Cophenetic Correlation:**
- Measures how well embedding distances match ground truth tree distances
- Range: -1 to 1
- **Target**: >0.7-0.8 for good preservation
- **Your values**: 0.22 (low, but improving from negative)

**Spearman Correlation:**
- Rank correlation between embedding and tree distances
- Range: -1 to 1
- **Target**: >0.7 for good preservation
- **Your values**: ~0.00 (very low, needs improvement)

### Collapse Detection

**Norm CV & Distance CV:**
- Coefficient of variation measures embedding diversity
- **Low values**: Indicates collapse (all embeddings similar)
- **Your values**: Increasing (0.27 → 0.59), indicating good diversity

**Collapse Flag:**
- `False` = No collapse detected ✓
- `True` = Embeddings are collapsing (bad)

### Manifold Validity

**Lorentz Norm:**
- Should be exactly -1.0 (for curvature=1.0)
- **Your values**: -1.000000 ± 0.000001 ✓ (perfect!)

**Max Violation:**
- How much embeddings violate the hyperboloid constraint
- **Your values**: < 1e-5 ✓ (excellent!)

## Current Status (02_stage, Epoch 5/20)

### ✅ Strengths:
1. **Manifold validity**: Perfect (-1.0 ± 0.000001)
2. **No collapse**: All embeddings show good diversity
3. **Hyperbolic space usage**: Radius is increasing, model is exploring
4. **Improving trend**: Cophenetic improved from negative to positive

### ⚠️ Areas to Monitor:
1. **Low hierarchy correlations**: 
   - Cophenetic: 0.22 (target: >0.7)
   - Spearman: ~0.00 (target: >0.7)
   - **Reason**: Early in training (only 25% complete)

2. **Rapid radius growth**: 
   - 2.23 → 16.29 in 5 epochs
   - Monitor for stability

3. **Declining cophenetic**: 
   - Peaked at epoch 2 (0.27), now at 0.22
   - May recover as training continues

## Recommendations

1. **Continue Training**: You're only 25% through stage 02. Correlations often improve in later epochs.

2. **Monitor Trends**: 
   - Watch if cophenetic correlation starts increasing again
   - Check if hyperbolic radius stabilizes
   - Ensure no collapse occurs

3. **Check Evaluation Setup**:
   - Eval sample size: 500 ✓ (good)
   - Ground truth distances: Loaded correctly ✓

4. **If Metrics Don't Improve**:
   - Consider learning rate reduction
   - Check if loss is decreasing
   - Verify training data quality

## Quick Reference

**Good Signs:**
- ✓ Manifold valid = True
- ✓ Collapse = False
- ✓ Radius increasing gradually
- ✓ Correlations improving over time

**Warning Signs:**
- ⚠️ Radius > 30-40 (may indicate instability)
- ⚠️ Correlations declining consistently
- ⚠️ Collapse = True
- ⚠️ Manifold violations > 1e-3

## Example Output

After running `visualize_metrics.py`, you'll get:
- A PNG file: `outputs/visualizations/02_stage_metrics.png`
- Console output with analysis and recommendations
- A summary table of all metrics

## Next Steps

1. Run visualization: `python scripts/visualize_metrics.py --stage 02_stage`
2. Check investigation: `python scripts/investigate_hierarchy.py`
3. Monitor training and re-run visualization after more epochs
4. Compare metrics across different stages

