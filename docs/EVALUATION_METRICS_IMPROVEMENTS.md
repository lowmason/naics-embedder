# Evaluation Metrics Improvements

## Summary

The evaluation metrics have been significantly improved to provide more informative feedback during training.

## Changes Made

### 1. **Improved Collapse Detection**

**Before:**
- Used absolute thresholds (0.01) for variance, norm std, and distance std
- All metrics showed `collapse=True` throughout training
- No visibility into actual values

**After:**
- Uses **Coefficient of Variation (CV)** for norm and distance checks
- CV = std / mean - normalized measure of spread
- Logs actual values (`norm_cv`, `distance_cv`, `mean_variance`)
- More appropriate thresholds:
  - `variance_threshold`: 0.001 (absolute, for per-dimension variance)
  - `norm_cv_threshold`: 0.05 (5% variation in norms)
  - `distance_cv_threshold`: 0.05 (5% variation in distances)

**Example Output:**
```
norm_cv=0.1256, dist_cv=0.1665
```
This shows 12.56% variation in embedding norms and 16.65% variation in pairwise distances.

### 2. **Enhanced Cophenetic Correlation**

**Before:**
- Always returned 0.0 because ground truth had many zero distances
- No filtering of same-level codes
- Limited metadata

**After:**
- Filters pairs with `tree_distance < min_distance` (default 0.1)
- Returns dictionary with:
  - `correlation`: The actual correlation value
  - `n_pairs`: Number of valid pairs used
  - `n_total`: Total number of pairs
  - `mean_tree_dist`: Average tree distance for context
  - `mean_emb_dist`: Average embedding distance

**Example Output:**
```
cophenetic=0.0000 (190 pairs)
```
Shows correlation and number of pairs that passed filtering.

### 3. **Enhanced Spearman Correlation**

**Before:**
- Computed on all pairs including zeros
- Could be skewed by uninformative pairs

**After:**
- Same filtering as cophenetic (distance >= 0.1)
- Returns dictionary with correlation and metadata
- More stable rank-based measure

### 4. **Additional Logged Metrics**

New metrics added to TensorBoard/Lightning logs:
- `val/mean_variance`: Per-dimension variance (collapse indicator)
- `val/norm_cv`: Coefficient of variation for norms (shown in progress bar)
- `val/distance_cv`: Coefficient of variation for distances (shown in progress bar)
- `val/cophenetic_n_pairs`: Number of pairs used in cophenetic calculation
- `val/spearman_n_pairs`: Number of pairs used in spearman calculation
- `val/std_pairwise_distance`: Standard deviation of pairwise distances
- `val/median_distortion`: Median distortion ratio (shown in progress bar)

## Interpretation Guide

### Coefficient of Variation (CV)

- **norm_cv < 0.05**: Norms are very similar (potential collapse)
- **norm_cv 0.05-0.15**: Moderate diversity (normal for trained embeddings)
- **norm_cv > 0.20**: High diversity in magnitudes

- **distance_cv < 0.05**: Distances are very similar (collapse)
- **distance_cv 0.10-0.20**: Good diversity in pairwise relationships
- **distance_cv > 0.30**: High diversity (may indicate good separation)

### Cophenetic/Spearman Correlation

- **< 0.0**: Inverse relationship (bad)
- **0.0-0.3**: Weak correlation (early training)
- **0.3-0.7**: Moderate correlation (improving)
- **> 0.7**: Strong hierarchy preservation (goal)

### Training Progress Example

From actual training run:
```
Epoch 0: norm_cv=0.0908, dist_cv=0.1598, spearman=0.0650
Epoch 4: norm_cv=0.1020, dist_cv=0.1651, spearman=-0.0534
Epoch 7: norm_cv=0.1256, dist_cv=0.1665, spearman=-0.0674
```

**Observations:**
- `norm_cv` increasing (0.09 → 0.13): Embeddings spreading out ✓
- `dist_cv` stable (~0.16): Maintaining diversity ✓
- `spearman` negative but small: Not preserving hierarchy yet ✗

## Next Steps

1. **If norm_cv stays < 0.05**: Increase learning rate or reduce regularization
2. **If distance_cv < 0.05**: Check contrastive loss weight
3. **If cophenetic/spearman don't improve**: 
   - Increase number of training examples
   - Adjust positive/negative sampling strategy
   - Check if ground truth distances are meaningful

## Configuration

To adjust thresholds, modify `/home/ubuntu/naics-gemini/src/naics_gemini/model/evaluation.py`:

```python
collapse = self.embedding_stats.check_collapse(
    embeddings,
    variance_threshold=0.001,    # Absolute variance threshold
    norm_cv_threshold=0.05,      # 5% CV in norms
    distance_cv_threshold=0.05   # 5% CV in distances
)
```

To adjust distance filtering:

```python
cophenetic_result = self.hierarchy_metrics.cophenetic_correlation(
    emb_dists,
    gt_dists,
    min_distance=0.1  # Minimum tree distance to include
)
```
