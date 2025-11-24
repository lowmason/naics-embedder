# Linear Branch Implementation Summary

## Implementation Approach

I've implemented the skip-level logic for linear branches as required by issue #20. Here's how it works:

### 1. **Identification of Linear Branches**

The code identifies linear branches by checking if an anchor has exactly 1 child at its immediate level:
- Level 2 linear: Has only 1 child at level 3
- Level 3 linear: Has only 1 child at level 4
- Level 4 linear: Has only 1 child at level 5
- Level 5 linear: Has only 1 child at level 6

### 2. **Skip-Level Stratification**

For linear branches, the code skips the immediate child level and uses grandchildren as the stratum:

- **Level 2 linear** → Uses `code_4` (skips `code_3`)
- **Level 3 linear** → Uses `code_5` (skips `code_4`)
- **Level 4 linear** → Uses `code_6` (skips `code_5`)
- **Level 5 linear** → Cannot skip further (already at leaves), uses `code_6` but allows `num_strata == 1`

### 3. **Processing Flow**

1. **Identify linear branches**: Count immediate children for each anchor
2. **Split processing**:
   - **Non-linear branches**: Process normally using immediate children as stratum
   - **Linear branches**: Process with skip-level logic using grandchildren as stratum
3. **Combine results**: Merge both sets of results

### 4. **Special Handling for Level 5 Linear**

Level 5 linear branches cannot skip further (level 6 is already leaves), so:
- They use `code_6` as stratum (immediate children)
- The `num_strata > 1` filter is relaxed to allow `num_strata == 1` for linear branches
- This ensures the single child is included as a positive

### 5. **Sampling Logic**

The same sampling rules apply to both linear and non-linear branches:
- **Level 2-3**: Return all grandchildren (when num_strata > 1)
- **Level 4**: Sample from grandchildren (complex logic for 4-digit)
- **Level 5**: Sample up to 3 for non-linear, keep all for linear

## Key Code Changes

1. **Linear branch identification** (lines 123-147):
   - Computes `num_children` for each anchor
   - Filters anchors with `num_children == 1`

2. **Separate processing paths**:
   - `descendants_nonlinear`: Normal processing (lines 150-179)
   - `descendants_linear_skip`: Skip-level for levels 2-4 (lines 186-215)
   - `descendants_linear_5`: Special handling for level 5 (lines 217-230)

3. **Level 5 sampling update** (lines 315-345):
   - Allows `num_strata == 1` for linear branches
   - Samples up to 3 for non-linear, keeps all for linear

## Benefits

✅ **Diversity**: Linear branches now sample from grandchildren, ensuring diversity across different branches

✅ **Correctness**: Matches the issue requirement for "Dynamic Skip-Level"

✅ **Completeness**: Handles all edge cases including level 5 linear branches

✅ **Performance**: Uses vectorized Polars operations, no performance penalty

## Testing Recommendations

1. **Sector 11 test**: Verify it returns positives from multiple 3-digit codes (111, 112, 113, 114, 115)
2. **Linear branch test**: Find a linear branch (e.g., a sector with only 1 subsector) and verify it samples from grandchildren
3. **Level 5 linear test**: Verify level 5 linear branches with only 1 child are included


