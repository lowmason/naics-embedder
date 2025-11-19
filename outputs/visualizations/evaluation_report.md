# Training Stages Evaluation Report

## Executive Summary

This report provides a comprehensive evaluation of the first three training stages: 01_text, 02_text, and 03_text.

## Per-Stage Analysis

### 01_text

**Configuration:**
- Positive relations: [1]
- Positive levels: None
- N positives: 512
- N negatives: 24

**Metrics Summary:**
- Epochs completed: 8
- Final epoch: 7


**Hierarchy Preservation:**
- Cophenetic correlation: 0.2237 → 0.2775

**Hyperbolic Radius:**
- Mean: 1.2372 → 10.2730
- Std: 0.0099 → 3.6780

**Collapse Detection:** 0 epochs with collapse detected

### 02_text

**Configuration:**
- Positive relations: [1, 2]
- Positive levels: None
- N positives: 256
- N negatives: 32

**Metrics Summary:**
- Epochs completed: 7
- Final epoch: 6


**Hierarchy Preservation:**
- Cophenetic correlation: 0.3050 → 0.2341

**Hyperbolic Radius:**
- Mean: 9.4280 → 14.8719
- Std: 4.4232 → 7.3046

**Collapse Detection:** 0 epochs with collapse detected

### 03_text

**Configuration:**
- Positive relations: [1, 2, 3, 4, 5, 6, 7]
- Positive levels: [4, 5, 6]
- N positives: 8
- N negatives: 24

**Metrics Summary:**
- Epochs completed: 6
- Final epoch: 5


**Hierarchy Preservation:**
- Cophenetic correlation: 0.2100 → 0.3191

**Hyperbolic Radius:**
- Mean: 11.0643 → 12.6289
- Std: 9.0349 → 5.9657

**Collapse Detection:** 0 epochs with collapse detected

## Comparative Analysis

### Loss Progression

Comparing training and validation loss across stages:

- **01_text**: 
- **02_text**: 
- **03_text**: 

### Hierarchy Preservation

- **01_text**: 0.2237 → 0.2775 (+0.0538)
- **02_text**: 0.3050 → 0.2341 (-0.0709)
- **03_text**: 0.2100 → 0.3191 (+0.1091)

## Key Findings

1. Training progression across stages
2. Loss trends and convergence
3. Hierarchy preservation improvements
4. Hyperbolic radius evolution

## Recommendations

Based on the analysis:
- Continue monitoring loss trends
- Track hierarchy preservation metrics
- Consider adjustments based on validation performance

