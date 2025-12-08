# -------------------------------------------------------------------------------------------------
# Difficulty-Based Negative Sampler for Phase 1 Curriculum
# -------------------------------------------------------------------------------------------------
"""
Select negatives using a difficulty curriculum that anneals from easy to hard.

Buckets by tree distance:
- Easy: d >= 6 (distant codes, trivial to distinguish)
- Semi-hard: d = 4-5 (cousins/2nd cousins, moderate challenge)
- Hard: d = 3 (close relatives, challenging)
- Masked: d = 2 (siblings, excluded entirely by Phase 1 sampling)

The difficulty ratios interpolate linearly across Phase 1 epochs:
- Start: 70% easy, 20% semi-hard, 10% hard
- End: 20% easy, 40% semi-hard, 40% hard
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from naics_embedder.utils.config import StreamingConfig

logger = logging.getLogger(__name__)


def _bucket_candidates(
    candidates: List[Dict[str, Any]],
    distance_lookup: Dict[Tuple[str, str], float],
    anchor_code: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Bucket candidates by tree distance into easy, semi-hard, and hard.

    Args:
        candidates: List of candidate negative dictionaries
        distance_lookup: Mapping from (anchor_code, negative_code) to tree distance
        anchor_code: The anchor code string

    Returns:
        Tuple of (easy, semi_hard, hard) candidate lists
    """
    easy: List[Dict[str, Any]] = []
    semi_hard: List[Dict[str, Any]] = []
    hard: List[Dict[str, Any]] = []

    for c in candidates:
        negative_code = c.get('negative_code', '')
        dist = distance_lookup.get((anchor_code, negative_code), 12.0)

        if dist >= 6:
            easy.append(c)
        elif dist >= 4:  # d = 4 or 5
            semi_hard.append(c)
        elif dist == 3:
            hard.append(c)
        # d = 2 (siblings) should already be excluded by Phase 1 sampling

    return easy, semi_hard, hard


def _interpolate_ratios(
    epoch_progress: float,
    cfg: StreamingConfig,
) -> Tuple[float, float, float]:
    """
    Interpolate difficulty ratios based on epoch progress.

    Args:
        epoch_progress: Progress through Phase 1 (0.0 to 1.0)
        cfg: Streaming configuration with ratio parameters

    Returns:
        Tuple of (easy_ratio, semi_ratio, hard_ratio) summing to 1.0
    """
    t = epoch_progress

    easy_ratio = cfg.phase1_easy_start + t * (cfg.phase1_easy_end - cfg.phase1_easy_start)
    semi_ratio = cfg.phase1_semi_start + t * (cfg.phase1_semi_end - cfg.phase1_semi_start)
    hard_ratio = 1.0 - easy_ratio - semi_ratio  # Derived to guarantee sum = 1.0

    return easy_ratio, semi_ratio, hard_ratio


def _sample_from_bucket(
    bucket: List[Dict[str, Any]],
    n_want: int,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Sample from a bucket, returning the sampled items and any shortfall.

    Args:
        bucket: List of candidates to sample from
        n_want: Number of items to sample
        rng: Random number generator

    Returns:
        Tuple of (sampled_items, shortfall)
    """
    n_avail = len(bucket)
    if n_avail == 0:
        return [], n_want

    n_take = min(n_want, n_avail)
    indices = rng.choice(n_avail, n_take, replace=False)
    sampled = [bucket[i] for i in indices]

    return sampled, n_want - n_take


def select_by_difficulty(
    candidates: List[Dict[str, Any]],
    n_select: int,
    distance_lookup: Dict[Tuple[str, str], float],
    anchor_code: str,
    epoch_progress: float,
    cfg: StreamingConfig,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    Select exactly n_select negatives using the difficulty curriculum.

    The curriculum anneals from easy-heavy to hard-heavy across Phase 1 epochs.
    If a bucket doesn't have enough candidates, the shortfall is filled from
    subsequent buckets (semi -> hard -> remaining).

    Args:
        candidates: List of candidate negative dictionaries
        n_select: Exact number of negatives to select
        distance_lookup: Mapping from (anchor_code, negative_code) to tree distance
        anchor_code: The anchor code string
        epoch_progress: Progress through Phase 1 (0.0 to 1.0)
        cfg: Streaming configuration with ratio parameters
        rng: Random number generator for sampling

    Returns:
        List of exactly n_select negative dictionaries
    """
    if not candidates:
        return []

    if n_select <= 0:
        return []

    # Bucket candidates by difficulty
    easy, semi_hard, hard = _bucket_candidates(candidates, distance_lookup, anchor_code)

    # Get interpolated ratios
    easy_ratio, semi_ratio, hard_ratio = _interpolate_ratios(epoch_progress, cfg)

    # Compute counts ensuring exact sum via remainder assignment
    n_easy = round(n_select * easy_ratio)
    n_semi = round(n_select * semi_ratio)
    n_hard = n_select - n_easy - n_semi  # Remainder guarantees exact total

    # Sample from each bucket with shortfall cascade
    selected: List[Dict[str, Any]] = []

    picked_easy, shortfall = _sample_from_bucket(easy, n_easy, rng)
    selected.extend(picked_easy)

    picked_semi, shortfall = _sample_from_bucket(semi_hard, n_semi + shortfall, rng)
    selected.extend(picked_semi)

    picked_hard, shortfall = _sample_from_bucket(hard, n_hard + shortfall, rng)
    selected.extend(picked_hard)

    # If still short, sample from any remaining candidates
    if shortfall > 0:
        already_selected = set(id(c) for c in selected)
        remaining = [c for c in candidates if id(c) not in already_selected]

        if remaining:
            extra, _ = _sample_from_bucket(remaining, shortfall, rng)
            selected.extend(extra)

    # Final safety check
    if len(selected) < n_select:
        logger.warning(
            f'Could not select {n_select} negatives for anchor {anchor_code}, '
            f'only {len(selected)} available'
        )

    return selected

