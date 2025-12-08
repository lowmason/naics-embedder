# -------------------------------------------------------------------------------------------------
# Tests for Phase 1 Difficulty Sampler
# -------------------------------------------------------------------------------------------------
"""
Tests for the difficulty curriculum-based negative sampling.

Tests cover:
1. Count guarantees - exactly n_select negatives returned
2. Annealing - ratios change correctly with epoch progress
3. Reproducibility - same seed produces identical results
4. Edge cases - empty buckets, insufficient candidates
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple

from naics_embedder.text_model.dataloader.difficulty_sampler import (
    select_by_difficulty,
    _bucket_candidates,
    _interpolate_ratios,
)
from naics_embedder.utils.config import StreamingConfig


# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_candidates() -> List[Dict]:
    """Create sample candidate negatives with various tree distances."""
    candidates = []
    # Easy negatives (d >= 6)
    for i in range(20):
        candidates.append({
            'negative_idx': i,
            'negative_code': f'easy_{i}',
            'relation_margin': 0.1,
            'distance_margin': 0.1,
        })
    # Semi-hard negatives (d = 4-5)
    for i in range(15):
        candidates.append({
            'negative_idx': 100 + i,
            'negative_code': f'semi_{i}',
            'relation_margin': 0.2,
            'distance_margin': 0.2,
        })
    # Hard negatives (d = 3)
    for i in range(10):
        candidates.append({
            'negative_idx': 200 + i,
            'negative_code': f'hard_{i}',
            'relation_margin': 0.3,
            'distance_margin': 0.3,
        })
    return candidates


@pytest.fixture
def distance_lookup() -> Dict[Tuple[str, str], float]:
    """Create distance lookup matching the sample candidates."""
    lookup = {}
    anchor = 'anchor_code'
    # Easy distances (d >= 6)
    for i in range(20):
        lookup[(anchor, f'easy_{i}')] = 6.0 + (i % 4)  # 6, 7, 8, 9
    # Semi-hard distances (d = 4-5)
    for i in range(15):
        lookup[(anchor, f'semi_{i}')] = 4.0 + (i % 2)  # 4, 5
    # Hard distances (d = 3)
    for i in range(10):
        lookup[(anchor, f'hard_{i}')] = 3.0
    return lookup


@pytest.fixture
def default_config() -> StreamingConfig:
    """Create default streaming config."""
    return StreamingConfig(
        phase1_easy_start=0.70,
        phase1_easy_end=0.20,
        phase1_semi_start=0.20,
        phase1_semi_end=0.40,
    )


# -------------------------------------------------------------------------------------------------
# Test: Count Guarantees
# -------------------------------------------------------------------------------------------------

class TestCountGuarantees:
    """Tests that exactly n_select negatives are returned."""

    @pytest.mark.unit
    def test_exact_count_returned(
        self, sample_candidates, distance_lookup, default_config
    ):
        """Verify exactly n_select negatives are returned."""
        rng = np.random.default_rng(42)
        n_select = 24

        result = select_by_difficulty(
            candidates=sample_candidates,
            n_select=n_select,
            distance_lookup=distance_lookup,
            anchor_code='anchor_code',
            epoch_progress=0.5,
            cfg=default_config,
            rng=rng,
        )

        assert len(result) == n_select

    @pytest.mark.unit
    def test_exact_count_at_start(
        self, sample_candidates, distance_lookup, default_config
    ):
        """Verify exact count at epoch start (easy-heavy)."""
        rng = np.random.default_rng(42)
        n_select = 20

        result = select_by_difficulty(
            candidates=sample_candidates,
            n_select=n_select,
            distance_lookup=distance_lookup,
            anchor_code='anchor_code',
            epoch_progress=0.0,
            cfg=default_config,
            rng=rng,
        )

        assert len(result) == n_select

    @pytest.mark.unit
    def test_exact_count_at_end(
        self, sample_candidates, distance_lookup, default_config
    ):
        """Verify exact count at epoch end (hard-heavy)."""
        rng = np.random.default_rng(42)
        n_select = 20

        result = select_by_difficulty(
            candidates=sample_candidates,
            n_select=n_select,
            distance_lookup=distance_lookup,
            anchor_code='anchor_code',
            epoch_progress=1.0,
            cfg=default_config,
            rng=rng,
        )

        assert len(result) == n_select

    @pytest.mark.unit
    def test_handles_n_select_larger_than_candidates(
        self, sample_candidates, distance_lookup, default_config
    ):
        """When n_select > len(candidates), return all available."""
        rng = np.random.default_rng(42)
        n_select = 100  # More than 45 candidates

        result = select_by_difficulty(
            candidates=sample_candidates,
            n_select=n_select,
            distance_lookup=distance_lookup,
            anchor_code='anchor_code',
            epoch_progress=0.5,
            cfg=default_config,
            rng=rng,
        )

        # Should return all available (with warning logged)
        assert len(result) <= len(sample_candidates)


# -------------------------------------------------------------------------------------------------
# Test: Annealing
# -------------------------------------------------------------------------------------------------

class TestAnnealing:
    """Tests that difficulty ratios anneal correctly."""

    @pytest.mark.unit
    def test_interpolate_ratios_at_start(self, default_config):
        """At epoch_progress=0, ratios should be start values."""
        easy, semi, hard = _interpolate_ratios(0.0, default_config)

        assert abs(easy - 0.70) < 1e-6
        assert abs(semi - 0.20) < 1e-6
        assert abs(hard - 0.10) < 1e-6
        assert abs(easy + semi + hard - 1.0) < 1e-6

    @pytest.mark.unit
    def test_interpolate_ratios_at_end(self, default_config):
        """At epoch_progress=1, ratios should be end values."""
        easy, semi, hard = _interpolate_ratios(1.0, default_config)

        assert abs(easy - 0.20) < 1e-6
        assert abs(semi - 0.40) < 1e-6
        assert abs(hard - 0.40) < 1e-6
        assert abs(easy + semi + hard - 1.0) < 1e-6

    @pytest.mark.unit
    def test_interpolate_ratios_at_midpoint(self, default_config):
        """At epoch_progress=0.5, ratios should be midpoint values."""
        easy, semi, hard = _interpolate_ratios(0.5, default_config)

        expected_easy = 0.70 + 0.5 * (0.20 - 0.70)  # 0.45
        expected_semi = 0.20 + 0.5 * (0.40 - 0.20)  # 0.30
        expected_hard = 1.0 - expected_easy - expected_semi  # 0.25

        assert abs(easy - expected_easy) < 1e-6
        assert abs(semi - expected_semi) < 1e-6
        assert abs(hard - expected_hard) < 1e-6
        assert abs(easy + semi + hard - 1.0) < 1e-6

    @pytest.mark.unit
    def test_ratios_always_sum_to_one(self, default_config):
        """Ratios should sum to 1.0 at any progress point."""
        for progress in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            easy, semi, hard = _interpolate_ratios(progress, default_config)
            assert abs(easy + semi + hard - 1.0) < 1e-6

    @pytest.mark.unit
    def test_more_easy_at_start(
        self, sample_candidates, distance_lookup, default_config
    ):
        """Early epochs should have more easy negatives."""
        rng = np.random.default_rng(42)
        n_select = 30

        result = select_by_difficulty(
            candidates=sample_candidates,
            n_select=n_select,
            distance_lookup=distance_lookup,
            anchor_code='anchor_code',
            epoch_progress=0.0,
            cfg=default_config,
            rng=rng,
        )

        # Count easy negatives (codes starting with 'easy_')
        easy_count = sum(1 for r in result if r['negative_code'].startswith('easy_'))

        # At start, 70% should be easy -> ~21 out of 30
        assert easy_count >= 15  # At least half should be easy

    @pytest.mark.unit
    def test_more_hard_at_end(
        self, sample_candidates, distance_lookup, default_config
    ):
        """Late epochs should have more hard negatives."""
        rng = np.random.default_rng(42)
        n_select = 30

        result = select_by_difficulty(
            candidates=sample_candidates,
            n_select=n_select,
            distance_lookup=distance_lookup,
            anchor_code='anchor_code',
            epoch_progress=1.0,
            cfg=default_config,
            rng=rng,
        )

        # Count hard negatives (codes starting with 'hard_')
        hard_count = sum(1 for r in result if r['negative_code'].startswith('hard_'))

        # At end, 40% should be hard -> ~12 out of 30
        # But we only have 10 hard candidates, so expect all of them
        assert hard_count >= 8


# -------------------------------------------------------------------------------------------------
# Test: Reproducibility
# -------------------------------------------------------------------------------------------------

class TestReproducibility:
    """Tests that same seed produces identical results."""

    @pytest.mark.unit
    def test_same_seed_same_result(
        self, sample_candidates, distance_lookup, default_config
    ):
        """Same seed should produce identical selections."""
        seed = 12345
        n_select = 20

        rng1 = np.random.default_rng(seed)
        result1 = select_by_difficulty(
            candidates=sample_candidates,
            n_select=n_select,
            distance_lookup=distance_lookup,
            anchor_code='anchor_code',
            epoch_progress=0.5,
            cfg=default_config,
            rng=rng1,
        )

        rng2 = np.random.default_rng(seed)
        result2 = select_by_difficulty(
            candidates=sample_candidates,
            n_select=n_select,
            distance_lookup=distance_lookup,
            anchor_code='anchor_code',
            epoch_progress=0.5,
            cfg=default_config,
            rng=rng2,
        )

        codes1 = [r['negative_code'] for r in result1]
        codes2 = [r['negative_code'] for r in result2]

        assert codes1 == codes2

    @pytest.mark.unit
    def test_different_seed_different_result(
        self, sample_candidates, distance_lookup, default_config
    ):
        """Different seeds should produce different selections."""
        n_select = 20

        rng1 = np.random.default_rng(111)
        result1 = select_by_difficulty(
            candidates=sample_candidates,
            n_select=n_select,
            distance_lookup=distance_lookup,
            anchor_code='anchor_code',
            epoch_progress=0.5,
            cfg=default_config,
            rng=rng1,
        )

        rng2 = np.random.default_rng(222)
        result2 = select_by_difficulty(
            candidates=sample_candidates,
            n_select=n_select,
            distance_lookup=distance_lookup,
            anchor_code='anchor_code',
            epoch_progress=0.5,
            cfg=default_config,
            rng=rng2,
        )

        codes1 = [r['negative_code'] for r in result1]
        codes2 = [r['negative_code'] for r in result2]

        # Very unlikely to be identical with different seeds
        assert codes1 != codes2


# -------------------------------------------------------------------------------------------------
# Test: Edge Cases
# -------------------------------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_empty_candidates(self, distance_lookup, default_config):
        """Empty candidates should return empty list."""
        rng = np.random.default_rng(42)

        result = select_by_difficulty(
            candidates=[],
            n_select=10,
            distance_lookup=distance_lookup,
            anchor_code='anchor_code',
            epoch_progress=0.5,
            cfg=default_config,
            rng=rng,
        )

        assert result == []

    @pytest.mark.unit
    def test_zero_n_select(
        self, sample_candidates, distance_lookup, default_config
    ):
        """n_select=0 should return empty list."""
        rng = np.random.default_rng(42)

        result = select_by_difficulty(
            candidates=sample_candidates,
            n_select=0,
            distance_lookup=distance_lookup,
            anchor_code='anchor_code',
            epoch_progress=0.5,
            cfg=default_config,
            rng=rng,
        )

        assert result == []

    @pytest.mark.unit
    def test_bucket_with_no_candidates(self, distance_lookup, default_config):
        """Handles buckets with no candidates gracefully."""
        # Only easy candidates
        easy_only = [
            {'negative_idx': i, 'negative_code': f'easy_{i}'}
            for i in range(10)
        ]

        rng = np.random.default_rng(42)

        result = select_by_difficulty(
            candidates=easy_only,
            n_select=8,
            distance_lookup=distance_lookup,
            anchor_code='anchor_code',
            epoch_progress=1.0,  # Would want 40% hard, but none available
            cfg=default_config,
            rng=rng,
        )

        # Should still return 8, filling from available
        assert len(result) == 8

    @pytest.mark.unit
    def test_missing_distance_uses_default(self, default_config):
        """Missing distances default to 12.0 (easy)."""
        candidates = [
            {'negative_idx': 1, 'negative_code': 'unknown_1'},
            {'negative_idx': 2, 'negative_code': 'unknown_2'},
        ]
        empty_lookup: Dict[Tuple[str, str], float] = {}

        easy, semi, hard = _bucket_candidates(candidates, empty_lookup, 'anchor')

        # All should be bucketed as easy (default distance 12.0 >= 6)
        assert len(easy) == 2
        assert len(semi) == 0
        assert len(hard) == 0


# -------------------------------------------------------------------------------------------------
# Test: Config Validation
# -------------------------------------------------------------------------------------------------

class TestConfigValidation:
    """Tests for config validation."""

    @pytest.mark.unit
    def test_valid_config_ratios(self):
        """Valid ratio configs should pass validation."""
        cfg = StreamingConfig(
            phase1_easy_start=0.60,
            phase1_easy_end=0.30,
            phase1_semi_start=0.30,
            phase1_semi_end=0.40,
        )
        # Should not raise
        assert cfg.phase1_easy_start == 0.60

    @pytest.mark.unit
    def test_invalid_start_ratios_sum(self):
        """Start ratios summing > 1.0 should fail validation."""
        with pytest.raises(ValueError, match='phase1_easy_start.*phase1_semi_start.*<= 1.0'):
            StreamingConfig(
                phase1_easy_start=0.70,
                phase1_semi_start=0.40,  # Sum = 1.1 > 1.0
            )

    @pytest.mark.unit
    def test_invalid_end_ratios_sum(self):
        """End ratios summing > 1.0 should fail validation."""
        with pytest.raises(ValueError, match='phase1_easy_end.*phase1_semi_end.*<= 1.0'):
            StreamingConfig(
                phase1_easy_end=0.50,
                phase1_semi_end=0.60,  # Sum = 1.1 > 1.0
            )

    @pytest.mark.unit
    def test_n_negatives_phase1_cannot_exceed_candidates(self):
        """n_negatives_phase1 > n_candidates should fail validation."""
        with pytest.raises(ValueError, match='n_negatives_phase1.*n_candidates'):
            StreamingConfig(
                n_candidates=24,
                n_negatives_phase1=48,  # More than candidates
            )

