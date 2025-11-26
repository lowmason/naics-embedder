'''
Unit tests for streaming dataset Phase 1 sampling with explicit exclusions.
'''

from naics_embedder.text_model.dataloader.streaming_dataset import (
    _compute_phase1_weights,
    _sample_negatives_phase1,
    _sample_negatives_sans_static,
)
from naics_embedder.utils.config import SansStaticConfig

def test_excluded_negatives_get_high_weight():
    '''Excluded negatives should be prioritized and flagged.'''

    anchor = '111'
    candidates = [
        {
            'negative_code': '222',
            'negative_idx': 0,
            'relation_margin': 0,
            'distance_margin': 4
        },
        {
            'negative_code': '333',
            'negative_idx': 1,
            'relation_margin': 0,
            'distance_margin': 6
        },
    ]
    distance_lookup = {
        (anchor, '222'): 4.0,
        (anchor, '333'): 6.0,
    }
    excluded_map = {anchor: {'333'}}
    code_to_idx = {'111': 0, '222': 1, '333': 2}

    weights, mask = _compute_phase1_weights(
        anchor_code=anchor,
        anchor_idx=0,
        candidate_negatives=candidates,
        distance_lookup=distance_lookup,
        excluded_map=excluded_map,
        code_to_idx=code_to_idx,
        alpha=1.5,
        exclusion_weight=100.0,
    )

    assert mask.tolist() == [False, True]
    assert weights[1] > weights[0]

def test_sample_marks_explicit_exclusions(monkeypatch):
    '''Sampled negatives should carry explicit_exclusion flag.'''

    anchor = '111'
    candidates = [
        {
            'negative_code': '222',
            'negative_idx': 0,
            'relation_margin': 0,
            'distance_margin': 4
        },
        {
            'negative_code': '333',
            'negative_idx': 1,
            'relation_margin': 0,
            'distance_margin': 6
        },
    ]
    distance_lookup = {
        (anchor, '222'): 4.0,
        (anchor, '333'): 6.0,
    }
    excluded_map = {anchor: {'333'}}
    code_to_idx = {'111': 0, '222': 1, '333': 2}

    # Fix RNG for deterministic sampling
    monkeypatch.setenv('PYTHONHASHSEED', '0')

    sampled = _sample_negatives_phase1(
        anchor_code=anchor,
        anchor_idx=0,
        candidate_negatives=candidates,
        n_negatives=2,
        distance_lookup=distance_lookup,
        excluded_map=excluded_map,
        code_to_idx=code_to_idx,
        alpha=1.5,
        exclusion_weight=100.0,
        seed=0,
    )

    assert len(sampled) == 2
    # Ensure the excluded negative is present and flagged
    excluded_flags = [neg['explicit_exclusion'] for neg in sampled]
    assert any(excluded_flags)

def test_sans_static_sampling_prefers_near_bucket():
    '''SANS static sampling should bias probabilities toward near negatives.'''

    anchor = '111'
    candidates = [
        {
            'negative_code': '1111',
            'negative_idx': 0,
            'relation_margin': 0,
            'distance_margin': 1,
        },
        {
            'negative_code': '2111',
            'negative_idx': 1,
            'relation_margin': 0,
            'distance_margin': 6,
        },
        {
            'negative_code': '3111',
            'negative_idx': 2,
            'relation_margin': 0,
            'distance_margin': 7,
        },
    ]
    distance_lookup = {
        (anchor, '1111'): 1.0,
        (anchor, '2111'): 6.0,
        (anchor, '3111'): 7.0,
    }

    sans_cfg = SansStaticConfig(
        near_distance_threshold=4.0,
        near_bucket_weight=0.8,
        far_bucket_weight=0.2,
    )

    sampled, metadata = _sample_negatives_sans_static(
        anchor_code=anchor,
        candidate_negatives=candidates,
        n_negatives=2,
        distance_lookup=distance_lookup,
        sans_cfg=sans_cfg,
        seed=0,
    )

    assert len(sampled) == 2
    assert metadata['candidates_near'] == 1
    assert metadata['candidates_far'] == 2
    assert metadata['effective_near_weight'] > metadata['effective_far_weight']
    assert metadata['sampled_near'] + metadata['sampled_far'] == len(sampled)
