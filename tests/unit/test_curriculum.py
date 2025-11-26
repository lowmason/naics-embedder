'''
Unit tests for curriculum scheduler weighting logic.
'''

import pytest

from naics_embedder.text_model.curriculum import CurriculumScheduler
from naics_embedder.utils.config import AnnealConfig

def test_inverse_tree_distance_weighting_masks_siblings():
    '''Phase 1 weighting should mask siblings and favor closer non-siblings.'''

    scheduler = CurriculumScheduler(
        max_epochs=10, tree_distance_alpha=2.0, sibling_distance_threshold=2.0
    )

    anchor = '111'
    negatives = ['1111', '112', '120']
    tree_distances = {
        (anchor, '1111'): 1.0,  # ancestor/descendant
        (anchor, '112'): 2.0,  # sibling
        (anchor, '120'): 4.0,  # cousin
    }

    weights = scheduler.get_negative_sample_weights(
        anchor_code=anchor,
        negative_codes=negatives,
        tree_distances=tree_distances,
        use_tree_distance=True,
        mask_siblings=True,
    )

    assert len(weights) == 3
    # Sibling masked to zero
    assert weights[1] == 0.0
    # Remaining weights should sum to 1
    assert abs(sum(weights) - 1.0) < 1e-6
    # Closer ancestor/descendant gets higher probability than distant cousin
    assert weights[0] > weights[2]

def test_tree_distance_weights_uniform_when_missing_distances():
    '''When tree distances are unavailable, fallback to uniform weights.'''

    scheduler = CurriculumScheduler(max_epochs=5)

    negatives = ['a', 'b', 'c']
    weights = scheduler.get_negative_sample_weights(
        anchor_code='root',
        negative_codes=negatives,
        tree_distances=None,
        use_tree_distance=True,
        mask_siblings=True,
    )

    assert len(weights) == 3
    assert all(abs(w - 1 / 3) < 1e-6 for w in weights)

def test_tree_distance_uniform_when_all_masked():
    '''If all weights are masked to zero, fall back to uniform sampling.'''

    scheduler = CurriculumScheduler(max_epochs=5, sibling_distance_threshold=2.0)

    negatives = ['sib1', 'sib2']
    tree_distances = {
        ('root', 'sib1'): 2.0,
        ('root', 'sib2'): 2.0,
    }

    weights = scheduler.get_negative_sample_weights(
        anchor_code='root',
        negative_codes=negatives,
        tree_distances=tree_distances,
        use_tree_distance=True,
        mask_siblings=True,
    )

    assert len(weights) == 2
    assert all(abs(w - 0.5) < 1e-6 for w in weights)

def test_two_phase_mode_enables_clustering_during_phase2():
    scheduler = CurriculumScheduler(max_epochs=10, phase_mode='two_phase')

    assert scheduler.get_phase(8) == 2
    flags = scheduler.get_curriculum_flags(8)
    assert flags['enable_clustering']
    assert scheduler.should_update_clustering(current_epoch=7, cluster_every_n_epochs=2)

def test_anneal_schedule_interpolates_alpha_and_router_mix():
    anneal_cfg = AnnealConfig(
        enabled=True,
        alpha_start=2.0,
        alpha_end=1.0,
        epochs=10,
        metric_name='adaptive_margin_mean',
        metric_threshold=0.2,
        metric_direction='below',
        router_mix_start=0.1,
        router_mix_end=0.6,
    )
    scheduler = CurriculumScheduler(max_epochs=10, anneal_config=anneal_cfg)

    scalars_mid = scheduler.get_schedule_scalars(5)
    assert pytest.approx(1.5, rel=1e-3) == scalars_mid['tree_distance_alpha']
    assert pytest.approx(0.35, rel=1e-3) == scalars_mid['router_mix_ratio']

    scheduler.update_metrics({'adaptive_margin_mean': 0.1})
    scalars_post = scheduler.get_schedule_scalars(5)
    assert scalars_post['anneal_progress'] == pytest.approx(1.0)
    assert scalars_post['tree_distance_alpha'] == pytest.approx(1.0)
