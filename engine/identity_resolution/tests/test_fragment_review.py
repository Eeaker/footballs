"""Safety checks for review-only occluded-fragment proposals."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.review_fragments import decide


def candidate(target, affinity, residual):
    return dict(target=target, affinity=affinity,
                bracket=dict(residual=residual), fused_score=.95)


def test_single_frame_cannot_produce_appearance_training_label():
    result = decide([candidate(7, .99, .1)], 1)
    assert result['interpolation'] == 7
    assert result['reid'] is None and result['fusion'] is None
    assert result['automatic_training_label'] is False


def test_ambiguous_candidates_are_rejected():
    result = decide([candidate(7, .96, .2), candidate(9, .95, .3)], 3)
    assert all(result[k] is None for k in ('interpolation', 'reid', 'fusion'))


def test_motion_cannot_override_bad_appearance_in_fusion():
    result = decide([candidate(5, .42, .55)], 6)
    assert result['interpolation'] == 5
    assert result['fusion'] is None and result['reid'] is None


def test_large_motion_error_rejects_fused_proposal():
    assert decide([candidate(5, .99, 3.)], 6)['fusion'] is None
