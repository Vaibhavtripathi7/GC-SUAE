import numpy as np

from utils.evaluation import nearest_endmember_sam_deg, spatial_coherence


def test_spatial_coherence_uniform_and_checkerboard():
    stride = 32
    coords = [(r * stride, c * stride) for r in range(4) for c in range(4)]
    uniform = np.zeros(16, dtype=int)
    assert spatial_coherence(uniform, coords, stride) == 1.0
    checker = np.array([(r + c) % 2 for r in range(4) for c in range(4)])
    assert spatial_coherence(checker, coords, stride) == 0.0


def test_nearest_endmember_sam_identity_and_scale_invariance():
    rng = np.random.default_rng(0)
    em = rng.random((6, 86))
    sams = nearest_endmember_sam_deg(em * 3.0, em)
    assert np.allclose(sams, 0.0, atol=0.05)  # arccos near 1 is numerically soft
    other = rng.random((3, 86))
    sams = nearest_endmember_sam_deg(other, em)
    assert sams.shape == (3,) and (sams > 0).all() and (sams < 90).all()
