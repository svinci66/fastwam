"""Statistical and weighting failure cases for the fixed offline diagnostic."""
import numpy as np
import torch

from experiments.robotwin.diagnose_natural60_reward import auc, donor_scores, unit_deltas, bootstrap_metrics
from experiments.robotwin.diagnose_awr_transmission import diagnostic_weights


def test_auc_orientation_and_ties():
    assert auc([1, 1, 0, 0], [2, 2, 1, 1]) == 1
    assert auc([1, 0], [2, 2]) == .5
    assert auc([1, 0], [0, 1]) == 0


def test_bootstrap_donor_excludes_all_copies_of_target():
    matrix = np.array([[100, 1, 3], [2, 200, 4], [5, 6, 300]], float)
    np.testing.assert_allclose(donor_scores(matrix), [2, 3, 5.5])
    np.testing.assert_allclose(donor_scores(matrix, [0, 0, 1, 2]), [2, 2, 8/3, 16/3])


def test_delta_cosine_does_not_reward_magnitude_accuracy():
    frames = np.arange(7*4, dtype=float).reshape(7, 4)
    a, am = unit_deltas(frames)
    b, bm = unit_deltas(20 + 3*frames)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(bm, 3*am, rtol=1e-6)


def test_return_gain_can_disappear_after_critic_adaptation():
    cfg = dict(beta=1., max_advantage_weight=20., normalize_advantage_weights=True)
    g = np.array([1., 0., 0.], np.float32)
    base = diagnostic_weights(g, np.zeros(3,np.float32), [[0, 1]], cfg)
    shifted = diagnostic_weights(g+2, np.ones(3,np.float32)*2, [[0, 1]], cfg)
    np.testing.assert_allclose(base[0], shifted[0])
    assert base[2].tolist() == [1, 1, 0]
    assert base[0][2] == 0


def test_macro_is_task_equal_and_donor_bootstrap_reproducible():
    d = {}
    for name, labels in [("a",[0,0,1,1]),("b",[0,1,1,1])]:
        values = np.array(labels,float)
        matrix = np.tile(values[:,None], (1,4))
        d[name] = dict(labels=labels, matrix=matrix, **{k:values for k in ["matched","time_reversed","latent_motion","action_variation","action_magnitude","negative_duration"]})
    a = bootstrap_metrics(d, 30)
    assert a == bootstrap_metrics(d,30)
    assert a["macro"]["matched"]["estimate"] == 1
    assert a["macro"]["content_delta"]["estimate"] == 0
