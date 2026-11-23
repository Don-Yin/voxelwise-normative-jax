"""Generic spline designs and held-out scores without imaging inputs."""
import numpy as np
import pytest

from shared.voxelwisenormative.basis import Basis
from shared.voxelwisenormative.batched_blr import design, fit_chunks, make_fitter
from shared.voxelwisenormative.cohort import standardise
from shared.voxelwisenormative.model import fit_score


def test_spline_column_linear_terms_and_learned_batches():
    x = np.array([[5, -2, 11], [8, -1, 20], [4, 1, 9], [10, 2, 15]], dtype=float)
    basis = Basis.fit(x, [72, "second", "second", 72], spline_column=1)
    assert basis.batch_levels == (72, "second")
    query = np.array([[3, 4, 17], [9, -4, 12]])
    phi = basis.apply(query, ["second", 72])
    assert phi.shape == (2, 12)
    spline_only = Basis.fit(x[:, 1:2]).apply(query[:, 1:2])
    np.testing.assert_allclose(phi[:, :7], spline_only[:, :7], atol=1e-12)
    np.testing.assert_allclose(phi[:, :7].sum(axis=1), 1, atol=1e-12)
    linear = [0, 2]
    np.testing.assert_allclose(phi[:, 7:9], (query[:, linear] - x.mean(0)[linear]) / x.std(0)[linear])
    np.testing.assert_array_equal(phi[:, 9:], [[1, 0, 1], [1, 1, 0]])


def test_declared_batches_keep_order_and_can_include_an_unobserved_level():
    x = np.arange(4)[:, None]
    basis = Basis.fit(x, ["c", "a", "a", "c"], batch_levels=("a", "b", "c"))
    np.testing.assert_array_equal(basis.apply(x[:2], ["b", "c"])[:, -3:], [[0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="unknown batch label 'd'"):
        basis.apply(x[:1], ["d"])


def test_fit_score_matches_the_ungrouped_pipeline_across_response_chunks():
    rng = np.random.default_rng(17)
    x = np.linspace(3, 9, 40)[:, None]
    y = x * np.array([0.2, -0.4, 0.1]) + rng.standard_normal((40, 3))
    held = np.arange(40) % 5 == 0
    basis = Basis.fit(x[~held])
    phi = basis.apply(x)
    assert phi.shape == (40, 8)
    np.testing.assert_array_equal(phi[:, -1], 1)
    train, test = standardise(y, held)
    _, _, expected = fit_chunks(make_fitter(), design(phi[~held], phi[held]), train, test, chunk=2)
    actual = fit_score(x, y, held, chunk=2)
    assert actual.shape == (8, 3) and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)


def test_fit_score_rejects_an_integer_mask_and_constant_training_responses():
    x = np.arange(4)[:, None]
    with pytest.raises(ValueError, match="row-aligned boolean mask"):
        fit_score(x, x, [0, 0, 0, 1])
    with pytest.raises(ValueError, match="positive training variance"):
        fit_score(x, np.zeros((4, 1)), [False, False, False, True])
