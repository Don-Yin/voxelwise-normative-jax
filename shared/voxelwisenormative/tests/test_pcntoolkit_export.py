"""Verify that native PCNtoolkit predicts from the exported fit, not a refit."""

import json

import numpy as np
import pytest

from shared.voxelwisenormative.basis import Basis
from shared.voxelwisenormative.batched_blr import design, fit_chunks, make_fitter
from shared.voxelwisenormative.cohort import standardise
from shared.voxelwisenormative.pcntoolkit_export import export_pcntoolkit


@pytest.mark.parametrize("with_batch", [False, True])
def test_export_round_trip(tmp_path, with_batch):
    """Keep raw-data scaling, spline order, and fitted batch codes after moving the folder."""
    pcn = pytest.importorskip("pcntoolkit")
    rng = np.random.default_rng(197)
    spline = np.linspace(5, 20, 48)
    x = np.column_stack((rng.normal(size=spline.size), spline, rng.uniform(size=spline.size)))
    y = 9 + spline[:, None] * [0.3, -1.5] + rng.normal(size=(len(x), 2))
    held = np.arange(len(x)) % 5 == 0
    batch = np.resize(np.array([72, "second"], dtype=object), len(x)) if with_batch else None
    if batch is not None:
        batch[0] = "declared but absent from training"
    train, test = (~held, held)
    b, bt = (None, None) if batch is None else (batch[train], batch[test])
    levels = ("second", "declared but absent from training", 72) if with_batch else None
    basis = Basis.fit(x[train], b, spline_column=1, batch_levels=levels)
    phi = design(basis.apply(x[train], b), basis.apply(x[test], bt))
    hyp, _, z = fit_chunks(make_fitter(), phi, *standardise(y, held), chunk=1)

    covariates = ["linear", "spline", "secondary linear"]
    responses = ["second-response", "first-response"]
    folder = export_pcntoolkit(tmp_path / "original", basis, hyp, x[train], y[train], b,
                               covariate_names=covariates, response_names=responses, batch_name="group")
    moved = folder.rename(tmp_path / "moved")
    metadata = json.loads((moved / "model" / "normative_model.json").read_text())
    assert metadata["is_fitted"] is True
    train_mean, train_sd = y[train].mean(0), y[train].std(0)
    for i, name in enumerate(responses):
        record = json.loads((moved / "model" / name / "regression_model.json").read_text())
        np.testing.assert_array_equal(record["model"]["hyp"], hyp[i])
        assert record["outscaler"]["m"] == train_mean[i]
        assert record["outscaler"]["s"] == train_sd[i]

    restored = pcn.NormativeModel.load(str(moved))
    if levels is not None:
        assert restored.batch_effects_maps["group"] == {str(level): i for i, level in enumerate(levels)}
    attrs = {"covariates": covariates, "response_vars": responses}
    labels = None if bt is None else np.asarray(bt, dtype=str)[:, None]
    if labels is not None:
        attrs["batch_effect_dims"] = ["group"]
    query = pcn.NormData.from_ndarrays("test", x[test], y[test], labels, attrs=attrs)
    actual = restored.predict(query).Z.sel(response_vars=responses).values
    np.testing.assert_allclose(actual, z, atol=1e-10, rtol=0, equal_nan=False)


@pytest.mark.parametrize("name", ["../escape", ".hidden", "normative_model.json"])
def test_invalid_model_names_do_not_create_outputs(tmp_path, name):
    """Reject hidden or reserved response paths before touching the destination."""
    x = np.arange(8, dtype=float)[:, None]
    target = tmp_path / "model"
    with pytest.raises(ValueError, match="response directory"):
        export_pcntoolkit(target, Basis.fit(x), np.ones((1, 4)), x, x, response_names=[name])
    assert not target.exists()


def test_export_rejects_existing_destination_and_non_homoskedastic_state(tmp_path):
    """An existing model is never overwritten, and additional parameters are not ignored."""
    x = np.arange(8, dtype=float)[:, None]
    basis = Basis.fit(x)
    with pytest.raises(FileExistsError, match="new output directory"):
        export_pcntoolkit(tmp_path, basis, np.ones((1, 4)), x, x)
    with pytest.raises(ValueError, match="homoskedastic warped hyperparameters"):
        export_pcntoolkit(tmp_path / "not-supported", basis, np.ones((1, 6)), x, x)


@pytest.mark.parametrize(("batch", "pattern"), [([1, "1"] * 4, "levels collide"),
                                                 ([1, 1.0, "1.0", "1.0"] * 2, "batch label 1.0 converts")])
def test_export_rejects_batch_label_conversion_that_changes_assignments(tmp_path, batch, pattern):
    """Numeric and string labels must keep their assignments in PCNtoolkit."""
    x = np.arange(8, dtype=float)[:, None]
    with pytest.raises(ValueError, match=pattern):
        export_pcntoolkit(tmp_path / "bad", Basis.fit(x, batch), np.ones((1, 4)), x, x, batch)
