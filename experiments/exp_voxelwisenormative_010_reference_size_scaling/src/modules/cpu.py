"""Measure the converged PCNtoolkit fit on nested voxel samples, one process per width."""

from __future__ import annotations

import logging
import time
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np

from shared.voxelwisenormative.basis import covariate_rows
from src.utils.measure import peak_rss_bytes, publish, runtime

from .cohorts import Cohort, load_cohort, mask_sample
from .grid import COLUMNS_SEED, MASK_VOXELS, Cell

logger = logging.getLogger(__name__)
VERSION = "1.4.0"
EPSILON = 1e-7
RSS_RETAINED_ARRAYS = 9  # train: Z, five centiles, logp, baseline logp, Yhat


def check_design(model, x, site, expected: np.ndarray) -> None:
    """Verify the 12 columns modulo a harmless permutation of the three site indicators."""
    actual, _ = model.Phi_Phi_var(x.values, site.values)
    if actual.shape != expected.shape:
        raise ValueError(f"PCNtoolkit design mismatch: {actual.shape} != {expected.shape}")
    np.testing.assert_allclose(actual[:, :-3], expected[:, :-3], rtol=0, atol=3e-12)
    actual_categories = {tuple(col) for col in actual[:, -3:].T}
    if actual_categories != {tuple(col) for col in expected[:, -3:].T}:
        raise ValueError(f"PCNtoolkit site indicators differ at shape {actual.shape}")


def worker(cell: Cell, width: int, out: Path, cohort: Cohort | None = None) -> Path:
    """Fit a width as one NormativeModel and save only fit metrics and finite checks."""
    logger.info("start pcntoolkit %s width=%d output=%s", cell.label, width, out)
    start = time.perf_counter()
    pcn = import_module("pcntoolkit")
    output = import_module("pcntoolkit.util.output").Output
    if version("pcntoolkit") != VERSION:
        raise ValueError(f"expected pcntoolkit {VERSION}, found {version('pcntoolkit')}")
    import_seconds = time.perf_counter() - start
    pre_input_peak = peak_rss_bytes()
    start = time.perf_counter()
    if cohort is None:
        cohort = load_cohort(cell)
    voxels = mask_sample(width)
    names = [f"v{v}" for v in voxels]
    attrs = {"covariates": ["age", "sex"], "batch_effect_dims": ["site"], "response_vars": names}
    x = np.column_stack((cohort.cov["age"], cohort.cov["sex"] == "M"))
    raw = cohort.responses(cohort.train, voxels)
    train, test = (
        pcn.NormData.from_ndarrays(tag, x[rows], y.astype(np.float64), cohort.cov["site"][rows, None],
                                  attrs=attrs)
        for tag, rows, y in zip(("train", "test"), (cohort.train, cohort.test), raw, strict=True)
    )
    del raw
    basis = cohort.basis()
    phi_train = basis.apply(*covariate_rows(cohort.cov, cohort.train))
    phi_test = basis.apply(*covariate_rows(cohort.cov, cohort.test))
    template = pcn.BLR(name="blr", fixed_effect=True, warp_name="WarpSinhArcsinh",
                       l_bfgs_b_epsilon=EPSILON, optimizer="l-bfgs-b",
                       basis_function_mean=pcn.BsplineBasisFunction(basis_column=0, degree=3, nknots=5))
    work = out.parents[3].parent / ".intermediates" / "pcntoolkit" / out.parent.name / out.stem
    model = pcn.NormativeModel(template, savemodel=False, evaluate_model=False, saveresults=False,
                               saveplots=False, save_dir=str(work), name=cell.label)
    setup_seconds = time.perf_counter() - start
    pre_fit_peak = peak_rss_bytes()
    blr = pcn.BLR
    orig_fit, orig_forward = blr.fit, blr.forward
    design_checks = set()
    heldout_phase = False

    def first_fit(self, x, site, maps, y):
        """Check the first fit's design once, not all voxels in the timed loop."""
        result = orig_fit(self, x, site, maps, y)
        if "train" not in design_checks:
            check_design(self, x, site, phi_train)
            design_checks.add("train")
        return result

    def first_prediction(self, x, site, y):
        """Check the held-out design once, outside the training prediction pass."""
        if heldout_phase and "test" not in design_checks:
            check_design(self, x, site, phi_test)
            design_checks.add("test")
        return orig_forward(self, x, site, y)

    previous = output.get_show_messages()
    output.set_show_messages(False)
    try:
        with patch.object(blr, "fit", first_fit), patch.object(blr, "forward", first_prediction):
            start = time.perf_counter()
            model.fit(train)
            fit_seconds = time.perf_counter() - start
            heldout_phase = True
            start = time.perf_counter()
            model.predict(test)
            predict_seconds = time.perf_counter() - start
            peak = peak_rss_bytes()
    finally:
        output.set_show_messages(previous)
    scores = test["Z"].sel(response_vars=names).values
    expected_shape = (len(cohort.test), width)
    if scores.shape != expected_shape or not np.isfinite(scores).all() or design_checks != {"train", "test"}:
        raise ValueError(f"invalid {cell.label} results: {scores.shape=}, expected finite {expected_shape}; {design_checks=}")
    error = cohort.reference.max_delta(voxels, scores) if cohort.reference is not None else None
    if error is not None and error > 1e-3:
        raise ValueError(f"PCNtoolkit at {cell.label}, width {width}, max |Δz| {error:g} exceeds 0.001")
    logger.info("pcntoolkit %s width=%d fit=%.3fs predict=%.3fs peak=%.1f mib",
                cell.label, width, fit_seconds, predict_seconds, peak / 2**20)
    return publish(out, {"schema_version": 1, "workload": "pcntoolkit", "dataset": cell.dataset,
                         "n": cell.n, "n_test": len(cohort.test), "columns": width, "mask_voxels": MASK_VOXELS,
                         "column_seed": COLUMNS_SEED, "pcntoolkit": VERSION, "epsilon": EPSILON,
                         "import_seconds": import_seconds, "setup_seconds": setup_seconds,
                         "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                         "seconds": fit_seconds + predict_seconds, "pre_input_peak_rss_bytes": pre_input_peak,
                         "pre_fit_peak_rss_bytes": pre_fit_peak, "peak_rss_bytes": peak,
                         "retained_training_arrays_bytes": RSS_RETAINED_ARRAYS * cell.n * 8 * width,
                         "design_columns": 12, "pinned_z_max_abs_delta": error, **runtime()})
