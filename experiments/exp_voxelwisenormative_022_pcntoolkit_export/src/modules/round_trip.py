"""Check JAX fits against a PCNtoolkit save, load and full predict on pinned IXI data."""
from __future__ import annotations

import json
import logging
import platform
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.artifacts import PCNTOOLKIT_REFERENCE, resolve
from shared.voxelwisenormative.basis import Basis
from shared.voxelwisenormative.batched_blr import design, fit_chunks, make_fitter
from shared.voxelwisenormative.cohort import (
    IXI_SITES,
    ixi_covariates,
    load_covariates,
    load_grey_matter,
    pinned_agreement,
    standardise,
)
from shared.voxelwisenormative.pcntoolkit_export import PCN_VERSION, export_pcntoolkit

logger = logging.getLogger(__name__)
N_VOXELS = 1000
ROUND_TRIP_TOLERANCE = 1e-10
FROZEN_REFERENCE_TOLERANCE = 2e-3  # separate optimizers; experiment 003 reported a gap of about 1.1e-3
SCALING_TOLERANCE = 1e-12


def _difference(actual: np.ndarray, expected: np.ndarray, label: str) -> dict[str, float]:
    """Summarize paired absolute differences without ignoring any scores."""
    if actual.shape != expected.shape or not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        raise ValueError(f"{label} scores differ in shape or are nonfinite: {actual.shape=}, {expected.shape=}")
    delta = np.abs(actual - expected)
    return {"median_abs_z": float(np.median(delta)), "p99_abs_z": float(np.quantile(delta, 0.99)),
            "max_abs_z": float(delta.max())}


def run(results: Path, artifacts_dir: Path | None) -> None:
    """Fit all frozen response columns, load the exported folder and verify held-out scores."""
    archive = resolve(PCNTOOLKIT_REFERENCE, "arrays.npz", root=artifacts_dir)["arrays.npz"]
    logger.info("[1/4] loading frozen ixi split and raw responses from %s", archive)
    cov = load_covariates(root=artifacts_dir)
    gm = load_grey_matter(mmap=True, root=artifacts_dir)
    x, sites = ixi_covariates(cov), cov["site"]
    with np.load(archive, allow_pickle=False) as ref:
        design_error = pinned_agreement(cov, ref)
        voxels, held = ref["voxels"], ref["test"]
        if voxels.shape != (N_VOXELS,) or np.unique(voxels).size != N_VOXELS:
            raise ValueError(f"expected {N_VOXELS} distinct voxel indices from {archive}, got {voxels.shape}")
        y = np.asarray(gm[:, voxels], dtype=np.float64)
        ytrain, ytest = standardise(y, held)
        scale_error = max(float(np.abs(ytrain - ref["y_train"]).max()), float(np.abs(ytest - ref["y_test"]).max()))
        if not scale_error <= SCALING_TOLERANCE:
            raise ValueError(f"raw responses do not reproduce scaling in {archive}: max error {scale_error:g}")
        d = design(ref["Phi_train"], ref["Phi_test"])
        frozen_z = ref["z_test"]

    logger.info("[2/4] fitting %d voxels with jax: %d training and %d held-out scans", y.shape[1], len(ytrain), len(ytest))
    hyp, _, z = fit_chunks(make_fitter(), d, ytrain, ytest)
    train = ~held
    basis = Basis.fit(x[train], sites[train], batch_levels=IXI_SITES)
    names = [f"v{voxel}" for voxel in voxels]
    covariates = ["age", "sex"]
    folder = results / "models" / datetime.now(UTC).strftime("ixi-%Y%m%d-%H%M%S-%f")
    logger.info("[3/4] exporting fitted parameters to %s", folder)
    pcn = import_module("pcntoolkit")
    output = import_module("pcntoolkit.util.output").Output
    show = output.get_show_messages()
    output.set_show_messages(False)
    try:
        path = export_pcntoolkit(folder, basis, hyp, x[train], y[train], sites[train],
                                 covariate_names=covariates, response_names=names, batch_name="site")
        logger.info("[4/4] loading %s and predicting with unmodified pcntoolkit", path)
        restored = pcn.NormativeModel.load(str(path))
        attrs = {"covariates": covariates, "response_vars": names, "batch_effect_dims": ["site"]}
        query = pcn.NormData.from_ndarrays("test", x[held], y[held], sites[held, None], attrs=attrs)
        predicted = restored.predict(query).Z.sel(response_vars=names).values
    finally:
        output.set_show_messages(show)

    response_files = len(list((path / "model").glob("*/regression_model.json")))
    if response_files != N_VOXELS:
        raise RuntimeError(f"expected {N_VOXELS} response files in {path}, found {response_files}")
    round_trip = _difference(predicted, z, "round-trip")
    vs_frozen = _difference(predicted, frozen_z, "frozen reference")
    for label, metrics, bound in (("round-trip", round_trip, ROUND_TRIP_TOLERANCE),
                                  ("frozen reference", vs_frozen, FROZEN_REFERENCE_TOLERANCE)):
        if metrics["max_abs_z"] > bound:
            raise RuntimeError(f"{label} z difference {metrics['max_abs_z']:g} exceeds {bound:g} for {path}")
    jax = import_module("jax")
    report = results / "json" / "round-trip.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "reference_artifact": PCNTOOLKIT_REFERENCE, "reference_payload": str(archive),
        "model_folder": str(path.relative_to(results)), "pcntoolkit": PCN_VERSION,
        "host": platform.system(), "architecture": platform.machine(), "device": str(jax.devices()[0]),
        "n_train": len(ytrain), "n_test": len(ytest), "n_voxels": len(voxels), "n_z": z.size,
        "response_files": response_files, "design_max_absdiff": design_error, "scaling_max_absdiff": scale_error,
        "round_trip": round_trip, "vs_frozen": vs_frozen,
        "tolerances": {"round_trip_max_abs_z": ROUND_TRIP_TOLERANCE,
                       "vs_frozen_max_abs_z": FROZEN_REFERENCE_TOLERANCE, "scaling_max_absdiff": SCALING_TOLERANCE},
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    logger.info("wrote %s: round-trip max |dz| %.6g; frozen reference max |dz| %.6g",
                report, round_trip["max_abs_z"], vs_frozen["max_abs_z"])
