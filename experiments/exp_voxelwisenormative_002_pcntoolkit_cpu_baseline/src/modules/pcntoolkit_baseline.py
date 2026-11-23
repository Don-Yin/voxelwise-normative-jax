"""PCNtoolkit 1.4.0 warped BLR, one model per voxel, timed and captured as the batched-fit reference."""

from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.artifacts import VOXEL_MATRIX, resolve, sha256
from shared.voxelwisenormative.cohort import (
    load_covariates,
    load_grey_matter,
    site_split,
)

logger = logging.getLogger(__name__)


def capture_blr(BLR, captured: dict, fit_seconds: list):
    """patch BLR.fit/forward to record the design, scaled data, fit and per-voxel fit time."""
    original_fit, original_forward = BLR.fit, BLR.forward

    def timed_fit(self, X_, be, be_maps, Y_):
        start = time.perf_counter()
        original_fit(self, X_, be, be_maps, Y_)
        fit_seconds.append(time.perf_counter() - start)
        Phi, _ = self.Phi_Phi_var(X_.values, be.values)
        captured.setdefault(self.name, {}).update(Phi_train=Phi, y_train=Y_.values.copy(), hyp=self.hyp.copy(), nlZ=float(self.nlZ))

    def capturing_forward(self, X_, be, Y_):
        Phi, _ = self.Phi_Phi_var(X_.values, be.values)
        captured[self.name].update(Phi_test=Phi, y_test=Y_.values.copy())
        return original_forward(self, X_, be, Y_)

    BLR.fit, BLR.forward = timed_fit, capturing_forward
    return original_fit, original_forward


def run(n_voxels: int, seed: int, epsilon: float, results: Path) -> None:
    """fit PCNtoolkit BLR per voxel, time it, and save the reference arrays."""
    from pcntoolkit import (  # pyright: ignore[reportMissingImports]
        BLR,
        BsplineBasisFunction,
        NormativeModel,
        NormData,
    )

    gm = load_grey_matter(mmap=True)
    cov = load_covariates()
    voxels = np.sort(np.random.default_rng(seed).choice(gm.shape[1], n_voxels, replace=False))
    Y = np.asarray(gm[:, voxels], dtype=np.float64)
    X = np.column_stack([cov["age"], (cov["sex"] == "M").astype(float)])
    site = cov["site"]
    test = site_split(site, seed)
    names = [f"v{v}" for v in voxels]
    attrs = {"covariates": ["age", "sex"], "batch_effect_dims": ["site"], "response_vars": names}
    train_data, test_data = (
        NormData.from_ndarrays(tag, X[rows], Y[rows], site[rows, None], subject_ids=cov["subject"][rows], attrs=attrs)
        for tag, rows in (("train", ~test), ("test", test))
    )
    captured: dict[str, dict] = {}
    fit_seconds: list[float] = []
    original_fit, original_forward = capture_blr(BLR, captured, fit_seconds)
    template = BLR(name="blr", fixed_effect=True, warp_name="WarpSinhArcsinh", l_bfgs_b_epsilon=epsilon,
                   basis_function_mean=BsplineBasisFunction(basis_column=0, degree=3, nknots=5))
    tag = f"blr-sinharcsinh-n{n_voxels}-seed{seed}" + ("" if epsilon == 0.1 else f"-eps{epsilon:g}")
    out = results / tag
    model = NormativeModel(template, savemodel=False, evaluate_model=False, saveresults=False,
                           saveplots=False, save_dir=str(out / "pcntoolkit"), name=tag)
    logger.info("fitting %d voxels, %d train, %d test, epsilon %g", n_voxels, (~test).sum(), test.sum(), epsilon)
    start = time.perf_counter()
    model.fit(train_data)
    wall_fit = time.perf_counter() - start
    BLR.fit = original_fit
    # fit() ends with predict() on the training set; the test pass below overwrites Phi_test/y_test.
    start = time.perf_counter()
    model.predict(test_data)
    wall_predict = time.perf_counter() - start
    BLR.forward = original_forward

    z = test_data["Z"].sel(response_vars=names).values
    for key in ("Phi_train", "Phi_test"):
        if any(not np.array_equal(captured[n][key], captured[names[0]][key]) for n in names):
            raise RuntimeError(f"{key} differs between voxels")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "arrays.npz", voxels=voxels, test=test,
        Phi_train=captured[names[0]]["Phi_train"], Phi_test=captured[names[0]]["Phi_test"],
        y_train=np.stack([captured[n]["y_train"] for n in names], 1),
        y_test=np.stack([captured[n]["y_test"] for n in names], 1),
        hyp=np.stack([captured[n]["hyp"] for n in names]), nlZ=np.array([captured[n]["nlZ"] for n in names]),
        z_test=z,
    )
    timing = {
        "pcntoolkit": "1.4.0", "l_bfgs_b_epsilon": epsilon, "n_train": int((~test).sum()), "n_test": int(test.sum()),
        "matrix_sha256": sha256(resolve(VOXEL_MATRIX, 'gm.npy')['gm.npy']),
        "n_voxels": n_voxels, "design_columns": int(captured[names[0]]["Phi_test"].shape[1]),
        "hyperparameters": int(captured[names[0]]["hyp"].size),
        "normativemodel_fit_seconds": wall_fit, "blr_fit_seconds_sum": float(sum(fit_seconds)),
        "blr_fit_seconds_per_voxel_median": float(np.median(fit_seconds)),
        "predict_test_seconds": wall_predict,
        "fit_seconds_per_voxel": wall_fit / n_voxels,
        "extrapolated_full_mask_hours": wall_fit / n_voxels * gm.shape[1] / 3600,
        "machine": f"{platform.machine()} {platform.processor()} {platform.platform()}",
        "threads": "PCNtoolkit fits voxels sequentially; numpy BLAS default threads",
    }
    (out / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    logger.info("%s", json.dumps(timing, indent=2))
