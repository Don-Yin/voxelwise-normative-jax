"""Frozen IXI inputs and unmodified serial PCNtoolkit BLR fits for CPU benchmarks."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np

from .artifacts import PCNTOOLKIT_REFERENCE, resolve
from .batched_blr import PENALTY
from .cohort import IXI_SITES, ixi_covariates, load_covariates, pinned_agreement

EPSILON = 1e-7
logger = logging.getLogger(__name__)


def cpu_host() -> dict:
    """Record the backend, hardware and CPU benchmark lock owner."""
    import jax

    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=True,
                             capture_output=True, text=True).stdout.strip()
    return {
        "cpu": cpu, "os": platform.system(), "machine": platform.machine(),
        "jax": jax.__version__, "jax_backend": jax.default_backend(),
        "thread_limits": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                         "VECLIB_MAXIMUM_THREADS")},
        "shared_timing_lock_owner": os.environ.get("VNJ_BENCH_LOCK_OWNER"),
    }


@dataclass(frozen=True)
class IXIInputs:
    """Aligned frozen responses, mean designs and pre-basis PCNtoolkit inputs."""

    archive: Path
    voxels: np.ndarray
    phi: np.ndarray
    phit: np.ndarray
    y: np.ndarray
    yt: np.ndarray
    x: np.ndarray
    xt: np.ndarray
    be: np.ndarray
    bet: np.ndarray


def load_ixi(n_voxels: int, *, root: Path | None = None) -> IXIInputs:
    """Use the first columns of the hash-verified frozen IXI reference."""
    archive = resolve(PCNTOOLKIT_REFERENCE, "arrays.npz", root=root)["arrays.npz"]
    cov = load_covariates(root=root)
    with np.load(archive, allow_pickle=False) as ref:
        if not 1 <= n_voxels <= ref["voxels"].size:
            raise ValueError(f"requested {n_voxels} voxels from {archive}; valid range 1..{ref['voxels'].size}")
        pinned_agreement(cov, ref)
        test, phi, phit = (ref[key] for key in ("test", "Phi_train", "Phi_test"))
        voxels = ref["voxels"][:n_voxels]
        y, yt = (np.ascontiguousarray(ref[key][:, :n_voxels]) for key in ("y_train", "y_test"))
    raw = ixi_covariates(cov)
    x = (raw - raw[~test].mean(axis=0)) / raw[~test].std(axis=0)
    site_ids = {name: i for i, name in enumerate(IXI_SITES)}
    be = np.array([site_ids[name] for name in cov["site"]])[:, None]
    return IXIInputs(archive, voxels, phi, phit, y, yt, x[~test], x[test], be[~test], be[test])


@dataclass(frozen=True)
class PCNReference:
    """Final PCNtoolkit parameters, evidence, held-out scores, and wall-time measurements."""

    hyp: np.ndarray
    objective: np.ndarray
    z: np.ndarray
    noise: np.ndarray
    noiset: np.ndarray
    fit_seconds: np.ndarray
    score_seconds: np.ndarray
    seconds: float
    design_error: float
    warnings: dict[str, int]


def fit_pcn(data: IXIInputs, *, ard: bool = False, heteroskedastic: bool = False) -> PCNReference:
    """Fit and score every voxel with serial PCNtoolkit BLR.fit and BLR.forward."""
    pcn = import_module("pcntoolkit")  # Optional benchmark dependency, not needed to fit in JAX.
    output = import_module("pcntoolkit.util.output").Output
    data_array = import_module("xarray").DataArray
    x, xt, be, bet, y, yt = map(data_array, (data.x, data.xt, data.be, data.bet, data.y, data.yt))
    be_maps = {"site": {name: i for i, name in enumerate(IXI_SITES)}}
    n = data.voxels.size
    fit_seconds, score_seconds, objective = np.empty(n), np.empty(n), np.empty(n)
    beta_columns = data.x.shape[1] + 1 if heteroskedastic else 1
    alpha_columns = data.phi.shape[1] if ard else 1
    hyp = np.empty((n, beta_columns + 2 + alpha_columns))
    z = np.empty_like(data.yt)
    show = output.get_show_messages()
    output.set_show_messages(False)
    logger.info("start serial pcntoolkit fits: %s ard=%s heteroskedastic=%s voxels=%d", data.archive, ard,
                heteroskedastic, n)
    model = None
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for i in range(n):
                model = pcn.BLR(
                    fixed_effect=True, heteroskedastic=heteroskedastic, ard=ard,
                    warp_name="WarpSinhArcsinh",
                    basis_function_mean=pcn.BsplineBasisFunction(basis_column=0, degree=3, nknots=5),
                    optimizer="l-bfgs-b", l_bfgs_b_epsilon=EPSILON, l_bfgs_b_l=PENALTY, l_bfgs_b_norm="l2",
                )
                start = time.perf_counter()
                model.fit(x, be, be_maps, y[:, i])
                fitted = time.perf_counter()
                z[:, i] = model.forward(xt, bet, yt[:, i]).values
                score_seconds[i] = time.perf_counter() - fitted
                fit_seconds[i] = fitted - start
                hyp[i], objective[i] = model.hyp, float(model.nlZ)
                if (i + 1) % max(1, n // 10) == 0 or i + 1 == n:
                    logger.info("[%d/%d] fitted pcntoolkit voxels", i + 1, n)
    finally:
        output.set_show_messages(show)
    seconds = time.perf_counter() - started
    if model is None:
        raise ValueError(f"no voxels to fit from {data.archive}")
    phi, noise = model.Phi_Phi_var(data.x, data.be)
    phit, noiset = model.Phi_Phi_var(data.xt, data.bet)
    design_error = float(max(np.abs(phi - data.phi).max(), np.abs(phit - data.phit).max()))
    if design_error > 1e-12:
        raise ValueError(f"pcntoolkit mean design differs from {data.archive} by {design_error:g}")
    bad = ~np.isfinite(hyp).all(axis=1) | ~np.isfinite(objective) | ~np.isfinite(z).all(axis=0)
    if bad.any():
        raise ValueError(f"nonfinite pcntoolkit fits for mask columns {data.voxels[bad].tolist()}")
    return PCNReference(hyp, objective, z, noise, noiset, fit_seconds, score_seconds, seconds,
                        design_error, dict(Counter(w.category.__name__ for w in caught)))
