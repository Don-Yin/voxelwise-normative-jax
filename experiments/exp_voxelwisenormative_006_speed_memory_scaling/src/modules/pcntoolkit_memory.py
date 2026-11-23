"""PCNtoolkit 1.4.0 process peak memory on synthetic data shaped like the IXI benchmark; no IXI data.

The model is exp_voxelwisenormative_002's converged BLR (warp, B-spline age, sex, site batch effect,
L-BFGS-B epsilon 1e-7). Covariates and responses are random but keep the benchmark's shape: 563
subjects in three sites, the same site-stratified 80/20 split (450 train, 113 test) and 12 design
columns. Memory depends on these shapes; timing here is only supporting evidence because convergence
depends on the values. One worker process per N; the launcher runs workers without a profiler,
because profiler samples would enter the measured resident memory.
"""

from __future__ import annotations

import json
import logging
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.cohort import site_split

logger = logging.getLogger(__name__)

N_SUBJECTS = 563
SITE_SIZES = (188, 188, 187)  # three synthetic sites; site_split takes 38 + 38 + 37 = 113 test subjects
SEED = 20261120
GRID = (100, 300, 1000, 3000, 10000)
# NormativeModel.fit ends with predict on the training set (normative_model.py:189), which keeps
# Z, five centiles, logp, baseline_logp and Yhat for every voxel: 9 float64 values per training subject.
RETAINED_ARRAYS_PER_SUBJECT = 9


def synthetic_cohort(n_voxels: int, seed: int = SEED) -> dict[str, np.ndarray]:
    """random age, sex, site and grey-matter-like responses with the benchmark's shapes."""
    rng = np.random.default_rng(seed)
    site = np.repeat(np.array([f"site{i}" for i in range(len(SITE_SIZES))]), SITE_SIZES)
    age = rng.uniform(20.0, 86.0, N_SUBJECTS)
    sex = rng.integers(0, 2, N_SUBJECTS).astype(np.float64)
    y = 0.5 - 0.003 * (age[:, None] - 50.0) + 0.05 * rng.standard_normal((N_SUBJECTS, n_voxels))
    return {"site": site, "age": age, "sex": sex, "y": y, "test": site_split(site, 0)}


def rss_bytes() -> int:
    """current resident memory of this process."""
    import psutil

    return int(psutil.Process().memory_info().rss)


def peak_rss_bytes() -> int:
    """process lifetime peak resident memory (macOS reports bytes, Linux KiB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def fit_one(n_voxels: int, out: Path, save_dir: Path) -> dict:
    """fit PCNtoolkit on one synthetic cohort and record time and resident memory around the fit."""
    import pcntoolkit  # pyright: ignore[reportMissingImports]
    from pcntoolkit import BLR, BsplineBasisFunction, NormativeModel, NormData  # pyright: ignore[reportMissingImports]

    cohort = synthetic_cohort(n_voxels)
    test = cohort["test"]
    X = np.column_stack([cohort["age"], cohort["sex"]])
    names = [f"v{i}" for i in range(n_voxels)]
    attrs = {"covariates": ["age", "sex"], "batch_effect_dims": ["site"], "response_vars": names}
    subjects = np.array([f"s{i:03d}" for i in range(N_SUBJECTS)])
    train = NormData.from_ndarrays("train", X[~test], cohort["y"][~test], cohort["site"][~test, None],
                                   subject_ids=subjects[~test], attrs=attrs)
    del cohort
    template = BLR(name="blr", fixed_effect=True, warp_name="WarpSinhArcsinh", l_bfgs_b_epsilon=1e-7,
                   basis_function_mean=BsplineBasisFunction(basis_column=0, degree=3, nknots=5))
    model = NormativeModel(template, savemodel=False, evaluate_model=False, saveresults=False,
                           saveplots=False, save_dir=str(save_dir), name=f"synthetic-n{n_voxels}")
    rss_before, peak_before = rss_bytes(), peak_rss_bytes()
    start = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - start
    peak_after, rss_after = peak_rss_bytes(), rss_bytes()
    first = model[names[0]]
    n_train = int((~test).sum())
    result = {
        "mode": "pcntoolkit-memory", "data": "synthetic, IXI benchmark shapes", "pcntoolkit": pcntoolkit.__version__,
        "n_voxels": n_voxels, "n_train": n_train, "n_test": int(test.sum()), "design_columns": int(first.D),
        "l_bfgs_b_epsilon": 1e-7, "seed": SEED,
        "rss_before_fit_bytes": rss_before, "peak_rss_before_fit_bytes": peak_before,
        "peak_rss_bytes": peak_after, "rss_after_fit_bytes": rss_after,
        "peak_set_during_fit": peak_after > peak_before,
        "retained_arrays_lower_bound_bytes": RETAINED_ARRAYS_PER_SUBJECT * n_train * 8 * n_voxels,
        "fit_seconds": fit_seconds, "fit_seconds_per_voxel": fit_seconds / n_voxels,
        "machine": f"{platform.machine()} {platform.platform()}", "python": platform.python_version(),
        "profiled": False, "result_path": str(out),
    }
    if result["design_columns"] != 12 or n_train != 450 or result["n_test"] != 113:
        raise RuntimeError(f"synthetic shapes differ from the benchmark: {result}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("n=%d peak rss %.1f MiB (before fit %.1f MiB), fit %.1f s", n_voxels, peak_after / 2**20,
                rss_before / 2**20, fit_seconds)
    return result


def launch(grid: tuple[int, ...], results: Path, intermediates: Path, entry: Path, profile: bool) -> Path:
    """run one unprofiled worker per N in a fresh process; profile one extra 1,000-voxel worker by default."""
    run_dir = results / "json" / "pcntoolkit-synthetic" / f"local-{time.time_ns()}"
    jobs = [(n, run_dir / f"n{n:06d}.json", ["--no-profile"]) for n in grid]
    if profile:  # profiler samples would enter the measured RSS, so its worker is kept apart
        jobs.append((1000, run_dir / "profiled" / "n001000.json", []))
    for i, (n, out, flags) in enumerate(jobs, 1):
        logger.info("[%d/%d] worker n=%d%s", i, len(jobs), n, " (profiled)" if not flags else "")
        command = [sys.executable, str(entry), "pcntoolkit-worker", "--n-voxels", str(n), "--out", str(out),
                   "--save-dir", str(intermediates / "pcntoolkit" / run_dir.name / f"n{n:06d}"), *flags]
        subprocess.run(command, check=True)
    return run_dir
