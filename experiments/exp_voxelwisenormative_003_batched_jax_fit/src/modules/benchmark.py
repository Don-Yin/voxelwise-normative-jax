"""time the batched fit and compare it with converged PCNtoolkit on the reference voxels."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.artifacts import PCNTOOLKIT_REFERENCE, resolve
from shared.voxelwisenormative.batched_blr import design, fit_chunks, make_fitter
from shared.voxelwisenormative.cohort import load_grey_matter, standardise

logger = logging.getLogger(__name__)


def inputs(mode: str, ref) -> tuple[np.ndarray, np.ndarray]:
    """scaled train/test data: the reference voxels (check) or every mask voxel (full)."""
    if mode == "check":
        return ref["y_train"], ref["y_test"]
    Y, Yt = standardise(load_grey_matter(), ref["test"])
    dy = float(np.abs(Y[:, ref["voxels"]] - ref["y_train"]).max())
    if dy > 1e-9:
        raise RuntimeError(f"standardisation differs from PCNtoolkit by {dy}")
    return Y, Yt


def agreement(z, fv, ref, idx) -> dict:
    """z and objective differences from PCNtoolkit on the reference voxels."""
    dz = np.abs(z[:, idx] - ref["z_test"])
    dobj = fv[idx] - ref["nlZ"]
    return {
        "check_voxels": int(len(idx)),
        "z_absdiff_median": float(np.median(dz)), "z_absdiff_p99": float(np.quantile(dz, 0.99)),
        "z_absdiff_max": float(dz.max()),
        "objective_minus_pcntoolkit_median": float(np.median(dobj)),
        "objective_minus_pcntoolkit_max": float(dobj.max()),
        "voxels_objective_worse_than_pcntoolkit_by_1e-6": int((dobj > 1e-6).sum()),
    }


def run(mode: str, n_iter: int, chunk: int, repeats: int, results: Path) -> dict:
    """fit in batch, time repeats after a compile warm-up, and write result.json."""
    import jax

    ref = np.load(resolve(PCNTOOLKIT_REFERENCE, "arrays.npz")["arrays.npz"])
    Y, Yt = inputs(mode, ref)
    d = design(ref["Phi_train"], ref["Phi_test"])
    fit = make_fitter(n_iter)
    device = str(jax.devices()[0])
    logger.info("%s: %d voxels, %d train, %d test on %s", mode, Y.shape[1], Y.shape[0], Yt.shape[0], device)
    width = min(chunk, Y.shape[1])
    fit_chunks(fit, d, Y[:, :width], Yt[:, :width], chunk)  # compile
    seconds = []
    hyp, fv, z = None, None, None
    for _ in range(max(repeats, 1)):
        start = time.perf_counter()
        hyp, fv, z = fit_chunks(fit, d, Y, Yt, chunk)
        seconds.append(time.perf_counter() - start)
        logger.info("[%d/%d] %.2f s", len(seconds), repeats, seconds[-1])
    assert hyp is not None and fv is not None and z is not None
    result = {
        "mode": mode, "device": device, "jax": jax.__version__, "n_train": int(Y.shape[0]),
        "n_test": int(Yt.shape[0]), "n_voxels": int(Y.shape[1]), "n_iter": n_iter, "chunk": chunk,
        "seconds": seconds, "seconds_best": min(seconds), "seconds_per_voxel": min(seconds) / Y.shape[1],
        "nonfinite_z": int((~np.isfinite(z)).sum()),
    }
    result.update(agreement(z, fv, ref, ref["voxels"] if mode == "full" else np.arange(Y.shape[1])))
    out = results / f"{mode}-{device.split(':')[0].replace(' ', '').lower()}-{Y.shape[1]}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if mode == "full":
        np.save(out / "z_test.npy", z.astype(np.float32))
        np.save(out / "hyp.npy", hyp.astype(np.float32))
    logger.info("%s", json.dumps(result, indent=2))
    return result
