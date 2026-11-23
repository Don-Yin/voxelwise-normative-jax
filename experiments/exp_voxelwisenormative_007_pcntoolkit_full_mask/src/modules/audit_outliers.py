"""Explain the full-mask comparison tail and reproduce the CPU optimiser stops."""

from __future__ import annotations

import json
import logging
import os
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import numpy as np

from shared.voxelwisenormative.artifacts import PCNTOOLKIT_REFERENCE, resolve
from shared.voxelwisenormative.cohort import standardise

from .campaign import check_job, inputs, read_calibration, reference
from .compare import CUDA
from .fit import SHAPE, fit_block
from .outlier_objective import HYP_NAMES, Objective, line_barrier, safety_bounds

logger = logging.getLogger(__name__)
LIMITS = np.array([0.001, 0.01, 0.1, 1.0])


def _native_json_scalar(value: object) -> bool | int | float:
    """Convert NumPy scalars without silently accepting unexpected types."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"unsupported audit JSON type: {type(value).__name__}")


def tails(cpu: np.ndarray, gpu: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """Count strict exceedances, and rank distinct columns by their worst test row."""
    if cpu.shape != gpu.shape or cpu.ndim != 2 or cpu.shape[1] < 10:
        raise ValueError(f"incompatible score matrices: {cpu.shape}, {gpu.shape}")
    rows = np.empty(cpu.shape[1], dtype=int)
    worst = np.empty(cpu.shape[1], dtype=np.float64)
    pair_counts = np.zeros(LIMITS.size, dtype=np.int64)
    voxel_counts = np.zeros_like(pair_counts)
    for lo in range(0, cpu.shape[1], 4096):
        hi = min(lo + 4096, cpu.shape[1])
        delta = np.abs(gpu[:, lo:hi].astype(np.float64) - cpu[:, lo:hi])
        if not np.isfinite(delta).all():
            raise ValueError(f"nonfinite difference in columns [{lo},{hi})")
        rows[lo:hi] = delta.argmax(axis=0)
        worst[lo:hi] = delta.max(axis=0)
        for i, limit in enumerate(LIMITS):
            pair_counts[i] += np.count_nonzero(delta > limit)
            voxel_counts[i] += np.count_nonzero(worst[lo:hi] > limit)
    counts = [
        {"threshold": float(t), "pairs": int(p), "pair_fraction": float(p / cpu.size),
         "voxels": int(v), "voxel_fraction": float(v / cpu.shape[1])}
        for t, p, v in zip(LIMITS, pair_counts, voxel_counts, strict=True)
    ]
    top = np.argsort(-worst, kind="stable")[:10]
    return counts, top, rows[top], worst[top]


def replay(gm, cov, ref, columns, out: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run the unmodified PCN evaluation and retain SciPy's stop record per voxel."""
    blr = import_module("pcntoolkit.regression_model.blr")
    original = blr.optimize.fmin_l_bfgs_b
    captured = {}

    def record(*args, **kwargs):
        """Forward the original optimiser call verbatim, including its restart policy."""
        owner = kwargs["func"].__self__
        name = owner.name
        trials = captured.setdefault(name, {"attempts": []})["attempts"]
        try:
            theta, objective, status = original(*args, **kwargs)
        except np.linalg.LinAlgError as exc:
            trials.append({"exception": type(exc).__name__, "message": str(exc)})
            raise
        task = status["task"]
        trials.append({
            "warnflag": int(status["warnflag"]), "task": task.decode() if isinstance(task, bytes) else str(task),
            "iterations": int(status["nit"]), "function_calls": int(status["funcalls"]),
            "gradient_inf": float(np.linalg.norm(status["grad"], ord=np.inf)),
        })
        captured[name].update(theta=theta.copy(), objective=float(objective))
        return theta, objective, status

    with patch.object(blr.optimize, "fmin_l_bfgs_b", record):
        z, weights, nlz = fit_block(gm, cov, ref["test"], columns, ref["Phi_train"], ref["Phi_test"], out)
    if set(captured) != {f"v{v}" for v in columns}:
        raise RuntimeError(f"unexpected CPU optimizer captures: {sorted(captured)}")
    return z, weights, nlz, captured


def read_archive(results: Path, array: str, merge: str, calibration: str):
    """Check job identity and load immutable score matrices without copying to RAM."""
    root = results / "merged" / check_job(array)
    path = root / "summary.json"
    try:
        with path.open() as stream:
            summary = json.load(stream)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot load merged result identity from {path}") from exc
    if (summary["array_job"], summary["merge_job"], summary["calibration_job"]) != (array, merge, calibration):
        raise ValueError(f"unmatched array/merge/calibration in {path}")
    types = {"z_test.npy": (113, SHAPE[1]), "nlZ.npy": (SHAPE[1],), "posterior_weights.npy": (SHAPE[1], 12)}
    cpu = {key: np.load(root / key, mmap_mode="r", allow_pickle=False) for key in types}
    for key, shape in types.items():
        if cpu[key].shape != shape or cpu[key].dtype != np.float64:
            raise ValueError(f"{root / key} must be float64 {shape}: got {cpu[key].shape}, {cpu[key].dtype}")
    paths = resolve(CUDA, "z_test.npy", "hyp.npy")
    gpu = {key: np.load(path, mmap_mode="r", allow_pickle=False) for key, path in paths.items()}
    if (gpu["z_test.npy"].shape, gpu["hyp.npy"].shape) != ((113, SHAPE[1]), (SHAPE[1], 4)):
        raise ValueError(f"mismatched CUDA archive dimensions at {paths}")
    return cpu, gpu, summary


def inspect(objective, row, cpu_theta, gpu_theta, cpu_z, gpu_z, archived_cpu, optimisation) -> dict:
    """Separate raw likelihood from the penalty and test both local starts fairly."""
    cpu_details = objective.describe(cpu_theta, row)
    gpu_details = objective.describe(gpu_theta, row)
    cpu_error = float(np.max(np.abs(objective.evaluate(cpu_theta)[3] - cpu_z)))
    gpu_error = float(np.max(np.abs(objective.evaluate(gpu_theta)[3] - gpu_z)))
    if gpu_error > 1e-4:
        raise ValueError(f"re-evaluation does not reproduce CUDA z: {gpu_error} at test row {row}")
    if abs(cpu_details["penalised_nll"] - optimisation["objective"]) > 1e-5:
        raise ValueError(f"common objective disagrees with PCNtoolkit: {cpu_details}, {optimisation}")
    verified = bool(cpu_error <= 1e-3 and abs(cpu_details["penalised_nll"] - archived_cpu) <= 1e-5)
    gap = gpu_details["penalised_nll"] - archived_cpu
    if abs(gap) <= 1e-5:
        winner = "equal_within_1e-5"
    else:
        winner = "pcntoolkit" if gap > 0 else "gpu"
    bounds = safety_bounds(cpu_theta, gpu_theta)
    first = objective.refine(cpu_theta, bounds, row)
    second = objective.refine(gpu_theta, bounds, row)
    names = ("noise_sd_warped", "warp_shift", "warp_shape", "weight_precision")
    return {
        "archived_pcn_penalised_nll": float(archived_cpu), "archived_pcn_z": float(cpu_z[row]),
        "archived_gpu_z": float(gpu_z[row]), "pcn_replay": cpu_details, "gpu_from_archive": gpu_details,
        "pcn_replay_matches_archive": verified, "pcn_optimizer_replayed": optimisation["attempts"],
        "max_recomputed_z_error_cpu": cpu_error, "max_recomputed_z_error_gpu": gpu_error,
        "gpu_minus_pcn_penalised_nll": float(gap), "better_archived_objective": winner,
        "gpu_minus_pcn_parameters": {name: gpu_details[name] - cpu_details[name] for name in names},
        "gpu_minus_pcn_theta": (gpu_theta - cpu_theta).tolist(),
        "analytically_refined_from_pcn": first, "analytically_refined_from_gpu": second,
        "refined_line_barrier_nats": line_barrier(objective, np.array(first["theta"]), np.array(second["theta"])),
    }


def audit(results: Path, array: str, merge: str, calibration: str) -> None:
    """Write counts and a convergence diagnostic for each top-ten voxel at Slurm."""
    cpu, gpu, archive = read_archive(results, array, merge, calibration)
    cpu_z, gpu_z = cpu["z_test.npy"], gpu["z_test.npy"]
    counts, columns, rows, maximum = tails(cpu_z, gpu_z)
    np.testing.assert_array_equal([rows[0], columns[0]], archive["z_all"]["worst_index"])
    np.testing.assert_allclose(maximum[0], archive["z_all"]["max_abs"], rtol=0, atol=1e-12)
    logger.info("full-mask strict exceedances: %s", json.dumps(counts))
    job = os.environ.get("SLURM_JOB_ID", "local")
    out = results / "audit" / job
    out.mkdir(parents=True, exist_ok=False)
    ref = reference(PCNTOOLKIT_REFERENCE)
    gm, cov, _ = inputs(ref, read_calibration(results, calibration))
    y, yt = standardise(gm[:, columns], ref["test"])
    logger.info("replaying the original PCNtoolkit fit on columns %s", columns.tolist())
    replay_z, replay_weights, nlz, status = replay(gm, cov, ref, columns, out / "pcntoolkit")
    phi, phit = ref["Phi_train"], ref["Phi_test"]
    gram = phi.T @ phi
    top = []
    for i, (v, row, diff) in enumerate(zip(columns, rows, maximum, strict=True)):
        opt = status[f"v{v}"]
        item = inspect(Objective(phi, gram, phit, y[:, i], yt[:, i]), row, opt["theta"],
                       gpu["hyp.npy"][v].astype(np.float64), cpu_z[:, v], gpu_z[:, v], cpu["nlZ.npy"][v], opt)
        item.update(mask_column=int(v), test_row=int(row), max_abs_delta_z=float(diff),
                    max_cpu_replay_delta_z=float(np.abs(replay_z[:, i] - cpu_z[:, v]).max()),
                    max_cpu_replay_delta_weight=float(np.abs(replay_weights[i] - cpu["posterior_weights.npy"][v]).max()),
                    cpu_replay_penalised_nll=float(nlz[i]))
        top.append(item)
        logger.info("[%d/10] mask column %d, row %d, |delta z| %.6g, gpu minus pcn objective %s, stop %s",
                    i + 1, v, row, diff, item["gpu_minus_pcn_penalised_nll"], opt["attempts"][-1]["task"])
    report = {"job": job, "array_job": array, "merge_job": merge, "calibration_job": calibration,
              "pairs": cpu_z.size, "voxels": cpu_z.shape[1], "test_rows": cpu_z.shape[0], "thresholds": counts,
              "hyperparameters": HYP_NAMES, "worst_voxels": top, "z_all": archive["z_all"]}
    output = out / "outliers.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False, default=_native_json_scalar) + "\n")
    temp.rename(output)
    logger.info("saved %s", output)
