"""Clearly separate measured timings, modelled memory and workload estimates."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from .grid import MASK_VOXELS, TRIALS


def power_fit(points: Iterable[tuple[int, float]]) -> dict | None:
    """Fit log(y) = intercept + slope * log(x) to positive values in one regime."""
    pairs = sorted(points)
    if len(pairs) < 3:
        return None
    x, y = np.asarray(pairs, dtype=float).T
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x <= 0) or np.any(y <= 0):
        raise ValueError(f"power fit requires positive finite input: {pairs}")
    if len(np.unique(x)) != len(x):
        raise ValueError(f"power fit has duplicate x values: {pairs}")
    lx, ly = np.log(x), np.log(y)
    exponent, offset = np.polyfit(lx, ly, 1)
    residual = np.sum((ly - (exponent * lx + offset)) ** 2)
    total = np.sum((ly - ly.mean()) ** 2)
    return {"exponent": float(exponent), "log_intercept": float(offset),
            "x_min": int(x.min()), "x_max": int(x.max()), "points": len(x),
            "log_r2": float(1.0 - residual / total) if total > 0 else None}


def cpu_projection(samples: list[dict]) -> dict:
    """Scale the largest timing linearly; regress monolithic RSS on sample width."""
    ordered = sorted(samples, key=lambda row: row["columns"])
    if len(ordered) != 3 or len({r["columns"] for r in ordered}) != 3:
        raise ValueError(f"CPU projection needs three distinct sample widths: {ordered}")
    x = np.array([row["columns"] for row in ordered], dtype=float)
    y = np.array([row["peak_rss_bytes"] for row in ordered], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    total = np.sum((y - y.mean()) ** 2)
    rss_r2 = float(1.0 - np.sum((y - (slope * x + intercept)) ** 2) / total) if total > 0 else None
    last = ordered[-1]
    estimate = float(intercept + slope * MASK_VOXELS)
    lower_bound = last["retained_training_arrays_bytes"] // last["columns"] * MASK_VOXELS
    if slope <= 0 or intercept < 0 or estimate < lower_bound:
        reason = "RSS fit is non-increasing, has a negative intercept, or falls below the retained-array bound"
    else:
        reason = None
    return {"sampled_voxels": last["columns"], "seconds_per_voxel": last["seconds"] / last["columns"],
            "projected_seconds": last["seconds"] / last["columns"] * MASK_VOXELS,
            "sample_peak_rss_bytes": last["peak_rss_bytes"],
            "projected_peak_rss_bytes": estimate if reason is None else None,
            "memory_projection_unavailable_reason": reason, "rss_slope_bytes_per_voxel": float(slope),
            "rss_intercept_bytes": float(intercept), "rss_linear_r2": rss_r2,
            "retained_arrays_lower_bound_bytes": lower_bound, "samples": ordered}


def n_powers(rows: list[dict]) -> dict:
    """Regress the real and artificial references separately; never treat them as one cohort."""
    powers = {}
    for dataset in ("ixi", "synthetic"):
        series = [r for r in rows if r["dataset"] == dataset]
        cpu = [r["pcntoolkit"] for r in series]
        pairs = list(zip((r["n"] for r in series), cpu, strict=True))
        powers[dataset] = {
            "cpu_time": power_fit((n, r["projected_seconds"]) for n, r in pairs),
            "cpu_memory": power_fit((n, r["projected_peak_rss_bytes"]) for n, r in pairs
                                    if r["projected_peak_rss_bytes"] is not None),
            **{precision: {field: power_fit((r["n"], r["gpu"][precision][field]) for r in series)
                           for field in ("seconds_median", "peak_bytes_in_use", "host_peak_rss_bytes")}
               for precision in ("float32", "float64")},
        }
    return powers


def estimate_cost(rows: list[dict], n: int) -> dict:
    """Return hours for an arbitrary N bracketed by the synthetic timing knots."""
    data = {r["n"]: r for r in rows if r["dataset"] == "synthetic"}
    knots = sorted(data)
    if not knots or not knots[0] <= n <= knots[-1]:
        raise ValueError(f"N={n} is outside measured synthetic sizes {knots}; a new measurement is needed")
    lo = max(k for k in knots if k <= n)
    hi = min(k for k in knots if k >= n)
    fits = TRIALS + 1

    def interpolate(a: float, b: float) -> float:
        """Interpolate positive elapsed time in log-size/log-time coordinates."""
        if lo == hi:
            return a
        f = math.log(n / lo) / math.log(hi / lo)
        return math.exp((1 - f) * math.log(a) + f * math.log(b))

    cpu = interpolate(*(data[k]["pcntoolkit"]["projected_seconds"] for k in (lo, hi)))
    gpu = {}
    for precision in ("float32", "float64"):
        hot = interpolate(*(data[k]["gpu"][precision]["seconds_median"] for k in (lo, hi)))
        cold = interpolate(*(data[k]["gpu"][precision]["warmup_seconds"] for k in (lo, hi)))
        gpu[precision] = {"one_warm_fit_gpu_hours": hot / 3600, "fit_and_64_refits_gpu_hours": fits * hot / 3600,
                          "with_first_cold_gpu_hours": (cold + TRIALS * hot) / 3600}
    return {"n": n, "n_test": data[lo]["n_test"], "mask_voxels": MASK_VOXELS, "fits": fits,
            "scope": "synthetic scenario; excludes input staging, preparation and queueing",
            "method": "measured synthetic size" if n == lo else "log-log interpolation of adjacent measured synthetic sizes",
            "synthetic_bracket": [lo, hi], "estimated_one_fit_cpu_hours": cpu / 3600,
            "estimated_fit_and_64_refits_cpu_hours": fits * cpu / 3600, "gpu": gpu}
