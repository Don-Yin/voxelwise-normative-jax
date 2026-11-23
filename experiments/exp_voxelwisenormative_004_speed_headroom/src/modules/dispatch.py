"""Run identical compiled voxel blocks on one or four devices and gather in mask order."""

from __future__ import annotations

from functools import cache

import jax
import numpy as np

from shared.voxelwisenormative.batched_blr import N_ITER, fit_chunks, make_fitter

from .optimizer import make_variant

# One-factor controls; this comparison lives only in the experiment.
VARIANTS = {
    "baseline": (np.float64, False, False, False),
    "autodiff-f64": (np.float64, False, False, False),
    "autodiff-f32": (np.float32, False, False, False),
    "autodiff-tf32": (np.float32, False, False, True),
    "analytic-f64": (np.float64, True, False, False),
    "analytic-f32": (np.float32, True, False, False),
    "early-f64": (np.float64, True, True, False),
    "early-f32": (np.float32, True, True, False),
    "early-tf32": (np.float32, True, True, True),
}


@cache
def kernel(variant: str):
    """reuse JAX compiled programs when a chunk width is unchanged."""
    _, analytic, early, tf32 = VARIANTS[variant]
    return make_fitter() if variant == "baseline" else make_variant(analytic=analytic, early=early, tf32=tf32)


def shards(voxels: int, chunk: int, devices) -> list:
    """share work evenly then split each device's contiguous shard into blocks."""
    return [(device, slice(lo, min(lo + chunk, end))) for i, device in enumerate(devices)
            for end in [(i + 1) * voxels // len(devices)]
            for lo in range(i * voxels // len(devices), end, chunk)]


def fit(data, variant: str, chunk: int, devices):
    """return host evidence, z and per-voxel iterations after synchronizing all devices."""
    fit_kernel = kernel(variant)
    if variant == "baseline" and len(devices) == 1:
        with jax.default_device(devices[0]):
            _, fv, z = fit_chunks(fit_kernel, data.design, data.y, data.yt, chunk)
        return fv, z, np.full(data.y.shape[1], N_ITER, dtype=np.int32)
    dtype, *_ = VARIANTS[variant]
    d = data.design
    host_design = tuple(np.asarray(a, dtype=dtype) for a in (d.lam, d.G, d.H))
    prepared = {device: tuple(jax.device_put(a, device) for a in host_design) for device in devices}
    outputs = []
    for device, cols in shards(data.y.shape[1], chunk, devices):
        y, yt = (jax.device_put(np.ascontiguousarray(a[:, cols], dtype=dtype), device) for a in (data.y, data.yt))
        outputs.append(fit_kernel(*prepared[device], y, yt))
    jax.block_until_ready(outputs)
    fv = np.concatenate([np.asarray(out[1]) for out in outputs])
    z = np.concatenate([np.asarray(out[2]) for out in outputs], axis=1)
    counts = (np.full(z.shape[1], N_ITER, dtype=np.int32) if variant == "baseline"
              else np.concatenate([np.asarray(out[3]) for out in outputs]))
    return fv, z, counts
