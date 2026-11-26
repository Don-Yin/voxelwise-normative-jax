"""K bootstrap refits of one voxel chunk in one compiled call: per-draw scaling, early-stop fit and z on the device."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .refit_inputs import Inputs

logger = logging.getLogger(__name__)
MEMORY_FIELDS = ("temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes",
                 "alias_size_in_bytes", "generated_code_size_in_bytes")
THRESHOLD = 1.96
Z_TOLERANCE = 0.01  # agreement of the original fit with the converged PCNtoolkit reference


@dataclass(frozen=True)
class Batch:
    """Host z for K refits, per-chunk loop lengths and the two timed phases."""

    z: np.ndarray  # (K, Nt, V)
    steps: np.ndarray  # (K, chunks) loop steps each refit needed in each chunk
    voxel_steps: np.ndarray  # (K,) summed per-voxel active steps
    device_window: tuple[float, float]  # epoch seconds from upload to all outputs ready
    compute_seconds: float
    gather_seconds: float


def kernel(dtype: str):
    """vmap exp_010's analytic early-stop fit over a leading refit axis, after per-draw float64 scaling."""
    import jax
    import jax.numpy as jnp

    from shared.voxelwisenormative.early_newton import make_variant

    fit = make_variant(analytic=True, early=True, tf32=False)

    def one(lam, G, H, rows, train, test):
        """Scale on this draw's rows only (population SD), fit, and score the fixed held-out rows."""
        y = train[rows].astype(jnp.float64)
        mean, sd = jnp.mean(y, axis=0), jnp.std(y, axis=0)
        bad = jnp.count_nonzero(~(jnp.isfinite(mean) & jnp.isfinite(sd) & (sd > 0)))
        x = ((y - mean) / sd).astype(dtype)
        xt = ((test.astype(jnp.float64) - mean) / sd).astype(dtype)
        _, _, z, counts = fit(lam, G, H, x, xt)
        return z, jnp.max(counts), jnp.sum(counts), bad

    return jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0, None, None)))


class Batcher:
    """Device-resident response chunks and one compiled executable per (K, chunk width)."""

    def __init__(self, inputs: Inputs, chunk: int, dtype: str, device) -> None:
        import jax

        self.inputs, self.dtype, self.device = inputs, np.dtype(dtype), device
        n, v = inputs.train.shape
        self.rows = np.vstack([inputs.rows, np.arange(n, dtype=np.int32)[None]])  # last row: the original fit
        self.bounds = [(lo, min(lo + chunk, v)) for lo in range(0, v, chunk)]
        self.train = [jax.device_put(np.ascontiguousarray(inputs.train[:, lo:hi]), device) for lo, hi in self.bounds]
        self.test = [jax.device_put(np.ascontiguousarray(inputs.test[:, lo:hi]), device) for lo, hi in self.bounds]
        self.jitted = kernel(dtype)
        self.compiled: dict[tuple[int, int], Callable] = {}
        self.reports: dict[str, dict] = {}

    def widths(self) -> dict[int, int]:
        """First chunk index of each distinct width."""
        first: dict[int, int] = {}
        for i, (lo, hi) in enumerate(self.bounds):
            first.setdefault(hi - lo, i)
        return first

    def compile(self, k: int) -> None:
        """Compile each chunk width for K refits; keep seconds and XLA's static memory plan per width."""
        import jax

        n, nt, d = self.inputs.train.shape[0], self.inputs.test.shape[0], self.inputs.lam.shape[1]
        sharding = jax.sharding.SingleDeviceSharding(self.device)
        specs = [jax.ShapeDtypeStruct(shape, self.dtype, sharding=sharding) for shape in ((k, d), (k, d, n), (k, nt, d))]
        specs.append(jax.ShapeDtypeStruct((k, n), np.int32, sharding=sharding))
        report = {}
        for width, i in self.widths().items():
            start = time.perf_counter()
            compiled = self.jitted.lower(*specs, self.train[i], self.test[i]).compile()
            stats = compiled.memory_analysis()
            self.compiled[k, width] = compiled
            report[str(width)] = {"compile_seconds": time.perf_counter() - start,
                                  **{field: int(getattr(stats, field)) for field in MEMORY_FIELDS}}
        self.reports[str(k)] = report
        logger.info("compiled k=%d for widths %s: %s", k, list(report), report)

    def dispatch(self, draws: np.ndarray, chunks: list[int] | None = None) -> list:
        """Upload K designs and enqueue one call per chunk; return device outputs without waiting."""
        import jax

        host = [self.inputs.lam[draws], self.inputs.G[draws], self.inputs.H[draws]]
        lam, G, H = (jax.device_put(a.astype(self.dtype), self.device) for a in host)
        rows = jax.device_put(self.rows[draws], self.device)
        k = len(draws)
        selected = range(len(self.bounds)) if chunks is None else chunks
        return [self.compiled[k, self.bounds[i][1] - self.bounds[i][0]](lam, G, H, rows, self.train[i], self.test[i])
                for i in selected]

    def warm(self, draws: np.ndarray) -> float:
        """Run each compiled width once, outside any timed pass."""
        import jax

        start = time.perf_counter()
        jax.block_until_ready(self.dispatch(draws, list(self.widths().values())))
        return time.perf_counter() - start

    def run(self, draws: np.ndarray, *, out_z: np.ndarray | None = None) -> Batch:
        """Time one synchronised batch, then gather z and loop counters, writing z to out_z if supplied."""
        import jax

        shape = (len(draws), self.inputs.test.shape[0], self.inputs.train.shape[1])
        if out_z is not None and (out_z.shape != shape or out_z.dtype != self.dtype):
            raise ValueError(f"out_z has {out_z.shape=} {out_z.dtype=}, expected {shape} {self.dtype}")
        epoch, start = time.time(), time.perf_counter()
        outs = self.dispatch(draws)
        jax.block_until_ready(outs)
        ready = time.perf_counter()
        z = np.empty(shape, dtype=self.dtype) if out_z is None else out_z
        for (lo, hi), out in zip(self.bounds, outs, strict=True):
            z[:, :, lo:hi] = np.asarray(out[0])
        steps = np.stack([np.asarray(out[1]) for out in outs], axis=1)
        voxel_steps = np.sum([np.asarray(out[2], dtype=np.int64) for out in outs], axis=0)
        bad = np.sum([np.asarray(out[3]) for out in outs], axis=0)
        end = time.perf_counter()
        if bad.any():
            raise ValueError(f"invalid draw scale at {bad.tolist()} voxels for draws {draws.tolist()}")
        return Batch(z, steps, voxel_steps, (epoch, epoch + ready - start), ready - start, end - ready)


def agreement(z: np.ndarray, ref: np.ndarray) -> tuple[bool, int, float, int]:
    """Bitwise equality, differing elements, max |Δz| and |z|>1.96 disagreements for a finite refit."""
    bits = np.dtype(f"u{z.dtype.itemsize}")
    differing = int(np.count_nonzero(z.view(bits) != ref.view(bits)))
    if not differing:
        return True, 0, 0.0, 0
    flips = np.count_nonzero((np.abs(z) > THRESHOLD) != (np.abs(ref) > THRESHOLD))
    return False, differing, float(np.max(np.abs(z.astype(np.float64, copy=False) - ref))), int(flips)


def original_check(batcher: Batcher, inputs: Inputs) -> dict:
    """Fit every reference row once and check the pinned PCNtoolkit voxels."""
    batch = batcher.run(np.asarray([len(inputs.rows)]))
    report = {"seconds_including_first_run": batch.compute_seconds + batch.gather_seconds,
              "flags": int(np.count_nonzero(np.abs(batch.z) > THRESHOLD)), "pinned_z_max_abs_delta": None}
    if inputs.pinned_voxels is not None:
        error = float(np.max(np.abs(batch.z[0][:, inputs.pinned_voxels] - inputs.pinned_z)))
        if not error <= Z_TOLERANCE:
            raise ValueError(f"original fit max |Δz| {error:g} against PCNtoolkit exceeds {Z_TOLERANCE}")
        report["pinned_z_max_abs_delta"] = error
    logger.info("original fit: %s", report)
    return report
