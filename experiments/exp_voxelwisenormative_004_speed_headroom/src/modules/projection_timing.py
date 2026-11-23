"""Time the same precomputed 12 × 450 projection as serial vectors and one A100 product."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.batched_blr import design


def check_product(matrix, response, observed):
    # Compare to a float64 reference against the absolute dot-product roundoff bound.
    # Some rotated coordinates nearly cancel; result-relative tolerances misclassify them.
    a, b = matrix.astype(np.float64), response.astype(np.float64)
    n_eps = 2 * a.shape[1] * np.finfo(np.float32).eps
    errors = (n_eps / (1 - n_eps)) * (abs(a) @ abs(b))
    if not np.all(abs(observed - a @ b) <= errors + np.finfo(np.float32).tiny):
        raise RuntimeError('projection differs from the float64 reference beyond FP32 dot-product roundoff')


def run_cpu(matrix, response):
    # Row-major columns-as-rows avoid charging Python copies on the serial-only path.
    rows = np.ascontiguousarray(response.T)
    loops = 80
    samples = []
    result = [matrix.dot(row) for row in rows]
    for i in range(loops + 2):
        start = time.perf_counter()
        result = [matrix.dot(row) for row in rows]
        elapsed = time.perf_counter() - start
        if i >= 2:
            samples.append(elapsed)
    check_product(matrix, response, np.stack(result).T)
    return samples, len(rows), platform.platform()


def run_gpu(matrix, response):
    import jax
    import jax.numpy as jnp

    devices = jax.devices('gpu')
    assert len(devices) == 1 and 'A100' in devices[0].device_kind, devices
    n = 165402
    w, r = jax.device_put((matrix, np.ascontiguousarray(response[:,np.arange(n)%response.shape[1]])), devices[0])

    @jax.jit
    def project(w, r):
        return jnp.dot(w, r, precision=jax.lax.Precision.HIGHEST)

    jax.block_until_ready((w,r))
    result = jax.block_until_ready(project(w,r))
    for _ in range(2):
        jax.block_until_ready(project(w,r))
    samples = []
    for i in range(24):
        start = time.perf_counter()
        result = jax.block_until_ready(project(w,r))
        samples.append(time.perf_counter()-start)
        print(f'[{i+1}/24] projection: {samples[-1]:.6f} s', flush=True)
    check_product(matrix, response, np.asarray(result[:,:response.shape[1]]))
    return samples, n, devices[0].device_kind


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--no-profile',action='store_true',required=True,help='component timers or external profiling')
    args = p.parse_args()
    assert hashlib.sha256(args.input.read_bytes()).hexdigest() == args.sha256
    with np.load(args.input) as d:
        phi, y = d['phi'], d['y']
    assert phi.shape == (450,12) and y.shape[0] == 450
    matrix = np.ascontiguousarray(design(phi,phi[:0]).G, dtype=np.float32)
    response = np.sinh(np.e*(np.arcsinh(y)+1)).astype(np.float32)
    samples, n, device = (run_cpu if args.cpu else run_gpu)(matrix,response)
    record = {'median_seconds':statistics.median(samples), 'seconds':samples,
              'measured_columns':n, 'unique_columns':y.shape[1], 'n_train':450, 'design_columns':12,
              'device':device, 'dtype':'float32', 'job_id':os.environ.get('SLURM_JOB_ID'),
              'operation':'G @ z only; precomputed shared basis, synthetic input; no transfers', 'input_sha256':args.sha256}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp = args.output.with_suffix('.partial')
    tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    tmp.replace(args.output)


if __name__ == '__main__':
    main()
