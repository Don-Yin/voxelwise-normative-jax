"""Warm timing windows for the code-explainer slides, using synthetic PCNtoolkit inputs."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from shared.voxelwisenormative.batched_blr import N_ITER, design
from shared.voxelwisenormative.early_newton import make_variant


def timed_parts(phi, responses, n_voxels, chunk, repeats, device):
    # H and y_test are dummy zero-row arrays: this profile only returns the optimizer outputs.
    wall = time.perf_counter
    decomp = []
    d = design(phi, phi[:0])
    for _ in range(100):
        start = wall()
        d = design(phi, phi[:0])
        decomp.append(wall() - start)
    pool = np.arange(n_voxels) % responses.shape[1]
    y = np.ascontiguousarray(responses[:, pool], dtype=np.float32)

    def put(a):
        return jax.device_put(np.asarray(a, dtype=np.float32), device)

    lam, gram_project, holdout_design = (put(a) for a in (d.lam,d.G,d.H))
    y_chunks = [put(y[:, start:start+chunk]) for start in range(0,n_voxels,chunk)]
    y_holdout = [put(np.empty((0,a.shape[1]))) for a in y_chunks]
    z_chunks = [jnp.sinh(jnp.exp(jnp.float32(1)) * (jnp.arcsinh(a)+1)) for a in y_chunks]
    # Runtime inputs prevent the compiler from folding the common starting values across voxels.
    precision = [put(np.full((2,z.shape[1]), np.e)) for z in z_chunks]
    jax.block_until_ready((z_chunks,y_holdout,precision))

    @jax.jit
    def posterior_once(lam, g, z, precision):
        beta, alpha = precision
        den = lam[:,None]*beta + alpha
        m = (beta*(g @ z)) / den
        return m, jnp.sum(jnp.log(den), axis=0)

    def warm_test(kernel, args_list, window):
        result = [kernel(*a) for a in args_list]
        jax.block_until_ready(result)
        elapsed, enqueue = [], []
        for i in range(repeats):
            start = wall()
            result = [kernel(*a) for a in args_list]
            enqueued = wall()
            jax.block_until_ready(result)
            elapsed.append(wall()-start)
            enqueue.append(enqueued-start)
            print(f'[{i+1}/{repeats}] {window}: {elapsed[-1]:.6f} s', flush=True)
        return elapsed, enqueue, result

    post_args = [(lam,gram_project,z,p) for z,p in zip(z_chunks,precision,strict=True)]
    post, _, post_result = warm_test(posterior_once, post_args, 'posterior')
    gram = phi.T @ phi
    _, q = np.linalg.eigh(gram)
    first = responses[:,0]
    z0 = np.sinh(np.e * (np.arcsinh(first)+1))
    dense_means = np.linalg.solve(np.e*(gram+np.eye(gram.shape[0])), np.e*(phi.T@z0))
    np.testing.assert_allclose(q @ np.asarray(post_result[0][0])[:,0], dense_means, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(float(post_result[0][1][0]), np.linalg.slogdet(np.e*(gram+np.eye(gram.shape[0])))[1], atol=2e-4)

    def compiler(early):
        fit = make_variant(analytic=True, early=early, tf32=False)

        @jax.jit
        def opt(lam,g,h,y,yt):
            hyp, _, _, counts = fit(lam,g,h,y,yt)
            return hyp, counts

        return opt

    cases = [(lam,gram_project,holdout_design,a,b) for a,b in zip(y_chunks,y_holdout,strict=True)]
    early, enqueue, results = warm_test(compiler(True), cases, 'optimizer early stop')
    fixed, _, fixed_results = warm_test(compiler(False), cases, 'optimizer fixed cap')
    counts = [np.asarray(count) for _,count in results]
    assert all(np.all(np.isfinite(np.asarray(hyp))) for hyp,_ in (*results,*fixed_results))
    assert all(np.all((count>0) & (count<=N_ITER)) for count in counts)
    return {'n_voxels':n_voxels, 'n_unique_responses': responses.shape[1],
            'n_train':phi.shape[0], 'design_columns':phi.shape[1], 'chunk':chunk,'repeats':repeats,
            'device': device.device_kind, 'jax': jax.__version__, 'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
            'medians_seconds': {'host_eig_and_rotation':statistics.median(decomp),
                                'rotated_posterior_pass':statistics.median(post),
                                'early_optimizer':statistics.median(early),
                                'fixed_optimizer':statistics.median(fixed),
                                'enqueue_only':statistics.median(enqueue)},
            'seconds':{'rotated_posterior_pass':post, 'early_optimizer':early, 'fixed_optimizer':fixed,
                       'enqueue_only':enqueue}, 'early_block_steps': [int(c.max()) for c in counts],
            'fixed_block_steps': [N_ITER]*len(counts)}


SYNTHETIC_SEED = 20261120


def synthetic_input() -> tuple[np.ndarray, np.ndarray]:
    """Generate a 450×12 design and 128 skewed response columns without loading data."""
    rng = np.random.default_rng(SYNTHETIC_SEED)
    phi = np.column_stack((np.ones(450), rng.normal(size=(450, 11))))
    y = 0.1 * phi[:, [1]] + rng.lognormal(sigma=0.45, size=(450, 128))
    y = (y - y.mean(0)) / y.std(0)
    return phi, y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-profile',action='store_true',help='use the enclosing profiler')
    p.add_argument('--smoke-cpu',action='store_true',help='run the small CPU fixture')
    a = p.parse_args()
    from src.utils.profiling import profiled

    phi, y = synthetic_input()
    with profiled(None if a.no_profile else a.output.parent / 'profile' / 'mechanism-timing.html'):
        result = measure(a, phi, y)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    p = a.output.with_suffix('.partial')
    p.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    p.replace(a.output)


def measure(a, phi, y):
    if a.smoke_cpu:
        result = timed_parts(phi, y, 131, 128, 1, jax.devices('cpu')[0])
    else:
        gpu = jax.devices('gpu')
        assert len(gpu) == 1 and 'A100' in gpu[0].device_kind, gpu
        result = timed_parts(phi, y, 165402, 32768, 7, gpu[0])
    result.update(synthetic_seed=SYNTHETIC_SEED, smoke_cpu=a.smoke_cpu)
    return result


if __name__ == '__main__':
    main()
