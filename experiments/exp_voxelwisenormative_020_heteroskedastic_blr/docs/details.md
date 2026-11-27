# Heteroskedastic warped Bayesian linear regression

## Inputs and method

The loader verifies the frozen IXI grey-matter reference `pcntoolkit-blr-reference-n1000-v1/arrays.npz`.

It pairs that reference with the `ixi-grey-matter-voxel-matrix-v1/covariates.tsv` table in the same artifact store.

| Block | Specification |
|---|---|
| Responses | Grey matter scaled with training means and standard deviations; held-out rows use the same scaling. |
| Mean | Cubic age spline, linear sex term, intercept and site indicators. |
| Noise | `log(beta) = Phi_var @ w`, with training-scaled age, sex and an intercept. Each voxel has its own `w`. |
| Prior | One scalar weight precision per voxel; automatic relevance determination is off. |
| Warp | The same sinh–arcsinh transform and log Jacobian in both implementations. |
| Objective | Negative log evidence minus the warp log Jacobian plus a penalty on the hyperparameter norm. |
| JAX math | Form every weighted Gram matrix with one matrix multiplication over precomputed subject outer products. |
| JAX optimiser | Batched Cholesky and triangular solves; exact JAX gradients and independent BFGS line searches. |
| PCNtoolkit optimiser | Unmodified serial `BLR.fit` and `BLR.forward`, using finite-difference L-BFGS-B. |

Both solvers start at ones; the independent fit compares their final held-out z-scores.

A second check gives JAX the PCNtoolkit parameters and compares both objective and z-scores without optimising.

## Measured CPU times and agreement

All keys refer to `.results/json/benchmark-1000.json`. Times are wall-clock seconds.

| JSON key | Value |
|---|---|
| `host.cpu` | Apple M4 Max |
| `host.jax_backend` | CPU |
| `pcntoolkit.version` | 1.4.0 |
| `host.jax` | 0.12.1 |
| `host.shared_timing_lock_owner` | `00-hetero-blr` |
| `dataset.n_voxels` | 1,000 |
| `dataset.n_train` | 450 |
| `dataset.n_test` | 113 |
| `dataset.mean_design_columns` | 12 |
| `dataset.mean_design_rank` | 10 |
| `dataset.noise_design_columns` | 3 |
| `pcntoolkit.design_max_absdiff` | 5.55 × 10⁻¹⁶ |
| `pcntoolkit.total_seconds` | 36.340 s |
| `jax_cpu.first_fit_with_precompute_seconds` | 1.009 s |
| `jax_cpu.warm_seconds` | 0.868, 0.654, 0.862 s |
| `jax_cpu.warm_seconds_median` | 0.862 s |
| `jax_cpu.warm_speedup_over_pcntoolkit` | 42.2× |
| `agreement.n_test_scores` | 113,000 |
| `agreement.z_tolerance` | 0.01 |
| `agreement.z_fraction_within_tolerance` | 100% |
| `agreement.z_absdiff_median` | 3.34 × 10⁻⁶ |
| `agreement.z_absdiff_p99` | 9.28 × 10⁻⁵ |
| `agreement.z_absdiff_max` | 0.001124 |
| `agreement.same_hyp_z_max_absdiff` | 8.03 × 10⁻¹⁴ |
| `agreement.same_hyp_objective_max_absdiff` | 1.82 × 10⁻¹² |
| `agreement.jax_voxels_at_grad_tol` | 1,000 |
| `agreement.jax_grad_tol` | 3 × 10⁻⁵ |

PCNtoolkit uses `pcntoolkit.l_bfgs_b_epsilon` for finite differences; both solvers use `pcntoolkit.penalty`.

The JAX iteration cap and precision are recorded in `jax_cpu.max_iters` and `jax_cpu.dtype`.

## Reproduce on CPU

Set `A` to the existing artifact store, not to a copy of the frozen arrays. Run from the project root:

```bash
A=/path/to/experiments/shared/artifacts
E=experiments/exp_voxelwisenormative_020_heteroskedastic_blr
C="$PWD/$E/.intermediates"
mkdir -p "$C/pytensor" "$C/matplotlib"
export PYTENSOR_FLAGS="base_compiledir=$C/pytensor" MPLCONFIGDIR="$C/matplotlib"
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
run=(micromamba run -n voxelwise-normative-jax python "$E/run.py")
"${run[@]}" --artifacts-dir "$A" --voxels 4 --repeats 1
VNJ_BENCH_LOCK_OWNER=local lockf -k /tmp/vnj-bench.lock "${run[@]}" --artifacts-dir "$A" --no-profile
```

The first command is a short default-profile check. The unprofiled full run waits for the shared CPU timing lock.

Both timers include held-out scoring and result collection, but exclude process startup and reading the input.

The JAX first fit includes compilation and outer-product precomputation. Warm fits reuse both.

The speed ratio compares one serial PCNtoolkit pass with the median warm JAX pass on the same CPU.

It includes optimizer and CPU parallelism differences; it does not isolate the matrix multiplication.

## Outputs and figure

| Output | Contents |
|---|---|
| `.results/json/benchmark-1000.json` | Input dimensions, host, timings and paired agreement. |
| `.results/tables/fit-arrays-1000.npz` | Paired z-scores, optimum parameters, objectives, gradients and serial voxel times. |
| `.results/profile/run.html` | Actual work, written by default; `--no-profile` only disables profiling. |
| `.results/00-cpu-fit-agreement.png` | Fit-and-score wall times and the empirical distribution of absolute z differences. |
| `.results/review/00-cpu-fit-agreement/` | Figure checker bundle: reports, source manifest, PNGs and vector PDF. |

The numerical entry exits nonzero if either agreement gate fails. Both tolerances are in the report.

To check and regenerate the summary figure:

```bash
S="$HOME/system-macos/active-work/skills/python-plotting-convention"
F=docs/figures/exp_voxelwisenormative_020.py
B="$E/.results/review/00-cpu-fit-agreement"
bash "$S/scripts/checker.sh" "$F" 16 --outdir "$B" && (
  PNG="$B/render/exp_voxelwisenormative_020-display.png"
  cp "$PNG" "$E/.results/00-cpu-fit-agreement.png"
  cp "$PNG" docs/assets/exp_voxelwisenormative_020-result.png
)
```
