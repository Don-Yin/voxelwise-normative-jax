# Why automatic relevance determination changes held-out scores

## Model and comparison

Automatic relevance determination (ARD) gives each mean coefficient its own inverse prior variance.

All subjects share one noise precision per voxel. Pass `ard=True` without `heteroskedastic=True`.

`pcntoolkit-blr-reference-n1000-v1/arrays.npz` supplies `voxels`, `test` and both mean designs.

Its `y_train` and `y_test` arrays contain responses scaled on training scans only.

`ixi-grey-matter-voxel-matrix-v1/covariates.tsv` supplies age, sex and site. We copy neither input.

Both implementations fit the same sinh-arcsinh response transform and linear mean model.

The objective is negative log evidence minus the log warp Jacobian, plus a parameter penalty.

PCNtoolkit uses serial `BLR.fit` and `BLR.forward` with finite-difference gradients.

JAX uses batched Cholesky solves, exact gradients and a separate search per voxel.

JAX tries the all-ones start, identity warp with several priors, and shifted warps.

It keeps the lowest finite objective for each voxel. The package needs no PCNtoolkit parameters.

## Results

Keys below refer to `.results/json/benchmark-1000.json` unless noted otherwise.

The sample has 1,000 voxels and 450/113 train/test scans (`dataset.{n_voxels,n_train,n_test}`).

The design has 12 columns and rank 10 (`dataset.{mean_design_columns,mean_design_rank}`).

| Fit and score on Mac CPU | Measured | Key |
|---|---:|---|
| PCNtoolkit, serial | 558.706 s | `pcntoolkit.total_seconds` |
| JAX, first call with compilation | 7.952 s | `jax_cpu.first_fit_with_precompute_seconds` |
| JAX, warm median | 9.126 s | `jax_cpu.warm_seconds_median` |
| PCNtoolkit time divided by warm JAX time | 61.2× | `jax_cpu.warm_speedup_over_pcntoolkit` |
| Maximum score gap at identical parameters | 2.50e−13 | `agreement.same_hyp_z_max_absdiff` |
| Maximum objective gap at identical parameters | 1.62e−12 | `agreement.same_hyp_objective_max_absdiff` |

Both timings used the shared lock. The ratio compares fit and score time, not identical scores.

The margins are 0.001 for the objective and 0.01 for scores (`agreement.{objective_tolerance,z_tolerance}`).

| JAX fits compared with PCNtoolkit | All ones | Deployed | Key |
|---|---:|---:|---|
| Voxels with any score gap above the margin | 211 | 194 | `voxels_gt_z_tolerance` |
| Maximum absolute score gap | 0.874 | 0.325 | `z_absdiff_max` |
| Voxels where JAX has the lower objective | 189 | 257 | `voxels_jax_objective_better` |
| Voxels where PCNtoolkit has the lower objective | 79 | 1 | `voxels_jax_objective_worse` |
| Largest JAX objective excess | 11.008 | 0.777 | `objective_jax_minus_pcn_max` |

The all-ones results are in `initialization_audit.all_ones`; the deployed results are in `agreement`.

Among the 211 original score gaps, JAX wins the objective in 92, PCNtoolkit in 79, with 40 near ties.

After restarts, among 194 score gaps, JAX wins in 156, PCNtoolkit in 1, with 37 near ties.

Those counts are under `z_disagreement_by_objective` for each fit. See `jax_cpu.starts` for the starts.

The remaining worse fit is voxel 82405, where `jax_objective - objective` is largest.

Those arrays are in `tables/fit-arrays-1000.npz` and `tables/pcntoolkit-1000.npz`, respectively.

Restarts improve 145 objectives beyond the margin (`initialization_audit.deployed_voxels_improving_all_ones`).

Starting from PCNtoolkit's fit gives no worse objectives but still leaves 97 voxels above the score margin.

That diagnostic uses `initialization_audit.pcn_seeded`; the deployed solver does not use it.

### What caused the largest gap?

`initialization_audit.worst_all_ones_voxel` identifies mask column 53684.

PCNtoolkit's objective there was 176.402710 (`pcn_objective`).

| JAX fit (`fits` key) | Objective | Max score gap | Smallest Hessian eigenvalue |
|---|---:|---:|---:|
| `all_ones` | 187.410900 | 0.874340 | 0.02346 |
| `pcn_seeded` | 176.402638 | 0.000746 | 0.02659 |
| `deployed` | 176.402638 | 0.000747 | 0.02659 |

The columns use `objective`, `z_max_absdiff_vs_pcn`, and `hessian_min_eigenvalue`.

Small gradients and positive curvature identify distinct local minima of the same fitting objective.

Restarts recover the better fit for this voxel, but finite restarts do not guarantee the global minimum.

The training objective does not constrain held-out scores to match PCNtoolkit's.

## Reproduce on Mac CPU

From the project root, run the benchmark under the shared timing lock:

```bash
H="$HOME/system-macos/active-work/projects/2026-11-22-hackathon-scientific-parallelization"
A="$H/experiments/shared/artifacts"
E=experiments/exp_voxelwisenormative_021_ard_blr
VNJ_BENCH_LOCK_OWNER=ard lockf -k /tmp/vnj-bench.lock \
  micromamba run -n voxelwise-normative-jax python -B "$E/run.py" --artifacts-dir "$A" --no-profile
```

`--phase pcn` runs just the reference. `--phase jax` verifies and reuses that reference.

Omit `--no-profile` to profile the work. For a short check, use `--voxels 4 --repeats 1`.

| Output under `.results/` | Content |
|---|---|
| `json/benchmark-1000.json` | Timings, agreement and restart audit. |
| `tables/pcntoolkit-1000.npz` | Reference: `voxels`, `hyp`, `objective`, `z`, and timings. |
| `tables/fit-arrays-1000.npz` | Aligned arrays: `all_ones_`, `pcn_seeded_`, and `jax_` prefixes. |
| `profile/run.html` | Profile of the short check. Wall-time measurements use no profiler. |
| `review/00-cpu-fit-and-score/` | Checker reports, figure renders, and source manifest. |

Regenerate the figure from the full run:

```bash
S="$HOME/system-macos/active-work/skills/python-plotting-convention"
B="$E/.results/review/00-cpu-fit-and-score"
bash "$S/scripts/checker.sh" docs/figures/exp_voxelwisenormative_021.py 16 --outdir "$B"
P="$B/render/exp_voxelwisenormative_021-display.png"
cp "$P" "$E/.results/00-cpu-fit-and-score.png"
cp "$P" docs/assets/exp_voxelwisenormative_021-result.png
```
