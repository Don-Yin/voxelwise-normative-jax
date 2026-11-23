# How closely do the CPU and batched fits agree across the full mask

The reference is PCNtoolkit 1.4.0 on supported Python 3.12–3.13, with finite-difference step 1e-7.

The small parity stage compares the CPU library across environments on 20 separately saved columns.

Use experiment 002 with `--n-voxels 20 --epsilon 1e-7` and freeze that comparison artifact locally.

The full fit gives each worker explicit mask columns. Merge checks exact coverage and finite scores.

Submit stages in `slurm/` with your own account and job IDs. Export those IDs explicitly.

The full check also requires the frozen full-mask batched fit from experiment 003.

To exercise the solvers with generated data and no GPU, run [the synthetic suite](../../../scripts/test.sh).

On two strongly skewed synthetic columns, JAX reached stationary points 148 and 162 nats above PCNtoolkit.

Two of 165,402 IXI voxels matched that pattern (`log b < -2`, `log β > 5`); we did not check their objective gaps.

[outlier audit](outlier-audit.md)

[stopping sweep](stopping-sweep.md)

[gradient benchmark](gradient-benchmark.md)
