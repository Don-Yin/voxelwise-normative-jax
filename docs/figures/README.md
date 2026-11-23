# figures

regenerate the result images with Matplotlib 3.10 or later. the [standard builder](result/standard.py) reads the small aggregate tables below and uses the [PCNtoolkit icon](../assets/pcntoolkit-icon.png) in experiment 010. no scans, fitted arrays or private plotting modules are needed.

```sh
python -m pip install 'matplotlib>=3.10'
python -B docs/figures/result/standard.py all
```

replace `all` with one experiment number, or use `--outdir <directory>` to write outside [the asset directory](../assets/). each result uses linear scales; cluster bins and workload names are categories, not continuous values.

| experiment | aggregate inputs and measure |
|---|---|
| [exp_002](../../experiments/exp_voxelwisenormative_002_pcntoolkit_cpu_baseline/docs/details.md) | [histogram counts](data/convergence.csv): paired objective and z-score changes between default and converged PCNtoolkit. |
| [exp_003](../../experiments/exp_voxelwisenormative_003_batched_jax_fit/docs/details.md) | [batched fit](data/batched-fit.csv): best of three warm full-mask JAX times and paired score differences on sampled voxels. |
| [exp_004](../../experiments/exp_voxelwisenormative_004_speed_headroom/docs/details.md) | [variant sweep](data/speed-and-error.csv): median whole-mask time and pinned z error; the split panels use the 0.01 error limit. |
| [exp_006](../../experiments/exp_voxelwisenormative_006_speed_memory_scaling/docs/details.md) | [voxel sweep](data/speed-memory.csv): median single-A100 time and peak live allocated bytes. |
| [exp_007](../../experiments/exp_voxelwisenormative_007_pcntoolkit_full_mask/docs/details.md) | [whole-mask summaries](data/full-mask-agreement.csv): two statistics from paired absolute z differences, with separately labelled scales. |
| [exp_010](../../experiments/exp_voxelwisenormative_010_reference_size_scaling/docs/details.md) | [timing samples](data/speed.csv) and [pooled flag counts](data/flag-stability.csv): one-core projections, A100 medians and the separate IXI refit run. |
| [exp_011](../../experiments/exp_voxelwisenormative_011_fragile_clusters/docs/details.md) | [cluster bins](data/fragile-share-no-core.csv): share of initially flagged clusters with no voxel flagged in all 64 refits. |
| [exp_013](../../experiments/exp_voxelwisenormative_013_gpu_headroom/docs/details.md) | [batch sweep](data/refit-batching.csv): timed work divided by 64 refits, cumulative allocator peak and disagreements with batch size one. |
| [exp_017](../../experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/docs/details.md) | [map quality](data/synthsr-map-quality.csv): two group medians; each map is compared with the mean of the other maps on the same mask. |

see [results and denominators](../004-results.md). the [social preview builder](logo/render-logo.sh) centres the existing voxel-brain icon with ImageMagick; see [figure marks](../licences.md#figure-marks).
