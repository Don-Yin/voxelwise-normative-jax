# results

## whole-mask runtime

[exp_010](../experiments/exp_voxelwisenormative_010_reference_size_scaling/docs/details.md) timed fit and score for a 165,402-voxel mask and 113 held-out rows. [timing table](figures/data/speed.csv)

| reference design | PCNtoolkit sample on one CPU core | projected full mask | measured on one A100, 64-bit | projected time ÷ measured time |
|---|---|---|---|---|
| IXI, 450 scans | 88.440 s / 256 voxels | 15.873 h | 2.779 s | about 20,600× |
| synthetic, 50,000 rows | 188.212 s / 24 voxels | 360.309 h | 302.031 s | about 4,300× |

PCNtoolkit timings include fit and prediction, not setup and imports. the A100 values are medians of three warm fits, including transfers and score collection but not compilation or input setup. multiply CPU seconds per sampled voxel by the full mask width, then compare in the same units. we did not time a full-mask single-core fit. the figure offsets the two markers within each design for legibility and does not connect the designs.

## numerical agreement

| comparison | paired sample | absolute z-score difference |
|---|---|---|
| [exp_003: batched fit](figures/data/batched-fit.csv) | 1,000 pinned mask voxels, 113 held-out scans | 99th percentile 0.0000609; maximum 0.00110 |
| [exp_004: fastest passing A100 variant](figures/data/speed-and-error.csv) | same pinned sample; maximum error limit 0.01 | maximum 0.003664; the two plotted TensorFloat-32 variants fail the limit |
| [exp_007: whole mask](figures/data/full-mask-agreement.csv) | 18,690,426 held-out score pairs | 99.9th percentile 0.000911; maximum 4.22 |

[exp_007 details](../experiments/exp_voxelwisenormative_007_pcntoolkit_full_mask/docs/details.md) describe a limitation: on two strongly skewed synthetic columns the solver reached stationary points about 148 and 162 nats worse than PCNtoolkit. two IXI voxels shared the fitted parameter pattern; their objective gaps were not checked. the inspected sample agreement does not guarantee a small error at every voxel.

## flag recurrence

an original flag is a held-out scan–voxel pair with `|z| > 1.96` before resampling. in both runs the denominator is 927,451 original flags pooled across 113 held-out scans, with 450 IXI reference scans. new flags are excluded. [flag count table](figures/data/flag-stability.csv)

| run | rule | count and share of original flags |
|---|---|---|
| [exp_010](../experiments/exp_voxelwisenormative_010_reference_size_scaling/docs/details.md) | flagged in fewer than 52 of 64 refits | 200,530 / 927,451 filtered (21.6%) |
| [exp_010](../experiments/exp_voxelwisenormative_010_reference_size_scaling/docs/details.md) | flagged in at least 52 of 64 refits | 726,921 / 927,451 kept (78.4%) |
| [exp_011, fold 0](../experiments/exp_voxelwisenormative_011_fragile_clusters/docs/details.md) | flagged in all 64 refits | 416,394 / 927,451 kept (44.9%) |

these are separate resampling runs, not shares of all voxels. the [cluster plot](figures/data/fragile-share-no-core.csv) uses a different denominator: initially flagged clusters, pooled by size, using 26-neighbour connectivity. a cluster has a core if at least one voxel in it is flagged in all 64 refits.

## other results

| evidence | result |
|---|---|
| [exp_002 PCNtoolkit convergence](figures/data/convergence.csv) | tightening the finite-difference step lowered the penalised objective on the pinned sample. the 3.4 h sequential extrapolation used default numerical-library threads, not the one-core protocol above. |
| [exp_006 voxel scaling](figures/data/speed-memory.csv) | at full mask width, 32-bit early stop took 1.674 s and 0.461 GB live device memory versus 7.114 s and 2.040 GB for the original 64-bit fit on one A100. |
| [exp_013 refit batching](figures/data/refit-batching.csv) | batches of eight cut the 32-bit time from 1.418 to 1.138 s/refit but changed 4,298 flag decisions across the 64 draws. 64-bit batching changed no flags in this sweep. |
| [exp_017 SynthSR quality](figures/data/synthsr-map-quality.csv) | median Pearson correlation with the leave-one-out grey-matter mean: 0.9014 for 57 retained crops and 0.9003 for 506 rebuilt crops. |

[all experiments](../experiments/README.md) · [measurement protocol](003-experiment-protocol.md#measure-and-plot) · [figure sources](figures/README.md)
