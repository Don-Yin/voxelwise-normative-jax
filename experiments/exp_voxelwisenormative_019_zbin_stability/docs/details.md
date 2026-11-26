# bootstrap flag stability by original |z|

## measured result

| original absolute z | original flags | unstable flags | pooled share | person p10 / p50 / p90 |
|---|---:|---:|---:|---|
| above 1.96 to below 2.5 | 670,779 | 200,479 | 29.8875% | 24.27% / 28.13% / 36.83% |
| 2.5 to below 3.1 | 196,897 | 51 | 0.0259% | 0% / 0% / 0% |
| 3.1 and above | 59,775 | 0 | 0% | 0% / 0% / 0% |
| all flags | 927,451 | 200,530 | 21.6216% | 17.52% / 20.69% / 27.21% |

all 113 held-out people had flags in every bin. the last column is the 10th, 50th and 90th percentile across people.

[aggregate receipt](../../../docs/figures/data/zbin-stability.json) · [regenerate the figure](../../../docs/figures/README.md)

## method

- use 450 public IXI reference scans, 113 held-out scans and 165,402 mask voxels. the run used one NVIDIA A100 on Hugging Face Jobs.
- reuse [exp_010](../../exp_voxelwisenormative_010_reference_size_scaling/README.md)'s 64-bit fit and site-stratified bootstrap.
- keep the held-out rows fixed. resample reference rows with fixed counts per site, then recompute response scaling on each draw.
- chain 0 uses `default_rng([20261124, 20261121, 450])`: the draw seed, the chain seed and the reference size.
- count only original flags (`|z| > 1.96`). mark one unstable if it recurs in fewer than 52 of 64 refits at the same threshold.
- exactly 1.96 is excluded. exactly 2.5 or 3.1 enters the higher bin. the bin depends on the original magnitude.
- divide each bin's unstable count by all its original flags for the pooled share. each person gets equal weight for the percentile summary.
- a person's fraction uses that person's original flags in the bin; omit people with no flags in that bin.
- require both pooled totals to match [exp_010](../../../docs/004-results.md#flag-recurrence), 200,530 unstable of 927,451 original flags.

## run

install the package and use your own Hugging Face Jobs account and bucket.

run from the repository root. the command submits a paid job.

prepare the inputs with the [local protocol](../../../docs/003-experiment-protocol.md#prepare-ixi-locally); the job stages no mask.

```sh
exp=experiments/exp_voxelwisenormative_019_zbin_stability
run=hugging-face-a100-zbin-benchmark
input="$exp/.intermediates/$run"
bucket='namespace/name'  # replace with your bucket
python -m shared.hugging_face_jobs.build_input --profile zbin-stability "$input"
hf jobs run --flavor a100-large --timeout 20m --attempts 1 -d --name "$run" \
  -v "$input:/code:ro" -v "hf://buckets/$bucket:/out" \
  python:3.14.3 bash /code/shared/hugging_face_jobs/container_entry.sh \
  --run-name "$run" "$exp" --run-name @RUN_NAME@
```

when the job succeeds, fetch the results from your own bucket. use a new run name for each repeat.

```sh
hf sync "hf://buckets/$bucket/$run" "$exp/.results/hugging-face/$run"
```

the CPU smoke run uses generated inputs. the bin tests check the thresholds and denominators separately.

```sh
JAX_PLATFORMS=cpu python "$exp/run.py" --synthetic --run-name synthetic-smoke
```

## outputs

| path under the run's output root | contents |
|---|---|
| `json/<run>/zbin-stability.json` | cohort totals, bin totals, across-person percentiles and timings |
| `profile/run.html` | default Python profile; use `--no-profile` on the entry to skip it |

original magnitudes and recurrence counts stay in memory. no individual score, count or fraction is shipped.
