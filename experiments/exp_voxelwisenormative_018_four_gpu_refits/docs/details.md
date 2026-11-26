# four A100s for independent refits

## measured result

one timed pass per device count on the same Hugging Face Jobs node with four NVIDIA A100 GPUs.

| setup | warm wall time | throughput |
|---|---:|---:|
| one A100, measured | 151.251 s | 0.423 refits/s |
| four A100s, measured | 38.915 s | 1.645 refits/s |
| ideal four-device scaling, one-device time ÷ 4 | 37.813 s | — |

four devices gave 3.887× the throughput of one, or 97.17% of ideal fourfold scaling. this run gives no estimate of repeat variability.

all 1,196,187,264 paired z-scores matched bit for bit across 64 refits; no `|z| > 1.96` flag changed.

[aggregate receipt](../../../docs/figures/data/four-gpu-refits.json) · [regenerate the figure](../../../docs/figures/README.md)

## comparison

- fit 450 IXI reference scans and score 113 held-out scans at 165,402 mask voxels using the verified split and draws from [exp_013](../../exp_voxelwisenormative_013_gpu_headroom/README.md).
- run all 64 bootstrap draws on device 0, then run the same draws in four contiguous groups of 16 on four devices.
- use the same 64-bit early-stopping Newton fit and 32,768-voxel chunks in both passes; fit one draw per call.
- compile and warm each device outside the timer. measure warm fit and host score collection, excluding input loading and verification.
- check the original fit against the pinned PCNtoolkit sample: maximum |Δz| was 0.001104, below the 0.01 limit.
- compare all finite score pairs in draw order. require maximum |Δz| ≤ 10⁻⁸ and zero threshold flips.

## one chunk on one A100

- a trace of draw 0, chunk 0 (32,768 voxels), outside timing had a 97.61% device-stream busy fraction.
- two groups around the Newton linear solve took 97.66% of GPU kernel time in 102 observed iterations.
- the largest individual kernel took 9.42% of summed GPU kernel time. groups of kernels and individual kernels are counted separately.

## run

install the package and use your own Hugging Face Jobs account and bucket.

run from the repository root. the command submits a paid job.

prepare the IXI matrix and pinned PCNtoolkit inputs using the [local protocol](../../../docs/003-experiment-protocol.md#prepare-ixi-locally).

the staging profile verifies both artifacts and stages just this experiment and the needed inputs; it omits the mask.

```sh
exp=experiments/exp_voxelwisenormative_018_four_gpu_refits
run=hugging-face-a100-four-device-benchmark
input="$exp/.intermediates/$run"
bucket='namespace/name'  # replace with your bucket
python -m shared.hugging_face_jobs.build_input --profile four-gpu-refits "$input"
hf jobs run --flavor a100x4 --timeout 30m --attempts 1 -d --name "$run" \
  -v "$input:/code:ro" -v "hf://buckets/$bucket:/out" \
  python:3.14.3 bash /code/shared/hugging_face_jobs/container_entry.sh \
  --run-name "$run" "$exp" --run-name @RUN_NAME@
```

when the job succeeds, fetch the results from your own bucket. use a new run name for each repeat.

```sh
hf sync "hf://buckets/$bucket/$run" "$exp/.results/hugging-face/$run"
```

for a local check without IXI or CUDA, use four logical CPU devices on generated inputs:

```sh
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  python "$exp/run.py" --synthetic
```

## outputs

| path under the run's output root | contents |
|---|---|
| `json/<run>/refits.json` | timings, device preparation and whole-mask score agreement |
| `json/<run>/trace.json` | device-stream activity and ranked kernel groups and individual kernels |
| `profile/run.html` | default Python profile; use `--no-profile` on the entry to skip it |
| `profile/jax-trace/` | trace of one warm call, outside timed work |

z-score arrays are compared in host memory and are not saved. only the aggregate excerpt and figure are shipped here.
