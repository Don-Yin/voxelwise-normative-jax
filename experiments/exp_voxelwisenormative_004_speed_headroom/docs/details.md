# Can early stopping reduce total time without failing the z check

The comparison uses the 1,000-column converged reference and checks all full-mask scores for finiteness.

A timed variant passes when its maximum absolute z error on the reference columns is at most 0.01.

One-GPU and four-GPU launchers are in `slurm/`. Run the synthetic unit tests from the project root.

The fastest passing variant used blocks of 32,768 columns on one A100.

TensorFloat-32 and a larger early-stop chunk exceeded the z tolerance in this benchmark.

The early-stop float32 route also exceeded the tolerance in the CPU comparison.

The mechanism probe generates its own input. For a local CPU run, use:

```sh
PYTHONPATH=.:../.. JAX_PLATFORMS=cpu python src/modules/mechanism_timing.py \
  --smoke-cpu --output .results/json/mechanism.json
```

[hugging face calibration](hugging-face-calibration.md)
