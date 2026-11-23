# Which initially flagged clusters are sensitive to the reference group

Run the full path in a Linux Slurm or Hugging Face job with the IXI artifacts and a CUDA-enabled JAX install.

```sh
python run.py --synthetic --resources
```

The synthetic path generates tiny fixtures and tests array schemas and cluster handling on CPU.

Original flags use strict `abs(z) > 1.96`. Connected components use a specified 3-D neighbourhood.

Use `--per-refit --fold <n>` to save all flags for one of the five folds.

`--z-summary` stores the original z and the minimum and maximum absolute z over refits.

`--refit-flag-cutoffs` stores flags at extra cutoffs; it also requires `--z-summary`.

A voxel kept in all 64 refits is distinct from a voxel kept in at least 52 of 64 refits.

The [cloud helpers](../../../shared/hugging_face_jobs/README.md) verify mounts and downloaded receipts.
