# Can similar original z-scores have different recurrence counts

Generate the full baseline fit in experiment 003 and a 64-draw archive in experiment 005.

Pass the baseline path and digest, bootstrap archive, and its separate receipt explicitly.

```sh
python run.py --baseline <z.npy> --baseline-sha256 <digest> \
  --bootstrap <bootstrap.npz> --receipt <result.json>
```

The scan verifies both inputs, checks cohort and voxel order, and uses CPU JAX to replay the draws.

Original candidates have absolute z between 3.0 and 3.1. A pair must differ by at most 0.02.

Ranking prioritizes pairs with at most 15 and at least 60 flags across 64 draws.
