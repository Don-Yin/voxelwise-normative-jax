# How does the size of the reference group change cost and recurrence

The real grid uses 25–450 IXI reference scans. The size of 50,000 comes from a generated design only.

CPU fits time fit and prediction on one core and project by mask width. GPU fits time one A100.

```sh
python run.py smoke --engine pcntoolkit
JAX_PLATFORMS=cpu python run.py smoke --engine jax --precision float64
JAX_PLATFORMS=cpu python run.py smoke --engine bootstrap
```

These smoke runs test the engines independently. They do not compare paired predictions.

At 450 IXI references, 21.6% of original flags recurred in fewer than 52 of 64 refits.

The denominator is original subject–voxel flags, pooled across 113 held-out scans.

For the full benchmark, configure your scheduler and follow `run.py --help` plus the wrappers in `slurm/`.

[complexity](complexity.md)
