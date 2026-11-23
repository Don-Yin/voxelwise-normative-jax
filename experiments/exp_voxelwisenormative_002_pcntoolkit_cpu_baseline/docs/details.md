# Does PCNtoolkit converge with its default optimizer step

Use the optional `pcntoolkit` install extra on Python 3.12 or 3.13. Supply the verified IXI matrix artifact.

| Setting | Value |
|---|---|
| Fit | Fixed-effect Bayesian linear regression and a SinhArcsinh warp |
| Design | Cubic age spline, sex, and fixed site effects |
| Split | 450 training rows and 113 held-out rows, stratified by site with seed 0 |
| Voxel selection | 1,000 mask columns drawn with seed 0 |

```sh
python run.py --n-voxels 1000
python run.py --n-voxels 1000 --epsilon 1e-7
```

The default command uses the old step of 0.1; the second supplies the reference used in experiment 003.

Sequential voxel fits took 74.3 s on the converged sample, extrapolating to 3.4 h for the full mask.

NumPy used its default BLAS thread settings.

Use [freeze-ixi.py](../../../scripts/freeze-ixi.py) to check and store the new comparison artifact locally.
