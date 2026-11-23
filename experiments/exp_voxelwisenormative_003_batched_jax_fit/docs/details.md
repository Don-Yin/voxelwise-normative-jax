# Can a batched fit match converged PCNtoolkit

The shared design is diagonalized once. JAX fits four hyperparameters per voxel with damped Newton steps.

The fit uses 150 iterations in float64. Each reported warm pass excludes compilation.

With the artifact from experiment 002 available, run from this directory:

```sh
JAX_PLATFORMS=cpu python run.py check --repeats 1
python run.py full
```

The check reports finite z, absolute z error and penalized objective differences for 1,000 columns.

Full mode also needs the IXI matrix and writes the fitted parameter and held-out score arrays locally.
