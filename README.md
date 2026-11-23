<p align="center"><img src="docs/figures/logo/banner.png" alt="" width="100%"></p>

# voxelwise-normative-jax

batched JAX reference fits and refit stability checks for grey-matter voxels.

<a href="docs/assets/exp_voxelwisenormative_010-result.png"><img alt="whole-mask fit times and original flag recurrence" src="docs/assets/exp_voxelwisenormative_010-result.png" width="600"></a>

on one NVIDIA A100, whole-mask fits were about 20,600× faster at 450 IXI references and 4,300× at 50,000 synthetic references against projected single-core PCNtoolkit times. [all experiments](experiments/README.md)

## install

install with `python -m pip install .`.

## use

<!-- readme-example -->
```python
import numpy as np
from shared.voxelwisenormative.batched_blr import design, fit_chunks, make_fitter

rng = np.random.default_rng(0)
age = np.linspace(-1.0, 1.0, 64)
phi = np.column_stack((np.ones_like(age), age))
y = age[:, None] * np.array([0.6, -0.3]) + rng.normal(size=(age.size, 2))
held = np.arange(age.size) % 4 == 0
y = (y - y[~held].mean(axis=0)) / y[~held].std(axis=0)
d = design(phi[~held], phi[held])
_, _, z = fit_chunks(make_fitter(), d, y[~held], y[held], chunk=2)
assert z.shape == y[held].shape and np.isfinite(z).all()
print(z.shape)
```
<!-- /readme-example -->

[licences](docs/licences.md): MIT; optional PCNtoolkit is GPL-3.0-only.

## documents

[context](docs/000-context.md) · [results](docs/004-results.md) · [open questions](docs/002-open-questions.md) · [decisions](docs/001-decisions.md) · [protocol](docs/003-experiment-protocol.md)
