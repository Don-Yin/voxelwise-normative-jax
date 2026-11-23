<p align="center"><img src="docs/figures/logo/banner.png" alt="" width="100%"></p>

# voxelwise-normative-jax

batched JAX normative modelling for numeric response matrices, with or without imaging.

<a href="docs/assets/exp_voxelwisenormative_010-result.png"><img alt="whole-mask fit times and original flag recurrence" src="docs/assets/exp_voxelwisenormative_010-result.png" width="600"></a>

on one NVIDIA A100, whole-mask fits were about 20,600× faster at 450 IXI references and 4,300× at 50,000 synthetic references against projected single-core PCNtoolkit times. the 450-reference A100 time was measured on a university HPC cluster, the 50,000-reference A100 time on a rented [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) A100. [all experiments](experiments/README.md)

## install

install with `python -m pip install .`.

## use

covariates and responses share rows. pass a boolean test mask and optional batch labels; omit `batch_levels` to learn levels from training rows.

<!-- readme-example -->
```python
import numpy as np
from shared.voxelwisenormative.model import fit_score

rng = np.random.default_rng(0)
exposure = np.linspace(-1.0, 1.0, 64)
x = np.column_stack((rng.normal(size=64), exposure))
batch = np.resize(["lab-a", "lab-b", "lab-c"], 64)
y = exposure[:, None] * np.array([0.6, -0.3]) + rng.normal(size=(64, 2))
held = np.arange(64) % 4 == 0
z = fit_score(x, y, held, batch=batch, spline_column=1, chunk=2)
assert z.shape == y[held].shape and np.isfinite(z).all()
print(z.shape)
```
<!-- /readme-example -->

[licences](docs/licences.md): MIT; optional PCNtoolkit is GPL-3.0-only.

## documents

[context](docs/000-context.md) · [results](docs/004-results.md) · [open questions](docs/002-open-questions.md) · [decisions](docs/001-decisions.md) · [protocol](docs/003-experiment-protocol.md)
