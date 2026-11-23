# experiment protocol

## install and test

use an isolated Python 3.13 environment and run commands from the repository root. see [package metadata](../pyproject.toml) for the runtime and extras.

```sh
python -m pip install '.[test]'
bash scripts/test.sh
JAX_PLATFORMS=cpu python -m pytest -q docs/tests/test_readme.py
```

install `'.[test,pcntoolkit]'` and run `bash scripts/test.sh --agreement-only` for the paired numerical test. it compares z-scores and penalised objectives on generated voxel columns using CPU JAX, not CUDA. real-data checks skip unless their local artifacts are supplied.

on a supported Linux NVIDIA host, install `'.[cuda12]'` for the separate [JAX CUDA runtime](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu). the A100 experiments use configured batch jobs; the example and synthetic tests do not.

## prepare IXI locally

[Brain Development's IXI Dataset](https://brain-development.org/ixi-dataset/) supplies the T1 archive and demographic workbook; see [data terms](licences.md#data). store them outside the repository. the crop step needs the FSL tools on `PATH` and `FSLDIR` set to the installed tools.

```sh
python -m pip install '.[ixi]'
bash scripts/download-ixi.sh '<ixi-data>'
python scripts/preprocess-ixi.py --archive '<ixi-data>/IXI-T1.tar' \
    --metadata '<ixi-data>/IXI.xls' --out '<ixi-crops>'
```

| next step | local product and instructions |
|---|---|
| structural preprocessing | [exp_000](../experiments/exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/docs/details.md) uses the crop manifest and an external voxel-based morphometry chain. |
| common voxel matrix | [exp_001](../experiments/exp_voxelwisenormative_001_ixi_voxel_matrix/docs/details.md) builds the mask, scan-by-voxel matrix and aligned covariates. |
| PCNtoolkit reference | [exp_002](../experiments/exp_voxelwisenormative_002_pcntoolkit_cpu_baseline/docs/details.md) fits the converged sampled reference; [exp_007](../experiments/exp_voxelwisenormative_007_pcntoolkit_full_mask/docs/details.md) describes the smaller calibration. |
| full fit | [exp_003](../experiments/exp_voxelwisenormative_003_batched_jax_fit/docs/details.md) produces full-mask scores for analyses that consume its fitted model. |
| alternate preprocessing | [exp_017](../experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/docs/details.md) adds SynthSR before building the grey-matter matrix. |

```sh
PYTHONPATH=. python scripts/freeze-ixi.py matrix '<matrix-output>'
PYTHONPATH=. python scripts/freeze-ixi.py reference '<reference-output>'
```

[the freezer](../scripts/freeze-ixi.py) verifies producer receipts, copies the payload to an ignored local artifact and refuses to overwrite it. the external chain, vendor tools and historical crops are not bundled; byte-for-byte reproduction is not guaranteed.

## measure and plot

| measure | protocol |
|---|---|
| elapsed fit and score | synchronised warm runs exclude compilation and input setup. use the experiment's stated median or minimum, not both interchangeably. |
| one-core time projection | multiply sampled PCNtoolkit fit and predict time per voxel by mask width; do not divide by an assumed core count. |
| peak device memory | report maximum live allocated bytes, not reserved accelerator memory or host process memory. |
| z-score agreement | compare paired held-out scores on the stated mask or pinned sample; report absolute differences, including the largest. |
| recurrence | count whether each originally flagged scan–voxel pair remains flagged on refits at the same cutoff, regardless of sign. |

[figure commands and sources](figures/README.md) link the shipped input tables; [results](004-results.md) state the workloads, denominators and timing rules.
