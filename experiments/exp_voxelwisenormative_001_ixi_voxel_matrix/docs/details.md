# How many observations and voxels go into the model

Build grey-matter derivatives with [experiment 000](../../exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/README.md).

`run.py` checks the index, template and each map digest before writing the C-order matrix.

`--subjects-sha256` and `--template-sha256` bind the inputs to the producer's recorded digests.

Run `python run.py --derivatives <ixi-derivatives>` with those two digest options.

The grey-matter mask is template intensity ≥ 0.2. The matrix is float32 and shares row order with the covariates.

Publish the local manifest from verified outputs using [freeze-ixi.py](../../../scripts/freeze-ixi.py).
