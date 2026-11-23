# context

voxelwise normative modelling asks how far a held-out grey-matter measurement is from its predicted value. a z-score expresses that difference in predictive standard deviations after a fitted response transformation.

| step | method |
|---|---|
| design | the IXI reference uses a cubic age spline, sex and fixed site effects. |
| fit | a warped Bayesian linear regression fits each voxel. JAX batches the fits and reuses the decomposition of the shared design. |
| score | predictive variance accounts for residual noise and uncertainty in the regression weights. |
| refit | draw reference scans with replacement within site and count how often the same held-out scan–voxel pair is flagged. |

JAX and [PCNtoolkit](https://github.com/amarquand/PCNtoolkit) fit the same penalised objective with different optimisers. [results](004-results.md#numerical-agreement) show where the scores agree and where they differ. flag recurrence measures sensitivity to the chosen reference sample, not disease or a clinical false-positive rate.
