# context

normative modelling asks how far a held-out observation is from its predicted value. each response column can be a voxel, a regional summary, or another numeric measurement. a z-score expresses the difference in predictive standard deviations after a fitted response transformation.

| step | method |
|---|---|
| design | learn a cubic spline for one numeric covariate using training rows only. include the other numeric covariates, an intercept and optional one-hot batch indicators. the IXI example uses an age spline, sex and site. |
| fit | a warped Bayesian linear regression fits each response column. JAX batches the fits and reuses the decomposition of the shared design. |
| score | predictive variance accounts for residual noise and uncertainty in the regression weights. |
| refit | in the site-stratified IXI runs, draw reference scans with replacement within site and count how often the same held-out scan–voxel pair is flagged. |

JAX and [PCNtoolkit](https://github.com/amarquand/PCNtoolkit) fit the same penalised objective with different optimisers. [results](004-results.md#numerical-agreement) show where the scores agree and where they differ. flag recurrence measures sensitivity to the chosen reference sample, not disease or a clinical false-positive rate.
