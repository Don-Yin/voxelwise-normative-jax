# Can every IXI scan be scored out of sample

Use the IXI matrix and PCNtoolkit comparison artifacts. The plan validates the inputs and skips model fitting.

```sh
python run.py --plan --run-name local-plan
```

The five folds are site-balanced, with sizes 113, 113, 113, 112 and 112.

Training statistics and the age basis are specific to each fold.

Run `slurm/crossfit-a100.sh` for all scores or `slurm/vbm-age-icelake.sh` for the separate linear age analysis.

The age analysis uses one linear regression per voxel with age, sex and site predictors.
