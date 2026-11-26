# PCNtoolkit model export

## Reproduce on a Mac CPU

Install the `pcntoolkit` extra in `voxelwise-normative-jax`. Set `A` to the existing directory of frozen artifacts; read them in place.

```sh
E=experiments/exp_voxelwisenormative_022_pcntoolkit_export
A=../2026-11-22-hackathon-scientific-parallelization/experiments/shared/artifacts
micromamba run -n voxelwise-normative-jax python "$E/run.py" --artifacts-dir "$A"
```

The default profile is `.results/profile/run.html`; `--no-profile` skips it. A failed check exits nonzero.

Read `arrays.npz` from `pcntoolkit-blr-reference-n1000-v1`.

Read `gm.npy` and `covariates.tsv` from `ixi-grey-matter-voxel-matrix-v1`.

| Input | Purpose |
|---|---|
| `arrays.npz` | `voxels`, `test`, `Phi_*`, `y_*` and `z_test` pin the sample, split, design, scaling and scores. |
| `gm.npy` | Use the selected responses in their original units. |
| `covariates.tsv` | Encode age, sex and scan site in cohort order. |

The saved model is under `.results/models/ixi-<time>/model/`.

| Output | Contents |
|---|---|
| `.results/json/round-trip.json` | Sample sizes, agreement checks and paired absolute z differences. |
| `normative_model.json` | Metadata, covariate scales and regression template. |
| `<response>/regression_model.json` | Fitted parameters, posterior precision and mean, and response scale. |

## What matched

The values come from `.results/json/round-trip.json`. The check ran with the shared `/tmp/vnj-bench.lock` held.

| Measure | Value | JSON key |
|---|---:|---|
| Training scans / held-out scans | 450 / 113 | `n_train` / `n_test` |
| Voxels / held-out scores | 1,000 / 113,000 | `n_voxels` / `n_z` |
| Saved response models | 1,000 | `response_files` |
| Same fit, maximum absolute z difference | 3.81 × 10⁻¹³ | `round_trip.max_abs_z` |
| Independent fits, maximum absolute z difference | 0.00110418 | `vs_frozen.max_abs_z` |

`round_trip` checks the **same** fit after loading. `vs_frozen` includes differences between optimizers.

Regenerate the figure after running the check:

```sh
S="$HOME/system-macos/active-work/skills/python-plotting-convention"
R="$E/.results/review/00-export-agreement"
P=exp_voxelwisenormative_022
micromamba run -n voxelwise-normative-jax \
    bash "$S/scripts/checker.sh" "docs/figures/$P.py" 8.5 --outdir "$R"
cp "$R/render/$P-display.png" "$E/.results/00-export-agreement.png"
cp "$R/render/$P-display.png" "docs/assets/$P-result.png"
```

## Use with a fitted regression

`export_pcntoolkit` takes the trained spline and batch basis, fitted parameters, and the same raw training rows. It does not optimize again.

```python
from pcntoolkit import NormativeModel
from shared.voxelwisenormative.pcntoolkit_export import export_pcntoolkit

path = export_pcntoolkit("new-model", basis, hyp, raw_x_train, raw_y_train, batch_train)
restored = NormativeModel.load(str(path))
result = restored.predict(test_data)
```

Use `Basis.fit` for `basis` and `fit_chunks(make_fitter(), ...)` for `hyp`. The `fit_score` shortcut returns only scores.

Fit on standardized responses, then pass the raw arrays for export. Keep the same row and column order.

Give `NormData` raw float64 values in covariate order. Use the same names as in the export.

Supply batch labels to `NormData` as a two-dimensional column. Make each label's string spelling unique and consistent.

Omit batch labels and batch names for an ungrouped model.

Load the parent folder from `export_pcntoolkit`, not its `model` subfolder. The export target must not already exist.

## Scope and extensions

PCNtoolkit 1.4.0 computes the posterior at the supplied parameters and writes its own files.

Export currently supports constant noise and a shared prior precision. These variants would need more state:

| Future option | Needed for export |
|---|---|
| Heteroskedastic: noise varies with covariates | Store the variance basis and coefficients to reconstruct the weighted posterior. |
| Custom variance design | Recreate the variance covariates at prediction time. |
| Automatic relevance determination (ARD) | Enable `ard` and store one prior precision per mean basis column in toolkit order. |

Check each extension by loading and predicting with the JAX parameters held fixed.
