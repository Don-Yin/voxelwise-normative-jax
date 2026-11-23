"""Rank same-person voxel pairs from the 113 held-out scores, using the original 64 draws."""
from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.artifacts import (
    PCNTOOLKIT_REFERENCE,
    VOXEL_MATRIX,
    resolve,
    sha256,
)
from shared.voxelwisenormative.batched_blr import design, fit_chunks, make_fitter

LOW, HI, MAX_GAP = 3., 3.1, .02
TARGET_LOW, TARGET_HIGH = 15, 60
PER_PERSON, N_TRIALS = 48, 64


def pick_columns(baseline):
    """Take evenly spaced quantiles of each person's eligible |z| range; never use refit outcomes."""
    absolute = np.abs(baseline)
    eligible = np.flatnonzero((absolute >= LOW) & (absolute <= HI))
    ordered = eligible[np.argsort(absolute[eligible], kind="stable")]
    ranks = np.linspace(0, len(ordered) - 1, min(PER_PERSON, len(ordered))).round().astype(int)
    return ordered[ranks], len(ordered)


def rank_pairs(base, flags, mapping, original_cols):
    for t, idx in enumerate(mapping):
        count = flags[:, t, :].sum(0)
        for i, j in combinations(idx, 2):
            gap = abs(abs(float(base[t, i])) - abs(float(base[t, j])))
            if gap > MAX_GAP:
                continue
            a, b = sorted((i, j), key=lambda col: (int(count[col]), int(original_cols[col])))
            low, high = int(count[a]), int(count[b])
            missed = max(0, low - TARGET_LOW, TARGET_HIGH - high)
            yield (missed, low - high, low, gap, t, a, b)


def scan(baseline_path: Path, baseline_sha256: str, bootstrap_path: Path, receipt_path: Path,
         output: Path, chunk: int) -> None:
    import jax
    import nibabel as nib
    from nibabel.affines import apply_affine

    jax.config.update("jax_platforms", "cpu")
    if output.exists():
        raise FileExistsError(output)
    if sha256(baseline_path) != baseline_sha256:
        raise ValueError(f"{baseline_path}: digest differs from the pinned baseline")
    try:
        info = json.loads(receipt_path.read_text())
        expected = info['arrays']['sha256']
        bitorder = info['trial_flags_encoding']['bitorder']
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f'{receipt_path}: invalid bootstrap receipt: {error}') from error
    digest = sha256(bootstrap_path)
    if digest != expected or bitorder != 'little':
        raise ValueError(f'{bootstrap_path}: bootstrap digest or bit order differs from the receipt')
    full_base = np.load(baseline_path, mmap_mode="r", allow_pickle=False)
    assert full_base.shape == (113, 165402) and np.isfinite(full_base).all()
    samples_by_person = [pick_columns(row) for row in full_base]
    columns = np.unique(np.concatenate([cols for cols, _ in samples_by_person]))
    mapping = [np.searchsorted(columns, cols) for cols, _ in samples_by_person]
    assert len(columns) > 0
    base = full_base[:, columns]
    with np.load(bootstrap_path, allow_pickle=False) as saved:
        ref_rows = saved["reference_cohort_rows"]
        samples = saved["sampled_cohort_rows"]
        archived_row = saved["baseline_z"][columns]
        packed = saved["packed_trial_flags"][:, columns // 8]
    bits = (packed >> (columns % 8)) & 1
    assert samples.shape == (N_TRIALS, 450)
    rows = np.searchsorted(ref_rows, samples)
    np.testing.assert_array_equal(ref_rows[rows], samples)
    t0 = info["test_row_zero_based"]
    np.testing.assert_allclose(base[t0], archived_row, atol=2e-5, rtol=0)
    with np.load(resolve(PCNTOOLKIT_REFERENCE, "arrays.npz")["arrays.npz"], allow_pickle=False) as ref:
        np.testing.assert_array_equal(ref_rows, np.flatnonzero(~ref["test"]))
        test_rows = np.flatnonzero(ref["test"])
        phi, phi_test = ref["Phi_train"], ref["Phi_test"]
    assert len(test_rows) == len(base) and test_rows[t0] == info["held_out_cohort_row_zero_based"]
    inputs = resolve(VOXEL_MATRIX, "gm.npy", "mask.nii.gz")
    gm = np.load(inputs["gm.npy"], mmap_mode="r", allow_pickle=False)
    assert gm.shape == (563, 165402)
    train_y = gm[np.ix_(ref_rows, columns)]
    test_y = gm[np.ix_(test_rows, columns)]
    fitter = make_fitter(n_iter=info["n_iter"])

    def score(draw):
        y = train_y[draw].astype(np.float64)
        mean, sd = y.mean(0), y.std(0)
        assert np.isfinite(sd).all() and (sd > 0).all()
        _, _, z = fit_chunks(fitter, design(phi[draw], phi_test), (y - mean) / sd, (test_y - mean) / sd, chunk=chunk)
        if not np.isfinite(z).all():
            raise ValueError("nonfinite replay score")
        return z

    original = score(np.arange(len(ref_rows)))
    np.testing.assert_allclose(original, base, atol=2e-5, rtol=0)
    print(f"[{len(base)}/{len(base)}] checked baselines; {len(columns)} model columns in the union", flush=True)
    # Scores store only each person's sampled columns; the dense batch is transient.
    replayed = [np.empty((N_TRIALS, len(idx))) for idx in mapping]
    for trial, draw in enumerate(rows):
        z = score(draw)
        np.testing.assert_array_equal(np.abs(z[t0]) > info["z_threshold_abs_strict"], bits[trial])
        for t, idx in enumerate(mapping):
            replayed[t][trial] = z[t, idx]
        print(f"[{trial + 1}/{N_TRIALS}] replayed fixed draw", flush=True)
    flags = np.zeros((N_TRIALS, len(base), len(columns)), dtype=bool)
    for t, idx in enumerate(mapping):
        flags[:, t, idx] = np.abs(replayed[t]) >= LOW
    ranks = sorted(rank_pairs(base, flags, mapping, columns))
    if not ranks:
        raise ValueError("no same-person pairs meet the baseline criteria")
    mask = nib.load(inputs["mask.nii.gz"])
    if not isinstance(mask, nib.Nifti1Image) or mask.affine is None:
        raise ValueError("expected the shared NIfTI mask with a voxel-to-world affine")
    affine = mask.affine
    ijk = np.argwhere(np.asarray(mask.dataobj) > 0)
    assert len(ijk) == gm.shape[1]

    def example(t, col):
        c = int(columns[col])
        i, = np.flatnonzero(mapping[t] == col)
        z = replayed[t][:, i]
        return {"column_zero_based": c, "baseline_z": float(base[t, col]),
                "refits_flagged": int((np.abs(z) >= LOW).sum()), "refit_z_signed": z.tolist(),
                "ijk_zero_based": ijk[c].tolist(), "mni_xyz_mm": apply_affine(affine, ijk[c]).tolist()}

    top = [{"held_out_test_row_zero_based": t, "held_out_cohort_row_zero_based": int(test_rows[t]),
            "abs_z_difference": gap, "worst_target_shortfall_refits": missed,
            "low": example(t, a), "high": example(t, b)} for missed, _, _, gap, t, a, b in ranks[:5]]
    result = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "held_out_people": len(base),
              "sources": {"baseline": baseline_path.name, "baseline_sha256": baseline_sha256,
                          "draws": bootstrap_path.name, "draws_sha256": digest},
              "baseline_abs_z_interval": [LOW, HI], "max_baseline_abs_z_difference": MAX_GAP,
              "max_candidates_per_person": PER_PERSON, "eligible_baseline_columns_by_test_row": [n for _, n in samples_by_person],
              "model_columns_replayed": len(columns), "pairs_ranked": len(ranks), "pairs_meeting_targets": sum(r[0] == 0 for r in ranks),
              "rule": "Schoolbook replay of fixed draws with trial-local scaling and the original basis. Select 48 evenly spaced baseline |z| quantiles per person. Rank eligible pairs by smallest worst shortfall from (low <= 15, high >= 60), largest count gap, lowest low count, smallest absolute-score difference, then indices.",
              "cutoff_inclusive_abs_z": LOW, "targets_count": [TARGET_LOW, TARGET_HIGH], "top_five": top}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_suffix(".tmp")
    stage.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    stage.replace(output)
    print(f"{output}: {len(ranks)} pairs, {result['pairs_meeting_targets']} met both targets", flush=True)
