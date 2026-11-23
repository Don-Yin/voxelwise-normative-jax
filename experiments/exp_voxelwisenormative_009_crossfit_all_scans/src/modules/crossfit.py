"""Five disjoint, site-balanced held-out sets. Fit the same warped BLR as exp_003/005."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.artifacts import (
    PCNTOOLKIT_REFERENCE,
    VOXEL_MATRIX,
    resolve,
    sha256,
)
from shared.voxelwisenormative.basis import (
    SITES,
    Basis,
    covariate_rows,
    pinned_agreement,
)
from shared.voxelwisenormative.batched_blr import design, fit_chunks, make_fitter
from shared.voxelwisenormative.cohort import load_covariates, standardise

FOLDS = 5
SHAPE = (563, 165_402)
LOGGER = logging.getLogger(__name__)


def assignments(sites: np.ndarray, seed: int) -> np.ndarray:
    """Stratify within sites and spread remainders across folds, with no subject duplication."""
    random = np.random.default_rng(seed)
    fold = np.full(len(sites), -1, dtype=np.int8)
    offset = 0
    for site in SITES:
        rows = random.permutation(np.flatnonzero(sites == site))
        fold[rows] = (np.arange(len(rows)) + offset) % FOLDS
        offset += len(rows)
        if np.ptp(np.bincount(fold[rows], minlength=FOLDS)) > 1:
            raise RuntimeError(f'unbalanced fold assignment at {site}')
    if np.any(fold < 0) or np.ptp(np.bincount(fold, minlength=FOLDS)) > 1:
        raise RuntimeError('invalid or unbalanced fold assignment')
    return fold


def run(results: Path, run_name: str, seed: int, chunk: int, plan_only: bool, focus_column: int) -> None:
    cov = load_covariates()
    if len(cov['subject']) != SHAPE[0] or len(set(cov['subject'])) != SHAPE[0]:
        raise ValueError('expected 563 unique scans (one per IXI subject)')
    with np.load(resolve(PCNTOOLKIT_REFERENCE, 'arrays.npz')['arrays.npz']) as reference:
        agreement = pinned_agreement(cov, reference)
    paths = resolve(VOXEL_MATRIX, 'gm.npy', 'mask.nii.gz')
    gm = np.load(paths['gm.npy'], mmap_mode='r', allow_pickle=False)
    if gm.shape != SHAPE:
        raise ValueError(f'expected voxel matrix {SHAPE}, got {gm.shape}')
    fold = assignments(cov['site'], seed)
    splits = []
    for k in range(FOLDS):
        held = fold == k
        train = ~held
        basis = Basis.fit(*covariate_rows(cov, train))
        phi = basis.apply(*covariate_rows(cov, slice(None)))
        splits.append((held, design(phi[train], phi[held]), basis))
    table = [{'fold': k, 'n_train': int((fold != k).sum()), 'n_test': int((fold == k).sum()),
              'site_test': {s: int(((fold == k) & (cov['site'] == s)).sum()) for s in SITES}}
             for k in range(FOLDS)]
    report = {'input_artifact': VOXEL_MATRIX, 'shape': list(SHAPE), 'seed': seed, 'input_dtype': str(gm.dtype),
              'row_order': 'same as IXI covariates.tsv', 'held_out_folds': table, 'design_max_abs_error_vs_exp002': agreement,
              'focus_column': focus_column, 'chunk': chunk, 'iterations': 150, 'mode': 'plan' if plan_only else 'full'}
    meta = results / 'json' / run_name
    if plan_only:
        meta.mkdir(parents=True, exist_ok=True)
        (meta / 'plan.json').write_text(json.dumps(report, indent=2) + '\n')
        LOGGER.info('validated 5 folds: %s; pinned design max error %.3g', [f['n_test'] for f in table], agreement)
        return

    import jax

    gpu = jax.devices('gpu')  # deliberately errors instead of falling back to CPU
    if jax.devices()[0].platform != 'gpu' or not gpu:
        raise RuntimeError('the full run requires a CUDA GPU and JAX GPU as the default device')
    fit = make_fitter()
    report.update(device=str(gpu[0]), device_kind=gpu[0].device_kind, jax=jax.__version__)
    final = results / 'arrays' / run_name
    stage = results.parent / '.intermediates' / run_name
    meta.mkdir(parents=True, exist_ok=True)
    if final.exists() or (meta / 'result.json').exists():
        raise FileExistsError(run_name)
    stage.mkdir(parents=True)
    z_file = stage / 'z_oof.npy'
    z = np.lib.format.open_memmap(z_file, mode='w+', dtype=np.float32, shape=SHAPE)
    written = np.zeros(SHAPE[0], dtype=bool)
    hyp_focus, scale_focus, ages_and_sex, knots = [], [], [], []
    for k, (held, rotated, basis) in enumerate(splits):
        begin = time.perf_counter()
        y, yt = standardise(gm, held)
        if not (np.isfinite(y).all() and np.isfinite(yt).all()):
            raise ValueError(f'non-finite fold {k} scaled response (check training variance)')
        hyp, obj, test_z = fit_chunks(fit, rotated, y, yt, chunk)
        if test_z.shape != (int(held.sum()), SHAPE[1]) or not all(np.isfinite(v).all() for v in (test_z, hyp, obj)):
            raise ValueError(f'non-finite or invalid fold {k} outputs')
        if written[held].any():
            raise RuntimeError(f'overlapping test rows in fold {k}')
        z[held] = test_z
        if not np.isfinite(z[held]).all():
            raise ValueError(f'non-finite fold {k} after writing float32')
        z.flush()
        written[held] = True
        focus_train = gm[~held, focus_column].astype(np.float64)
        scale_focus.append((focus_train.mean(), focus_train.std()))
        hyp_focus.append(hyp[focus_column].copy())
        ages_and_sex.append(np.stack([basis.mean, basis.sd]))
        knots.append(basis.knots)
        table[k]['seconds_including_scale_fit_write'] = time.perf_counter() - begin
        LOGGER.info('[%d/%d] %d test scans scored', k + 1, FOLDS, int(held.sum()))
        del y, yt, hyp, obj, test_z
    if not written.all():
        raise RuntimeError('one or more subjects did not receive a held-out score')
    del z
    np.savez_compressed(stage / 'index.npz', fold=fold, subject=cov['subject'], focus_column=focus_column,
                        focus_hyp=hyp_focus, focus_y_mean_sd=scale_focus, x_mean_sd=ages_and_sex, age_knots=knots)
    final.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(final)
    report['outputs'] = {p.name: {'file': str(p.relative_to(results)), 'sha256': sha256(p)} for p in final.iterdir()}
    temp = meta / 'result.tmp'
    temp.write_text(json.dumps(report, indent=2) + '\n')
    temp.replace(meta / 'result.json')
