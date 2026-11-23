"""Unthresholded OLS age contrast for the full IXI grey-matter mask."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from shared.voxelwisenormative.artifacts import VOXEL_MATRIX, resolve, sha256
from shared.voxelwisenormative.cohort import load_covariates

LOGGER = logging.getLogger(__name__)
COLUMNS = ('intercept', 'age_centered_years', 'sex_M', 'site_HH', 'site_IOP')
EXAMPLE = 'IXI129-Guys-0775-T1'


def age_design(cov):
    age, sex, site = (cov[k] for k in ('age', 'sex', 'site'))
    if not np.isfinite(age).all() or set(sex) != {'F', 'M'} or set(site) != {'Guys', 'HH', 'IOP'}:
        raise ValueError('unexpected IXI covariates')
    center = float(age.mean())
    x = np.column_stack((np.ones(len(age)), age-center, sex == 'M', site == 'HH', site == 'IOP'))
    return x, center


def ols_age(x, responses, chunk):
    """Return coefficients and signed t; reuse the one QR factorization across all voxels."""
    n, p = x.shape
    if responses.shape[0] != n or n <= p or np.linalg.matrix_rank(x) != p or chunk < 1:
        raise ValueError('invalid or rank-deficient OLS design')
    q, r = np.linalg.qr(x, mode='reduced')
    inv_r = np.linalg.solve(r, np.eye(p))
    variance_factor = float(inv_r[1] @ inv_r[1])  # age diagonal of (XᵀX)⁻¹
    coef = np.empty((p, responses.shape[1]), dtype=np.float64)
    t_age = np.empty(responses.shape[1], dtype=np.float64)
    for start in range(0, responses.shape[1], chunk):
        s = slice(start, start+chunk)
        y = np.asarray(responses[:, s], dtype=np.float64)
        b = inv_r @ (q.T @ y)
        residual = y - x @ b
        mse = np.einsum('ij,ij->j', residual, residual)/(n-p)
        if np.any(mse <= 0) or not np.isfinite(mse).all():
            raise ValueError(f'invalid residual variance in voxel columns starting at {start}')
        coef[:, s], t_age[s] = b, b[1]/np.sqrt(mse*variance_factor)
    if not np.isfinite(t_age).all() or not np.isfinite(coef).all():
        raise ValueError('non-finite OLS output')
    return coef, t_age


def fit_vbm(out: Path, chunk: int, focus_column: int) -> None:
    if out.exists():
        raise FileExistsError(out)
    cov = load_covariates()
    gm_file = resolve(VOXEL_MATRIX, 'gm.npy')['gm.npy']
    gm = np.load(gm_file, mmap_mode='r', allow_pickle=False)
    if gm.shape != (563, 165402) or len(set(cov['subject'])) != len(gm):
        raise ValueError('expected 563 distinct IXI rows and 165,402 mask columns')
    x, center = age_design(cov)
    b, t = ols_age(x, gm, chunk)
    person, = np.flatnonzero(cov['subject'] == EXAMPLE)
    low, high = np.quantile(cov['age'], [0., 1.])
    # Export only the fitted line for the highlighted voxel, never the raw responses.
    row_at_center = x[person].copy()
    row_at_center[1] = 0
    intercept = float(row_at_center @ b[:, focus_column])
    meta = {'input_artifact': VOXEL_MATRIX, 'shape': list(gm.shape), 'column_order': 'raw mask C order',
            'model': 'grey_matter(v) ~ age + sex + site', 'design_columns': COLUMNS, 'dof': len(x)-len(COLUMNS),
            'age_center_years': center, 't_display_quantiles_abs': dict(zip(('p95','p99','max'),
                np.quantile(abs(t), [.95,.99,1.]).tolist(), strict=True)),
            'focus': {'column_zero_based': focus_column, 'conditional_subject': EXAMPLE,
                      'age_bounds_years': [float(low), float(high)], 'age_center_years': center,
                      'predicted_at_center': intercept, 'beta_age_per_year': float(b[1, focus_column]),
                      't_age': float(t[focus_column]), 'sex': str(cov['sex'][person]), 'site': str(cov['site'][person])}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.{out.name}-', dir=out.parent) as scratch:
        staged = Path(scratch)/'ready'
        staged.mkdir()
        for name, values in (('beta_age.npy', b[1]), ('t_age.npy', t)):
            np.save(staged/name, values, allow_pickle=False)
        meta['sha256'] = {p.name: sha256(p) for p in sorted(staged.iterdir())}
        (staged/'evidence.json').write_text(json.dumps(meta, indent=2)+'\n')
        staged.rename(out)
    LOGGER.info('Saved %d age effects and t statistics (%d residual degrees of freedom): %s', len(t), meta['dof'], out)
