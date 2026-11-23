"""Reproduce experiment 002's PCNtoolkit 1.4.0 age/sex/site design, fitted on the supplied training rows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

SITES = ('HH', 'Guys', 'IOP')  # exact order in the pinned Phi_train/Phi_test from experiment 002


@dataclass(frozen=True)
class Basis:
    mean: np.ndarray
    sd: np.ndarray
    knots: np.ndarray

    @classmethod
    def fit(cls, ages, sex, sites):
        """Estimate scaling and padded knots from reference subjects only."""
        if set(sites) != set(SITES):
            raise ValueError(f'training lacks an IXI site: {sorted(set(sites))}')
        x = np.column_stack((ages, np.asarray(sex) == 'M'))
        mean, sd = x.mean(axis=0), x.std(axis=0)
        if not np.isfinite(x).all() or np.any(sd == 0):
            raise ValueError('invalid age/sex training scale')
        age = (x[:, 0] - mean[0]) / sd[0]
        padding = 0.05 * np.ptp(age)
        lo, hi = age.min() - padding, age.max() + padding
        knots = np.r_[np.repeat(lo, 3), np.linspace(lo, hi, 5), np.repeat(hi, 3)]
        return cls(mean, sd, knots)

    def apply(self, ages, sex, sites):
        """Construct the fixed twelve-column design, extrapolating age when needed."""
        x = (np.column_stack((ages, np.asarray(sex) == 'M')) - self.mean) / self.sd
        categories = np.asarray(sites)[:, None] == np.array(SITES)
        if not np.isfinite(x).all() or not np.all(categories.sum(axis=1) == 1):
            raise ValueError('invalid age/sex/site design rows')
        age = BSpline.design_matrix(x[:, 0], self.knots, 3, extrapolate=True).toarray()
        return np.column_stack((age, x[:, 1], np.ones(len(x)), categories))


def covariate_rows(cov, selector):
    """Select corresponding ages, sexes and sites in model argument order."""
    return tuple(cov[key][selector] for key in ('age', 'sex', 'site'))


def average_site_rows(basis: Basis, ages, sexes, cohort_sites: np.ndarray) -> np.ndarray:
    """Query rows with the three site indicators replaced by whole-cohort proportions."""
    phi = basis.apply(ages, sexes, ["Guys"] * len(ages))
    phi[:, -len(SITES):] = [(cohort_sites == site).mean() for site in SITES]
    return phi


def pinned_agreement(cov, reference):
    """Check the original IXI split's design against the archived PCNtoolkit arrays."""
    test = reference['test']
    basis = Basis.fit(*covariate_rows(cov, ~test))
    actual = basis.apply(*covariate_rows(cov, slice(None)))
    error = max(float(np.max(np.abs(reference['Phi_train'] - actual[~test]))),
                float(np.max(np.abs(reference['Phi_test'] - actual[test]))))
    if error > 1e-12:
        raise ValueError(f'reproduced covariate basis differs from experiment 002 by {error:g}')
    return error
