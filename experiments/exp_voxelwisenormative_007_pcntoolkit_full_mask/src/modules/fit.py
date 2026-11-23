"""The experiment 002 PCNtoolkit fit, retaining original-design posterior weights."""

from __future__ import annotations

import logging
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np

logger = logging.getLogger(__name__)
EPSILON = 1e-7
DESIGN_ATOL = 3e-12
SHAPE = (563, 165_402)
DESIGN_COLS = 12
BLOCK = 256


def fit_block(gm, cov, test, voxels, phi_train, phi_test, out: Path):
    """Fit a small group exactly as experiment 002; keep z, posterior mean and objective."""
    if version("pcntoolkit") != "1.4.0":
        raise RuntimeError(f"expected PCNtoolkit 1.4.0; got {version('pcntoolkit')}")
    pcn = import_module("pcntoolkit")
    BLR = pcn.BLR
    Output = import_module("pcntoolkit.util.output").Output

    Y = np.asarray(gm[:, voxels], dtype=np.float64)
    X = np.column_stack([cov["age"], (cov["sex"] == "M").astype(float)])
    site = cov["site"]
    names = [f"v{v}" for v in voxels]
    attrs = {"covariates": ["age", "sex"], "batch_effect_dims": ["site"], "response_vars": names}
    train_data, test_data = (
        pcn.NormData.from_ndarrays(name, X[rows], Y[rows], site[rows, None],
                                  subject_ids=cov["subject"][rows], attrs=attrs)
        for name, rows in (("train", ~test), ("test", test))
    )
    template = BLR(name="blr", fixed_effect=True, warp_name="WarpSinhArcsinh", l_bfgs_b_epsilon=EPSILON,
                   basis_function_mean=pcn.BsplineBasisFunction(basis_column=0, degree=3, nknots=5))
    model = pcn.NormativeModel(template, savemodel=False, evaluate_model=False, saveresults=False,
                           saveplots=False, save_dir=str(out), name=out.name)
    # Check the real designs at the fit/predict boundaries, including site encoding and scaling.
    orig_fit, orig_forward = BLR.fit, BLR.forward
    passed_test = set()

    def check_fit(self, X_, be, be_maps, Y_):
        """Inspect the fitted design without modifying the optimiser."""
        orig_fit(self, X_, be, be_maps, Y_)
        phi, _ = self.Phi_Phi_var(X_.values, be.values)
        np.testing.assert_allclose(phi, phi_train, rtol=0, atol=DESIGN_ATOL, err_msg=f"train design: {self.name}")

    def check_forward(self, X_, be, Y_):
        """Inspect the design before predicting the held-out outcomes."""
        if X_.shape[0] == len(phi_test):
            phi, _ = self.Phi_Phi_var(X_.values, be.values)
            np.testing.assert_allclose(phi, phi_test, rtol=0, atol=DESIGN_ATOL, err_msg=f"test design: {self.name}")
            passed_test.add(self.name)
        return orig_forward(self, X_, be, Y_)

    prior = Output.get_show_messages()
    Output.set_show_messages(False)
    try:
        with patch.object(BLR, "fit", check_fit), patch.object(BLR, "forward", check_forward):
            model.fit(train_data)
            model.predict(test_data)
    finally:
        Output.set_show_messages(prior)
    if passed_test != set(names):
        raise RuntimeError(f"test design was not checked for {sorted(set(names) - passed_test)}")
    z = test_data["Z"].sel(response_vars=names).values
    weights = np.stack([model[n].m for n in names])
    objective = np.array([model[n].nlZ for n in names])
    for name, values, shape in (
        ("z_test", z, (113, len(voxels))), ("posterior_weights", weights, (len(voxels), DESIGN_COLS)),
        ("nlZ", objective, (len(voxels),)),
    ):
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"{name}, voxels {voxels[0]}–{voxels[-1]}: shape {values.shape}, expected finite {shape}")
    return z, weights, objective


def fit_groups(gm, cov, ref, voxels, out: Path):
    """Fit voxel blocks with bounded memory and preserve mask column order."""
    results = []
    for start in range(0, len(voxels), BLOCK):
        block = voxels[start:start + BLOCK]
        results.append(fit_block(gm, cov, ref["test"], block, ref["Phi_train"], ref["Phi_test"], out))
        logger.info("[%d/%d] fitted through mask column %d", min(start + BLOCK, len(voxels)), len(voxels), block[-1])
    return tuple(np.concatenate([r[i] for r in results], axis=1 if i == 0 else 0) for i in range(3))
