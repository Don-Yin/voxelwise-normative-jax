"""Synthetic checks for the CPU gradient, solver sweep and the unmodified PCNtoolkit fit path."""

from __future__ import annotations

import json
import tempfile
import unittest
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np
from src.modules import stopping_sweep
from src.modules.fit import EPSILON
from src.modules.gradient_benchmark import annotate, selection
from src.modules.gradient_fit import compile_value_and_grad, timed_fit, verify_gradient
from src.modules.outlier_objective import Objective
from src.modules.stopping_sweep import cell_record, summarise, sweep

from shared.voxelwisenormative.cohort import standardise


def synthetic_problem(folder: Path, *, skewed: bool = True):
    """Run the unmodified PCNtoolkit fit and capture the designs the benchmark reuses."""
    pcn = import_module("pcntoolkit")
    rng = np.random.default_rng(80)
    age = rng.uniform(18, 85, 563)
    sex = np.array(["F", "M"])[rng.integers(0, 2, len(age))]
    cov = {"age": age, "sex": sex, "subject": np.array([f"synthetic-{i}" for i in range(len(age))]),
           "site": np.array(["A", "B", "C"])[rng.integers(0, 3, len(age))]}
    if skewed:
        gm = 0.01 * age[:, None] + rng.lognormal(size=(len(age), 2))
    else:
        gm = 0.01 * age[:, None] + rng.gamma(16.0, 0.25, size=(len(age), 2))
    gm = gm.astype(np.float32)
    test = np.arange(len(age)) % 5 == 0
    names = ["v0", "v1"]
    columns = {"covariates": ["age", "sex"], "batch_effect_dims": ["site"], "response_vars": names}
    x = np.column_stack([age, (sex == "M").astype(float)])
    train, held = (pcn.NormData.from_ndarrays(name, x[rows], gm[rows].astype(np.float64), cov["site"][rows, None],
                                            subject_ids=cov["subject"][rows], attrs=columns)
                   for name, rows in (("train", ~test), ("test", test)))
    template = pcn.BLR(name="blr", fixed_effect=True, warp_name="WarpSinhArcsinh", l_bfgs_b_epsilon=EPSILON,
                       basis_function_mean=pcn.BsplineBasisFunction(basis_column=0, degree=3, nknots=5))
    ref = {"test": test}
    actual_fit, actual_forward = pcn.BLR.fit, pcn.BLR.forward

    def capture_fit(model, x, be, maps, y):
        actual_fit(model, x, be, maps, y)
        ref["Phi_train"] = model.Phi_Phi_var(x.values, be.values)[0]

    def capture_forward(model, x, be, y):
        if len(x) == test.sum():
            ref["Phi_test"] = model.Phi_Phi_var(x.values, be.values)[0]
        return actual_forward(model, x, be, y)

    model = pcn.NormativeModel(template, savemodel=False, evaluate_model=False, saveresults=False,
                               saveplots=False, save_dir=str(folder / "ref"), name="ref")
    with patch.object(pcn.BLR, "fit", capture_fit), patch.object(pcn.BLR, "forward", capture_forward):
        model.fit(train)
        model.predict(held)
    z_ref = held["Z"].sel(response_vars=names).values
    loss_ref = np.array([model[name].nlZ for name in names])
    return cov, gm, test, ref, z_ref, loss_ref


def gradient_binder(gm, test, ref):
    """Use JAX when installed; otherwise exercise the same fprime path with the audit derivative."""
    y, yt = standardise(gm, test)
    if find_spec("jax"):
        return y, yt, compile_value_and_grad(ref["Phi_train"], y[:, 0])

    def bind(x, y):
        independent = Objective(x, x.T @ x, ref["Phi_test"], y, yt[:, 0])
        return lambda theta: independent.evaluate(theta)[1:3]

    return y, yt, bind


class GradientBenchmarkTests(unittest.TestCase):
    def test_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'outliers.json'
            path.write_text(json.dumps({'array_job': '1', 'merge_job': '2', 'calibration_job': '3',
                                       'job': '4', 'worst_voxels': [{'mask_column': i} for i in (*range(1, 10), 82300)]}))
            with patch('src.modules.gradient_benchmark.resolve', return_value={'outliers.json': path}):
                cols, calibration, worst = selection('4', ('1', '2', '3'))
        self.assertEqual((len(cols), len(calibration), len(worst)), (522, 512, 10))
        self.assertIn(82300, worst)

    @unittest.skipUnless(find_spec('jax') and find_spec('pcntoolkit'), 'install the pcntoolkit extra for paired agreement')
    def test_two_voxel_jax_agreement(self):
        import jax

        from shared.voxelwisenormative.batched_blr import design, fit_chunks, make_fitter

        self.assertEqual(jax.default_backend(), 'cpu')
        with tempfile.TemporaryDirectory() as folder:
            _, gm, held, ref, z_pcn, loss_pcn = synthetic_problem(Path(folder), skewed=False)
        y, yt = standardise(gm, held)
        basis = design(ref['Phi_train'], ref['Phi_test'])
        _, loss, z = fit_chunks(make_fitter(), basis, y, yt, chunk=2)
        self.assertEqual(z.shape, z_pcn.shape)
        self.assertEqual(loss.shape, loss_pcn.shape)
        self.assertTrue(all(np.isfinite(a).all() for a in (loss, loss_pcn, z, z_pcn)))
        dz, dloss = float(np.abs(z - z_pcn).max()), float((loss - loss_pcn).max())
        print(f'CPU JAX vs PCNtoolkit: max |Δz|={dz:.12g}; max JAX minus PCN objective={dloss:.12g}; '
              f'max absolute objective difference={float(np.abs(loss - loss_pcn).max()):.12g}')
        self.assertLessEqual(dz, 0.01)
        self.assertLessEqual(dloss, 1e-6)

    @unittest.skipUnless(find_spec('jax') and find_spec('pcntoolkit'), 'install the pcntoolkit extra for the skewed fixture')
    def test_skewed_fixture_reaches_a_worse_stationary_point(self):
        from shared.voxelwisenormative.batched_blr import design, fit_chunks, make_fitter

        with tempfile.TemporaryDirectory() as folder:
            _, gm, held, ref, _, loss_pcn = synthetic_problem(Path(folder), skewed=True)
        y, yt = standardise(gm, held)
        phi, phit = ref['Phi_train'], ref['Phi_test']
        hyp, _, _ = fit_chunks(make_fitter(), design(phi, phit), y, yt, chunk=2)
        gram = phi.T @ phi
        for col, theta in enumerate(hyp):
            _, value, grad, _ = Objective(phi, gram, phit, y[:, col], yt[:, col]).evaluate(theta)
            norm = float(np.linalg.norm(grad, ord=np.inf))
            gap = float(value - loss_pcn[col])
            print(f'Skewed voxel {col}: gradient infinity norm={norm:.12g}; JAX minus PCN objective={gap:.12g} nats')
            self.assertTrue(np.isfinite([norm, gap]).all())
            self.assertLess(norm, 1e-6)
            self.assertGreater(gap, 1.0)

    def test_failure_constant_ends_the_search(self):
        """One trial at PCNtoolkit's failure constant stops L-BFGS-B at a large gradient; factr and maxls do not matter."""
        from scipy.optimize import fmin_l_bfgs_b

        big = float(np.finfo(np.float64).max)

        def make(wall):
            def fun(x):
                if x[0] > wall:
                    return big, np.zeros(2)
                return 0.5 * (x[0] - 10) ** 2 + 200 * x[1] ** 2, np.array([x[0] - 10, 400 * x[1]])

            return fun

        start = np.array([-60.0, 3.0])
        _, _, free = fmin_l_bfgs_b(make(np.inf), start, factr=10.0, maxls=50)
        self.assertLess(np.abs(free["grad"]).max(), 1e-5)
        for factr, maxls in ((1e7, 20), (1e3, 50), (10.0, 20), (10.0, 50)):
            _, _, info = fmin_l_bfgs_b(make(0.0), start, factr=factr, maxls=maxls)
            self.assertIn("RELATIVE REDUCTION", str(info["task"]))
            self.assertGreater(np.abs(info["grad"]).max(), 100)

    @unittest.skipUnless(find_spec("jax"), "JAX not installed in this local runtime")
    def test_compiled_gradient(self):
        rng = np.random.default_rng(79)
        x, xt = rng.normal(size=(450, 12)), rng.normal(size=(113, 12))
        y, yt = rng.normal(size=450), rng.normal(size=113)
        bind = compile_value_and_grad(x, y)
        ad = bind(x, y)
        exact = Objective(x, x.T @ x, xt, y, yt)
        for theta in (np.ones(4), np.array([0.3, -0.25, 0.1, -0.4]), np.zeros(4)):
            _, loss, grad, _ = exact.evaluate(theta)
            actual, actual_grad = ad(theta)
            np.testing.assert_allclose(actual, loss, rtol=1e-10)
            np.testing.assert_allclose(actual_grad, grad, rtol=1e-8, atol=1e-9)

    @unittest.skipUnless(find_spec("pcntoolkit"), "PCNtoolkit not installed in this local runtime")
    def test_installed_fit_and_counters(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            cov, gm, test, ref, z_ref, loss_ref = synthetic_problem(out)
            ids = np.arange(gm.shape[1])
            z, loss, counts = timed_fit(gm, cov, ref, ids, out / "approx", None)
            np.testing.assert_array_equal(z, z_ref)
            np.testing.assert_array_equal(loss, loss_ref)
            y, yt, bind = gradient_binder(gm, test, ref)
            verify_gradient(bind, ref["Phi_train"], ref["Phi_test"], y, yt, ids)
            z2, loss2, counts2 = timed_fit(gm, cov, ref, ids, out / "exact", bind)
            np.testing.assert_allclose(z2, z_ref, atol=1e-3)
            np.testing.assert_allclose(loss2, loss_ref, atol=1e-5)
            for a, b in zip(counts, counts2, strict=True):
                a, b = a["attempts"][-1], b["attempts"][-1]
                self.assertEqual(a["function_calls"], 5 * a["gradient_calls"])
                self.assertEqual(b["function_calls"], b["gradient_calls"])
                self.assertGreaterEqual(a["function_calls"], 5)
            annotate(counts2[0], 82300, (set(), {82300}), z2[:, 0], z_ref[:, 0], loss2[0])
            self.assertTrue(counts2[0]["audited_outlier"])
            self.assertLess(counts2[0]["max_abs_delta_z_vs_gpu"], 1e-3)
            self.assertNotIn("objective_trace", counts[0]["attempts"][-1])
            z3, loss3, counts3 = timed_fit(gm, cov, ref, ids, out / "tight", bind, {"factr": 10.0, "maxls": 50})
            records = []
            for j, item in enumerate(counts3):
                trial = item["attempts"][-1]
                self.assertEqual(trial["solver"]["factr"], 10.0)
                self.assertEqual((trial["solver"]["maxls"], trial["solver"]["pgtol"]), (50, 1e-5))
                self.assertEqual(len(trial["objective_trace"]), trial["iterations"] + 1)
                self.assertEqual(len(trial["step_trace"]), trial["iterations"])
                if trial["task"].startswith("CONVERGENCE"):
                    self.assertAlmostEqual(trial["objective_trace"][-1], loss3[j], places=9)
                records.append(cell_record(item["attempts"], j, z3[:, j], z_ref[:, j], loss3[j], loss_ref[j]))
                self.assertEqual(records[-1]["ftol"], 10.0 * np.finfo(float).eps)
            self.assertEqual(summarise(records)["n_voxels"], len(ids))

    @unittest.skipUnless(find_spec("pcntoolkit"), "PCNtoolkit not installed in this local runtime")
    def test_sweep_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            cov, gm, test, ref, z_ref, loss_ref = synthetic_problem(out)
            _, _, bind = gradient_binder(gm, test, ref)
            audit = {"worst_voxels": [{"mask_column": 100 + j, "gpu_from_archive": {"penalised_nll": float(loss_ref[j])}}
                                      for j in range(gm.shape[1])]}
            versions = {"pcntoolkit": "1.4.0", "scipy": "1.18.1", "jax": "synthetic", "jaxlib": "synthetic"}
            with (patch.object(stopping_sweep, "package_versions", return_value=versions),
                  patch.object(stopping_sweep, "read_audit", return_value=audit),
                  patch.object(stopping_sweep, "load_selected", return_value=(gm, cov, ref, z_ref, loss_ref, z_ref)),
                  patch.object(stopping_sweep, "compile_value_and_grad", return_value=bind)):
                sweep(out, "1", "2", "3", "4")
            receipt = json.loads(next(out.glob("stopping-sweep/*/sweep.json")).read_text())
            self.assertEqual(len(receipt["summary"]), 7)
            original = receipt["per_voxel"][stopping_sweep.ORIGINAL]
            self.assertEqual([r["cpu_archive_max_abs_delta_z"] for r in original], [0.0, 0.0])
            for rows in receipt["per_voxel"].values():
                self.assertEqual([r["mask_column"] for r in rows], [100, 101])
                self.assertTrue(all(len(r["objective_trace"]) == r["iterations"] + 1 for r in rows))


if __name__ == "__main__":
    unittest.main()
