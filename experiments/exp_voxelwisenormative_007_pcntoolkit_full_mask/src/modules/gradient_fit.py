"""Supply a CPU JAX gradient to the exact PCNtoolkit/SciPy fit at runtime."""

from __future__ import annotations

import inspect
from importlib import import_module
from time import perf_counter
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy.optimize import _lbfgsb_py

from shared.voxelwisenormative.batched_blr import LOG2PI, PENALTY

from .fit import EPSILON, fit_block


def compile_value_and_grad(phi, sample_y):
    """Warm up float64 AD once; bind each installed fit's actual design and response."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve

    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"requires JAX CPU, got {jax.default_backend()}")
    n, d = phi.shape
    ident = jnp.eye(d, dtype=jnp.float64)

    def objective(theta, x, gram, y):
        """Warped negative log marginal likelihood and the same L2 penalty."""
        log_beta, shift, log_shape, log_alpha = theta
        beta, shape, alpha = jnp.exp(log_beta), jnp.exp(log_shape), jnp.exp(log_alpha)
        u = shape * (jnp.arcsinh(y) + shift)
        warped = jnp.sinh(u)
        chol = jnp.linalg.cholesky(beta * gram + alpha * ident)
        w = cho_solve((chol, True), beta * (x.T @ warped))
        e = warped - x @ w
        logdet = 2 * jnp.log(jnp.diag(chol)).sum()
        logjac = log_shape + jnp.logaddexp(u, -u) - jnp.log(2) - 0.5 * jnp.log1p(y**2)
        evidence = (n * (LOG2PI - log_beta) - d * log_alpha + beta * (e @ e) + alpha * (w @ w) + logdet) / 2
        # The norm's zero subgradient at the origin avoids a spurious 0 / 0 there.
        sq_norm = theta @ theta
        safe_norm = jnp.sqrt(jnp.asarray(jnp.where(sq_norm > 0, sq_norm, 1.0)))
        norm = jnp.where(sq_norm > 0, safe_norm, 0.0)
        return evidence - logjac.sum() + PENALTY * norm

    def device_args(x, y):
        return jax.device_put((np.ascontiguousarray(x), x.T @ x, np.ascontiguousarray(y)))

    start = np.ones(4)
    args = device_args(phi, sample_y)
    compiled = jax.jit(jax.value_and_grad(objective)).lower(start, *args).compile()
    compiled(start, *args)[1].block_until_ready()

    def bind(x, y):
        data = device_args(x, y)
        return lambda theta: compiled(theta, *data)

    return bind


def expect_original_call(kwargs):
    """Refuse any call other than PCNtoolkit's finite-difference fit of this warped BLR."""
    func, (_, _, var, strength, norm) = kwargs["func"], kwargs["args"]
    owner = func.__self__
    if (kwargs.get("approx_grad") is not True or "fprime" in kwargs or kwargs.get("epsilon") != EPSILON
            or owner.ard or owner.warp_reparam or type(owner.warp).__name__ != "WarpSinhArcsinh"
            or np.any(var) or strength != PENALTY or norm.upper() != "L2"):
        raise ValueError("unexpected optimiser, warp, variance model or penalty in PCNtoolkit BLR")


def iteration_watch(rec, callback):
    """Record each accepted iterate's objective and step length; forward to PCNtoolkit's own callback."""
    last = []

    def watch(intermediate_result):
        x = np.copy(intermediate_result.x)
        rec["objective_trace"].append(float(intermediate_result.fun))
        rec["step_trace"].append(float(np.linalg.norm(x - last[0])))
        last[:] = [x]
        if callback is not None:
            callback(x)

    return watch, last


def observe_optimizer(kwargs, trials, bind, solver=None):
    """Call fmin_l_bfgs_b; expose the nfev and njev counters it normally hides.

    A solver override (factr, maxls or pgtol) also records the objective at every iteration.
    """
    # Read counters through a scoped wrapper without changing SciPy's algorithm or options.
    native = _lbfgsb_py._minimize_lbfgsb
    expect_original_call(kwargs)
    func, (x, y) = kwargs["func"], kwargs["args"][:2]
    names = ("m", "factr", "pgtol", "maxfun", "maxiter", "maxls")
    defaults = inspect.signature(_lbfgsb_py.fmin_l_bfgs_b).parameters
    rec = {"function_calls": 0, "gradient_calls": None, "invalid_trial_gradients": 0, "sentinel_evaluations": 0,
           "solver": {n: defaults[n].default for n in names} | (solver or {})}
    trials.append(rec)
    invalid_point = None
    last: list = []
    if solver is not None:
        rec["objective_trace"], rec["step_trace"] = [], []
        watch, last = iteration_watch(rec, kwargs.get("callback"))
        kwargs.update(solver, callback=watch)

    def counted(theta, *args):
        nonlocal invalid_point
        rec["function_calls"] += 1
        value = func(theta, *args)
        if solver is not None and not rec["objective_trace"]:
            rec["objective_trace"].append(float(value))  # the start point; later entries follow each iteration
            last[:] = [theta.copy()]
        # PCNtoolkit substitutes a constant for numerical failures in loglik.
        failed = value == np.finfo(np.float64).max
        rec["sentinel_evaluations"] += int(failed)
        invalid_point = theta.copy() if failed and bind is not None else None
        return value

    def observe(*args, **options):
        result = native(*args, **options)
        if result.nfev != rec["function_calls"]:
            raise RuntimeError("SciPy's objective counter differs from the observed calls")
        rec["gradient_calls"] = int(result.njev)
        return result

    start = perf_counter()
    try:
        kwargs["func"] = counted
        if bind is not None:
            value_and_grad = bind(x, y)

            def fprime(theta, *_):
                if invalid_point is not None and np.array_equal(theta, invalid_point):
                    rec["invalid_trial_gradients"] += 1
                    return np.zeros_like(theta)
                grad = np.asarray(value_and_grad(theta)[1])
                if not np.isfinite(grad).all():
                    raise FloatingPointError(f"nonfinite JAX gradient at theta={theta.tolist()}")
                return grad

            kwargs.update(approx_grad=False, fprime=fprime)
        with patch.object(_lbfgsb_py, "_minimize_lbfgsb", observe):
            result = _lbfgsb_py.fmin_l_bfgs_b(**kwargs)
    except Exception as exc:
        rec.update(task=type(exc).__name__, error=str(exc))
        raise
    finally:
        rec["optimizer_seconds"] = perf_counter() - start
    theta, _, info = result
    task = info["task"]
    grad = info["grad"]
    rec.update(task=task.decode() if isinstance(task, bytes) else str(task), warnflag=int(info["warnflag"]),
               iterations=int(info["nit"]), gradient_inf=float(np.max(np.abs(grad))) if np.isfinite(grad).all() else None,
               theta=theta.tolist())
    return result


def timed_fit(gm, cov, ref, indices, out, bind, solver=None):
    """Keep the original preprocessing, model, restarts and z prediction in every condition."""
    blr = import_module("pcntoolkit.regression_model.blr")
    original_fit = blr.BLR.fit
    observed = {}

    def call(**kwargs):
        record = observed[kwargs["func"].__self__.name]
        return observe_optimizer(kwargs, record["attempts"], bind, solver)

    def fit(model, *args, **kwargs):
        if model.name in observed:
            raise RuntimeError(f"unexpected duplicate fit for {model.name}")
        record: dict[str, Any] = {"attempts": []}
        observed[model.name] = record
        start = perf_counter()
        original_fit(model, *args, **kwargs)
        record["blr_fit_seconds"] = perf_counter() - start

    with patch.object(blr.optimize, "fmin_l_bfgs_b", call), patch.object(blr.BLR, "fit", fit):
        z, _, objective = fit_block(gm, cov, ref["test"], indices, ref["Phi_train"], ref["Phi_test"], out)
    if set(observed) != {f"v{i}" for i in indices}:
        raise RuntimeError(f"unexpected fitted voxel set: {list(observed)}")
    return z, objective, [observed[f"v{i}"] for i in indices]


def verify_gradient(bind, phi, phit, y, yt, ids):
    """Check AD against the installed scalar loss and the independent audit derivative."""
    from .outlier_objective import Objective

    blr = import_module("pcntoolkit.regression_model.blr")
    gram, no_variance = phi.T @ phi, np.zeros((len(phi), 1))
    loss_gap = grad_gap = 0.0
    for i in ids:
        exact = Objective(phi, gram, phit, y[:, i], yt[:, i])
        native = blr.BLR(name="objective_check", warp_name="WarpSinhArcsinh")
        ad = bind(phi, y[:, i])
        for theta in (np.ones(4), np.array([0.3, -0.25, 0.1, -0.4]), np.zeros(4)):
            loss = native.penalized_loglik(theta, phi, y[:, i], no_variance, PENALTY, "L2")
            _, independent, expected, _ = exact.evaluate(theta)
            value, grad = ad(theta)
            np.testing.assert_allclose([value, independent], loss, rtol=1e-9, atol=1e-7)
            np.testing.assert_allclose(grad, expected, rtol=1e-7, atol=1e-7)
            loss_gap = max(loss_gap, abs(float(value) - float(loss)))
            grad_gap = max(grad_gap, float(np.max(np.abs(grad - expected))))
    return {"checked_voxels": len(ids), "max_abs_loss_gap": loss_gap, "max_abs_grad_gap": grad_gap}
