"""Export a fitted homoskedastic warped regression through PCNtoolkit's native save API."""
from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from unicodedata import normalize

import numpy as np

from .basis import Basis
from .batched_blr import PENALTY

PCN_VERSION = "1.4.0"
_SPLINE_DEGREE = 3
_HYP_COUNT = 4


def _names(names: Sequence[str] | None, width: int, prefix: str) -> list[str]:
    """Choose the toolkit's default column names, or validate those supplied."""
    values = [f"{prefix}_{i}" for i in range(width)] if names is None else list(names)
    if isinstance(names, str) or len(values) != width or any(not isinstance(n, str) or not n for n in values):
        raise ValueError(f"expected {width} nonempty {prefix} column names, got {names!r}")
    return values


def _validate_names(covariates: list[str], responses: list[str], batch_names: list[str]) -> None:
    """Reject collisions in data columns and reserved or hidden response directories."""
    names = covariates + responses + batch_names
    if any(not isinstance(n, str) or not n for n in names) or len(names) != len(set(names)):
        raise ValueError(f"covariate, response and batch column names must be nonempty and distinct: {names!r}")
    folders = {"normative_model.json"}
    for name in responses:
        folder = normalize("NFC", name).casefold()
        if name.startswith(".") or any(c in name for c in "/\\\0:") or folder in folders:
            raise ValueError(f"invalid or duplicate PCNtoolkit response directory {name!r}")
        folders.add(folder)


def _pack_batches(levels: tuple, labels) -> tuple[list[str], np.ndarray | None]:
    """Check that string conversion preserves the fitted batch assignments."""
    names = np.asarray(levels, dtype=str).tolist()
    if len(names) != len(set(names)):
        raise ValueError(f"PCNtoolkit converts batch labels to strings; levels collide: {levels!r}")
    if labels is None:
        return names, None
    raw = np.asarray(labels, dtype=object)
    converted = raw.astype(str)
    spellings = dict(zip(levels, names, strict=True))
    for label, spelling in zip(raw, converted, strict=True):
        if spelling != spellings[label]:
            raise ValueError(f"batch label {label!r} converts to {spelling!r}, expected {spellings[label]!r}")
    return names, converted[:, None]


def _column_scalers(scaler, names: list[str]) -> dict:
    """Convert a vector of fitted scales to PCNtoolkit's one-scaler-per-column state."""
    state = scaler.to_dict()
    return {name: type(scaler).from_dict({**state, **{k: state[k][i] for k in ("m", "s", "min", "max")}})
            for i, name in enumerate(names)}


def export_pcntoolkit(
    path: str | Path,
    basis: Basis,
    hyp: np.ndarray,
    covariates_train: np.ndarray,
    responses_train: np.ndarray,
    batch_train=None,
    *,
    covariate_names: Sequence[str] | None = None,
    response_names: Sequence[str] | None = None,
    batch_name: str = "batch_effect_0",
) -> Path:
    """Save to a new folder from ``Basis.fit`` and ``fit_chunks(make_fitter(), ...)``.

    Pass the same raw training rows (not the held-out rows) and response order as
    the fit, with one hyperparameter row per response. The posterior is rebuilt
    at those fixed parameters; no optimizer is run. PCNtoolkit 1.4.0 is needed
    only when calling this function. Load with ``NormativeModel.load(str(path))``
    and predict from raw, float64 values in the same covariate order.
    """
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"PCNtoolkit export requires a new output directory: {path}")
    x, y, hyp = (np.asarray(a, dtype=np.float64) for a in (covariates_train, responses_train, hyp))
    phi = basis.apply(x, batch_train)
    if y.ndim != 2 or y.shape[0] != len(x) or not y.shape[1] or not np.isfinite(y).all():
        raise ValueError(f"expected finite training responses with shape ({len(x)}, n_responses > 0), got {y.shape}")
    if hyp.shape != (y.shape[1], _HYP_COUNT) or not np.isfinite(hyp).all():
        raise ValueError(f"expected finite homoskedastic warped hyperparameters with shape "
                         f"{(y.shape[1], _HYP_COUNT)}, got {hyp.shape}")
    covariates = _names(covariate_names, x.shape[1], "covariate")
    responses = _names(response_names, y.shape[1], "response_var")
    batch_names = [batch_name] if basis.batch_levels else []
    _validate_names(covariates, responses, batch_names)
    levels, labels = _pack_batches(basis.batch_levels, batch_train)

    try:
        installed = version("pcntoolkit")
    except PackageNotFoundError as exc:
        raise ImportError(f"install the pcntoolkit extra to export (requires version {PCN_VERSION})") from exc
    if installed != PCN_VERSION:
        raise RuntimeError(f"PCNtoolkit export requires version {PCN_VERSION}, found {installed}")
    pcn = import_module("pcntoolkit")
    standard_scaler = import_module("pcntoolkit.math_functions.scaler").StandardScaler

    y_scaler = standard_scaler()
    y_scaler.fit(y)
    bad_scale = np.flatnonzero(~np.isfinite(y_scaler.m) | ~np.isfinite(y_scaler.s) | (y_scaler.s <= 0))
    if bad_scale.size:
        raise ValueError(f"invalid training response scale for {responses[int(bad_scale[0])]!r}")
    y_scaled = y_scaler.transform(y)
    x_scaler = standard_scaler()
    x_scaler.m, x_scaler.s = basis.mean, basis.sd
    x_scaler.min, x_scaler.max = x.min(0), x.max(0)

    i = int(basis.spline_column)
    degree = _SPLINE_DEGREE
    knot_count = basis.knots.size - 2 * degree
    lo = float((x_scaler.min[i] - basis.mean[i]) / basis.sd[i])
    hi = float((x_scaler.max[i] - basis.mean[i]) / basis.sd[i])
    template = pcn.BLR(
        fixed_effect=bool(levels), warp_name="WarpSinhArcsinh", l_bfgs_b_l=PENALTY,
        basis_function_mean=pcn.BsplineBasisFunction(
            basis_column=i, degree=degree, nknots=knot_count,
            min=lo, max=hi, knots=basis.knots.tolist(), is_fitted=True,
        ),
    )
    # Our basis puts the spline first; PCNtoolkit expands its column in place.
    width = knot_count + degree - 1
    if i:
        phi = np.column_stack((phi[:, width:width + i], phi[:, :width], phi[:, width + i:]))
    model = pcn.NormativeModel(template, save_dir=str(path), saveplots=False, saveresults=False, evaluate_model=False)
    attrs = {"covariates": covariates, "response_vars": responses}
    if batch_names:
        attrs["batch_effect_dims"] = batch_names
    data = pcn.NormData.from_ndarrays("train", x, y, labels, attrs=attrs)
    model.register_data_info(data)
    if levels:
        model.unique_batch_effects[batch_name] = levels
        model.batch_effects_maps[batch_name] = {level: j for j, level in enumerate(levels)}
        for level in levels:
            model.batch_effect_counts[batch_name].setdefault(level, 0)
    model.inscalers = _column_scalers(x_scaler, covariates)
    model.outscalers = _column_scalers(y_scaler, responses)

    for j, name in enumerate(responses):
        regressor = model[name]
        regressor.be_maps = model.batch_effects_maps
        regressor.var_D = 1  # the unused variance design for constant noise
        # This evaluates the native posterior and objective, but does not optimize hyp.
        regressor.nlZ = regressor.penalized_loglik(hyp[j], phi, y_scaled[:, j], regularizer_strength=PENALTY, norm="L2")
        if not np.isfinite(regressor.nlZ) or regressor.nlZ == np.finfo(np.float64).max:
            raise ValueError(f"cannot compute a finite PCNtoolkit posterior for response {name!r}: {hyp[j]!r}")
        regressor.n_hyp = _HYP_COUNT
        regressor.gamma = hyp[j, 1:3].copy()
        regressor.is_fitted = True

    model.is_fitted = True
    path.mkdir(parents=True)
    model.save()
    return path
