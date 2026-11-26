"""Plot measured CPU times and paired held-out scores for the heteroskedastic fit."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from result.pcn_cpu_comparison import build_comparison

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "exp_voxelwisenormative_020_heteroskedastic_blr" / ".results"


def build_figure(*, target_width_cm=16):
    """Draw the frozen-width experiment from its locally generated results."""
    report_path = RESULTS / "json" / "benchmark-1000.json"
    paired_path = RESULTS / "tables" / "fit-arrays-1000.npz"
    raw_report = report_path.read_bytes()
    try:
        report = json.loads(raw_report)
    except json.JSONDecodeError as exc:
        raise ValueError(f"cannot read the benchmark report {report_path}: {exc}") from exc
    cpu = report["jax_cpu"]
    times = report["pcntoolkit"]["total_seconds"], cpu["first_fit_with_precompute_seconds"], cpu["warm_seconds_median"]
    with np.load(paired_path, allow_pickle=False) as pairs:
        gaps = np.abs(pairs["jax_z"] - pairs["pcn_z"])
    fig, meta = build_comparison(times, gaps, target_width_cm=target_width_cm)
    return fig, {**meta, "input_sha256": {
        str(report_path.relative_to(ROOT)): sha256(raw_report).hexdigest(),
        str(paired_path.relative_to(ROOT)): sha256(paired_path.read_bytes()).hexdigest(),
    }}
