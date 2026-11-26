"""Plot CPU fit times and paired held-out scores for automatic relevance determination."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from result.pcn_cpu_comparison import build_comparison

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "exp_voxelwisenormative_021_ard_blr" / ".results"


def build_figure(*, target_width_cm=16):
    """Compare the deployed JAX fit with serial PCNtoolkit at the frozen voxel width."""
    report_path = RESULTS / "json" / "benchmark-1000.json"
    jax_path = RESULTS / "tables" / "fit-arrays-1000.npz"
    pcn_path = RESULTS / "tables" / "pcntoolkit-1000.npz"
    raw = report_path.read_bytes()
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"cannot read the benchmark report {report_path}: {exc}") from exc
    cpu = report["jax_cpu"]
    times = report["pcntoolkit"]["total_seconds"], cpu["first_fit_with_precompute_seconds"], cpu["warm_seconds_median"]
    with np.load(jax_path, allow_pickle=False) as pairs, np.load(pcn_path, allow_pickle=False) as pcn:
        if not np.array_equal(pairs["voxels"], pcn["voxels"]):
            raise ValueError(f"voxel order differs between {jax_path} and {pcn_path}")
        gaps = np.abs(pairs["jax_z"] - pcn["z"])
    figure, metadata = build_comparison(times, gaps, target_width_cm=target_width_cm)
    return figure, {**metadata, "input_sha256": {
        str(path.relative_to(ROOT)): sha256(content).hexdigest()
        for path, content in ((report_path, raw), (jax_path, jax_path.read_bytes()), (pcn_path, pcn_path.read_bytes()))
    }}
