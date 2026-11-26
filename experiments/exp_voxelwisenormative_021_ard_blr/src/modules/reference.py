"""Measure and save one serial PCNtoolkit ARD reference, or verify a saved run."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.artifacts import PCNTOOLKIT_REFERENCE, sha256
from shared.voxelwisenormative.batched_blr import PENALTY
from shared.voxelwisenormative.pcn_cpu_reference import (
    EPSILON,
    IXIInputs,
    PCNReference,
    fit_pcn,
)


def reference(data: IXIInputs, out: Path, *, replay: bool) -> PCNReference:
    """Time the original model unless replaying the identical saved reference inputs."""
    identity = {
        "artifact": PCNTOOLKIT_REFERENCE, "input_sha256": sha256(data.archive),
        "versions": {name: version(name) for name in ("pcntoolkit", "numpy", "scipy")},
        "epsilon": EPSILON, "penalty": PENALTY,
    }
    if replay:
        try:
            with np.load(out, allow_pickle=False) as f:
                saved = json.loads(str(f["identity"]))
                if saved != identity or not np.array_equal(f["voxels"], data.voxels):
                    raise ValueError(f"reference identity or voxel order differs: {saved!r} != {identity!r}")
                return PCNReference(f["hyp"], f["objective"], f["z"], f["noise"], f["noiset"], f["fit_seconds"],
                                    f["score_seconds"], float(f["seconds"]), float(f["design_error"]),
                                    json.loads(str(f["warnings"])))
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"cannot replay {out}: {exc}") from exc
    fitted = fit_pcn(data, ard=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, allow_pickle=False, voxels=data.voxels, identity=json.dumps(identity),
            hyp=fitted.hyp, objective=fitted.objective, z=fitted.z, noise=fitted.noise, noiset=fitted.noiset,
            fit_seconds=fitted.fit_seconds, score_seconds=fitted.score_seconds, seconds=fitted.seconds,
            design_error=fitted.design_error, warnings=json.dumps(fitted.warnings),
        )
    temporary.replace(out)
    return fitted
