"""Exercise the new narrow staging profiles with generated stand-ins and no upload."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from shared.hugging_face_jobs import build_input as staging


@pytest.mark.parametrize(("profile", "experiment", "module"), [
    ("four-gpu-refits", "018_four_gpu_refits", "compare"),
    ("zbin-stability", "019_zbin_stability", "zbins"),
])
def test_maskless_ixi_mount(tmp_path, monkeypatch, profile, experiment, module):
    """A mount contains just the expected inputs and imports without the original checkout."""
    source, output = tmp_path / "source", tmp_path / "mount"
    folder = f"{staging.CODE_PREFIX}{experiment}"
    spec = staging.PROFILES[profile]
    assert tuple(spec.code) == (experiment,)
    for path in ["shared/__init__.py", *staging.JOB_FILES,
                 *(f"{folder}/{entry}" for entry in spec.code[experiment]),
                 *(f"shared/voxelwisenormative/{name}.py" for name in spec.modules)]:
        path = Path(path)
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staging.ROOT / path, target)

    matrix_id = "ixi-grey-matter-voxel-matrix-v1"
    ref_id = "pcntoolkit-blr-reference-n1000-v1"
    base = source / "experiments/shared/artifacts"
    matrix, reference = base / matrix_id, base / ref_id
    matrix.mkdir(parents=True)
    reference.mkdir()
    np.save(matrix / "gm.npy", np.ones((3, 4), dtype=np.float32))
    (matrix / "covariates.tsv").write_text("subject\nIXI901-HH-9001-T1\nIXI902-Guys-9002-T1\nIXI903-IOP-9003-T1\n")
    np.savez(reference / "arrays.npz", test=np.array([False, False, True]))
    for artifact, payloads, extra in (
        (matrix, ("gm.npy", "covariates.tsv"), {"source_dataset": "https://brain-development.org/ixi-dataset/"}),
        (reference, ("arrays.npz",), {"input_artifact": matrix_id}),
    ):
        files = [{"path": name, "sha256": staging.sha256(artifact / name)} for name in payloads]
        (artifact / "manifest.json").write_text(json.dumps({**extra, "files": files}))

    monkeypatch.setattr(staging, "MATRIX_SHAPE", (3, 4))
    rows = staging.build(output, source, profile)
    assert rows == staging.verify(output)
    assert {r["path"] for r in rows if r["role"] == "data"} == {
        f"experiments/shared/artifacts/{artifact}/{name}"
        for artifact, names in ((matrix_id, ("manifest.json", "gm.npy", "covariates.tsv")),
                                (ref_id, ("manifest.json", "arrays.npz")))
        for name in names
    }
    assert not tuple(output.rglob("mask.nii.gz"))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (output / folder, output)))}
    subprocess.run([sys.executable, "-B", "-c", f"import src.modules.{module}"], cwd=output, env=env, check=True)

    (matrix / "gm.npy").write_bytes(b"wrong payload")
    with pytest.raises(ValueError, match="gm.npy does not match"):
        staging.build(tmp_path / "bad", source, profile)
