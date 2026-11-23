"""exp_011's real-IXI guards: a Slurm job, or a Hugging Face job with its marker; nothing else opens real input.

The scorer runs against a stand-in for JAX with one A100, so a job the guard lets through reaches the end of
`cuda_scorer` without a GPU, and no IXI file is read. Run from the project root:
PYTHONPATH="$PWD/$exp:$PWD" micromamba run -n voxelwise-normative-jax python -m unittest discover -s "$exp/tests" -v
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import types
import unittest
from unittest import mock

import run
from src.modules import study

HF_ID = "1234567890abcdef12345678"
SLURM_ID = "81595615"
HF_ENV = {"HUGGING_FACE_JOB_ID": HF_ID}
SLURM_ENV = {"SLURM_JOB_ID": SLURM_ID}
SLURM_PREFIX = "test-environments/voxelwise-normative-jax"
HF_PREFIX = "test-environments/hugging-face-a100-venv"
DEVICE = "NVIDIA A100-SXM4-80GB"


def fake_jax() -> types.SimpleNamespace:
    """JAX with one A100 and the version of the Slurm receipts."""
    return types.SimpleNamespace(__version__="0.12.1",
                                 devices=lambda: [types.SimpleNamespace(platform="gpu", device_kind=DEVICE)])


def scorer(env: dict[str, str], platform: str) -> tuple:
    """`cuda_scorer` under this environment and platform, with no GPU and no fitter."""
    with (mock.patch.dict(os.environ, env, clear=True), mock.patch.object(sys, "platform", platform),
          mock.patch.dict(sys.modules, {"jax": fake_jax()}), mock.patch.object(study, "make_fitter", lambda: None)):
        return study.cuda_scorer(32768)


def parse(prefix: str, env: dict[str, str], platform: str = "linux") -> str:
    """The empty string when `run.parse_args()` accepts this interpreter environment, else the error it printed."""
    stderr = io.StringIO()
    with (mock.patch.object(sys, "argv", ["run.py", "--synthetic"]), mock.patch.object(sys, "prefix", prefix),
          mock.patch.object(sys, "platform", platform), mock.patch.dict(os.environ, env, clear=True),
          contextlib.redirect_stderr(stderr)):
        try:
            run.parse_args()
        except SystemExit:
            return stderr.getvalue()
    return ""


class RealInputGuard(unittest.TestCase):
    def refused(self, env: dict[str, str], platform: str = "linux") -> None:
        with self.assertRaisesRegex(RuntimeError, "real IXI input can only be opened"):
            scorer(env, platform)

    def test_mac_refuses(self):
        self.refused(SLURM_ENV, "darwin")
        self.refused(HF_ENV, "darwin")

    def test_linux_without_a_marker_refuses(self):
        self.refused({})
        self.refused({"JOB_ID": HF_ID})  # the platform's own variable is not the marker

    def test_slurm_job_passes_and_its_receipt_is_unchanged(self):
        _, runtime = scorer(SLURM_ENV, "linux")
        self.assertEqual(runtime, {"device": DEVICE, "jax": "0.12.1", "slurm_job": SLURM_ID})

    def test_hugging_face_marker_passes_and_is_recorded_under_its_own_key(self):
        _, runtime = scorer(HF_ENV, "linux")
        self.assertEqual(runtime, {"device": DEVICE, "jax": "0.12.1", "slurm_job": None, "hugging_face_job": HF_ID})

    def test_malformed_marker_refuses(self):
        for value in (HF_ID[:-1], HF_ID.upper(), HF_ID + "0", "", SLURM_ID):
            self.refused({"HUGGING_FACE_JOB_ID": value})

    def test_both_variables_refuse(self):
        self.refused({**SLURM_ENV, **HF_ENV})


class EnvironmentName(unittest.TestCase):
    """run.py: voxelwise-normative-jax as before, or only the environment container_entry.sh builds when the marker is present."""

    def test_slurm_environment(self):
        self.assertEqual(parse(SLURM_PREFIX, SLURM_ENV), "")
        self.assertEqual(parse(SLURM_PREFIX, {}), "")  # unchanged: the name is the whole check without a marker

    def test_container_environment_needs_the_marker(self):
        self.assertIn("run in voxelwise-normative-jax", parse(HF_PREFIX, {}))
        self.assertEqual(parse(HF_PREFIX, HF_ENV), "")

    def test_a_marker_accepts_only_the_container_environment(self):
        self.assertIn("run in hugging-face-a100-venv", parse(SLURM_PREFIX, HF_ENV))

    def test_a_marker_that_is_no_hugging_face_job_refuses_in_either_environment(self):
        for prefix in (SLURM_PREFIX, HF_PREFIX):
            for env, platform in (({"HUGGING_FACE_JOB_ID": "abc"}, "linux"), (HF_ENV, "darwin"), ({**SLURM_ENV, **HF_ENV}, "linux")):
                self.assertIn("does not identify a Hugging Face job", parse(prefix, env, platform))

    def test_the_laptop_environment_refuses(self):
        self.assertIn("run in voxelwise-normative-jax", parse("test-environments/cpu-only", {}, "darwin"))


if __name__ == "__main__":
    unittest.main()
