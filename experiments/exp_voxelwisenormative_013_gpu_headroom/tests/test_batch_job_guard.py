"""exp_013's real-IXI guard: a Slurm job, or a Hugging Face job with its marker; nothing else opens real input.

A job the guard lets through is stopped at the first artifact lookup, so no IXI file is read. Run from this folder:
JAX_PLATFORMS=cpu micromamba run -n emergence python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT), str(EXPERIMENT.parents[1])]
inputs = importlib.import_module("shared.voxelwisenormative.refit_inputs")
receipts = importlib.import_module("src.utils.receipts")

HF_ID = "1234567890abcdef12345678"
SLURM_ID = "81595615"
HF_ENV = {"HUGGING_FACE_JOB_ID": HF_ID}
SLURM_ENV = {"SLURM_JOB_ID": SLURM_ID}


class Reached(Exception):
    """The guard let the job through: the next step, the artifact lookup, was reached."""


def read_ixi(env: dict[str, str], platform: str) -> None:
    def lookup(*_args, **_kwargs):
        raise Reached

    with (mock.patch.dict(os.environ, env, clear=True), mock.patch.object(sys, "platform", platform),
          mock.patch.object(inputs, "resolve", lookup)):
        inputs.read_ixi()


def runtime(env: dict[str, str], platform: str = "linux") -> dict:
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(sys, "platform", platform):
        return receipts.runtime()


class RealInputGuard(unittest.TestCase):
    def refused(self, env: dict[str, str], platform: str = "linux") -> None:
        with self.assertRaisesRegex(RuntimeError, "real IXI input can only be opened"):
            read_ixi(env, platform)

    def test_mac_refuses(self):
        self.refused(SLURM_ENV, "darwin")
        self.refused(HF_ENV, "darwin")

    def test_linux_without_a_marker_refuses(self):
        self.refused({})
        self.refused({"JOB_ID": HF_ID})  # the platform's own variable is not the marker

    def test_slurm_job_passes(self):
        with self.assertRaises(Reached):
            read_ixi(SLURM_ENV, "linux")

    def test_hugging_face_marker_passes(self):
        with self.assertRaises(Reached):
            read_ixi(HF_ENV, "linux")

    def test_malformed_marker_refuses(self):
        for value in (HF_ID[:-1], HF_ID.upper(), HF_ID + "0", "", SLURM_ID):
            self.refused({"HUGGING_FACE_JOB_ID": value})

    def test_both_variables_refuse(self):
        self.refused({**SLURM_ENV, **HF_ENV})


class Receipt(unittest.TestCase):
    def test_slurm_receipt_has_no_hugging_face_key(self):
        record = runtime(SLURM_ENV)
        self.assertEqual(record["slurm_job_id"], SLURM_ID)
        self.assertNotIn("hugging_face_job", record)

    def test_hugging_face_receipt_leaves_every_slurm_field_empty(self):
        record = runtime(HF_ENV)
        self.assertEqual(record["hugging_face_job"], HF_ID)
        self.assertEqual([record[key] for key in ("slurm_job_id", "slurm_array_job_id", "slurm_array_task_id")], [None] * 3)

    def test_an_invalid_marker_is_not_recorded(self):
        self.assertNotIn("hugging_face_job", runtime(HF_ENV, "darwin"))
        self.assertNotIn("hugging_face_job", runtime({"HUGGING_FACE_JOB_ID": "abc"}))


if __name__ == "__main__":
    unittest.main()
