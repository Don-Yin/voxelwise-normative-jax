"""The rule behind every real-IXI guard: Slurm or a valid Hugging Face marker on Linux, nothing else, never both."""

from __future__ import annotations

import unittest
from pathlib import Path

from shared.voxelwisenormative.batch_job import (
    HUGGING_FACE_ENVIRONMENT,
    SLURM_ENVIRONMENT,
    BatchJob,
    batch_job,
    expected_environment,
    hugging_face_job,
    require_batch_job,
)

ROOT = Path(__file__).resolve().parents[3]  # run from the project root with PYTHONPATH=$PWD

HF_ID = "1234567890abcdef12345678"  # invented ID
SLURM_ID = "81595615"


class Accepted(unittest.TestCase):
    def test_slurm_job_on_linux(self):
        self.assertEqual(batch_job({"SLURM_JOB_ID": SLURM_ID}, "linux"), BatchJob("slurm", SLURM_ID))
        self.assertIsNone(hugging_face_job({"SLURM_JOB_ID": SLURM_ID}, "linux"))
        self.assertEqual(expected_environment({"SLURM_JOB_ID": SLURM_ID}, "linux"), SLURM_ENVIRONMENT)

    def test_hugging_face_marker_on_linux(self):
        env = {"HUGGING_FACE_JOB_ID": HF_ID}
        self.assertEqual(batch_job(env, "linux"), BatchJob("hugging_face", HF_ID))
        self.assertEqual(hugging_face_job(env, "linux"), HF_ID)
        self.assertEqual(expected_environment(env, "linux"), HUGGING_FACE_ENVIRONMENT)
        self.assertEqual(require_batch_job("real IXI input", env, "linux").job_id, HF_ID)


class Refused(unittest.TestCase):
    def refused(self, env, platform="linux"):
        self.assertIsNone(batch_job(env, platform), (env, platform))
        with self.assertRaisesRegex(RuntimeError, "real IXI input can only be opened"):
            require_batch_job("real IXI input", env, platform)

    def test_mac(self):
        self.refused({"SLURM_JOB_ID": SLURM_ID}, "darwin")
        self.refused({"HUGGING_FACE_JOB_ID": HF_ID}, "darwin")

    def test_linux_without_a_marker(self):
        self.refused({})
        self.refused({"JOB_ID": HF_ID, "PATH": "/usr/bin"})  # the platform's own variable is not the marker

    def test_malformed_marker(self):
        for value in (HF_ID[:23], HF_ID + "0", HF_ID.upper(), "g" + HF_ID[1:], HF_ID + "\n", " " + HF_ID, "", SLURM_ID):
            self.refused({"HUGGING_FACE_JOB_ID": value})

    def test_malformed_slurm_id(self):
        for value in ("", "abc", "12a", "-12", "4159 5615", "４１５９５６１５"):
            self.refused({"SLURM_JOB_ID": value})

    def test_both_variables(self):
        self.refused({"SLURM_JOB_ID": SLURM_ID, "HUGGING_FACE_JOB_ID": HF_ID})
        self.refused({"SLURM_JOB_ID": "abc", "HUGGING_FACE_JOB_ID": HF_ID})

    def test_message_shows_what_was_found(self):
        with self.assertRaisesRegex(RuntimeError, "HUGGING_FACE_JOB_ID='abc'"):
            require_batch_job("real IXI input", {"HUGGING_FACE_JOB_ID": "abc"}, "linux")


class Environment(unittest.TestCase):
    def test_without_a_marker_variable_the_slurm_environment(self):
        self.assertEqual(expected_environment({}, "linux"), SLURM_ENVIRONMENT)
        self.assertEqual(expected_environment({}, "darwin"), SLURM_ENVIRONMENT)

    def test_a_marker_variable_that_is_no_job_is_an_error(self):
        for env, platform in (({"HUGGING_FACE_JOB_ID": HF_ID}, "darwin"), ({"HUGGING_FACE_JOB_ID": "abc"}, "linux"),
                              ({"HUGGING_FACE_JOB_ID": HF_ID, "SLURM_JOB_ID": SLURM_ID}, "linux")):
            with self.assertRaisesRegex(ValueError, "does not identify a Hugging Face job"):
                expected_environment(env, platform)

    def test_the_container_entry_builds_the_environment_the_guards_expect(self):
        entry = (ROOT / "shared/hugging_face_jobs/container_entry.sh").read_text()
        self.assertIn(f"VENV=${{HF_VENV_DIR:-/opt/{HUGGING_FACE_ENVIRONMENT}}}", entry)


if __name__ == "__main__":
    unittest.main()
