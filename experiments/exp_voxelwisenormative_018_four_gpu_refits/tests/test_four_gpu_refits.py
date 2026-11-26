"""Regression checks for shard failures, z agreement, and the Hugging Face-only gate."""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP), str(EXP.parents[1])]
compare = importlib.import_module("src.modules.compare")


class Agreement(unittest.TestCase):
    """Match by draw and require numerical agreement as well as unchanged calls."""

    def test_threshold_flip_is_rejected_even_below_absolute_error_tolerance(self):
        ref = np.array([[[1.96]], [[0.5]]])
        actual = ref.copy()
        actual[0, 0, 0] = np.nextafter(1.96, np.inf)
        report = compare.compare_z(ref, actual)
        self.assertFalse(report["passed"])
        self.assertEqual(report["bitwise_equal_refits"], 1)
        self.assertEqual(report["differing_elements"], 1)
        self.assertEqual(report["flag_disagreements"], 1)
        self.assertLess(report["max_abs_dz"], report["max_abs_dz_allowed"])

    def test_non_finite_scores_are_rejected(self):
        ref = np.array([[[0.5]]])
        with self.assertRaisesRegex(ValueError, "non-finite z at draw 0"):
            compare.compare_z(ref, np.array([[[np.nan]]]))


class Trace(unittest.TestCase):
    """A compiler command buffer is a group of CUDA kernels, not one kernel."""

    def test_groups_do_not_masquerade_as_kernels(self):
        summary = importlib.import_module("shared.voxelwisenormative.refit_trace").hottest_kernels
        args = {"kernel_details": "grid:1,1,1", "hlo_op": "command_buffer_21"}
        report = summary([{"args": args, "name": name, "dur": duration}
                          for name, duration in (("reduce", 3), ("concatenate", 8), ("reduce", 2))]
                         + [{"name": "MemcpyH2D", "dur": 50}])
        self.assertEqual(report["kernel_duration_us"], 13)
        self.assertEqual([r["total_us"] for r in report["top_kernels"]], [8, 5])
        self.assertEqual(report["top_hlo_groups"][0]["distinct_kernels"], 2)
        self.assertEqual(report["top_hlo_groups"][0]["calls"], 3)


class Launcher(unittest.TestCase):
    """Accept only the actual Hugging Face marker, not a Slurm or conflicting marker."""

    def test_real_guard_is_hugging_face_only(self):
        job_id = "1234567890abcdef12345678"
        valid = {"HUGGING_FACE_JOB_ID": job_id, "JOB_ID": job_id}
        with patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, valid, clear=True):
                self.assertEqual(compare.require_hugging_face_wrapper(), job_id)
            for env in ({"SLURM_JOB_ID": "42"}, {**valid, "SLURM_CPUS_PER_TASK": "1"}, {**valid, "JOB_ID": "other"}):
                with (patch.dict(os.environ, env, clear=True),
                      self.assertRaisesRegex(RuntimeError, "require the Hugging Face wrapper")):
                    compare.require_hugging_face_wrapper()

    def test_worker_failure_is_not_hidden(self):
        inputs = SimpleNamespace(rows=np.arange(4), test=np.empty((1, 1)))
        batchers = [Mock(inputs=inputs, device=SimpleNamespace(id=i)) for i in range(4)]
        batchers[1].run.side_effect = ValueError("refit failed on device 1")
        for i in (0, 2, 3):
            batchers[i].run.return_value = SimpleNamespace(compute_seconds=0.1, gather_seconds=0.01)
        with self.assertRaisesRegex(ValueError, "refit failed on device 1"):
            compare.refit_pass(batchers)


if __name__ == "__main__":
    unittest.main()
