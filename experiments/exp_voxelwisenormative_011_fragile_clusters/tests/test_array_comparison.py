"""array_comparison.py and jobs/compare_arrays.py: what counts as identical, and what a difference reports."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from src.modules import array_comparison as ac
from src.modules.inputs import THRESHOLD

JOB = Path(__file__).resolve().parents[1] / "jobs" / "compare_arrays.py"


def save(folder: Path, name: str = "z-summary.npz", **arrays: Any) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / name, **arrays)
    return folder


class CompareArray(unittest.TestCase):
    def test_equal_arrays_are_identical(self):
        z = np.linspace(-3, 3, 24).reshape(4, 6)
        row = ac.compare_array(z, z.copy())
        self.assertEqual((row["differ"], row["elements"], row["shape"], row["dtype"]), (0, 24, (4, 6), "float64"))
        self.assertNotIn("first", row)

    def test_one_last_bit_is_found_with_its_place_size_and_no_flag_moving(self):
        z = np.linspace(-3, 3, 24).reshape(4, 6)
        other = z.copy()
        other[2, 5] = np.nextafter(other[2, 5], np.inf)
        row = ac.compare_array(z, other)
        self.assertEqual((row["differ"], row["first"], row["flags_changed"]), (1, (2, 5), 0))
        self.assertEqual(row["largest_difference"], float(other[2, 5] - z[2, 5]))

    def test_a_flag_that_moves_at_the_threshold_is_counted(self):
        z = np.array([[THRESHOLD, 0.5]])
        other = np.array([[np.nextafter(THRESHOLD, np.inf), 0.5]])  # 1.96 is not above 1.96; the next float is
        self.assertEqual(ac.compare_array(z, other)["flags_changed"], 1)
        self.assertEqual(ac.compare_array(-z, -other)["flags_changed"], 1, "the sign of z does not matter")

    def test_bits_are_compared_not_values(self):
        row = ac.compare_array(np.array([0.0, np.nan, 1.0]), np.array([-0.0, np.nan, 1.0]))
        self.assertEqual((row["differ"], row["first"]), (1, (0,)))  # -0.0 == 0.0 as values; NaN with the same bits is the same
        self.assertEqual(ac.compare_array(np.array([np.nan]), np.array([np.nan]))["differ"], 0)

    def test_packed_flags_report_bytes_and_bits(self):
        old = np.zeros((2, 3, 4), dtype=np.uint8)
        new = old.copy()
        new[1, 2, 3] = 0b10100100  # three flags in one byte
        new[0, 0, 0] = 1
        row = ac.compare_array(old, new, flags_packed=True)
        self.assertEqual((row["differ"], row["flag_bits"], row["first"]), (2, 4, (0, 0, 0)))
        self.assertNotIn("largest_difference", row)

    def test_dtype_or_shape_mismatch_is_a_difference_without_a_count(self):
        for other in (np.zeros((4, 6), dtype=np.float32), np.zeros((4, 5))):
            row = ac.compare_array(np.zeros((4, 6)), other)
            self.assertIsNone(row["differ"])
            self.assertIn("baseline float64 (4, 6)", str(row["mismatch"]))

    def test_other_types_are_refused(self):
        with self.assertRaisesRegex(ValueError, "cannot compare the bits"):
            ac.compare_array(np.array(["a"]), np.array(["a"]))


class CompareRuns(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.z = np.linspace(-3, 3, 24).reshape(4, 6)
        self.rows = np.arange(4, dtype=np.int32)
        self.flags = np.packbits(np.abs(self.z) > THRESHOLD, axis=-1, bitorder="little")

    def run_folder(self, name: str, z: np.ndarray | None = None) -> Path:
        folder = save(self.base / name, "refit-flags.npz", refit_flags_packed=self.flags[None], held_out_cohort_rows_zero_based=self.rows)
        return save(folder, original_z=self.z if z is None else z, held_out_cohort_rows_zero_based=self.rows)

    def test_two_runs_with_the_same_arrays_are_identical_although_written_separately(self):
        rows = ac.compare_runs(self.run_folder("old"), self.run_folder("new"))
        self.assertEqual([(r["file"], r["key"]) for r in rows], [("refit-flags.npz", "refit_flags_packed"),
                         ("refit-flags.npz", "held_out_cohort_rows_zero_based"), ("z-summary.npz", "original_z"),
                         ("z-summary.npz", "held_out_cohort_rows_zero_based")])
        self.assertTrue(ac.identical(rows))
        self.assertEqual(ac.render(rows, "old", "new")[-1], "all 4 arrays of 2 files are bit-for-bit identical")

    def test_a_changed_array_is_named_and_counted(self):
        other = self.z.copy()
        other[1, 1] += 1e-12
        rows = ac.compare_runs(self.run_folder("old"), self.run_folder("new", other))
        self.assertFalse(ac.identical(rows))
        text = "\n".join(ac.render(rows, "old", "new"))
        self.assertRegex(text, r"z-summary\.npz\s+original_z\s+float64 \(4, 6\)\s+DIFFERENT: 1 of 24 elements \(4\.17%\), first at \(1, 1\); "
                               r"largest difference 1e-12, entries whose flag at 1\.96 changes: 0")
        self.assertEqual(text.splitlines()[-1], "1 of 4 arrays differ")

    def test_a_key_in_only_one_file_and_a_missing_file(self):
        old, new = self.run_folder("old"), self.run_folder("new")
        save(new, original_z=self.z, held_out_cohort_rows_zero_based=self.rows, refit_max_abs_z=self.z)
        rows = ac.compare_runs(old, new, ["z-summary.npz"])
        self.assertEqual([r["mismatch"] for r in rows if r["differ"] is None], ["only in the new file"])
        self.assertIn("DIFFERENT: only in the new file", "\n".join(ac.render(rows, "old", "new")))
        with self.assertRaisesRegex(FileNotFoundError, "cutoff-flags.npz"):
            ac.compare_runs(old, new, ["cutoff-flags.npz"])


class Command(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("compare_arrays", JOB)
        assert spec and spec.loader
        cls.job = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.job)

    def run_job(self, *arguments: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = self.job.main([*arguments, "--no-profile"])
        return status, out.getvalue()

    def test_exit_status_says_whether_every_array_is_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            z = np.linspace(-3, 3, 24).reshape(4, 6)
            same = save(Path(tmp) / "same", original_z=z)
            also = save(Path(tmp) / "also", original_z=z.copy())
            other = save(Path(tmp) / "other", original_z=z + 1e-9)
            self.assertEqual(self.run_job("--baseline", str(same), "--new", str(also), "--files", "z-summary.npz")[0], 0)
            status, text = self.run_job("--baseline", str(same), "--new", str(other), "--files", "z-summary.npz")
            self.assertEqual(status, 1)
            self.assertIn("DIFFERENT: 24 of 24 elements (100%), first at (0, 0); largest difference 1e-09", text)

    def test_a_run_that_is_not_there_stops_with_where_it_looked(self):
        with self.assertRaisesRegex(SystemExit, "no folder with refit-flags.npz, z-summary.npz for 'hugging-face-a100-nowhere'"):
            self.run_job("--baseline", "hugging-face-a100-nowhere")


if __name__ == "__main__":
    unittest.main()
