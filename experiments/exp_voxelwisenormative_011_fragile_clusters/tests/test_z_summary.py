"""--z-summary: the outputs it must not touch, the one file it adds, and the arithmetic that makes every threshold available."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import run
from src.modules import study
from src.modules.inputs import THRESHOLD, draws, synthetic
from src.modules.publish import Outputs
from src.modules.z_summary import (
    CUTOFF_FILE,
    FILE,
    MAX_CUTOFFS,
    ZSummary,
    validate_cutoffs,
)
from unchanged_outputs import EXPECTED_OUTPUTS, WALL_CLOCK, digest, synthetic_runs

SLURM_PREFIX = "test-environments/voxelwise-normative-jax"


def receipt(results: Path, name: str) -> dict:
    return json.loads((results / "json" / name / "result.json").read_text())


def without_run_specifics(text: dict) -> dict:
    """A receipt without its name, outputs, wall-clock seconds and --z-summary entry: what two runs must share."""
    return {key: value for key, value in text.items() if key not in {"run_name", "outputs", "z_summary", *WALL_CLOCK}}


def parse(*arguments: str) -> tuple[argparse.Namespace | None, str]:
    """(namespace, '') when run.parse_args() accepts the arguments in a Slurm job, else (None, what it printed)."""
    stderr = io.StringIO()
    with (mock.patch.object(sys, "argv", ["run.py", *arguments]), mock.patch.object(sys, "prefix", SLURM_PREFIX),
          mock.patch.object(sys, "platform", "linux"), mock.patch.dict(os.environ, {"SLURM_JOB_ID": "81595615"}, clear=True),
          contextlib.redirect_stderr(stderr)):
        try:
            return run.parse_args(), ""
        except SystemExit:
            return None, stderr.getvalue()


def accepted(*arguments: str) -> argparse.Namespace:
    namespace, error = parse(*arguments)
    if namespace is None:
        raise AssertionError(f"run.py refused {arguments}: {error}")
    return namespace


class OptionOff(unittest.TestCase):
    def test_every_output_is_what_it_was_before_the_option_existed(self):
        """tests/data/unchanged-outputs.json was made by the code from before --z-summary; see unchanged_outputs.py."""
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / ".results"
            synthetic_runs(results)
            self.assertEqual(digest(results), EXPECTED_OUTPUTS)


class OptionOn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.results = Path(tmp.name) / ".results"
        study.run_per_refit(cls.results, "synthetic-plain", 32768, use_fixture=True)
        study.run_per_refit(cls.results, "synthetic-z", 32768, use_fixture=True, z_summary=True)

    def archive(self, name: str, file: str) -> dict[str, np.ndarray]:
        with np.load(self.results / "arrays" / name / file, allow_pickle=False) as arrays:
            return dict(arrays)

    def test_flags_and_receipt_are_the_plain_run_plus_one_file(self):
        plain, with_z = self.archive("synthetic-plain", "refit-flags.npz"), self.archive("synthetic-z", "refit-flags.npz")
        self.assertEqual(list(plain), list(with_z))
        for key in plain:
            np.testing.assert_array_equal(with_z[key], plain[key])
        old, new = receipt(self.results, "synthetic-plain"), receipt(self.results, "synthetic-z")
        self.assertEqual(without_run_specifics(new), without_run_specifics(old))
        self.assertEqual(list(old["outputs"]), ["refit-flags.npz"])
        self.assertEqual(list(new["outputs"]), ["refit-flags.npz", FILE])
        self.assertNotIn("z_summary", old)
        file = self.results / "arrays" / "synthetic-z" / FILE
        self.assertEqual(new["outputs"][FILE]["bytes"], file.stat().st_size)

    def test_summary_is_the_extremes_of_the_refits_and_the_fit_without_resampling(self):
        data = synthetic()
        samples, _ = draws(data, verify_original=False)
        z = np.stack([study.scores_for(data, rows, study.fixture_score, include_site=True) for rows in samples])
        original = study.scores_for(data, np.arange(len(data.train_rows)), study.fixture_score, include_site=True)
        summary = self.archive("synthetic-z", FILE)
        self.assertEqual(list(summary), ["original_z", "refit_min_abs_z", "refit_max_abs_z", "held_out_cohort_rows_zero_based"])
        self.assertEqual({key: str(summary[key].dtype) for key in list(summary)[:3]}, dict.fromkeys(list(summary)[:3], "float64"))
        np.testing.assert_array_equal(summary["original_z"], original)
        np.testing.assert_array_equal(summary["refit_min_abs_z"], np.abs(z).min(axis=0))
        np.testing.assert_array_equal(summary["refit_max_abs_z"], np.abs(z).max(axis=0))
        np.testing.assert_array_equal(summary["held_out_cohort_rows_zero_based"], data.test_rows)

    def test_any_threshold_gives_the_voxels_flagged_in_every_refit(self):
        data = synthetic()
        samples, _ = draws(data, verify_original=False)
        z = np.stack([study.scores_for(data, rows, study.fixture_score, include_site=True) for rows in samples])
        smallest = self.archive("synthetic-z", FILE)["refit_min_abs_z"]
        for threshold in (0.5, 1.0, THRESHOLD, 2.5, 3.0):
            every = (np.abs(z) > threshold).all(axis=0)
            np.testing.assert_array_equal(smallest > threshold, every)
            self.assertTrue(every.any() and not every.all(), f"threshold {threshold} would test nothing here")
        counts = self.archive("synthetic-z", "refit-flags.npz")["recurrence_counts"]
        np.testing.assert_array_equal(smallest > THRESHOLD, counts == len(samples))

    def test_the_receipt_says_what_was_compared(self):
        entry = receipt(self.results, "synthetic-z")["z_summary"]
        counts = self.archive("synthetic-z", "refit-flags.npz")["recurrence_counts"]
        self.assertEqual((entry["file"], entry["dtype"]), (FILE, "float64"))
        self.assertEqual(entry["check"], {"threshold": THRESHOLD, "entries": counts.size,
                                          "above_threshold_in_every_refit": int((counts == 64).sum()), "entries_that_differ": 0})
        self.assertGreater(entry["original_fit_seconds"], 0)


class OtherModes(unittest.TestCase):
    def test_a_reference_subset_shares_its_single_fit_with_the_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / ".results"
            study.run_per_refit(results, "synthetic-n24-z", 32768, use_fixture=True, reference_n=24, z_summary=True)
            with np.load(results / "arrays/synthetic-n24-z/refit-flags.npz") as flags, np.load(results / "arrays/synthetic-n24-z" / FILE) as z:
                np.testing.assert_array_equal(np.unpackbits(flags["original_flags_packed"], axis=-1, count=320, bitorder="little"),
                                              np.abs(z["original_z"]) > THRESHOLD)
            self.assertEqual(receipt(results, "synthetic-n24-z")["reference_n"], 24)

    def test_a_later_fold_keeps_its_own_scans(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / ".results"
            study.run_per_refit(results, "synthetic-fold-2-z", 32768, use_fixture=True, fold=2, z_summary=True)
            with np.load(results / "arrays/synthetic-fold-2-z/refit-flags.npz") as flags, np.load(results / "arrays/synthetic-fold-2-z" / FILE) as z:
                np.testing.assert_array_equal(z["held_out_cohort_rows_zero_based"], flags["held_out_cohort_rows_zero_based"])
                self.assertEqual(z["original_z"].shape, (6, 320))


class SelfCheck(unittest.TestCase):
    def test_a_summary_that_disagrees_with_the_counts_is_refused(self):
        z = np.array([[2.5, 1.0], [3.0, 2.0]])
        summary = ZSummary(z, 1.0)
        summary.add(np.abs(z))
        summary.add(np.abs(z) + 0.1)
        counts = (summary.smallest > THRESHOLD).astype(np.uint8) * 2
        self.assertEqual(summary.check(counts)["entries_that_differ"], 0)
        counts[0, 1] = 2  # flagged in every refit although its smallest |z| is 1.0
        with self.assertRaisesRegex(ValueError, "differ in 1 of 4"):
            summary.check(counts)

    def test_a_run_whose_check_fails_publishes_nothing(self):
        def never_lowers_the_minimum(self, _):
            self.refits += 1

        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / ".results"
            with mock.patch.object(ZSummary, "add", never_lowers_the_minimum), self.assertRaisesRegex(ValueError, "differ in"):
                study.run_per_refit(results, "synthetic-broken", 32768, use_fixture=True, z_summary=True)
            self.assertFalse(any(results.rglob("*")) if results.exists() else False)
            self.assertFalse((Path(tmp) / ".intermediates" / "synthetic-broken").exists())

    def test_float64_keeps_the_voxel_just_above_the_threshold_that_float32_loses(self):
        just_above = np.nextafter(THRESHOLD, 3.0)
        summary = ZSummary(np.full((1, 1), 2.0), 1.0)
        summary.add(np.full((1, 1), just_above))
        self.assertEqual(summary.check(np.ones((1, 1), dtype=np.uint8))["above_threshold_in_every_refit"], 1)
        self.assertFalse(np.float32(just_above) > np.float32(THRESHOLD))  # why the arrays are not float32

    def test_wrong_inputs_are_refused(self):
        with self.assertRaisesRegex(ValueError, "float64"):
            ZSummary(np.zeros((2, 3), dtype=np.float32), 1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            ZSummary(np.array([[np.nan]]), 1.0)
        summary = ZSummary(np.ones((2, 3)), 1.0)
        with self.assertRaisesRegex(ValueError, "must be float64"):
            summary.add(np.ones((2, 4)))
        with self.assertRaisesRegex(ValueError, "0 refits added"):
            summary.check(np.zeros((2, 3), dtype=np.uint8))


CUTOFFS = (2.3, 2.6, 3.0)


def without_timing(entry: dict) -> dict:
    """A receipt's `z_summary` entry without the cutoff part and the single fit's seconds."""
    return {key: value for key, value in entry.items() if key not in {"cutoff_flags", "original_fit_seconds"}}


class Cutoffs(unittest.TestCase):
    """--refit-flag-cutoffs: every refit's flags and the single fit's at stricter cutoffs, checked against the z summary."""

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.results = Path(tmp.name) / ".results"
        study.run_per_refit(cls.results, "synthetic-z", 32768, use_fixture=True, z_summary=True)
        study.run_per_refit(cls.results, "synthetic-cutoffs", 32768, use_fixture=True, z_summary=True, cutoffs=CUTOFFS)
        data = synthetic()
        samples, _ = draws(data, verify_original=False)
        cls.abs_z = np.abs(np.stack([study.scores_for(data, rows, study.fixture_score, include_site=True) for rows in samples]))
        cls.original = study.scores_for(data, np.arange(len(data.train_rows)), study.fixture_score, include_site=True)

    def archive(self, name: str, file: str) -> dict[str, np.ndarray]:
        with np.load(self.results / "arrays" / name / file, allow_pickle=False) as arrays:
            return dict(arrays)

    def test_flags_at_each_cutoff_are_the_direct_comparison_of_float64_abs_z(self):
        cutoff = self.archive("synthetic-cutoffs", CUTOFF_FILE)
        summary = self.archive("synthetic-cutoffs", FILE)
        self.assertEqual(list(cutoff), ["cutoffs", "refit_flags_packed", "original_flags_packed", "held_out_cohort_rows_zero_based"])
        self.assertEqual((cutoff["cutoffs"].dtype, cutoff["cutoffs"].tolist()), (np.float64, list(CUTOFFS)))
        self.assertEqual((cutoff["refit_flags_packed"].shape, cutoff["refit_flags_packed"].dtype), ((3, 64, 4, 40), np.uint8))
        self.assertEqual(cutoff["original_flags_packed"].shape, (3, 4, 40))
        plain = self.archive("synthetic-cutoffs", "refit-flags.npz")["refit_flags_packed"]
        for k, threshold in enumerate(CUTOFFS):
            np.testing.assert_array_equal(cutoff["refit_flags_packed"][k], np.packbits(self.abs_z > threshold, axis=-1, bitorder="little"))
            np.testing.assert_array_equal(cutoff["original_flags_packed"][k], np.packbits(np.abs(self.original) > threshold, axis=-1, bitorder="little"))
            flags = np.unpackbits(cutoff["refit_flags_packed"][k], axis=-1, count=320, bitorder="little").astype(bool)
            np.testing.assert_array_equal(flags.all(axis=0), summary["refit_min_abs_z"] > threshold)
            np.testing.assert_array_equal(flags.any(axis=0), summary["refit_max_abs_z"] > threshold)
            self.assertTrue(flags.all(axis=0).any() and (flags.any(axis=0) & ~flags.all(axis=0)).any(), f"cutoff {threshold} would test nothing")
            self.assertFalse((cutoff["refit_flags_packed"][k] & ~plain).any(), "a stricter cutoff flags a subset of the flags at 1.96")

    def test_the_other_outputs_are_those_of_the_run_without_cutoffs(self):
        for file in ("refit-flags.npz", FILE):
            old, new = self.archive("synthetic-z", file), self.archive("synthetic-cutoffs", file)
            self.assertEqual(list(old), list(new))
            for key in old:
                np.testing.assert_array_equal(new[key], old[key])
        old, new = receipt(self.results, "synthetic-z"), receipt(self.results, "synthetic-cutoffs")
        self.assertEqual(list(old["outputs"]), ["refit-flags.npz", FILE])
        self.assertEqual(list(new["outputs"]), ["refit-flags.npz", FILE, CUTOFF_FILE])
        self.assertEqual(without_timing(new["z_summary"]), without_timing(old["z_summary"]))
        self.assertNotIn("cutoff_flags", old["z_summary"])

    def test_the_receipt_names_the_cutoffs_and_what_was_checked(self):
        entry = receipt(self.results, "synthetic-cutoffs")["z_summary"]["cutoff_flags"]
        self.assertEqual((entry["file"], entry["cutoffs"]), (CUTOFF_FILE, list(CUTOFFS)))
        self.assertEqual((entry["check"]["refits"], entry["check"]["entries_per_cutoff"], entry["check"]["entries_that_differ"]), (64, 1280, 0))
        self.assertEqual(set(entry["arrays"]), {"cutoffs", "refit_flags_packed", "original_flags_packed"})

    def test_a_later_fold_and_a_reference_subset_work_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / ".results"
            study.run_per_refit(results, "synthetic-fold-2-c", 32768, use_fixture=True, fold=2, z_summary=True, cutoffs=CUTOFFS)
            study.run_per_refit(results, "synthetic-n24-c", 32768, use_fixture=True, reference_n=24, z_summary=True, cutoffs=(2.5,))
            with np.load(results / "arrays/synthetic-fold-2-c" / CUTOFF_FILE) as fold, np.load(results / "arrays/synthetic-n24-c" / CUTOFF_FILE) as subset:
                self.assertEqual(fold["refit_flags_packed"].shape, (3, 64, 6, 40))
                self.assertEqual(subset["refit_flags_packed"].shape, (1, 64, 4, 40))

    def run_with_one_flag_changed(self, level: int, refit: int, scan: int, voxel: int) -> None:
        """A run whose cutoff code flips one bit of one refit at `level` (0 to 2 the cutoffs, 3 the flags at 1.96)."""
        real_add = ZSummary.add

        def add_and_flip(summary, abs_z):
            real_add(summary, abs_z)
            if summary.flags is not None and summary.refits == refit + 1:
                summary.flags[level, refit, scan, voxel // 8] ^= 1 << (voxel % 8)

        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / ".results"
            with mock.patch.object(ZSummary, "add", add_and_flip):
                try:
                    study.run_per_refit(results, "synthetic-broken", 32768, use_fixture=True, z_summary=True, cutoffs=CUTOFFS)
                finally:
                    self.assertFalse(any(results.rglob("*")) if results.exists() else False, "nothing is written when the check fails")
                    self.assertFalse((Path(tmp) / ".intermediates" / "synthetic-broken").exists())

    def test_a_flag_that_breaks_the_and_stops_the_run_before_anything_is_written(self):
        every = np.argwhere((self.abs_z > CUTOFFS[1]).all(axis=0))[0]  # flagged in all 64 refits at 2.6: clearing one bit changes the AND
        with self.assertRaisesRegex(ValueError, "at cutoff 2.6 the AND of the refits' flags differs from refit_min_abs_z > 2.6 in 1 entries"):
            self.run_with_one_flag_changed(1, 9, int(every[0]), int(every[1]))

    def test_a_flag_that_breaks_the_or_stops_the_run(self):
        never = np.argwhere(~(self.abs_z > CUTOFFS[2]).any(axis=0))[0]  # flagged in no refit at 3.0: setting one bit changes the OR
        with self.assertRaisesRegex(ValueError, "at cutoff 3 the OR of the refits' flags differs from refit_max_abs_z > 3 in 1 entries"):
            self.run_with_one_flag_changed(2, 3, int(never[0]), int(never[1]))

    def test_flags_at_1_96_that_differ_from_the_runs_own_stop_the_run(self):
        flagged = (self.abs_z > THRESHOLD).sum(axis=0)
        some = np.argwhere((flagged > 1) & (flagged < 63))[0]  # neither the AND nor the OR can see one flipped bit here
        with self.assertRaisesRegex(ValueError, "flags at 1.96 that the cutoff code makes differ from refit_flags_packed in 1 entries"):
            self.run_with_one_flag_changed(3, 5, int(some[0]), int(some[1]))

    def test_cutoffs_are_validated(self):
        self.assertEqual(validate_cutoffs((2.3, 3.09)), (2.3, 3.09))
        self.assertEqual(validate_cutoffs((), z_summary=False), ())
        too_many = tuple(2 + i / 10 for i in range(1, MAX_CUTOFFS + 2))
        for bad in ((THRESHOLD,), (1.5,), (2.3, 2.3), (float("nan"),), (float("inf"),), too_many):
            with self.subTest(bad), self.assertRaisesRegex(ValueError, "distinct finite values above 1.96"):
                validate_cutoffs(bad)
        with self.assertRaisesRegex(ValueError, "cutoffs need z_summary"):
            validate_cutoffs((2.5,), z_summary=False)
        with self.assertRaisesRegex(ValueError, "cutoffs need z_summary"):
            study.run_per_refit(Path("unused"), "synthetic-x", 32768, use_fixture=True, cutoffs=(2.5,))

    def test_command_line(self):
        for fold in range(5):
            namespace = accepted("--synthetic", "--per-refit", "--z-summary", "--fold", str(fold), "--refit-flag-cutoffs", "2.3,2.5,3.0,3.1,3.5")
            self.assertEqual((namespace.refit_flag_cutoffs, namespace.fold), ((2.3, 2.5, 3.0, 3.1, 3.5), fold))
        self.assertEqual(accepted("--synthetic", "--per-refit", "--z-summary").refit_flag_cutoffs, ())
        for arguments, message in ((("--refit-flag-cutoffs", "2.5"), "cutoffs need z_summary"),
                                   (("--z-summary", "--refit-flag-cutoffs", "1.5"), "distinct finite values above 1.96"),
                                   (("--z-summary", "--refit-flag-cutoffs", "2.5,2.5"), "distinct finite values above 1.96"),
                                   (("--z-summary", "--refit-flag-cutoffs", "2.5;3"), "expected numbers separated by commas")):
            with self.subTest(arguments):
                self.assertIn(message, parse("--synthetic", "--per-refit", *arguments)[1])


class Contract(unittest.TestCase):
    def test_the_archive_is_written_exactly_when_it_was_asked_for(self):
        flags, counts, rows = np.zeros((1, 1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), np.zeros(1, dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / ".results"
            with self.assertRaisesRegex(ValueError, "z_arrays is missing"):
                Outputs(results, "a", refits_only=True, z_summary=True).publish_refits(flags, counts, rows, {})
            with self.assertRaisesRegex(ValueError, "z_arrays is given"):
                Outputs(results, "b", refits_only=True).publish_refits(flags, counts, rows, {}, {"original_z": counts})
            with self.assertRaisesRegex(ValueError, "cutoff_arrays is missing"):
                Outputs(results, "c", refits_only=True, z_summary=True, cutoff_flags=True).publish_refits(flags, counts, rows, {}, {"original_z": counts})
            with self.assertRaisesRegex(ValueError, "cutoff_arrays is given"):
                Outputs(results, "d", refits_only=True, z_summary=True).publish_refits(flags, counts, rows, {}, {"original_z": counts}, {"cutoffs": counts})
            self.assertFalse(results.exists())
        self.assertEqual(set(Outputs(Path("r"), "n", refits_only=True, z_summary=True).destinations), {"refit-flags.npz", FILE})
        self.assertEqual(set(Outputs(Path("r"), "n", refits_only=True).destinations), {"refit-flags.npz"})

    def test_command_line(self):
        self.assertTrue(accepted("--synthetic", "--per-refit", "--z-summary").z_summary)
        self.assertFalse(accepted("--synthetic", "--per-refit").z_summary)
        for fold in range(5):
            namespace = accepted("--synthetic", "--per-refit", "--z-summary", "--fold", str(fold))
            self.assertEqual((namespace.z_summary, namespace.fold), (True, fold))
        self.assertIn("--z-summary needs --per-refit", parse("--synthetic", "--z-summary")[1])
        self.assertIn("--reference-n needs --per-refit, fold 0", parse("--synthetic", "--per-refit", "--z-summary", "--fold", "2", "--reference-n", "20")[1])


if __name__ == "__main__":
    unittest.main()
