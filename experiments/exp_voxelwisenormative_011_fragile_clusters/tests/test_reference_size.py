"""--reference-n: the age-stratified subset, the unchanged 450-person draws, and synthetic runs end to end."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from src.modules import reference_size
from src.modules.inputs import DRAW_DIGEST, draws, synthetic
from src.modules.reference_size import age_stratified, rank_fetched
from src.modules.study import run, run_per_refit

from shared.voxelwisenormative.artifacts import PCNTOOLKIT_REFERENCE, resolve
from shared.voxelwisenormative.basis import SITES
from shared.voxelwisenormative.cohort import load_covariates


def frozen_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cohort rows, ages and sites of the 450 frozen references; no grey matter is read."""
    with np.load(resolve(PCNTOOLKIT_REFERENCE, "arrays.npz")["arrays.npz"], allow_pickle=False) as archive:
        rows = np.flatnonzero(~archive["test"])
    cov = load_covariates()
    return rows, cov["age"][rows], cov["site"][rows]


def load_run(results: Path, name: str) -> tuple[dict[str, np.ndarray], dict]:
    """Every array and the receipt of one finished per-refit run."""
    with np.load(results / "arrays" / name / "refit-flags.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    return arrays, json.loads((results / "json" / name / "result.json").read_text())


@unittest.skipUnless(os.getenv('VOXELWISE_TEST_IXI') == '1', 'requires locally prepared IXI artifacts')
class FrozenReferenceTest(unittest.TestCase):
    def test_450_keeps_every_reference_in_order_and_the_original_draws(self):
        rows, ages, _ = frozen_reference()
        keep = age_stratified(ages, 450)
        np.testing.assert_array_equal(keep, np.arange(450))
        _, digest = draws(replace(synthetic(), train_rows=rows[keep]), verify_original=True)
        self.assertEqual(digest, DRAW_DIGEST)

    def test_smaller_subsets_keep_age_range_sites_and_one_per_stratum(self):
        _, ages, sites = frozen_reference()
        order = np.argsort(ages, kind="stable")
        rank = np.empty(450, dtype=int)
        rank[order] = np.arange(450)
        for n in (50, 100, 200):
            keep = age_stratified(ages, n)
            np.testing.assert_array_equal(keep, age_stratified(ages, n))
            self.assertEqual(len(np.unique(keep)), n)
            self.assertTrue(np.all(np.diff(keep) > 0))
            self.assertEqual((ages[keep].min(), ages[keep].max()), (ages.min(), ages.max()))
            self.assertEqual(set(sites[keep]), set(SITES))
            edges = np.arange(n - 1) * 448 // (n - 2)
            inner = np.sort(rank[keep])[1:-1] - 1
            np.testing.assert_array_equal(np.searchsorted(edges, inner, side="right") - 1, np.arange(n - 2))
            self.assertLess(abs(np.median(ages[keep]) - np.median(ages)), 2.0)
        with self.assertRaises(ValueError):
            age_stratified(ages, 451)


class SyntheticRunTest(unittest.TestCase):
    def test_full_size_equals_the_per_refit_run_and_the_full_site_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / ".results"
            run(results, "synthetic-main", 32768, use_fixture=True)
            run_per_refit(results, "synthetic-refits", 32768, use_fixture=True)
            run_per_refit(results, "synthetic-n24", 32768, use_fixture=True, reference_n=24)
            with np.load(results / "arrays/synthetic-main/flags.npz") as main:
                full_site = main["site_flags_packed"][0]
            before, old = load_run(results, "synthetic-refits")
            after, new = load_run(results, "synthetic-n24")
        for key in before:
            np.testing.assert_array_equal(after[key], before[key])
        np.testing.assert_array_equal(after["original_flags_packed"], full_site)
        np.testing.assert_array_equal(after["reference_cohort_rows_zero_based"], np.arange(24))
        self.assertEqual(new["bootstrap_cohort_rows_sha256"], old["bootstrap_cohort_rows_sha256"])
        self.assertEqual((new["reference_n"], new["references"]), (24, 24))
        self.assertNotIn("reference_n", old)

    def test_small_reference_run_and_its_ranking(self):
        data = synthetic()
        on_planes = np.isin(np.argwhere(data.mask)[:, 2], data.axial_indices)
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / ".results"
            run_per_refit(results, "synthetic-n12", 32768, use_fixture=True, reference_n=12)
            arrays, receipt = load_run(results, "synthetic-n12")
            fetched = Path(tmp) / "fetch"
            fetched.mkdir()
            shutil.copy(results / "arrays/synthetic-n12/refit-flags.npz", fetched)
            shutil.copy(results / "json/synthetic-n12/result.json", fetched)
            lines = rank_fetched(fetched, on_planes).read_text().splitlines()
        keep = age_stratified(data.reference_ages, 12)
        np.testing.assert_array_equal(arrays["reference_cohort_rows_zero_based"], data.train_rows[keep])
        self.assertEqual(arrays["refit_flags_packed"].shape, (64, 4, 40))
        self.assertEqual(arrays["original_flags_packed"].shape, (4, 40))
        bits = np.unpackbits(arrays["refit_flags_packed"], axis=-1, count=320, bitorder="little")
        np.testing.assert_array_equal(bits.sum(axis=0), arrays["recurrence_counts"])
        self.assertEqual((receipt["references"], receipt["reference_n"], receipt["reference_pool"]), (12, 12, 24))
        self.assertEqual(set(receipt["reference_site_counts"]), set(SITES))
        original = np.unpackbits(arrays["original_flags_packed"], axis=-1, count=320, bitorder="little").astype(bool)
        removed = (original & (arrays["recurrence_counts"] < 52)).sum(axis=1)
        header, *rows = [line.split("\t") for line in lines]
        self.assertEqual(header[:2], ["rank", "held_out_row_zero_based"])
        got = [(int(r[1]), int(r[header.index("removed_voxels")])) for r in rows]
        self.assertEqual(got, sorted(enumerate(removed.tolist()), key=lambda p: (-p[1], p[0])))


class FullSizeRankingTest(unittest.TestCase):
    """--original-z: a full-size run has no original flags, so the ranking takes them from the z of its one fit."""

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.base = Path(tmp.name)
        cls.data = synthetic()
        cls.on_planes = np.isin(np.argwhere(cls.data.mask)[:, 2], cls.data.axial_indices)
        results = cls.base / ".results"
        # at 24 references, the whole synthetic pool, the subset run is the full-size run plus its own original flags
        run_per_refit(results, "synthetic-n24", 32768, use_fixture=True, reference_n=24)
        run_per_refit(results, "synthetic-plain", 32768, use_fixture=True)
        run_per_refit(results, "synthetic-z", 32768, use_fixture=True, z_summary=True)
        run_per_refit(results, "synthetic-fold-2-z", 32768, use_fixture=True, fold=2, z_summary=True)
        cls.folders = {}
        for name in ("synthetic-n24", "synthetic-plain", "synthetic-z", "synthetic-fold-2-z"):
            folder = cls.base / f"fetch-{name}"
            folder.mkdir()
            shutil.copy(results / "arrays" / name / "refit-flags.npz", folder)
            shutil.copy(results / "json" / name / "result.json", folder)
            cls.folders[name] = folder
        shutil.copy(results / "arrays/synthetic-z/z-summary.npz", cls.folders["synthetic-z"])
        shutil.copy(results / "arrays/synthetic-fold-2-z/z-summary.npz", cls.folders["synthetic-fold-2-z"])
        cls.z_npy = cls.base / "z_test.npy"
        with np.load(results / "arrays/synthetic-z/z-summary.npz") as z:
            np.save(cls.z_npy, z["original_z"].astype(np.float32))

    def rank(self, name: str, original_z: Path | None = None) -> str:
        return rank_fetched(self.folders[name], self.on_planes, original_z).read_text()

    def test_z_test_and_z_summary_give_the_ranking_of_the_run_with_its_own_original_flags(self):
        own = self.rank("synthetic-n24")
        with mock.patch.object(reference_size, "N_TRAIN", 24):
            self.assertEqual(self.rank("synthetic-plain", self.z_npy), own)
        self.assertEqual(self.rank("synthetic-z", self.folders["synthetic-z"] / "z-summary.npz"), own)

    def test_a_z_summary_fits_another_fold_but_z_test_does_not(self):
        folder = self.folders["synthetic-fold-2-z"]
        self.assertIn("rank\theld_out_row_zero_based", self.rank("synthetic-fold-2-z", folder / "z-summary.npz"))
        with mock.patch.object(reference_size, "N_TRAIN", 22), self.assertRaisesRegex(ValueError, "is the fit of fold 0"):
            rank_fetched(folder, self.on_planes, self.z_npy)

    def test_exactly_one_source_of_the_one_fit(self):
        with self.assertRaisesRegex(ValueError, "has its own original flags"):
            rank_fetched(self.folders["synthetic-n24"], self.on_planes, self.z_npy)
        with self.assertRaisesRegex(ValueError, "give --original-z"):
            rank_fetched(self.folders["synthetic-plain"], self.on_planes)

    def test_scans_that_are_not_the_archives_are_refused(self):
        folder = self.folders["synthetic-z"]
        with np.load(folder / "z-summary.npz") as z:
            parts = dict(z)
        parts["held_out_cohort_rows_zero_based"] = parts["held_out_cohort_rows_zero_based"][::-1].copy()
        other = self.base / "other-scans.npz"
        np.savez(other, **parts)
        with self.assertRaisesRegex(ValueError, "holds other scans"):
            rank_fetched(folder, self.on_planes, other)
        shuffled = self.base / "shuffled-z_test.npy"
        np.save(shuffled, parts["original_z"][::-1].copy())  # z_test.npy has no row labels: the flag counts must still follow the refits'
        with mock.patch.object(reference_size, "N_TRAIN", 24), self.assertRaisesRegex(ValueError, "are the rows the same scans"):
            rank_fetched(self.folders["synthetic-plain"], self.on_planes, shuffled)

    def test_wrong_shape_and_a_changed_archive_are_refused(self):
        short = self.base / "short.npy"
        np.save(short, np.zeros((3, 320), dtype=np.float32))
        with mock.patch.object(reference_size, "N_TRAIN", 24), self.assertRaisesRegex(ValueError, "has shape"):
            rank_fetched(self.folders["synthetic-plain"], self.on_planes, short)
        copy = self.base / "fetch-changed"
        shutil.copytree(self.folders["synthetic-z"], copy)
        (copy / "refit-flags.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "is not the output its receipt describes"):
            rank_fetched(copy, self.on_planes, copy / "z-summary.npz")


if __name__ == "__main__":
    unittest.main()
