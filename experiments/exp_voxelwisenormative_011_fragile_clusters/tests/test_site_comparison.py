"""Two sites' per-refit exports of one fold: what the comparison reports, and each reason it refuses to compare."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from src.modules import site_comparison as sc
from src.modules.clusters import ClusterGrid

from shared.voxelwisenormative.artifacts import sha256

REFITS, SCANS, SHAPE = 64, 8, (5, 6, 7)
VOXELS = SHAPE[0] * SHAPE[1] * SHAPE[2]
BASE = {"mode": "ixi-cuda-refits", "held_out_scans": SCANS, "voxel_columns": VOXELS, "references": 450, "bootstrap_refits": REFITS,
        "bootstrap_seed": 1, "bootstrap_cohort_rows_sha256": "d", "optimizer_iterations": 150, "flag_rule": "abs(z) > 1.96",
        "mask_column_order": "C-order true voxels", "packed_axis": -1, "packed_bitorder": "little",
        "runtime": {"device": "NVIDIA A100-SXM4-80GB"}}
NO_PLANES = np.zeros(VOXELS, dtype=bool)


def random_flags(seed: int = 0) -> np.ndarray:
    """(refits, scans, voxels) booleans; each scan has its own flag rate so the ranking has something to rank."""
    rate = np.linspace(0.1, 0.5, SCANS)[None, :, None]
    return np.random.default_rng(seed).random((REFITS, SCANS, VOXELS)) < rate


def make_site(label: str, flags: np.ndarray, receipt: dict | None = None, rows: np.ndarray | None = None) -> sc.Export:
    """One site's loaded run: BASE plus the receipt fields that differ, the flags packed as the exports are."""
    return sc.Export(label, BASE | (receipt or {}), np.packbits(flags, axis=-1, bitorder="little"), flags.sum(axis=0, dtype=np.uint8),
                     np.arange(SCANS, dtype=np.int32) * 3 if rows is None else rows)


def flipped(flags: np.ndarray, count: int, seed: int = 1) -> np.ndarray:
    """The flags with exactly `count` entries toggled."""
    out = flags.copy().reshape(-1)
    out[np.random.default_rng(seed).choice(out.size, count, replace=False)] ^= True
    return out.reshape(flags.shape)


GRID = ClusterGrid(np.ones(SHAPE, dtype=bool), (1, 3))


class Report(unittest.TestCase):
    def test_identical_exports_agree_everywhere(self):
        flags = random_flags()
        report = sc.compare(make_site("a", flags), make_site("b", flags), GRID)
        self.assertEqual(report["flags"]["flags_that_differ"], 0)
        self.assertEqual(report["recurrence"]["entries_that_differ"], 0)
        self.assertEqual(report["recurrence"]["stable_class_changed"], 0)
        self.assertEqual(report["ranking"]["largest_cluster_spearman"], 1.0)
        self.assertEqual(report["ranking"]["scans_with_a_different_rank"], 0)
        self.assertEqual(report["ranking"]["top_k_shared"], {"1": 1, "5": 5})
        self.assertEqual(report["not_compared"], [])

    def test_known_number_of_changed_flags(self):
        flags = random_flags()
        other = flipped(flags, 25)
        report = sc.compare(make_site("a", flags), make_site("b", other), GRID)
        self.assertEqual(report["flags"]["flags_that_differ"], 25)
        self.assertEqual(report["flags"]["flag_entries"], flags.size)
        total = report["flags"]["flags_all_refits"]
        self.assertEqual((total["a"], total["b"]), (int(flags.sum()), int(other.sum())))
        gaps = report["recurrence"]
        self.assertTrue(0 < gaps["entries_that_differ"] <= 25 and gaps["max_gap"] <= 25)
        self.assertEqual(gaps["gap_1"] + gaps["gap_2"] + gaps["gap_3_or_more"], gaps["entries_that_differ"])

    def test_text_names_both_sites_and_what_differs(self):
        flags = random_flags()
        text = "\n".join(sc.render(sc.compare(make_site("a", flags), make_site("b", flipped(flags, 5)), GRID), "a", "b"))
        self.assertIn("individual flags that differ (context): 5 of", text)
        self.assertIn("fragile-cluster ranking", text)

    def test_json_serialisable(self):
        flags = random_flags()
        json.dumps(sc.compare(make_site("a", flags), make_site("b", flipped(flags, 5)), GRID), allow_nan=False)


class Persons(unittest.TestCase):
    def test_kept_share_and_classes_against_a_direct_count(self):
        flags = random_flags()
        a, b = make_site("a", flags), make_site("b", flipped(flags, 10))
        single_z = np.where(flags[0] | flags[1], 2.5, 0.5)  # the one fit flags what refits 1 or 2 flag
        on_planes = np.zeros(VOXELS, dtype=bool)
        on_planes[::2] = True
        table = sc.people(a, b, [0, 5], single_z, on_planes)
        self.assertEqual([p["held_out_row"] for p in table], [0, 5])
        for person in table:
            row = person["held_out_row"]
            single = single_z[row] > 1.96
            counts = a.counts[row]
            mine = person["a"]
            self.assertEqual(mine["single_fit_flags"], int(single.sum()))
            self.assertEqual(mine["single_fit_kept_in_all_refits"], int((single & (counts == REFITS)).sum()))
            self.assertEqual(mine["single_fit_stable"] + mine["single_fit_fragile"], mine["single_fit_flags"])
            self.assertEqual(mine["stable_on_planes"], int((single & (counts >= 52) & on_planes).sum()))
            self.assertEqual(mine["refit_1_flags"], int(flags[0, row].sum()))
            self.assertAlmostEqual(mine["kept_share"], mine["single_fit_kept_in_all_refits"] / mine["single_fit_flags"])
        lines = sc.render_people(table, "a", "b")
        self.assertTrue(lines[0].startswith("held-out row 0 (cohort row 0)"))

    def test_a_number_that_differs_is_marked(self):
        flags = random_flags()
        table = sc.people(make_site("a", flags), make_site("b", flags[::-1]), [7], None, NO_PLANES)
        a, b = table[0]["a"], table[0]["b"]
        lines = sc.render_people(table, "a", "b")
        marked = [line for line in lines if line.endswith("!")]
        self.assertEqual(len(marked), sum(a[k] != b[k] for k in a if k != "cohort_row"))
        self.assertEqual(len(marked), 1)  # reversing the refits keeps the counts and only changes the first refit's flags
        self.assertIn("flags in refit 1", marked[0])

    def test_without_the_single_fit_only_the_refit_numbers(self):
        flags = random_flags()
        table = sc.people(make_site("a", flags), make_site("b", flags), [3], None, NO_PLANES)
        self.assertNotIn("single_fit_flags", table[0]["a"])
        self.assertEqual(table[0]["a"]["voxels_in_all_refits"], int((flags[:, 3].sum(axis=0) == REFITS).sum()))

    def test_refusals(self):
        flags = random_flags()
        a, b = make_site("a", flags), make_site("b", flags)
        fold2_a, fold2_b = make_site("a", flags, {"fold": 2}), make_site("b", flags, {"fold": 2})
        with self.assertRaisesRegex(ValueError, "outside 0..7"):
            sc.people(a, b, [8], None, NO_PLANES)
        with self.assertRaisesRegex(ValueError, "fold 0"):
            sc.people(fold2_a, fold2_b, [0], np.zeros((SCANS, VOXELS)), NO_PLANES)
        with self.assertRaisesRegex(ValueError, "must be"):
            sc.people(a, b, [0], np.zeros((SCANS, VOXELS + 1)), NO_PLANES)


class SameExperiment(unittest.TestCase):
    def setUp(self):
        self.flags = random_flags()

    def test_different_fold_or_draws_or_reference_or_rows(self):
        a = make_site("a", self.flags)
        for change in ({"fold": 1}, {"bootstrap_seed": 2}, {"bootstrap_cohort_rows_sha256": "e"}, {"reference_n": 100}, {"references": 449}):
            with self.subTest(change), self.assertRaisesRegex(ValueError, "not the same experiment"):
                sc.require_same_experiment(a, make_site("b", self.flags, change))
        with self.assertRaisesRegex(ValueError, "cohort rows differ"):
            sc.require_same_experiment(a, make_site("b", self.flags, rows=np.arange(SCANS, dtype=np.int32)))

    def test_a_receipt_written_before_fold_and_reference_n_existed(self):
        """Slurm's per-refit receipt of job 81589484 has neither key: that is fold 0 without a smaller reference."""
        a = make_site("a", self.flags)
        self.assertEqual(sc.require_same_experiment(a, make_site("b", self.flags, {"fold": 0, "reference_n": None})), [])
        with self.assertRaisesRegex(ValueError, "fold is 0 at a and 1 at b"):
            sc.require_same_experiment(a, make_site("b", self.flags, {"fold": 1}))

    def test_a_field_only_one_receipt_records_is_reported_not_compared(self):
        notes = sc.require_same_experiment(make_site("a", self.flags), make_site("b", self.flags, {"reference_cohort_rows_sha256": "x"}))
        self.assertEqual(notes, ["reference_cohort_rows_sha256 is recorded by b only, so it is not compared"])


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.results = Path(self.tmp.name)
        self.flags = random_flags()

    def write(self, result_dir: Path, array_dir: Path | None = None) -> Path:
        """result.json in `result_dir` and refit-flags.npz in `array_dir` (default: beside it); returns the archive."""
        array_dir = array_dir or result_dir
        result_dir.mkdir(parents=True)
        array_dir.mkdir(parents=True, exist_ok=True)
        site = make_site("x", self.flags)
        archive = array_dir / "refit-flags.npz"
        np.savez_compressed(archive, refit_flags_packed=site.packed, recurrence_counts=site.counts, held_out_cohort_rows_zero_based=site.rows)
        record = {"sha256": sha256(archive), "bytes": archive.stat().st_size}
        (result_dir / "result.json").write_text(json.dumps(BASE | {"outputs": {"refit-flags.npz": record}}))
        return archive

    def test_a_name_is_looked_up_as_a_folder_a_fetch_folder_or_the_archive(self):
        flat = self.results / "fetch-81589484"
        self.write(flat)
        archived = self.results / "hugging-face" / "run-a"
        self.write(archived / "json" / "run-a", archived / "arrays" / "run-a")
        expected = (flat / "result.json", flat / "refit-flags.npz")
        self.assertEqual(sc.locate(self.results, "81589484"), expected)
        self.assertEqual(sc.locate(self.results, str(flat)), expected)
        self.assertEqual(sc.locate(self.results, "run-a"), (archived / "json" / "run-a" / "result.json", archived / "arrays" / "run-a" / "refit-flags.npz"))
        with self.assertRaises(FileNotFoundError):
            sc.locate(self.results, "nothing-here")

    def test_load_checks_the_receipt_and_the_counts(self):
        archive = self.write(self.results / "fetch-ok")
        loaded = sc.load("ok", "ok", self.results)
        self.assertEqual(loaded.packed.shape, (REFITS, SCANS, (VOXELS + 7) // 8))
        archive.write_bytes(archive.read_bytes() + b"\0")
        with self.assertRaisesRegex(ValueError, "is not the export"):
            sc.load("ok", "ok", self.results)

    def test_counts_that_are_not_the_sum_of_the_refits(self):
        archive = self.write(self.results / "fetch-bad")
        with np.load(archive) as arrays:
            parts = dict(arrays)
        parts["recurrence_counts"] = parts["recurrence_counts"].copy()
        parts["recurrence_counts"][0, 0] ^= 1
        np.savez_compressed(archive, **parts)
        result = self.results / "fetch-bad" / "result.json"
        receipt = json.loads(result.read_text())
        receipt["outputs"]["refit-flags.npz"] = {"sha256": sha256(archive), "bytes": archive.stat().st_size}
        result.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "recurrence_counts differ from the sum"):
            sc.load("bad", "bad", self.results)


if __name__ == "__main__":
    unittest.main()
