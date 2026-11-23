"""Hand-computed clusters, exact rule boundaries and an independent flood-fill check of the fragile-cluster share."""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import naive_stability as naive
import numpy as np
from src.modules import share_inputs
from src.modules.cluster_share import (
    band_of,
    count_summary,
    curve,
    readings,
    shares,
    size_bins,
    summarise,
)
from src.modules.clusters import ClusterGrid
from src.modules.inputs import N_FOLDS, N_REFITS, N_SCANS
from src.modules.share_inputs import (
    CUTOFF_RUN,
    RUN,
    check_cohort,
    open_fold,
    read_cutoff_fold,
    read_fold,
)
from src.modules.share_run import definitions
from src.modules.stability import best_overlap, cropped, grids, person_columns

from shared.voxelwisenormative.artifacts import sha256

SHAPE = (6, 6, 5)


def flags(*points: tuple[int, int, int]) -> np.ndarray:
    """Voxels as mask columns of the all-true 6 x 6 x 5 mask (C order)."""
    volume = np.zeros(SHAPE, dtype=bool)
    for point in points:
        volume[point] = True
    return volume.ravel()


class ConnectivityTest(unittest.TestCase):
    def test_neighbour_counts_join_what_they_should(self):
        mask = np.ones(SHAPE, dtype=bool)
        face, edge, corner = flags((1, 1, 1), (1, 1, 2)), flags((1, 1, 1), (1, 2, 2)), flags((1, 1, 1), (2, 2, 2))
        found = {n: [ClusterGrid(mask, (), n).components(row)[1] for row in (face, edge, corner)] for n in (6, 18, 26)}
        self.assertEqual(found, {6: [1, 2, 2], 18: [1, 1, 2], 26: [1, 1, 1]})
        with self.assertRaisesRegex(ValueError, "connectivity must be one of"):
            ClusterGrid(mask, (), 8)

    def test_cropping_changes_neither_labels_nor_numbering(self):
        rng = np.random.default_rng(3)
        mask = np.zeros((9, 10, 8), dtype=bool)
        mask[2:7, 3:9, 1:6] = rng.random((5, 6, 5)) > 0.2
        row = rng.random(int(mask.sum())) > 0.5
        for n in (6, 26):
            full, small = ClusterGrid(mask, (), n).components(row), ClusterGrid(cropped(mask), (), n).components(row)
            np.testing.assert_array_equal(full[0], small[0])
            self.assertEqual(full[1], small[1])


class PersonTest(unittest.TestCase):
    def setUp(self):
        self.grid = ClusterGrid(np.ones(SHAPE, dtype=bool), (), 26)
        a, b, c = [(1, 1, 1), (1, 1, 2), (1, 1, 3)], [(1, 2, 1), (1, 2, 2)], [(4, 4, 4)]
        self.z = np.zeros(180)
        self.z[flags(*a)], self.z[flags(*b)], self.z[flags(*c)] = 3.0, -3.0, 2.5
        self.refits = np.stack([flags(*a, *b, *c), flags((1, 1, 1), (1, 1, 2)), flags(), flags(*a, (1, 1, 4), *c)])

    def test_hand_counted_clusters(self):
        """A (3 voxels) and C (1) are positive, B (2, a face neighbour of A) negative; refit 0 joins A and B as one component."""
        got = person_columns(self.grid, self.z, self.refits)
        self.assertEqual(got["sign"].tolist(), [1, 1, -1])
        self.assertEqual(got["voxels"].tolist(), [3, 1, 2])
        self.assertEqual(got["core_voxels"].tolist(), [0, 0, 0])
        self.assertEqual(got["recurrence_sum"].tolist(), [8, 2, 2])
        self.assertEqual(got["refit_voxels"].tolist(), [[3, 2, 0, 3], [1, 0, 0, 1], [2, 0, 0, 0]])
        np.testing.assert_allclose(got["refit_jaccard"], [[0.6, 2 / 3, 0, 0.75], [1, 0, 0, 1], [0.4, 0, 0, 0]], rtol=1e-6)
        np.testing.assert_allclose(got["jaccard_mean"], [(0.6 + 2 / 3 + 0.75) / 4, 0.5, 0.1])

    def test_all_refits_keep_a_voxel_as_core(self):
        always = np.stack([flags((4, 4, 4), (1, 1, 1))] * 4)
        self.assertEqual(person_columns(self.grid, self.z, always)["core_voxels"].tolist(), [1, 1, 0])

    def test_higher_cutoff_leaves_no_cluster_and_empty_columns(self):
        got = person_columns(self.grid, self.z, self.refits, threshold=3.1)
        self.assertEqual((got["voxels"].shape, got["refit_voxels"].shape, got["jaccard_mean"].shape), ((0,), (0, 4), (0,)))

    def test_overlap_takes_the_best_component_and_zero_without_one(self):
        clusters = np.array([1, 1, 1, 1, 0, 2, 0, 0])  # cluster 1 has 4 voxels, cluster 2 one
        refit = np.array([1, 1, 0, 0, 0, 0, 0, 2])  # components of 2 voxels (in cluster 1) and 1 voxel outside both
        np.testing.assert_allclose(best_overlap(clusters, 2, refit, 2), [2 / 4, 0])

    def test_random_volumes_agree_with_plain_flood_fill(self):
        for seed in range(4):
            rng = np.random.default_rng(seed)
            mask = rng.random((7, 8, 6)) > 0.15
            n, base = int(mask.sum()), rng.normal(size=int(mask.sum())) * 1.4
            refits = np.stack([(np.abs(base) > 1.96) & (rng.random(n) > 0.25) | (rng.random(n) > 0.96) for _ in range(6)])
            for neighbours, grid in grids(mask).items():
                fast = person_columns(grid, base, refits)
                pos, n_pos = grid.components(base > 1.96)
                neg, _ = grid.components(base < -1.96)
                labels = np.where(neg > 0, neg + n_pos, pos)
                first = dict(zip(*np.unique(labels, return_index=True), strict=True))
                plain = naive.person(mask, base, refits, refits.sum(axis=0), neighbours)
                self.assertEqual(len(plain), len(fast["voxels"]))
                self.assertGreater(len(plain), 5, f"seed {seed}, {neighbours} neighbours: too few clusters to compare")
                for i, sign in enumerate(fast["sign"]):
                    want = plain[(int(sign), int(first[i + 1]))]
                    self.assertEqual((fast["voxels"][i], fast["core_voxels"][i]), (want["voxels"], want["core_voxels"]))
                    self.assertEqual(fast["refit_voxels"][i].tolist(), want["refit_voxels"])
                    np.testing.assert_allclose(fast["refit_jaccard"][i], want["refit_jaccard"], atol=1e-7)
                    self.assertAlmostEqual(fast["jaccard_mean"][i], want["jaccard_mean"], places=12)


def columns(sizes, **over):
    """Per-cluster columns for 64 refits; every cluster fully kept unless `over` says otherwise."""
    size = np.asarray(sizes, dtype=np.int32)
    base = {"voxels": size, "core_voxels": size.copy(), "recurrence_sum": size * N_REFITS,
            "refit_voxels": np.repeat(size[:, None], N_REFITS, axis=1).astype(np.uint16),
            "refit_jaccard": np.ones((len(size), N_REFITS), dtype=np.float32), "jaccard_mean": np.ones(len(size))}
    return {**base, **over}


class RuleTest(unittest.TestCase):
    def test_bands_of_the_mean(self):
        got = band_of(np.array([0.0, 0.59, 0.6, 0.74, 0.75, 0.84, 0.85, 1.0]))
        self.assertEqual(got.tolist(), [0, 0, 1, 1, 2, 2, 3, 3])

    def test_each_rule_at_its_boundary(self):
        refit_voxels = np.full((9, N_REFITS), 4, dtype=np.uint16)
        refit_voxels[1, 5] = 0  # vanishes once
        refit_voxels[2, :3] = 1  # fewer than half in 3 refits
        refit_voxels[3, :4] = 1  # ... in 4
        refit_voxels[4, :10] = 2  # exactly half is not fewer than half
        mean = np.array([1, 1, 1, 1, 1, 0.6, 0.6 - 1e-6, 0.75, 0.75 - 1e-6])
        c = columns([4] * 9, refit_voxels=refit_voxels, jaccard_mean=mean, core_voxels=np.array([4, 4, 4, 4, 4, 4, 4, 0, 4], dtype=np.int32))
        got = {rule: np.flatnonzero(bad).tolist() for rule, bad in readings(c).items()}
        self.assertEqual(got["no_core"], [7])
        self.assertEqual(got["vanishes"], [1])
        self.assertEqual(got["loses_half"], [3])
        self.assertEqual(got["jaccard_not_trusted"], [6])
        self.assertEqual(got["jaccard_not_stable"], [5, 6, 8])
        self.assertEqual(got["low_reselection"], [])

    def test_reselection_rate_two_thirds_is_not_above_two_thirds(self):
        sums = np.array([128, 129, 191, 192, 0], dtype=np.int32)  # 3 voxels: rate 128/192 = 2/3 exactly, then just above
        got = readings(columns([3] * 5, recurrence_sum=sums))["low_reselection"]
        self.assertEqual(got.tolist(), [True, False, False, False, True])


class ShareTest(unittest.TestCase):
    def test_pooled_per_person_and_volume_shares(self):
        got = shares(np.array([1, 1, 5, 10, 3]), np.array([0, 0, 0, 7, 9]), np.array([True, False, True, False, False]))
        self.assertEqual((got["clusters"], got["fragile"], got["people_with_clusters"]), (5, 2, 3))
        self.assertAlmostEqual(got["pooled_share"], 0.4)
        self.assertAlmostEqual(got["pooled_x_of_5"], 2.0)
        self.assertEqual(got["per_person_share"], {"q1": 0.0, "median": 0.0, "q3": 1 / 3})
        self.assertEqual((got["flagged_voxels"], got["fragile_voxels"]), (20, 6))
        self.assertAlmostEqual(got["fragile_volume_share"], 0.3)
        self.assertEqual(got["per_person_fragile_volume_share"]["median"], 0.0)

    def test_size_bins_double_and_curve_counts_every_cluster(self):
        low, high = size_bins(40)
        self.assertEqual((low.tolist(), high.tolist()), ([1, 2, 3, 5, 9, 17, 33], [1, 2, 4, 8, 16, 32, 64]))
        got = curve(np.array([1, 2, 3, 4, 5, 33, 64]), np.array([True, False, True, True, False, False, True]), high)
        self.assertEqual(got["clusters"], [1, 1, 2, 1, 0, 0, 2])
        self.assertEqual(got["fragile"], [1, 0, 2, 0, 0, 0, 1])
        self.assertEqual(got["share"], [1.0, 0.0, 1.0, 0.0, None, None, 0.5])

    def test_distribution_of_a_per_refit_count(self):
        got = count_summary(np.array([0, 0, 2, 64]))
        self.assertEqual((got or {})["histogram"][:3] + (got or {})["histogram"][-1:], [2, 0, 1, 1])
        self.assertEqual((got or {})["mean"], 16.5)
        self.assertIsNone(count_summary(np.array([], dtype=int)))

    def two_folds(self):
        """Ten clusters in two folds of two people each; every fold has clusters of every size class."""
        sizes = [1, 3, 12, 60, 150, 2, 7, 120, 1, 55]
        mean = np.array([1, 0.5, 1, 0.9, 0.2, 1, 0.7, 1.0, 0.55, 1.0])
        core = np.array([1, 0, 12, 60, 150, 2, 0, 120, 1, 55], dtype=np.int32)
        c = columns(sizes, jaccard_mean=mean, core_voxels=core)
        c["fold"] = np.array([0] * 5 + [1] * 5, dtype=np.int8)
        c["cohort_row"] = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3], dtype=np.int32)
        return c

    def test_scopes_add_up_and_folds_are_compared(self):
        got = summarise(self.two_folds(), [0, 1])
        pooled = {k: got["scopes"][k]["readings"]["no_core"]["size_classes"]["all"] for k in ("fold0", "fold1", "all")}
        self.assertEqual([pooled[k]["clusters"] for k in ("fold0", "fold1", "all")], [5, 5, 10])
        self.assertEqual([pooled[k]["fragile"] for k in ("fold0", "fold1", "all")], [1, 1, 2])
        self.assertEqual(got["scopes"]["all"]["people"], 4)
        self.assertEqual(got["scopes"]["all"]["readings"]["no_core"]["size_classes"]["ge10"]["clusters"], 5)
        self.assertEqual(got["fold_spread"]["readings"]["no_core"]["all"]["range"], 0.0)
        self.assertEqual(got["scopes"]["all"]["jaccard_bands"]["all"], {
            "clusters": 10, "not_trusted": 3, "doubtful": 1, "stable": 0, "highly_stable": 6,
            "share": {"not_trusted": 0.3, "doubtful": 0.1, "stable": 0.0, "highly_stable": 0.6}})
        self.assertEqual(sum(got["scopes"]["all"]["jaccard_counts"]["all"]["recoveries"]["histogram"]), 10)
        self.assertEqual(sum(got["size_curve"]["scopes"]["all"]["no_core"]["clusters"]), 10)

    def test_a_fold_without_a_size_class_is_an_error_not_a_gap(self):
        c = self.two_folds()
        c["voxels"][c["fold"] == 1] = np.minimum(c["voxels"][c["fold"] == 1], 99)
        with self.assertRaisesRegex(ValueError, "a fold has no cluster of class ge100"):
            summarise(c, [0, 1])


VOXELS, PEOPLE, CUTOFFS = 100, 3, (2.3, 3.1)


def packed(values: np.ndarray) -> np.ndarray:
    return np.packbits(values, axis=-1, bitorder="little")


def write_run(root: Path, cutoffs: bool = True, tamper: Callable[[dict], None] = lambda arrays: None) -> dict:
    """A tiny fetched run (100 voxels, 3 people) with its receipt, consistent unless `tamper` edits the arrays before they are saved.

    Voxel 0 is flagged in every refit at every cutoff and voxel 2 in none, so the AND and the OR can each be broken on purpose.
    """
    rng = np.random.default_rng(0)
    name = (CUTOFF_RUN if cutoffs else RUN).format(fold=0)
    base = root / "hugging-face" / name
    for folder in ("json", "arrays"):
        (base / folder / name).mkdir(parents=True)
    refit_abs = np.abs(rng.normal(size=(N_REFITS, PEOPLE, VOXELS))) * 1.4
    refit_abs[:, :, 0], refit_abs[:, :, 2] = 5.0, 0.1
    z, rows = rng.normal(size=(PEOPLE, VOXELS)) * 1.8, np.array([3, 40, 500], dtype=np.int32)
    arrays = {"refit-flags.npz": {"refit_flags_packed": packed(refit_abs > 1.96), "recurrence_counts": (refit_abs > 1.96).sum(axis=0).astype(np.uint8),
                                  "held_out_cohort_rows_zero_based": rows},
              "z-summary.npz": {"original_z": z, "refit_min_abs_z": refit_abs.min(axis=0), "refit_max_abs_z": refit_abs.max(axis=0),
                                "held_out_cohort_rows_zero_based": rows.copy()}}
    if cutoffs:
        arrays["cutoff-flags.npz"] = {"cutoffs": np.array(CUTOFFS), "refit_flags_packed": np.stack([packed(refit_abs > t) for t in CUTOFFS]),
                                      "original_flags_packed": np.stack([packed(np.abs(z) > t) for t in CUTOFFS]), "held_out_cohort_rows_zero_based": rows}
    tamper(arrays)
    outputs = {}
    for file, data in arrays.items():
        path = base / "arrays" / name / file
        np.savez_compressed(path, **data)
        outputs[file] = {"path": f"arrays/{name}/{file}", "sha256": sha256(path), "bytes": path.stat().st_size}
    receipt = {"run_name": name, "fold": 0, "folds": N_FOLDS, "bootstrap_refits": N_REFITS, "voxel_columns": VOXELS, "flag_rule": "abs(z) > 1.96",
               "mask_column_order": "C-order true voxels", "packed_axis": -1, "packed_bitorder": "little", "held_out_scans": PEOPLE, "references": 450,
               "z_summary": {"check": {"entries_that_differ": 0}}, "runtime": {"hugging_face_job": "test"}, "outputs": outputs}
    (base / "json" / name / "result.json").write_text(json.dumps(receipt))
    return arrays


class FetchedInputsTest(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(share_inputs, "N_VOXELS", VOXELS)
        patch.start()
        self.addCleanup(patch.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_flags_at_each_cutoff_come_back_with_the_single_fit_z(self):
        arrays = write_run(self.root)
        run = open_fold(self.root, 0, cutoffs=True)
        self.assertEqual(set(run.files), {"refit-flags.npz", "z-summary.npz", "cutoff-flags.npz"})
        for level, cutoff in enumerate(CUTOFFS):
            got = read_cutoff_fold(run, cutoff)
            np.testing.assert_array_equal(got.packed, arrays["cutoff-flags.npz"]["refit_flags_packed"][level])
            np.testing.assert_array_equal(got.z, arrays["z-summary.npz"]["original_z"])
            self.assertEqual(got.rows.tolist(), [3, 40, 500])

    def test_the_1_96_reader_accepts_a_consistent_z_summary_run(self):
        arrays = write_run(self.root, cutoffs=False)
        got = read_fold(open_fold(self.root, 0))
        np.testing.assert_array_equal(got.packed, arrays["refit-flags.npz"]["refit_flags_packed"])

    def test_a_cutoff_the_file_does_not_hold_is_refused(self):
        write_run(self.root)
        with self.assertRaisesRegex(ValueError, "do not hold 2.5 exactly once"):
            read_cutoff_fold(open_fold(self.root, 0, cutoffs=True), 2.5)

    def test_a_run_opened_without_its_cutoffs_cannot_give_them(self):
        write_run(self.root)
        with self.assertRaisesRegex(ValueError, "cannot read the receipt"):
            open_fold(self.root, 0)  # the z-summary run of this fold is not there
        run = dataclasses.replace(open_fold(self.root, 0, cutoffs=True), cutoff_flags=None)
        with self.assertRaisesRegex(ValueError, "without its cutoff-flags.npz"):
            read_cutoff_fold(run, 2.3)

    def test_a_file_that_is_not_the_one_in_the_receipt_is_refused(self):
        write_run(self.root)
        path = next(self.root.rglob("cutoff-flags.npz"))
        path.write_bytes(path.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "is not the file its receipt describes"):
            open_fold(self.root, 0, cutoffs=True)

    def test_each_consistency_check_catches_its_own_break(self):
        def flip(file: str, key: str, index: tuple, bit: int):
            def tamper(arrays: dict) -> None:
                arrays[file][key][index] ^= bit
            return tamper

        def other_scans(arrays: dict) -> None:
            arrays["cutoff-flags.npz"]["held_out_cohort_rows_zero_based"] = np.array([3, 40, 501], dtype=np.int32)

        cases = {"the single fit's flags": flip("cutoff-flags.npz", "original_flags_packed", (0, 1, 3), 1),
                 "the AND of the refits' flags": flip("cutoff-flags.npz", "refit_flags_packed", (0, 7, 0, 0), 1),  # voxel 0 loses one refit
                 "the OR of the refits' flags": flip("cutoff-flags.npz", "refit_flags_packed", (0, 3, 1, 0), 4),  # voxel 2 gains one
                 "the padding bits": flip("cutoff-flags.npz", "refit_flags_packed", (0, 3, 1, -1), 128),
                 "lists other scans": other_scans}
        for broken, tamper in cases.items():
            with self.subTest(broken), tempfile.TemporaryDirectory() as tmp:
                write_run(Path(tmp), tamper=tamper)
                run = open_fold(Path(tmp), 0, cutoffs=True)
                with self.assertRaisesRegex(ValueError, broken):
                    read_cutoff_fold(run, 2.3)

    def test_definitions_name_the_cutoff_of_the_run(self):
        for cutoff in (1.96, 3.1):
            got = definitions(cutoff)
            self.assertEqual(got["cluster"], f"connected component of the single fit's map, positive (z > {cutoff:g}) and negative (z < -{cutoff:g}) separately")
            self.assertIn(f"abs(z) > {cutoff:g} in each of the 64 refits", got["refit_flags"])
        self.assertEqual(definitions(3.1)["readings"], definitions(1.96)["readings"])


class CohortTest(unittest.TestCase):
    def test_every_scan_once_and_no_repeats(self):
        rows = np.array_split(np.arange(N_SCANS), 5)
        check_cohort(dict(enumerate(rows)))
        check_cohort({0: rows[0], 1: rows[1]})
        with self.assertRaisesRegex(ValueError, "repeat"):
            check_cohort({0: rows[0], 1: rows[0]})
        with self.assertRaisesRegex(ValueError, "not 563"):
            check_cohort({k: np.arange(k * 100, k * 100 + 100) for k in range(5)})


if __name__ == "__main__":
    unittest.main()
