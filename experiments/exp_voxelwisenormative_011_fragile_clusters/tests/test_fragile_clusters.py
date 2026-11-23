"""Known voxel patterns test the agreed ranking and control definitions independently of CUDA."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from src.modules.clusters import (
    Cluster,
    ClusterGrid,
    bootstrap_ranking,
    contrast_ranking,
)
from src.modules.inputs import DRAW_DIGEST, draws, fold_split, read_mask, synthetic
from src.modules.study import flags_for, run, run_per_refit, site_fits, validate

from shared.voxelwisenormative.artifacts import (
    PCNTOOLKIT_REFERENCE,
    VOXEL_MATRIX,
    resolve,
)


class ClustersTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.ones((6, 6, 5), dtype=bool)
        self.mask[5, 5, 3] = False
        self.grid = ClusterGrid(self.mask, (0, 1, 2, 3, 4))

    def flags(self, *points):
        """Encode a manually specified voxel set in mask-column order."""
        mask = np.zeros_like(self.mask)
        for point in points:
            mask[point] = True
        return mask[self.mask]

    def test_three_dimensional_face_connectivity(self):
        cluster = self.flags((0, 1, 1), (1, 1, 1), (1, 1, 2), (2, 2, 2))
        self.assertEqual(self.grid.largest(cluster), Cluster(3, (0, 2, 1, 0, 0)))
        self.assertEqual(self.grid.largest(self.flags()), Cluster(0, (0, 0, 0, 0, 0)))
        tie = self.flags((0, 0, 0), (0, 1, 0), (3, 2, 2), (3, 3, 2))
        self.assertEqual(self.grid.largest(tie), Cluster(2, (2, 0, 0, 0, 0)))

    def test_bootstrap_rank_prefers_cluster_over_more_isolated_flags(self):
        triple = self.flags((1, 1, 1), (1, 1, 2), (1, 1, 3))
        stable = self.flags((1, 1, 4))
        distant = self.flags((5, 5, 4))
        scattered = self.flags((0, 0, 0), (0, 0, 1), (3, 2, 2), (5, 5, 4), (5, 1, 3))
        fourth = self.flags((0, 4, 4))
        first = np.stack([triple | stable, scattered, triple | fourth, stable])
        counts = first.astype(np.uint8)
        counts[0, stable] = 52
        counts[0, distant] = 7
        counts[0, self.flags((1, 1, 2))] = 51
        counts[3, stable] = 64
        rows = bootstrap_ranking(counts, first, self.grid)
        self.assertEqual([r["held_out_row_zero_based"] for r in rows], [2, 0, 1, 3])
        self.assertEqual([r["largest_cluster_voxels"] for r in rows], [3, 3, 2, 0])
        self.assertEqual([r["first_unstable_voxels"] for r in rows], [4, 3, 5, 0])
        self.assertEqual(rows[1]["ever_unstable_voxels"], 4)
        self.assertEqual(rows[1]["largest_cluster_axial_voxels"], "0,1,1,1,0")

    def test_inconsistent_recurrence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent recurrence"):
            bootstrap_ranking(np.zeros((1, self.grid.n), dtype=np.uint8),
                              self.flags((0, 0, 0))[None], self.grid)

    def test_scanner_flip_does_not_join_additions_and_losses(self):
        gained = self.flags((1, 1, 1), (1, 1, 2), (1, 1, 3))
        lost = self.flags((1, 2, 1), (1, 2, 2))  # shares faces with gained, but the direction differs
        shared = self.flags((0, 0, 0))
        scattered = self.flags((0, 0, 1), (4, 0, 0), (4, 4, 0), (0, 4, 4), (0, 0, 4), (4, 4, 4))
        rows = contrast_ranking("without-hh", "no-site", np.stack([gained | shared, scattered]),
                                np.stack([lost | shared, self.flags()]), self.grid)
        self.assertEqual([r["held_out_row_zero_based"] for r in rows], [0, 1])
        self.assertEqual(rows[0]["flipped_voxels"], 5)
        self.assertEqual(rows[0]["largest_flip_cluster_voxels"], 3)
        self.assertEqual(rows[0]["new_largest_cluster_voxels"], 3)
        self.assertEqual(rows[0]["removed_largest_cluster_voxels"], 2)
        self.assertEqual(rows[0]["rank_within_contrast"], 1)


class ProtocolTest(unittest.TestCase):
    @unittest.skipUnless(os.getenv('VOXELWISE_TEST_IXI') == '1', 'requires locally prepared IXI artifacts')
    def test_original_draw_digest_and_mask_without_raw_ixi(self):
        path = resolve(PCNTOOLKIT_REFERENCE, "arrays.npz")["arrays.npz"]
        with np.load(path, allow_pickle=False) as archive:
            ref = np.flatnonzero(~archive["test"])
        samples, digest = draws(replace(synthetic(), train_rows=ref), verify_original=True)
        self.assertEqual(samples.shape, (64, 450))
        self.assertEqual(digest, DRAW_DIGEST)
        path = resolve(VOXEL_MATRIX, "mask.nii.gz")["mask.nii.gz"]
        mask, cuts, heights = read_mask(path)
        self.assertEqual(mask.sum(), 165402)
        self.assertEqual(len(cuts), len(heights))
        self.assertEqual(len(cuts), 7)

    def test_per_refit_mode_agrees_with_main_run(self):
        """The extra export must repeat the main run's refits bit for bit and sum to its counts."""
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / ".results"
            run(results, "synthetic-main", 32768, use_fixture=True)
            run_per_refit(results, "synthetic-refits", 32768, use_fixture=True)
            with np.load(results / "arrays/synthetic-main/flags.npz") as main:
                counts, first = main["recurrence_counts"], main["first_three_flags_packed"]
            with np.load(results / "arrays/synthetic-refits/refit-flags.npz") as refits:
                packed, again = refits["refit_flags_packed"], refits["recurrence_counts"]
            receipt = json.loads((results / "json/synthetic-refits/result.json").read_text())
        self.assertEqual(packed.shape, (64, 4, 40))
        bits = np.unpackbits(packed, axis=-1, count=320, bitorder="little")
        np.testing.assert_array_equal(bits.sum(axis=0), counts)
        np.testing.assert_array_equal(again, counts)
        np.testing.assert_array_equal(packed[:3], first)
        self.assertEqual(list(receipt["outputs"]), ["refit-flags.npz"])
        self.assertEqual(receipt["mode"], "synthetic-refits")

    @unittest.skipUnless(os.getenv('VOXELWISE_TEST_IXI') == '1', 'requires locally prepared IXI artifacts')
    def test_folds_keep_the_frozen_split_and_hold_out_every_scan_once(self):
        """Fold 0 must be the original 113 scans (old results check it); the five folds cover all 563 scans once."""
        path = resolve(PCNTOOLKIT_REFERENCE, "arrays.npz")["arrays.npz"]
        with np.load(path, allow_pickle=False) as archive:
            held = archive["test"]
        splits = [fold_split(held, k) for k in range(5)]
        np.testing.assert_array_equal(splits[0][1], np.flatnonzero(held))
        np.testing.assert_array_equal(splits[0][0], np.flatnonzero(~held))
        self.assertEqual([len(test) for _, test in splits], [113, 113, 113, 112, 112])
        np.testing.assert_array_equal(np.sort(np.concatenate([test for _, test in splits])), np.arange(563))
        for train, test in splits:
            np.testing.assert_array_equal(np.sort(np.concatenate([train, test])), np.arange(563))
        _, digest = draws(replace(synthetic(), train_rows=splits[0][0]), verify_original=True)
        self.assertEqual(digest, DRAW_DIGEST)
        with self.assertRaises(ValueError):
            fold_split(held, 5)

    def test_per_refit_fold_holds_out_its_own_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / ".results"
            run_per_refit(results, "synthetic-fold-2", 32768, use_fixture=True, fold=2)
            with np.load(results / "arrays/synthetic-fold-2/refit-flags.npz") as refits:
                rows, packed = refits["held_out_cohort_rows_zero_based"], refits["refit_flags_packed"]
            receipt = json.loads((results / "json/synthetic-fold-2/result.json").read_text())
        np.testing.assert_array_equal(rows, fold_split(np.arange(28) >= 24, 2)[1])
        self.assertEqual(packed.shape, (64, 6, 40))
        self.assertEqual((receipt["fold"], receipt["references"]), (2, 22))

    def test_site_variants_keep_same_test_rows_and_fixed_no_site_design(self):
        data = synthetic()
        validate(data)
        cases = site_fits(data)
        self.assertEqual([c.name for c in cases], ["full-site", "no-site", "without-guys", "without-hh", "without-iop"])
        self.assertEqual([c.control for c in cases], [None, "full-site", "no-site", "no-site", "no-site"])
        self.assertEqual([len(c.train) for c in cases], [24, 24, 16, 16, 16])
        self.assertEqual([c.include_site for c in cases], [True, False, False, False, False])
        for case, site in zip(cases[2:], ("Guys", "HH", "IOP"), strict=True):
            self.assertNotIn(site, data.reference_sites[case.train])

    def test_response_scaling_with_duplicate_rows_and_strict_cutoff(self):
        data = synthetic()
        sample = np.array([0, 1, 2, 0, 1])
        selected = data.train[sample].astype(np.float64)
        mean, sd = selected.mean(axis=0), selected.std(axis=0)

        def score(rotated, y, yt):
            np.testing.assert_allclose(y, (selected - mean) / sd)
            np.testing.assert_allclose(yt, (data.test - mean) / sd)
            self.assertEqual(rotated.H.shape, (len(data.test), 9))
            z = np.zeros_like(yt)
            beyond = np.nextafter(1.96, np.inf)
            z[0, :4] = [1.96, -1.96, beyond, -beyond]
            return z

        got = flags_for(data, sample, score, include_site=False)
        self.assertEqual(got[0, :4].tolist(), [False, False, True, True])
        self.assertEqual(int(got.sum()), 2)
