"""No real MRI files or CUDA required: test cases, data identity, recurrence and projections."""

from __future__ import annotations

import tempfile
import unittest
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np
from src.modules.bootstrap import instability
from src.modules.cohorts import (
    bootstrap_draw,
    mask_sample,
    nested_train,
    synthetic_cohort,
)
from src.modules.collect import previous_voxel_sweep
from src.modules.gpu import TRAIN_ELEMENTS_CAP, chunk_width
from src.modules.grid import (
    BOOT_CASES,
    CHAINS,
    CPU_CASES,
    GPU_CASES,
    MASK_VOXELS,
    V_SWEEP,
)
from src.modules.scaling import cpu_projection, estimate_cost, power_fit


class DesignTest(unittest.TestCase):
    def test_calibration_indices(self):
        self.assertEqual([(c.dataset, c.n, w) for c, w in CPU_CASES[12:15]],
                         [("ixi", 450, w) for w in (16, 64, 256)])
        self.assertEqual([(c.dataset, c.n, w) for c, w in CPU_CASES[33:36]],
                         [("synthetic", 50_000, w) for w in (4, 12, 24)])
        for start, n in ((8, 450), (22, 50_000)):
            self.assertEqual([(c.n, p) for c, p in GPU_CASES[start:start+2]], [(n, "float32"), (n, "float64")])
        self.assertEqual((BOOT_CASES[10][0].n, BOOT_CASES[10][1]), (450, 0))
        self.assertEqual(len(BOOT_CASES), 11)
        self.assertEqual(chunk_width(450), 32_768)
        self.assertLessEqual(50_000 * chunk_width(50_000), TRAIN_ELEMENTS_CAP)

    def test_nested_stratified_chains_and_fixed_holdout(self):
        cohort = synthetic_cohort(450)
        sites = cohort.cov["site"]
        chains = []
        for seed in CHAINS:
            sets = [set(nested_train(sites, cohort.train, n, seed)) for n in (25, 50, 100, 200, 450)]
            self.assertTrue(all(a < b for a, b in pairwise(sets)))
            self.assertEqual(sets[-1], set(cohort.train))
            self.assertTrue(all(not rows.intersection(cohort.test) for rows in sets))
            chains.append(sets[0])
        self.assertEqual(len({frozenset(s) for s in chains}), 3)
        sampled = bootstrap_draw(cohort, np.random.default_rng(91))
        self.assertLess(len(set(sampled)), len(cohort.train))
        self.assertLessEqual(set(sampled), set(cohort.train))
        np.testing.assert_array_equal(np.unique(sites[sampled], return_counts=True),
                                      np.unique(sites[cohort.train], return_counts=True))

    def test_synthetic_values_agree_at_any_voxel_width_or_n(self):
        small, large = synthetic_cohort(450), synthetic_cohort(50_000)
        columns = mask_sample(4)
        self.assertLess(set(columns), set(mask_sample(24)))
        sy, st = small.responses(small.train, columns)
        ly, lt = large.responses(large.train, columns[::-1])
        np.testing.assert_array_equal(sy, ly[:450, ::-1])
        np.testing.assert_array_equal(st, lt[:, ::-1])

    def test_baseline_denominator_and_strict_threshold(self):
        original = np.array([[True, True, False], [True, False, False]])
        recurrences = np.array([[51, 52, 21], [64, 0, 65]])
        with self.assertRaisesRegex(ValueError, "invalid baseline"):
            instability(original, recurrences)
        recurrences[1, 2] = 1
        result = instability(original, recurrences)
        self.assertEqual(result["baseline_flags"], 3)
        self.assertEqual(result["unstable_flags"], 1)
        self.assertEqual(result["unstable_fraction"], 1 / 3)


class ProjectionTest(unittest.TestCase):
    def test_n_powers_and_cpu_projection(self):
        samples = [{"columns": w, "peak_rss_bytes": 100_000_000 + 3000 * w, "seconds": 0.5 * w,
                    "retained_training_arrays_bytes": 1000 * w} for w in (16, 64, 256)]
        projected = cpu_projection(samples)
        self.assertAlmostEqual(projected["projected_peak_rss_bytes"], 100_000_000 + 3000 * MASK_VOXELS, delta=0.01)
        self.assertEqual(projected["projected_seconds"], 0.5 * MASK_VOXELS)
        power = power_fit([(n, 3 * n**1.2) for n in (25, 50, 450, 5000)])
        assert power is not None
        self.assertAlmostEqual(power["exponent"], 1.2)
        flat = [dict(row, peak_rss_bytes=100_000_000) for row in samples]
        self.assertIsNone(cpu_projection(flat)["projected_peak_rss_bytes"])

    def test_conditional_cost_and_voxel_sweep(self):
        rows = [{"dataset": "synthetic", "n": n, "n_test": 113,
                 "pcntoolkit": {"projected_seconds": n / 2},
                 "gpu": {p: {"seconds_median": n / 50, "warmup_seconds": n / 45}
                         for p in ("float32", "float64")}} for n in (450, 10_000, 50_000)]
        exact = estimate_cost(rows, 50_000)
        self.assertEqual(exact["estimated_fit_and_64_refits_cpu_hours"], 65 * 25_000 / 3600)
        self.assertEqual(exact["gpu"]["float32"]["fit_and_64_refits_gpu_hours"], 65 * 1000 / 3600)
        interpolated = estimate_cost(rows, 25_000)
        self.assertAlmostEqual(interpolated["estimated_fit_and_64_refits_cpu_hours"], 65 * 12_500 / 3600)
        with self.assertRaisesRegex(ValueError, "outside measured synthetic sizes"):
            estimate_cost(rows, 60_000)
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / 'gpu-scaling.csv'
            fixture.write_text('variant,n_voxels,seconds_median,peak_bytes_in_use,host_peak_rss_bytes\n' + ''.join(
                f'{variant},{n},{n / 1000},{n * 8},{n * 16}\n'
                for variant in ('early-f32', 'baseline') for n in V_SWEEP))
            with patch('src.modules.collect.resolve', return_value={fixture.name: fixture}):
                self.assertEqual({r['n_voxels'] for r in previous_voxel_sweep()['rows']}, set(V_SWEEP))
