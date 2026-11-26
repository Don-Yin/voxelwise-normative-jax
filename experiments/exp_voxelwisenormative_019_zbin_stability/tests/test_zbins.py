"""Boundary, weighting and retention checks without any IXI data or GPU."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from src.modules.zbins import reconcile, run, summarize_bins

from shared.voxelwisenormative import reference_bootstrap as boot
from shared.voxelwisenormative.reference_grid import Z_THRESHOLD, Cell


def test_bins_use_the_original_magnitudes_and_a_person_specific_denominator():
    z = np.array([[1.96, -2.0, -2.5, 3.1, 4], [2.49, 2.4, 3.09, 1, 0], [1, 0, 0, 0, 0]])
    recurrence = np.array([[0, 51, 52, 51, 64], [51, 64, 51, 0, 0], [0, 0, 0, 0, 0]])
    baseline = np.abs(z) > Z_THRESHOLD
    rows = summarize_bins(baseline, np.abs(z[baseline]), recurrence)
    assert [(r["abs_z_lower"], r["abs_z_upper_exclusive"]) for r in rows] == [(1.96, 2.5), (2.5, 3.1), (3.1, None)]
    assert [(r["unstable_flags"], r["baseline_flags"]) for r in rows] == [(2, 3), (1, 2), (1, 2)]
    assert [r["unstable_fraction"] for r in rows] == [2 / 3, 1 / 2, 1 / 2]
    assert [r["held_out_people_with_flags"] for r in rows] == [2, 2, 1]
    np.testing.assert_allclose([r["per_person_fraction_p10_p50_p90"] for r in rows],
                               [[0.55, 0.75, 0.95], [0.1, 0.5, 0.9], [0.5, 0.5, 0.5]])


def test_gate_refuses_either_wrong_total():
    reconcile([{"unstable_flags": 200_530, "baseline_flags": 927_451}])
    for unstable, baseline in ((200_529, 927_451), (200_530, 927_452)):
        with pytest.raises(ValueError, match="expected 200,530 / 927,451"):
            reconcile([{"unstable_flags": unstable, "baseline_flags": baseline}])


def test_real_mode_refuses_slurm(tmp_path):
    with (patch.dict("os.environ", {"SLURM_JOB_ID": "42"}, clear=True), patch("sys.platform", "linux"),
          pytest.raises(RuntimeError, match="require the Hugging Face wrapper")):
        run(tmp_path / "receipt.json", smoke=False)
    assert not tuple(tmp_path.iterdir())


def test_shared_worker_retains_only_flagged_baseline_magnitudes(tmp_path):
    z = np.zeros((113, 32))
    z[0, :4] = [1.96, -2.0, 2.5, -3.1]
    refit_z = np.zeros_like(z)
    refit_z[0, [0, 2, 3]] = [3.5, 4.5, 5.5]
    device = SimpleNamespace(device_kind="fake-cpu", memory_stats=dict)
    out = tmp_path / "receipt.json"

    def summarize(baseline, original_abs_flags, counts):
        np.testing.assert_array_equal(np.flatnonzero(baseline), [1, 2, 3])
        np.testing.assert_array_equal(original_abs_flags, [2.0, 2.5, 3.1])
        np.testing.assert_array_equal(counts[baseline], [0, 2, 2])
        return [{"baseline_flags": int(baseline.sum())}]

    with (patch.object(boot, "open_fitter", return_value=(None, device, "test")),
          patch.object(boot, "scores", side_effect=[(z, 0), (refit_z, 0), (refit_z, 0)])):
        boot.worker(Cell("synthetic", 25), 0, out, smoke=True, summarize_magnitudes=summarize)
    assert json.loads(out.read_text())["baseline_abs_z_bins"] == [{"baseline_flags": 3}]
