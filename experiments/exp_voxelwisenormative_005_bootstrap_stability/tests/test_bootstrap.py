"""Check the bootstrap contract without treating a toy fixture as GPU evidence."""

import base64
import json
import logging
from contextlib import closing
from http.client import HTTPConnection

import numpy as np
from src.modules import experiment
from src.modules.live import live_view
from src.modules.model import Inputs, bootstrap_rows, scale, survives
from src.modules.slice_view import DIMMED, raster


def test_draws_are_paired_with_replacement():
    """Repeated rows are allowed, and running a shorter demo preserves trial prefixes."""
    a = bootstrap_rows(20261121, 64, 450)
    b = bootstrap_rows(20261121, 8, 450)
    np.testing.assert_array_equal(a[:8], b)
    assert a.shape == (64, 450)
    assert 0 <= a.min() <= a.max() < 450
    assert all(len(np.unique(sample)) < 450 for sample in a)


def test_scales_only_the_sampled_reference():
    """Duplicates contribute to the scaling; altering the held-out value changes only its own z."""
    data = np.array([[0, 10], [0, 10], [9, 1]], dtype=np.float32)
    y, yt = scale(data, np.array([100, 200]))
    np.testing.assert_allclose(y.mean(axis=0), [0, 0], atol=1e-15)
    np.testing.assert_allclose(y.std(axis=0), [1, 1])
    np.testing.assert_allclose(yt, [(np.array([100, 200]) - [3, 7]) / data.std(axis=0)])
    np.testing.assert_array_equal(y, scale(data, -np.ones(2))[0])


def test_survival_includes_the_cutoff_not_new_flags():
    """Exactly 80% passes, and a later refit may bring an original flag back."""
    flags = np.array([True, True, False])
    np.testing.assert_array_equal(survives(flags, np.array([4, 3, 5]), 5), [True, False, False])
    np.testing.assert_array_equal(survives(flags, np.array([3, 4, 4]), 5), [False, True, False])


def test_payload_and_updates_agree(tmp_path, monkeypatch):
    """One completed frame per trial; flags are strict and raw sampling indices exclude test rows."""
    columns = np.array([[0, 1, 2, -1]])
    data = Inputs(np.zeros((6, 4)), np.zeros(4), np.ones((6, 1)), np.ones((1, 1)),
                  np.array([0, 1, 2, 4, 5, 6]), 3, columns)
    z = iter(map(np.array, [[2, -2, 1.96, -1.96], [1.96, -2, 2, -2], [2, 2, 0, 0]]))
    monkeypatch.setattr(experiment, "gpu_fitter", lambda: (None, "fixture", "fixture"))
    monkeypatch.setattr(experiment, "load_inputs", lambda: data)
    monkeypatch.setattr(experiment, "score", lambda *args: (next(z), 0.2))
    frames = []
    report_path = experiment.run(tmp_path, "fixture", 2, 8, 4, lambda *s: frames.append(s))
    report = json.loads(report_path.read_text())
    assert report["baseline"] == {"flagged_whole_mask": 2, "flagged_slice": 2, "fit_seconds_including_compile": 0.2}
    assert [s[:3] for s in frames] == [(0, 2, 2), (1, 1, 2), (2, 1, 2)]
    assert report["baseline_flags_with_any_opposite_sign_trial"] == 1
    with np.load(tmp_path / report["arrays"]["path"]) as saved:
        flags = np.unpackbits(saved["packed_trial_flags"], axis=1, bitorder="little", count=4)
        np.testing.assert_array_equal(flags, [[0, 1, 1, 1], [1, 1, 0, 0]])
        np.testing.assert_array_equal(saved["running_survival_fraction"], [[0, 1, 1, 1], [.5, 1, .5, .5]])
        assert not (saved["sampled_cohort_rows"] == 3).any()
        np.testing.assert_array_equal(saved["trial_slice_rgb"], [s[-1] for s in frames])
        np.testing.assert_array_equal(saved["trial_slice_rgb"][-1, 0, 0], DIMMED)
        np.testing.assert_array_equal(saved["reference_rgb"], raster(columns, np.array([2, -2, 1.96, -1.96])))


def test_loopback_view_publishes_atomic_snapshots(caplog, monkeypatch):
    """The browser route updates the exact raster and count, without needing a GPU in tests."""
    caplog.set_level(logging.INFO)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "")
    with live_view(0, 2) as publish:
        port = int(caplog.messages[-1].rsplit(":", 1)[-1].rstrip("/"))
        with closing(HTTPConnection("127.0.0.1", port, timeout=2)) as client:
            client.request("GET", "/")
            with client.getresponse() as response:
                assert b"/layout.json" in response.read()
            client.request("GET", "/layout.json")
            with client.getresponse() as response:
                layout = json.load(response)
            assert layout["slice_box_xywh_px"] == [220, 180, 552, 664]
            assert layout["stable_box_xywh_px"] == [1148, 180, 552, 664]
            assert layout["tight_crop_source_xywh_voxels"] == [26, 1, 69, 83]
            current = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
            publish(1, 3, 5, 255 - current, current)
            client.request("GET", "/state")
            with client.getresponse() as response:
                state = json.load(response)
        assert (state["trial"], state["surviving"], state["baseline"]) == (1, 3, 5)
        assert (state["cutoff"], state["recurrence"]) == (1.96, .8)
        assert base64.b64decode(state["current"]) == current.tobytes()
