"""Local checks: sample and trace parsing, the exp_005 draw digest, and a synthetic CPU sweep end to end."""

from __future__ import annotations

import gzip
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT), str(EXPERIMENT.parents[1])]

# The experiment and project roots are put on sys.path first, so these are imported by name.
read_samples, summarize = (getattr(importlib.import_module("src.modules.gpu_samples"), name) for name in ("read_samples", "summarize"))
DRAW_DIGEST, draws = (getattr(importlib.import_module("src.modules.inputs"), name) for name in ("DRAW_DIGEST", "draws"))
device_busy = importlib.import_module("src.modules.trace").device_busy


class Parsing(unittest.TestCase):
    """Raw evidence parsers used only by collect."""

    def test_samples_drop_only_a_cut_final_line(self):
        """Windows select samples by epoch; a SIGTERM-truncated last line is dropped."""
        lines = ["2026/11/21 12:00:00.000, 10, 5, 1000, 100.5, 1410, 40",
                 "2026/11/21 12:00:00.100, 90, 50, 2000, 300.0, 1410, 41",
                 "2026/11/21 12:00:00.200, 100, 60, 3000, 350.0, 1395, 42", "2026/11/21 12:00"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "s.csv")
            path.write_text("\n".join(lines) + "\n")
            samples = read_samples(path, 0.0)
        start = samples["epoch"][1]
        summary = summarize(samples, [[start - 0.01, start + 0.15]])
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["gpu_util_mean"], 95.0)
        self.assertEqual(summary["memory_used_mib_max"], 3000.0)
        self.assertEqual(summarize(samples, [[0.0, 1.0]]), {"samples": 0})

    def test_device_busy_unions_stream_events(self):
        """Overlapping kernels merge; gaps over 20 us are counted; host events are ignored."""
        events = [{"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "/device:GPU:0"}},
                  {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "/host:CPU"}},
                  {"ph": "M", "name": "thread_name", "pid": 1, "tid": 7, "args": {"name": "Stream #13(Compute)"}},
                  {"ph": "M", "name": "thread_name", "pid": 1, "tid": 8, "args": {"name": "XLA Ops"}},
                  {"ph": "X", "pid": 1, "tid": 7, "ts": 0, "dur": 10}, {"ph": "X", "pid": 1, "tid": 7, "ts": 5, "dur": 10},
                  {"ph": "X", "pid": 1, "tid": 7, "ts": 45, "dur": 5}, {"ph": "X", "pid": 1, "tid": 7, "ts": 52, "dur": 48},
                  {"ph": "X", "pid": 1, "tid": 8, "ts": 0, "dur": 100}, {"ph": "X", "pid": 2, "tid": 1, "ts": 0, "dur": 999}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "perfetto_trace.json.gz")
            with gzip.open(path, "wt") as handle:
                json.dump({"traceEvents": events}, handle)
            busy = device_busy(path)
        self.assertEqual(busy["stream_events"], 4)
        self.assertEqual(busy["busy_us"], 68.0)
        self.assertAlmostEqual(busy["busy_fraction"], 0.68)
        self.assertEqual((busy["gaps"], busy["gaps_over_20us"]), (2, 1))
        self.assertEqual(busy["device_threads"], {"Stream #13(Compute)": 4, "XLA Ops": 1})


class Draws(unittest.TestCase):
    """The recipe reproduces exp_005's frozen draws from the split alone (no grey matter is read)."""

    @unittest.skipUnless(os.getenv('VOXELWISE_TEST_IXI') == '1', 'requires locally prepared IXI artifacts')
    def test_exp005_digest(self):
        """Only the boolean split of the pinned reference artifact is needed."""
        from shared.voxelwisenormative.artifacts import PCNTOOLKIT_REFERENCE, resolve

        with np.load(resolve(PCNTOOLKIT_REFERENCE, "arrays.npz")["arrays.npz"]) as ref:
            rows = np.flatnonzero(~ref["test"])
        indices, digest = draws(rows, 64, verify=True)
        self.assertEqual((indices.shape, digest), ((64, 450), DRAW_DIGEST))


class SyntheticSweep(unittest.TestCase):
    """The whole local path on CPU: compile, original fit, sweep, receipts, traces and collect."""

    def test_float64_sweep_and_collect(self):
        """K=1 repeats bit for bit; every K covers every draw; collect writes both tables."""
        from src.modules.collect import collect
        from src.modules.sweep import run

        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            receipt = json.loads(run(results, "synthetic-test", "float64", 64, smoke=True).read_text())
            passes = {(p["label"], p["k"]): p for p in receipt["passes"]}
            self.assertEqual(list(passes), [("reference", 1), ("repeat", 1), ("batched", 2), ("batched", 4),
                                            ("batched", 8)])
            self.assertEqual(passes["repeat", 1]["agreement_with_k1"]["bitwise_equal_refits"], receipt["refits"])
            self.assertTrue(all(len(p["batches"]) * p["k"] == receipt["refits"] for p in passes.values()))
            self.assertEqual(receipt["chunk_widths"], [64, 64, 64, 64, 44])
            tables = collect(results, "synthetic-test")
            self.assertTrue((tables / "sweep.csv").exists() and (tables / "trace.csv").exists())
            with self.assertRaises(FileExistsError):
                run(results, "synthetic-test", "float64", 64, smoke=True)
            # One run per precision (two Hugging Face jobs): the same receipts in json/<run>-float64/ give the same table.
            (results / "json" / "synthetic-test").rename(results / "json" / "synthetic-test-float64")
            by_precision = collect(results, "synthetic-test")
            self.assertEqual(by_precision, tables)
            self.assertEqual((by_precision / "sweep.csv").read_text().count("\n"), len(receipt["passes"]) + 1)
            with self.assertRaises(FileNotFoundError):
                collect(results, "synthetic-other")


if __name__ == "__main__":
    unittest.main()
