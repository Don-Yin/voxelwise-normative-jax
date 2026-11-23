"""fetch.py on small hand-made run folders: what it places, and each reason it places nothing."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT)]
fetch = importlib.import_module("shared.hugging_face_jobs.fetch")

JOB = "1234567890abcdef12345678"
A100 = {"available": True, "devices": [{"name": "NVIDIA A100-SXM4-80GB", "driver_version": "595.112.07"}]}


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


def host(run: str, experiment: str, args: str, **changes) -> dict:
    """A v3 host record of a finished run on one A100."""
    record = {"run_name": run, "experiment": experiment, "entry_args": args, "dry_run": False, "exit_code": 0, "gpu": A100,
              "environment": {"JOB_ID": JOB}, "hugging_face_job": JOB, "slurm_variables": [],
              "python": {"prefix": "test-environments/hugging-face-a100-venv"}}
    return record | changes


def exp011_run(base: Path, run: str = "hugging-face-a100-per-refit-fold1", **host_changes) -> Path:
    """json/<run>/{result.json, host record}, arrays/<run>/refit-flags.npz and logs/<run>.log of a fold 1 run."""
    folder = base / run
    npz = folder / "arrays" / run / "refit-flags.npz"
    npz.parent.mkdir(parents=True)
    npz.write_bytes(b"not really an npz")
    receipt = {"mode": "ixi-cuda-refits", "fold": 1, "references": 450, "held_out_scans": 113, "bootstrap_refits": 64,
               "refit_seconds": [30.0] + [7.5] * 63,
               "runtime": {"device": "NVIDIA A100-SXM4-80GB", "jax": "0.12.1", "slurm_job": None, "hugging_face_job": JOB},
               "outputs": {"refit-flags.npz": {"sha256": hashlib.sha256(npz.read_bytes()).hexdigest(), "bytes": npz.stat().st_size}}}
    write_json(folder / "json" / run / "result.json", receipt)
    write_json(folder / "json" / run / fetch.HOST_FILE,
               host(run, fetch.EXP011, f"run.py --per-refit --fold 1 --run-name {run} --resources", **host_changes))
    (folder / "logs").mkdir()
    (folder / "logs" / f"{run}.log").write_text("[1/64] refit\n")
    return folder


def exp011_z_run(base: Path, run: str = "hugging-face-a100-z-summary-fold0", fold: int = 0, z_changes: dict | None = None,
                 arguments: str | None = None, cutoffs: tuple[float, ...] = (), cutoff_changes: dict | None = None) -> Path:
    """A `--z-summary` run of 4 scans x 24 voxels with real npz files: 64 refits, the first 8 voxels flagged in every one.

    With `cutoffs` also `--refit-flag-cutoffs` and its cutoff-flags.npz (`cutoff_changes` replaces arrays in it).
    """
    import numpy as np

    rng = np.random.default_rng(fold)
    z = rng.normal(size=(64, 4, 24))
    z[:, :, :8] = rng.uniform(2.0, 4.0, size=(64, 4, 8)) * rng.choice([-1, 1], size=(64, 4, 8))
    abs_z = np.abs(z)
    counts = (abs_z > 1.96).sum(axis=0).astype(np.uint8)
    rows = np.arange(4, dtype=np.int32) * 5 + fold
    folder = base / run
    arrays = folder / "arrays" / run
    arrays.mkdir(parents=True)
    np.savez_compressed(arrays / "refit-flags.npz", refit_flags_packed=np.packbits(abs_z > 1.96, axis=-1, bitorder="little"),
                        recurrence_counts=counts, held_out_cohort_rows_zero_based=rows)
    z_arrays = {"original_z": z[0], "refit_min_abs_z": abs_z.min(axis=0), "refit_max_abs_z": abs_z.max(axis=0),
                "held_out_cohort_rows_zero_based": rows} | (z_changes or {})
    np.savez(arrays / "z-summary.npz", **z_arrays)
    names = ["refit-flags.npz", "z-summary.npz"]
    if cutoffs:
        def pack(flags):
            return np.packbits(flags, axis=-1, bitorder="little")

        cutoff_arrays = {"cutoffs": np.array(cutoffs), "refit_flags_packed": np.stack([pack(abs_z > c) for c in cutoffs]),
                         "original_flags_packed": np.stack([pack(np.abs(z[0]) > c) for c in cutoffs]),
                         "held_out_cohort_rows_zero_based": rows} | (cutoff_changes or {})
        np.savez_compressed(arrays / "cutoff-flags.npz", **cutoff_arrays)
        names.append("cutoff-flags.npz")
    outputs = {name: {"sha256": hashlib.sha256((arrays / name).read_bytes()).hexdigest(), "bytes": (arrays / name).stat().st_size}
               for name in names}
    receipt = {"mode": "ixi-cuda-refits", "fold": fold, "references": 450, "held_out_scans": 4, "bootstrap_refits": 64,
               "flag_rule": "abs(z) > 1.96", "refit_seconds": [8.0] + [7.5] * 63, "outputs": outputs,
               "runtime": {"device": "NVIDIA A100-SXM4-80GB", "jax": "0.12.1", "slurm_job": None, "hugging_face_job": JOB},
               "z_summary": {"check": {"above_threshold_in_every_refit": int((counts == 64).sum())}}}
    if cutoffs:
        receipt["z_summary"]["cutoff_flags"] = {"file": "cutoff-flags.npz", "cutoffs": list(cutoffs), "check": {"entries_that_differ": 0}}
    write_json(folder / "json" / run / "result.json", receipt)
    option = f" --refit-flag-cutoffs {','.join(f'{c:g}' for c in cutoffs)}" if cutoffs else ""
    write_json(folder / "json" / run / fetch.HOST_FILE,
               host(run, fetch.EXP011, arguments or f"run.py --per-refit --fold {fold} --z-summary{option} --run-name {run} --resources"))
    (folder / "logs").mkdir()
    (folder / "logs" / f"{run}.log").write_text("[1/64] refit\n")
    return folder


def exp010_run(base: Path, run: str = "hugging-face-a100-bench-large-n50000-float32", log_timed: str = "170.778") -> Path:
    folder = base / run
    rss = 37954760704
    row = {"schema_version": 1, "workload": "batched-early", "device": "NVIDIA A100-SXM4-80GB", "warmup_seconds": 202.54036,
           "seconds_repeats": [170.77779, 171.18476, 168.99117], "seconds_median": 170.77779, "host_peak_rss_bytes": rss,
           **dict.fromkeys(fetch.SLURM_FIELDS)}
    write_json(folder / "json" / "gpu" / run / "synthetic-n50000-float32.json", row)
    write_json(folder / "json" / run / fetch.HOST_FILE, host(run, fetch.EXP010, f"run.py --resources gpu --index 22 --run-name {run}",
                                                          hugging_face_job=None))
    (folder / "logs").mkdir()
    (folder / "logs" / f"{run}.log").write_text(
        f"INFO x [1/4] synthetic-n50000 float32 warmup 202.540s\nINFO x [2/4] synthetic-n50000 float32 timed {log_timed}s\n"
        "INFO x [3/4] synthetic-n50000 float32 timed 171.185s\nINFO x [4/4] synthetic-n50000 float32 timed 168.991s\n"
        f"INFO run resources: peak_rss_self={rss / 2**20:.1f}MiB\n")
    return folder


def exp013_run(base: Path, run: str = "hugging-face-a100-gpu-headroom-float32", sampler_stderr: str = "", with_samples: bool = True) -> Path:
    folder = base / run
    samples = f"profile/gpu-samples/{run}/float32.csv"
    if with_samples:
        (folder / samples).parent.mkdir(parents=True)
        (folder / samples).write_text("2026-11-21 22:30:00.000, 0 %\n")
    row = {"gpu_sampler": {"exit_before_stop": None, "stderr": sampler_stderr, "path": samples},
           "passes": [{"label": "reference", "k": 1, "status": "ok"}, {"label": "batched", "k": 64, "status": "oom"}],
           "hugging_face_job": JOB, **dict.fromkeys(fetch.SLURM_FIELDS)}
    write_json(folder / "json" / run / "float32.json", row)
    write_json(folder / "json" / run / fetch.HOST_FILE, host(run, fetch.EXP013, f"run.py --resources sweep --precision float32 --run-name {run}"))
    (folder / "logs").mkdir(exist_ok=True)
    (folder / "logs" / f"{run}.log").write_text("log\n")
    return folder


def exp004_run(base: Path, run: str = "hugging-face-a100-mechanism-timing", logged: str = "0.900000") -> Path:
    folder = base / run
    seconds = {"rotated_posterior_pass": [0.0006, 0.0007], "early_optimizer": [0.9, 0.91], "fixed_optimizer": [1.39, 1.4],
               "enqueue_only": [0.9, 0.91]}
    receipt = {"smoke_cpu": False, "slurm_job_id": None, "repeats": 2, "seconds": seconds,
               "medians_seconds": {key: sum(values) / 2 for key, values in seconds.items()}}
    write_json(folder / "json" / run / "mechanism-timing.json", receipt)
    write_json(folder / "json" / run / fetch.HOST_FILE, host(run, fetch.EXP004, "src/modules/mechanism_timing.py --smoke-cpu", hugging_face_job=None))
    (folder / "logs").mkdir()
    lines = [f"[{i}/2] {title}: {value:.6f} s" for title, key in fetch.LOG_TITLES.items() for i, value in enumerate(seconds[key], 1)]
    lines[2] = f"[1/2] optimizer early stop: {logged} s"
    (folder / "logs" / f"{run}.log").write_text("\n".join(lines) + "\n")
    return folder


class Placement(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "project"
        self.results = self.root / "experiments" / fetch.EXP011 / ".results"
        self.run_name = "hugging-face-a100-per-refit-fold1"
        self.source = exp011_run(Path(self.tmp.name) / "fetched")

    def fetch(self, **kwargs):
        return fetch.fetch(kwargs.pop("run", self.run_name), kwargs.pop("job_id", JOB), kwargs.pop("source", self.source), self.root)

    def test_archive_links_flat_folder_and_record(self):
        notes = self.fetch()
        self.assertTrue(any("sha256" in note for note in notes))
        archive = self.results / "hugging-face" / self.run_name
        record = json.loads((archive / fetch.ARCHIVE_RECORD).read_text())
        self.assertEqual((record["run_name"], record["job_id"]), (self.run_name, JOB))
        self.assertEqual(sorted(record["files"]), sorted(str(p) for p in fetch.files_of(self.source)))
        merged = self.results / "arrays" / self.run_name / "refit-flags.npz"
        self.assertEqual(merged.stat().st_ino, (archive / "arrays" / self.run_name / "refit-flags.npz").stat().st_ino)
        self.assertFalse((self.results / fetch.ARCHIVE_RECORD).exists())
        flat = self.results / f"fetch-{self.run_name}"
        self.assertEqual(sorted(p.name for p in flat.iterdir()), sorted(["result.json", "refit-flags.npz", fetch.HOST_FILE, f"{self.run_name}.log"]))
        self.assertEqual(json.loads((flat / "result.json").read_text())["fold"], 1)

    def test_second_fetch_is_refused_and_changes_nothing(self):
        self.fetch()
        before = sorted(str(p) for p in self.results.rglob("*"))
        with self.assertRaisesRegex(fetch.FetchError, "already fetched"):
            self.fetch()
        self.assertEqual(sorted(str(p) for p in self.results.rglob("*")), before)

    def test_existing_file_with_other_content_stops_everything_first(self):
        clash = self.results / "arrays" / self.run_name / "refit-flags.npz"
        clash.parent.mkdir(parents=True)
        clash.write_bytes(b"someone else's bytes")
        with self.assertRaisesRegex(fetch.FetchError, "refusing to overwrite 1 existing files"):
            self.fetch()
        self.assertFalse((self.results / "hugging-face").exists())
        self.assertEqual(clash.read_bytes(), b"someone else's bytes")

    def test_wrong_job_id(self):
        with self.assertRaisesRegex(fetch.FetchError, "JOB_ID is"):
            self.fetch(job_id='87654321fedcba0987654321')
        self.assertFalse(self.results.exists())

    def test_run_that_did_not_finish_cleanly(self):
        for name, changes in (("exit code", {"exit_code": 1}), ("check-only", {"dry_run": True}),
                              ("slurm variables", {"slurm_variables": ["SLURM_JOB_ID"]}),
                              ("slurm environment", {"python": {"prefix": "test-environments/voxelwise-normative-jax"}}),
                              ("two GPUs", {"gpu": {"devices": A100["devices"] * 2}}),
                              ("another GPU", {"gpu": {"devices": [{"name": "NVIDIA L4", "driver_version": "1"}]}})):
            with self.subTest(name):
                source = exp011_run(Path(self.tmp.name) / name.replace(" ", "-"), self.run_name, **changes)
                with self.assertRaises(fetch.FetchError):
                    self.fetch(source=source)
        failed = exp011_run(Path(self.tmp.name) / "failed", self.run_name)
        (failed / "ENTRY-FAILED.txt").write_text("entry stopped with status 2")
        with self.assertRaisesRegex(fetch.FetchError, "entry script failed"):
            self.fetch(source=failed)
        self.assertFalse(self.results.exists())

    def test_receipt_that_does_not_match_its_files_or_the_host_record(self):
        (self.source / "arrays" / self.run_name / "refit-flags.npz").write_bytes(b"changed after the receipt")
        with self.assertRaisesRegex(fetch.FetchError, "does not match result.json"):
            self.fetch()
        marker = exp011_run(Path(self.tmp.name) / "marker", self.run_name, hugging_face_job=None)
        with self.assertRaisesRegex(fetch.FetchError, "hugging_face_job is"):
            self.fetch(source=marker)
        fold = exp011_run(Path(self.tmp.name) / "fold", self.run_name)
        write_json(fold / "json" / self.run_name / fetch.HOST_FILE, host(self.run_name, fetch.EXP011, f"run.py --per-refit --fold 2 --run-name {self.run_name}"))
        with self.assertRaisesRegex(fetch.FetchError, "receipt fold 1, arguments say 2"):
            self.fetch(source=fold)

    def test_environment_name_is_the_one_the_guards_expect(self):
        batch_job = importlib.import_module("shared.voxelwisenormative.batch_job")
        self.assertEqual(fetch.ENVIRONMENT, batch_job.HUGGING_FACE_ENVIRONMENT)

    def test_run_names(self):
        for name in ("a100-81589484-fold1", "hugging-face-a100-", "hugging-face-a100-X", "../hugging-face-a100-x"):
            with self.assertRaisesRegex(fetch.FetchError, "run name must look like"):
                fetch.fetch(name, None, self.source, self.root)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "the z-summary check reads npz files with numpy")
class ZSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "project"
        self.results = self.root / "experiments" / fetch.EXP011 / ".results"
        self.run_name = "hugging-face-a100-z-summary-fold0"

    def fetch(self, source: Path):
        return fetch.fetch(self.run_name, JOB, source, self.root)

    def test_every_place_gets_the_file_and_the_note_says_what_was_compared(self):
        notes = self.fetch(exp011_z_run(self.base / "ok"))
        self.assertTrue(any(note.startswith("z-summary.npz: float64 (4, 24) x 3, min <= max, refit_min_abs_z > 1.96 equals "
                                            "recurrence_counts == 64 in all 96 entries") for note in notes), notes)
        archive = self.results / "hugging-face" / self.run_name
        for place in (archive, self.results):
            self.assertTrue((place / "arrays" / self.run_name / "z-summary.npz").is_file(), place)
        self.assertTrue((self.results / f"fetch-{self.run_name}" / "z-summary.npz").is_file())
        self.assertEqual((self.results / "arrays" / self.run_name / "z-summary.npz").stat().st_ino,
                         (archive / "arrays" / self.run_name / "z-summary.npz").stat().st_ino)

    def test_a_summary_that_disagrees_with_the_counts_is_not_placed(self):
        import numpy as np

        ok = exp011_z_run(self.base / "ok")
        with np.load(ok / "arrays" / self.run_name / "z-summary.npz") as z:
            smallest = z["refit_min_abs_z"].copy()
        smallest[0, 0] = 1.0  # flagged in every refit by the counts, but its smallest |z| says otherwise
        bad = exp011_z_run(self.base / "bad", z_changes={"refit_min_abs_z": smallest})
        with self.assertRaisesRegex(fetch.FetchError, "differ in 1 entries"):
            self.fetch(bad)
        self.assertFalse(self.results.exists())

    def test_wrong_arrays_are_not_placed(self):
        import numpy as np

        ok = exp011_z_run(self.base / "ok")
        with np.load(ok / "arrays" / self.run_name / "z-summary.npz") as z:
            smallest = z["refit_min_abs_z"]
        cases = {"not float64": {"refit_min_abs_z": smallest.astype(np.float32)},
                 "holds a NaN": {"refit_max_abs_z": np.full(smallest.shape, np.nan)},
                 "above the largest": {"refit_max_abs_z": smallest - 0.5},
                 "different scans": {"held_out_cohort_rows_zero_based": np.arange(4, dtype=np.int32)}}
        for message, changes in cases.items():
            with self.subTest(message), self.assertRaisesRegex(fetch.FetchError, message):
                self.fetch(exp011_z_run(self.base / message.replace(" ", "-"), z_changes=changes))
        self.assertFalse(self.results.exists())

    def test_arguments_outputs_and_receipt_must_agree_about_the_option(self):
        source = exp011_z_run(self.base / "plain-arguments", arguments=f"run.py --per-refit --fold 0 --run-name {self.run_name} --resources")
        with self.assertRaisesRegex(fetch.FetchError, "--z-summary is False in the arguments, True in the outputs"):
            self.fetch(source)
        self.assertFalse(self.results.exists())

    def test_a_receipt_file_that_is_not_what_the_receipt_says(self):
        source = exp011_z_run(self.base / "changed")
        (source / "arrays" / self.run_name / "z-summary.npz").write_bytes(b"changed after the receipt")
        with self.assertRaisesRegex(fetch.FetchError, "z-summary.npz does not match result.json"):
            self.fetch(source)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "the cutoff check reads npz files with numpy")
class CutoffFlags(unittest.TestCase):
    CUTOFFS = (2.05, 3.0, 3.5)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "project"
        self.results = self.root / "experiments" / fetch.EXP011 / ".results"
        self.run_name = "hugging-face-a100-cutoff-flags-fold0"
        ok = exp011_z_run(self.base / "ok", self.run_name, cutoffs=self.CUTOFFS)
        import numpy as np

        with np.load(ok / "arrays" / self.run_name / "cutoff-flags.npz") as cut:
            self.stored = {key: cut[key].copy() for key in cut.files}

    def fetch(self, **changes):
        source = exp011_z_run(Path(tempfile.mkdtemp(dir=self.base)), self.run_name, cutoffs=self.CUTOFFS, **changes)
        return fetch.fetch(self.run_name, JOB, source, self.root)

    def bits(self, level: int):
        """The stored refit flags at cutoff `level` as booleans (refits, scans, voxels)."""
        import numpy as np

        return np.unpackbits(self.stored["refit_flags_packed"][level], axis=-1, count=24, bitorder="little").astype(bool)

    def flipped(self, level: int, refit: int, scan: int, voxel: int):
        flags = self.stored["refit_flags_packed"].copy()
        flags[level, refit, scan, voxel // 8] ^= 1 << (voxel % 8)
        return {"refit_flags_packed": flags}

    def test_the_file_is_placed_everywhere_and_the_note_says_what_was_compared(self):
        notes = self.fetch()
        note = next(note for note in notes if note.startswith("cutoff-flags.npz: 3 cutoffs 2.05,3,3.5, uint8 (3, 64, 4, 3)"))
        self.assertIn("nested in 1.96", note)
        archive = self.results / "hugging-face" / self.run_name
        for place in (archive, self.results):
            self.assertTrue((place / "arrays" / self.run_name / "cutoff-flags.npz").is_file(), place)
        self.assertTrue((self.results / f"fetch-{self.run_name}" / "cutoff-flags.npz").is_file())
        self.assertTrue((self.results / f"fetch-{self.run_name}" / "z-summary.npz").is_file())

    def test_a_flag_that_breaks_the_and_is_not_placed(self):
        import numpy as np

        scan, voxel = (int(i) for i in np.argwhere(self.bits(0).all(axis=0))[0])  # flagged in all 64 refits: clearing one changes the AND
        with self.assertRaisesRegex(fetch.FetchError, "at 2.05 the AND of the refits' flags is not refit_min_abs_z > 2.05"):
            self.fetch(cutoff_changes=self.flipped(0, 7, scan, voxel))
        self.assertFalse(self.results.exists())

    def test_a_flag_that_breaks_the_or_is_not_placed(self):
        import numpy as np

        scan, voxel = (int(i) for i in np.argwhere(~self.bits(2).any(axis=0))[0])  # flagged in no refit: setting one changes the OR
        with self.assertRaisesRegex(fetch.FetchError, "at 3.5 the OR of the refits' flags is not refit_max_abs_z > 3.5"):
            self.fetch(cutoff_changes=self.flipped(2, 4, scan, voxel))
        self.assertFalse(self.results.exists())

    def test_a_flag_that_only_the_nesting_sees_is_not_placed(self):
        """Set at 3.0 for one refit, not set at 2.05 for it, with the voxel's other refits keeping the AND and the OR as they were."""
        import numpy as np

        looser, stricter = self.bits(0), self.bits(1)
        candidates = np.argwhere(~looser & ~stricter & (stricter.sum(axis=0) > 0) & (stricter.sum(axis=0) < 63) & (looser.sum(axis=0) > 0))
        refit, scan, voxel = (int(i) for i in candidates[0])
        with self.assertRaisesRegex(fetch.FetchError, "cutoff 3 flags something its next looser cutoff does not"):
            self.fetch(cutoff_changes=self.flipped(1, refit, scan, voxel))
        self.assertFalse(self.results.exists())

    def test_the_single_fits_flags_must_be_those_of_original_z(self):
        single = self.stored["original_flags_packed"].copy()
        single[0, 0, 0] ^= 1
        with self.assertRaisesRegex(fetch.FetchError, "at 2.05 the single fit's flags are not abs\\(original_z\\) > 2.05"):
            self.fetch(cutoff_changes={"original_flags_packed": single})
        self.assertFalse(self.results.exists())

    def test_other_wrong_files_are_not_placed(self):
        import numpy as np

        stored = self.stored["refit_flags_packed"]
        cases = {"stored cutoffs": {"cutoffs": np.array([2.05, 3.0, 3.6])}, "flag arrays are": {"refit_flags_packed": stored[:, :63]},
                 "flag arrays are ": {"refit_flags_packed": stored.astype(np.int64)},
                 "hold different scans": {"held_out_cohort_rows_zero_based": np.arange(4, dtype=np.int32)}}
        for message, changes in cases.items():
            with self.subTest(message), self.assertRaisesRegex(fetch.FetchError, message.strip()):
                self.fetch(cutoff_changes=changes)
            self.setUp()
        self.assertFalse(self.results.exists())

    def test_arguments_outputs_and_receipt_must_agree_about_the_cutoffs(self):
        no_option = f"run.py --per-refit --fold 0 --z-summary --run-name {self.run_name} --resources"
        with self.assertRaisesRegex(fetch.FetchError, "--refit-flag-cutoffs is False in the arguments, True in the outputs and True in the receipt"):
            self.fetch(arguments=no_option)
        changed = f"run.py --per-refit --fold 0 --z-summary --refit-flag-cutoffs 2.05,3,3.6 --run-name {self.run_name} --resources"
        with self.assertRaisesRegex(fetch.FetchError, "the receipt's cutoffs \\[2.05, 3.0, 3.5\\] or its own check disagree"):
            self.fetch(arguments=changed)
        without_z = f"run.py --per-refit --fold 0 --refit-flag-cutoffs 2.05,3,3.5 --run-name {self.run_name} --resources"
        with self.assertRaisesRegex(fetch.FetchError, "--z-summary is False in the arguments"):
            self.fetch(arguments=without_z)
        self.assertFalse(self.results.exists())

    def test_a_run_without_cutoffs_is_checked_as_before(self):
        notes = fetch.fetch(self.run_name, JOB, exp011_z_run(self.base / "plain", self.run_name), self.root)
        self.assertFalse(any(note.startswith("cutoff-flags.npz") for note in notes))
        self.assertFalse((self.results / f"fetch-{self.run_name}" / "cutoff-flags.npz").exists())


class Experiments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "project"

    def fetch(self, run: str, source: Path):
        return fetch.fetch(run, None, source, self.root)

    def test_exp010_v2_receipts_match_the_log_and_land_where_collect_reads_them(self):
        run = "hugging-face-a100-bench-large-n50000-float32"
        notes = self.fetch(run, exp010_run(self.base / "ok"))
        self.assertTrue(any("v2 record" in note for note in notes))
        results = self.root / "experiments" / fetch.EXP010 / ".results"
        self.assertTrue((results / "json" / "gpu" / run / "synthetic-n50000-float32.json").is_file())
        self.assertFalse((results / f"fetch-{run}").exists(), "only exp_011 gets a flat folder")

    def test_exp010_log_that_differs_from_the_receipt(self):
        with self.assertRaisesRegex(fetch.FetchError, "log timings"):
            self.fetch("hugging-face-a100-bench-large-n50000-float32", exp010_run(self.base / "bad", log_timed="170.900"))
        self.assertFalse((self.root / "experiments").exists())

    def test_exp010_receipt_with_a_slurm_id(self):
        run = "hugging-face-a100-bench-large-n50000-float32"
        source = exp010_run(self.base / "slurm")
        path = source / "json" / "gpu" / run / "synthetic-n50000-float32.json"
        write_json(path, json.loads(path.read_text()) | {"slurm_job_id": "81580226"})
        with self.assertRaisesRegex(fetch.FetchError, "Slurm-free"):
            self.fetch(run, source)

    def test_exp013_sampler_must_have_ended_cleanly_and_left_its_samples(self):
        run = "hugging-face-a100-gpu-headroom-float32"
        notes = self.fetch(run, exp013_run(self.base / "ok"))
        self.assertTrue(any("'oom'" in note for note in notes))
        with self.assertRaisesRegex(fetch.FetchError, "sampler is missing or failed"):
            self.fetch(run, exp013_run(self.base / "stderr", sampler_stderr="Failed to initialize NVML"))
        with self.assertRaisesRegex(fetch.FetchError, "missing or empty"):
            self.fetch(run, exp013_run(self.base / "nosamples", with_samples=False))

    def test_exp004_log_against_receipt(self):
        run = "hugging-face-a100-mechanism-timing"
        self.fetch(run, exp004_run(self.base / "ok"))
        self.assertTrue((self.root / "experiments" / fetch.EXP004 / ".results" / "json" / run / "mechanism-timing.json").is_file())
        with self.assertRaisesRegex(fetch.FetchError, "log and receipt differ"):
            self.fetch(run, exp004_run(self.base / "bad", logged="0.912345"))

    def test_hard_links_survive_a_file_system_that_refuses_them(self):
        real, calls = os.link, []

        def refuse(source, target, *args, **kwargs):
            calls.append(target)
            raise OSError("cross-device link")

        os.link = refuse
        self.addCleanup(setattr, os, "link", real)
        self.fetch("hugging-face-a100-mechanism-timing", exp004_run(self.base / "ok"))
        results = self.root / "experiments" / fetch.EXP004 / ".results"
        self.assertTrue(calls)
        self.assertEqual((results / "json" / "hugging-face-a100-mechanism-timing" / "mechanism-timing.json").read_text(),
                         (results / "hugging-face" / "hugging-face-a100-mechanism-timing" / "json" / "hugging-face-a100-mechanism-timing"
                          / "mechanism-timing.json").read_text())


if __name__ == "__main__":
    unittest.main()
