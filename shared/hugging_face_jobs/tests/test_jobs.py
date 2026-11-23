"""Checks of the printed hf commands: nothing after the image collides with an hf option, the numbers match the Slurm scripts."""

from __future__ import annotations

import importlib
import re
import shlex
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT)]

jobs = importlib.import_module("shared.hugging_face_jobs.jobs")


def words(command: str) -> list[str]:
    """The shell words of a printed command, line continuations joined."""
    return shlex.split(command.replace("\\\n", " "))


def after_image(command: str) -> list[str]:
    """What the container receives: the words after the image."""
    tokens = words(command)
    return tokens[tokens.index(jobs.IMAGE) + 1 :]


def slurm_minutes(script: str) -> int:
    """The `#SBATCH --time=HH:MM:SS` of a Slurm script, in minutes."""
    match = re.search(r"^#SBATCH\s+--time=(\d+):(\d+):(\d+)", (ROOT / script).read_text(), re.MULTILINE)
    assert match, script
    hours, minutes, seconds = map(int, match.groups())
    return hours * 60 + minutes + (seconds > 0)


class Commands(unittest.TestCase):
    """Every command the module prints."""

    def test_no_word_after_the_image_is_an_hf_option(self):
        """`hf jobs run` takes its own options anywhere in the line, so `--dry-run` for the container would be swallowed."""
        for command in [job.command() for job in jobs.JOBS] + [jobs.check_command(*check) for check in jobs.CHECKS]:
            clash = jobs.HF_OPTIONS.intersection(after_image(command))
            self.assertFalse(clash, f"{clash} would be taken by hf, not passed to the container: {command}")

    def test_the_check_only_flag_reaches_the_container(self):
        for check in jobs.CHECKS:
            self.assertIn("--check-only", after_image(jobs.check_command(*check)))
        self.assertNotIn("--check-only", [w for job in jobs.JOBS for w in after_image(job.command())])

    def test_options_before_the_image(self):
        for job in jobs.JOBS:
            tokens = words(job.command())
            before = tokens[: tokens.index(jobs.IMAGE)]
            self.assertEqual(before[before.index("--flavor") + 1], jobs.FLAVOR)
            self.assertEqual(before[before.index("--timeout") + 1], f"{job.timeout_minutes}m")
            self.assertEqual(before[before.index("--name") + 1], job.name)
            self.assertIn("-d", before)
            self.assertEqual(before.count("-v"), 2, "the code mount and the /out bucket, nothing else")
            self.assertNotIn(".results", " ".join(before))

    def test_run_names(self):
        names = [job.name for job in jobs.JOBS] + [name for name, _, _ in jobs.CHECKS]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertTrue(name.startswith("hugging-face-a100-") and re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) and len(name) <= 64)
        for job in jobs.JOBS:
            tokens = after_image(job.command())
            self.assertEqual(tokens[tokens.index("--run-name") + 1], job.name)

    def test_the_z_summary_jobs_are_one_per_fold_and_cost_at_most_1_25_dollars_each(self):
        z_jobs = [job for job in jobs.JOBS if "--z-summary" in job.args and "--refit-flag-cutoffs" not in job.args]
        self.assertEqual([job.name for job in z_jobs], [f"hugging-face-a100-z-summary-fold{fold}" for fold in range(5)])
        for fold, job in enumerate(z_jobs):
            arguments = after_image(job.command())
            self.assertEqual(arguments[arguments.index("--fold") + 1], str(fold))
            self.assertIn("--per-refit", arguments)
            self.assertEqual((job.timeout_minutes, round(job.max_cost, 2), jobs.INPUT_OF[job.experiment]), (30, 1.25, "IXI"))

    def test_the_cutoff_flag_jobs_are_one_per_fold_with_the_slurm_scripts_cutoffs_and_cost_at_most_6_25_dollars(self):
        cutoff_jobs = [job for job in jobs.JOBS if "--refit-flag-cutoffs" in job.args]
        self.assertEqual([job.name for job in cutoff_jobs], [f"hugging-face-a100-cutoff-flags-fold{fold}" for fold in range(5)])
        script = (ROOT / cutoff_jobs[0].slurm_script).read_text()
        self.assertIn(f"--refit-flag-cutoffs {jobs.CUTOFFS} ", script)
        for fold, job in enumerate(cutoff_jobs):
            arguments = after_image(job.command())
            self.assertEqual(arguments[arguments.index("--fold") + 1], str(fold))
            self.assertEqual(arguments[arguments.index("--refit-flag-cutoffs") + 1], jobs.CUTOFFS)
            self.assertTrue({"--per-refit", "--z-summary"} <= set(arguments))
            self.assertEqual((job.timeout_minutes, round(job.max_cost, 2), jobs.INPUT_OF[job.experiment]), (30, 1.25, "IXI"))
        text = jobs.render("SYN", "IXI", only="cutoff-flags")
        self.assertIn("total, 5 a100-large jobs at their timeouts: $6.25, plus 1 cpu-basic checks", text)

    def test_only_keeps_the_checks_and_jobs_it_names(self):
        text = jobs.render("SYN", "IXI", only="z-summary")
        self.assertEqual((text.count(f"--flavor {jobs.FLAVOR}"), text.count(f"--flavor {jobs.CPU_FLAVOR}")), (5, 1))
        self.assertIn("total, 5 a100-large jobs at their timeouts: $6.25, plus 1 cpu-basic checks", text)
        everything = jobs.render("SYN", "IXI")
        self.assertEqual((everything.count(f"--flavor {jobs.FLAVOR}"), everything.count(f"--flavor {jobs.CPU_FLAVOR}")),
                         (len(jobs.JOBS), len(jobs.CHECKS)))

    def test_timeout_is_the_slurm_limit_plus_ten_minutes(self):
        for job in jobs.JOBS:
            self.assertEqual(job.slurm_minutes, slurm_minutes(job.slurm_script), job.name)
            self.assertEqual(job.timeout_minutes, job.slurm_minutes + 10)
            self.assertLessEqual(job.max_cost, job.timeout_minutes / 60 * 2.5 + 1e-9)

    def test_arguments_are_those_of_the_slurm_script(self):
        """Every flag and subcommand of a row is in the script it names (or the runtime.sh that script sources)."""
        for job in jobs.JOBS:
            script = ROOT / job.slurm_script
            text = script.read_text() + "".join(p.read_text() for p in script.parent.glob("runtime.sh"))
            for word in (a for a in job.args if re.fullmatch(r"--?[a-z][a-z-]*|[a-z]+", a)):
                self.assertIn(word, text, f"{job.name}: {word} is in neither {job.slurm_script} nor its runtime.sh")


class Requirements(unittest.TestCase):
    """requirements-a100.txt: the Slurm voxelwise-normative-jax versions, as the entry script reads them."""

    def pins(self) -> dict[str, str]:
        lines = [re.sub(r"\s*#.*", "", line) for line in jobs.REQUIREMENTS.read_text().splitlines() if not line.lstrip().startswith("#")]
        return {m.group(1).lower().replace("_", "-"): m.group(2) for line in lines if (m := re.fullmatch(r"([A-Za-z0-9_.\[\]-]+)==(\S+)", line.strip()))}

    def test_versions_reported_from_slurm_py314(self):
        """Source: dist-info names in .../conda-envs/voxelwise-normative-jax/lib/python3.14/site-packages, listed 2026-11-21 21:34."""
        expected = {"numpy": "2.5.3", "nibabel": "5.4.2", "scipy": "1.18.1", "jax[cuda12]": "0.12.1", "jaxlib": "0.12.1",
                    "jax-cuda12-plugin": "0.12.1", "jax-cuda12-pjrt": "0.12.1", "ml-dtypes": "0.7.0", "opt-einsum": "3.4.0",
                    "nvidia-cublas-cu12": "12.9.3.13", "nvidia-cuda-runtime-cu12": "12.9.79", "nvidia-cudnn-cu12": "9.28.1.4",
                    "nvidia-cusolver-cu12": "11.7.5.82"}
        self.assertEqual({name: version for name, version in self.pins().items() if name in expected}, expected)

    def test_the_other_ten_are_those_of_the_first_hugging_face_job(self):
        """Source C: json/<run>/pip-freeze.txt of hugging-face-a100-mechanism-timing, 2026-11-21 22:03 (23 packages besides pip)."""
        extras = {"nvidia-cuda-cccl-cu12": "12.9.27", "nvidia-cuda-cupti-cu12": "12.9.79", "nvidia-cuda-nvcc-cu12": "12.9.86",
                  "nvidia-cuda-nvrtc-cu12": "12.9.86", "nvidia-cufft-cu12": "11.4.1.4", "nvidia-cusparse-cu12": "12.5.10.65",
                  "nvidia-nccl-cu12": "2.32.6", "nvidia-nvjitlink-cu12": "12.9.86", "nvidia-nvshmem-cu12": "3.8.0",
                  "packaging": "26.3"}
        pins = self.pins()
        self.assertEqual({name: pins.get(name) for name in extras}, extras)
        self.assertEqual(len(pins), 23)

    def test_nothing_is_unresolved_so_no_command_allows_unpinned(self):
        self.assertEqual(jobs.unresolved(), set())
        for job in jobs.JOBS:
            self.assertNotIn("--allow-unpinned", job.command())


if __name__ == "__main__":
    unittest.main()
