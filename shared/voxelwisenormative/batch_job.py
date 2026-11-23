"""Which batch job may open real IXI input: a Slurm Slurm job, or a Hugging Face job that carries the marker.

Every guard of real IXI input calls `require_batch_job`. It accepts exactly one of

- Linux and `SLURM_JOB_ID` a decimal string: a Slurm Slurm job;
- Linux and `HUGGING_FACE_JOB_ID` a 24-character lowercase hexadecimal string, the form of a Hugging Face job id:
  `shared/hugging_face_jobs/container_entry.sh` sets it, from the platform's own `JOB_ID`, and nothing else does.

A Mac, a Linux shell with neither variable, a malformed value and both variables together are refused. A Hugging Face
job never sets `SLURM_JOB_ID`: its receipts leave every Slurm field empty and record the id under `hugging_face_job`.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass

SLURM_VARIABLE = "SLURM_JOB_ID"
HUGGING_FACE_VARIABLE = "HUGGING_FACE_JOB_ID"
HUGGING_FACE_ID = re.compile(r"[0-9a-f]{24}")
SLURM_ENVIRONMENT = "voxelwise-normative-jax"
HUGGING_FACE_ENVIRONMENT = "hugging-face-a100-venv"  # the virtual environment container_entry.sh builds


@dataclass(frozen=True)
class BatchJob:
    """The batch job running this process: its platform ("slurm" or "hugging_face") and that platform's id for it."""

    platform: str
    job_id: str


def batch_job(env: Mapping[str, str] | None = None, platform_name: str | None = None) -> BatchJob | None:
    """The running batch job, or None wherever real IXI input must not be opened."""
    env = os.environ if env is None else env
    platform_name = sys.platform if platform_name is None else platform_name
    slurm, hugging_face = env.get(SLURM_VARIABLE, ""), env.get(HUGGING_FACE_VARIABLE, "")
    if platform_name != "linux" or (slurm and hugging_face):
        return None
    if slurm.isascii() and slurm.isdecimal():
        return BatchJob("slurm", slurm)
    if HUGGING_FACE_ID.fullmatch(hugging_face):
        return BatchJob("hugging_face", hugging_face)
    return None


def require_batch_job(what: str, env: Mapping[str, str] | None = None, platform_name: str | None = None) -> BatchJob:
    """The running batch job; RuntimeError naming `what` and what was found where the input must not be opened."""
    job = batch_job(env, platform_name)
    if job is None:
        env = os.environ if env is None else env
        found = f"platform {sys.platform if platform_name is None else platform_name!r}, " + ", ".join(
            f"{name}={env.get(name)!r:.40}" for name in (SLURM_VARIABLE, HUGGING_FACE_VARIABLE))
        raise RuntimeError(f"{what} can only be opened by a Linux Slurm job ({SLURM_VARIABLE} a decimal number) or a "
                           f"Hugging Face job ({HUGGING_FACE_VARIABLE} 24 hexadecimal characters, never together with "
                           f"{SLURM_VARIABLE}); found {found}; locally use the synthetic fixture")
    return job


def hugging_face_job(env: Mapping[str, str] | None = None, platform_name: str | None = None) -> str | None:
    """The id for a receipt's `hugging_face_job` key; None outside a valid Hugging Face job."""
    job = batch_job(env, platform_name)
    return job.job_id if job is not None and job.platform == "hugging_face" else None


def expected_environment(env: Mapping[str, str] | None = None, platform_name: str | None = None) -> str:
    """The interpreter environment name exp_011's run.py accepts: container_entry.sh's in a Hugging Face job, else voxelwise-normative-jax.

    A marker variable that is set but does not identify a Hugging Face job (malformed, a Mac, or Slurm as well) is an error.
    """
    env = os.environ if env is None else env
    if env.get(HUGGING_FACE_VARIABLE):
        if hugging_face_job(env, platform_name) is None:
            raise ValueError(f"{HUGGING_FACE_VARIABLE} is set but does not identify a Hugging Face job: it needs Linux, "
                             f"24 lowercase hexadecimal characters and no {SLURM_VARIABLE}")
        return HUGGING_FACE_ENVIRONMENT
    return SLURM_ENVIRONMENT
