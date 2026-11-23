"""100 ms nvidia-smi samples of the allocated GPU, joined later to each pass's device windows.

slurm-profile samples GPUs every 30 s and only through PyTorch; a pass here lasts seconds to minutes, so the
utilisation per K is this experiment's own measurand. The run stores raw samples; `collect` summarises them.
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Self

import numpy as np

logger = logging.getLogger(__name__)
FIELDS = ("timestamp", "utilization.gpu", "utilization.memory", "memory.used", "power.draw", "clocks.sm",
          "temperature.gpu")
PERIOD_MS = 100


def query(arguments: list[str]) -> str:
    """One bounded nvidia-smi CSV query."""
    return subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader"], capture_output=True, text=True,
                          check=True, timeout=60).stdout.strip()


class Sampler:
    """Run `nvidia-smi -lms` on the one visible GPU; record, never raise, how the sampler ended."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.report: dict = {}

    def __enter__(self) -> Self:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or "," in visible:
            raise RuntimeError(f"sampling needs exactly one CUDA_VISIBLE_DEVICES entry, got {visible!r}")
        listed = query(["--query-gpu=uuid"]).splitlines()
        # A device cgroup shows only the allocated GPU; otherwise CUDA_VISIBLE_DEVICES names it on the node.
        selector = listed[0].strip() if len(listed) == 1 else visible
        identity = query(["--query-gpu=index,uuid,name", "-i", selector])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("x")
        self.process = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits",
             "-lms", str(PERIOD_MS), "-i", selector], stdout=self.handle, stderr=subprocess.PIPE, text=True)
        offset = datetime.now().astimezone().utcoffset()
        self.report = {"cuda_visible_devices": visible, "gpus_listed": len(listed), "selector": selector,
                       "identity": identity, "period_ms": PERIOD_MS, "fields": list(FIELDS),
                       "utc_offset_seconds": offset.total_seconds() if offset else 0.0}
        logger.info("sampling %s every %d ms into %s", identity, PERIOD_MS, self.path)
        return self

    def __exit__(self, *_exc) -> bool:
        early = self.process.poll()
        self.process.terminate()
        try:
            _, err = self.process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            _, err = self.process.communicate()
        self.handle.close()
        self.report.update({"exit_before_stop": early, "returncode": self.process.returncode,
                            "stderr": (err or "").strip()[-2000:]})
        return False


def read_samples(path: Path, utc_offset_seconds: float) -> dict[str, np.ndarray]:
    """Parse samples to epoch seconds and floats; only a final line cut by SIGTERM may be dropped."""
    tz = timezone(timedelta(seconds=utc_offset_seconds))
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if rows and len(rows[-1]) != len(FIELDS):
        logger.info("dropping a final partial sample line in %s: %s", path, rows[-1])
        rows = rows[:-1]
    columns: dict[str, list] = {field: [] for field in FIELDS}
    for line, row in enumerate(rows, 1):
        if len(row) != len(FIELDS):
            raise ValueError(f"{path}:{line}: expected {len(FIELDS)} fields, got {row}")
        for field, value in zip(FIELDS, row, strict=True):
            columns[field].append(value.strip())
    out = {field: np.asarray(columns[field], dtype=float) for field in FIELDS[1:]}
    out["epoch"] = np.asarray([datetime.strptime(s, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=tz).timestamp()
                               for s in columns["timestamp"]])
    return out


def summarize(samples: dict[str, np.ndarray], windows: list[list[float]]) -> dict:
    """Utilisation, power and clocks over the samples that fall inside any of the windows."""
    epoch = samples["epoch"]
    inside = np.zeros(epoch.shape, dtype=bool)
    for start, end in windows:
        inside |= (epoch >= start) & (epoch <= end)
    if not inside.any():
        return {"samples": 0}
    util, power = samples["utilization.gpu"][inside], samples["power.draw"][inside]
    return {"samples": int(inside.sum()), "gpu_util_mean": float(util.mean()), "gpu_util_median": float(np.median(util)),
            "gpu_util_p10": float(np.quantile(util, 0.1)),
            "memory_controller_util_mean": float(samples["utilization.memory"][inside].mean()),
            "power_w_mean": float(power.mean()), "power_w_max": float(power.max()),
            "sm_clock_mhz_median": float(np.median(samples["clocks.sm"][inside])),
            "memory_used_mib_max": float(samples["memory.used"][inside].max())}
