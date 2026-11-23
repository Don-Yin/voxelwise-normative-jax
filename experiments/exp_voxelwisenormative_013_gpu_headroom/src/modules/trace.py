"""JAX profiler trace of one chunk call, and the device-stream busy fraction read from its perfetto export."""

from __future__ import annotations

import gzip
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .batched import Batcher

logger = logging.getLogger(__name__)
GAP_US = 20.0  # idle gaps longer than this are launch or host-synchronisation stalls, not back-to-back kernels


def capture(batcher: Batcher, k: int, results: Path, relative: Path) -> dict:
    """Trace the first voxel chunk for draws 0..K-1 with an already compiled executable."""
    import jax

    folder = results / relative
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with jax.profiler.trace(str(folder), create_perfetto_trace=True):
        jax.block_until_ready(batcher.dispatch(np.arange(k), [0]))
    seconds = time.perf_counter() - start
    files = sorted(folder.glob("plugins/profile/*/perfetto_trace.json.gz"))
    if len(files) != 1:
        raise RuntimeError(f"expected one perfetto trace under {folder}, found {files}")
    lo, hi = batcher.bounds[0]
    logger.info("traced k=%d chunk 0 (%d voxels) in %.2fs including the profiler: %s", k, hi - lo, seconds, files[0])
    return {"k": k, "chunk_index": 0, "width": hi - lo, "seconds_with_profiler": seconds,
            "perfetto": str(files[0].relative_to(results))}


def union(spans: list[tuple[float, float]]) -> tuple[float, np.ndarray]:
    """Busy microseconds of sorted intervals and the idle gaps between them."""
    busy, gaps = 0.0, []
    start, end = spans[0]
    for s, e in spans[1:]:
        if s > end:
            busy += end - start
            gaps.append(s - end)
            start, end = s, e
        else:
            end = max(end, e)
    return busy + end - start, np.asarray(gaps)


def device_busy(path: Path) -> dict:
    """Union of device stream activity over its own span; also lists every device thread found."""
    with gzip.open(path, "rt") as handle:
        data = json.load(handle)
    events = data["traceEvents"] if isinstance(data, dict) else data
    meta = [e for e in events if e.get("ph") == "M"]
    processes = {e["pid"]: e["args"]["name"] for e in meta if e.get("name") == "process_name"}
    threads = {(e["pid"], e["tid"]): e["args"]["name"] for e in meta if e.get("name") == "thread_name"}
    devices = {pid for pid, name in processes.items() if name.startswith("/device:GPU")}
    by_thread: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and e.get("pid") in devices:
            ts = float(e["ts"])
            by_thread[threads.get((e["pid"], e["tid"]), f"tid {e['tid']}")].append((ts, ts + float(e.get("dur", 0.0))))
    report = {"device_threads": {name: len(spans) for name, spans in sorted(by_thread.items())}}
    spans = sorted(s for name, rows in by_thread.items() if "Stream" in name for s in rows)
    if not spans:
        return {**report, "stream_events": 0}
    busy, gaps = union(spans)
    span = max(e for _, e in spans) - spans[0][0]
    long = gaps[gaps > GAP_US]
    return {**report, "stream_events": len(spans), "span_us": span, "busy_us": busy, "busy_fraction": busy / span,
            "gaps": int(gaps.size), "gaps_over_20us": int(long.size), "gap_us_over_20us_total": float(long.sum()),
            "gap_us_median": float(np.median(gaps)) if gaps.size else 0.0}
