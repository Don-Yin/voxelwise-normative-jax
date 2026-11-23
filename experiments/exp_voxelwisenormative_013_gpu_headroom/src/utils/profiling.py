"""default-on run profiling helpers, scoped to a run rather than to an import."""

from __future__ import annotations

import logging
import resource
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def _rss_mb(usage: resource.struct_rusage) -> float:
    """ru_maxrss as mebibytes (linux kilobytes, darwin bytes)."""
    raw = usage.ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


@contextmanager
def run_resources(*, enabled: bool = True) -> Iterator[None]:
    """log wall/cpu interval and peak RSS for the wrapped block; no-op when disabled."""
    if not enabled:
        yield
        return
    wall0 = time.perf_counter()
    self0 = resource.getrusage(resource.RUSAGE_SELF)
    child0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        yield
    finally:
        self1 = resource.getrusage(resource.RUSAGE_SELF)
        child1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        user = (self1.ru_utime - self0.ru_utime) + (child1.ru_utime - child0.ru_utime)
        system = (self1.ru_stime - self0.ru_stime) + (child1.ru_stime - child0.ru_stime)
        logger.info("run resources: wall=%.2fs cpu=%.2fs (user=%.2fs sys=%.2fs) peak_rss_self=%.1fMiB "
                    "peak_rss_children=%.1fMiB", time.perf_counter() - wall0, user + system, user, system,
                    _rss_mb(self1), _rss_mb(child1))


@contextmanager
def profiled(out: Path | None) -> Iterator[None]:
    """profile the wrapped block and write an html report to out; no-op when out is None."""
    if out is None:
        yield
        return
    # lazy: importing this module never starts or needs a profiler
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()
    try:
        yield
    finally:
        profiler.stop()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(profiler.output_html())
        logger.info("wrote profile to %s", out)
