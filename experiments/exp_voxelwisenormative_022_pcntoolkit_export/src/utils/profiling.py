"""Default-on run profiling, scoped to a run rather than to an import."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def profiled(out: Path | None) -> Iterator[None]:
    """Profile the wrapped block and report even on failure, unless explicitly disabled."""
    if out is None:
        yield
        return

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
