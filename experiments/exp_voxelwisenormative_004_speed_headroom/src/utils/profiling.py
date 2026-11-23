"""default-on run profiling helpers, scoped to a run rather than to an import."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def profiled(out: Path | None) -> Iterator[None]:
    """profile the wrapped block and write an html report to out; no-op when out is None."""
    if out is None:
        yield
        return

    # imported lazily so an experiment that never profiles carries no pyinstrument dependency
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

