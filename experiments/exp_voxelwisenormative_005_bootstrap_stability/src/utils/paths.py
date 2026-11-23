"""single source of truth for this experiment's filesystem paths."""

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS = EXPERIMENT_DIR / ".results"


def results_dir(*parts: str) -> Path:
    """return a nested .results/ subdirectory, creating it if needed."""
    target = RESULTS.joinpath(*parts)
    target.mkdir(parents=True, exist_ok=True)
    return target
