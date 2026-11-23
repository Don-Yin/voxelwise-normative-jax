"""Unit tests for the independent control comparison's command and numerical primitives."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from src.modules.compare import correlation


def test_requires_independent_directory():
    folder = Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, '-B', str(folder / 'run.py'), 'compare', '--run-dir', 'unread'],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 2
    assert '--comparison-dir' in done.stderr


def test_constant_correlation_refused():
    with np.errstate(invalid='ignore', divide='ignore'), pytest.raises(ValueError, match='constant vector'):
        correlation(np.ones(4), np.arange(4.0))


def test_pearson_correlation():
    x = np.arange(4.0)
    assert correlation(x, 2 * x + 1) == pytest.approx(1)
    assert correlation(x, -x) == pytest.approx(-1)
