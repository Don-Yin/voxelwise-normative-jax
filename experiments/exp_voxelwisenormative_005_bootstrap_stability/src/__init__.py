"""Import shared fitting code from the owning project, not from the caller's cwd."""

import sys

from .utils.paths import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
