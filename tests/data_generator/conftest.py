"""Path setup for data_generator tests.

The parent `tests/conftest.py` already adds `src/` to sys.path. We add the
repo root here so `scripts/migrate_pairs_to_v1.py` can be imported.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
