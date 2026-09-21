"""eNRRCrew upgraded application package."""

import os

# joblib's Windows physical-core probe depends on the optional `wmic` executable.
# Use a conservative physical-core estimate while preserving an explicit user value.
_logical_cores = os.cpu_count() or 1
_default_worker_limit = max(1, _logical_cores // 2)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_default_worker_limit))

__version__ = "0.5.1"
