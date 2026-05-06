"""FastSAM adapter placeholder.

The package is structured for FastSAM integration, but the repository does not
yet declare a FastSAM dependency or checkpoint format. Add concrete loading and
inference functions here when FastSAM is added to the project dependencies.
"""

from __future__ import annotations


def raise_fastsam_not_configured() -> None:
    """Raise a clear error for unfinished FastSAM workflows."""
    raise NotImplementedError("FastSAM support is not configured in this repository yet.")
