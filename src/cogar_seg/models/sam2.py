"""SAM2 adapter placeholder.

The package is structured for SAM2 integration, but the repository does not yet
declare a SAM2 dependency or checkpoint format. Add concrete loading and
inference functions here when SAM2 is added to the project dependencies.
"""

from __future__ import annotations


def raise_sam2_not_configured() -> None:
    """Raise a clear error for unfinished SAM2 workflows."""
    raise NotImplementedError("SAM2 support is not configured in this repository yet.")
