"""Approval state of registered model versions (shared by serving and governance).

A version may be served or rolled back to only if it is APPROVED:

* ``approval_status = approved`` tag, written by ``fraud-detection promote`` together with
  who approved it, when and why; or
* a *legacy release*: the frozen Phase 2 model registered by ``track-frozen``
  (tag ``source = frozen_snapshot``) before this approval workflow existed. It is
  reported as "approved (legacy release)" - no approval record is invented for it.
"""

from __future__ import annotations

APPROVED = "approved"
PENDING = "pending"
LEGACY = "approved (legacy release)"


def approval_state(tags: dict) -> str:
    status = tags.get("approval_status")
    if status:
        return status
    if tags.get("source") == "frozen_snapshot":
        return LEGACY
    return PENDING


def is_approved(tags: dict) -> bool:
    return approval_state(tags) in (APPROVED, LEGACY)
