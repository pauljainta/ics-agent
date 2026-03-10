"""PLC adapter interface.

This is intentionally read-only by default.
"""


def read_tags() -> list[dict]:
    """Return placeholder PLC tag data."""
    return []
