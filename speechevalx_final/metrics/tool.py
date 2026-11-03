from __future__ import annotations

from typing import Any


def normalize_value(value: Any) -> Any:
    """Normalize only unambiguous scalar representations (for example 5 and '5')."""
    if isinstance(value, str):
        value = value.strip()
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    if isinstance(value, dict):
        return {key: normalize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    return value
