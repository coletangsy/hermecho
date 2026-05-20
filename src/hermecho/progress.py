"""Structured progress events for machine consumers."""
from __future__ import annotations

import json
from typing import Any

PROGRESS_PREFIX = "HERMECHO_PROGRESS "


def emit_progress(
    stage: str,
    status: str,
    message: str,
    **fields: Any,
) -> None:
    """Print one machine-readable progress event without replacing human logs."""
    event = {
        "stage": stage,
        "status": status,
        "message": message,
    }
    event.update({key: value for key, value in fields.items() if value is not None})
    print(f"{PROGRESS_PREFIX}{json.dumps(event, ensure_ascii=False)}")
