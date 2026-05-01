"""
Helpers for importing the optional Google Gemini SDK.
"""
from __future__ import annotations

import sys
from typing import Any, Tuple


def load_google_genai() -> Tuple[Any, Any]:
    """
    Import ``google-genai`` lazily and raise a helpful error if unavailable.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Google Gemini SDK is not installed for the active Python "
            f"interpreter ({sys.executable}). Install dependencies with "
            "`python -m pip install -r requirements.txt` or install "
            "`google-genai` directly."
        ) from exc
    return genai, types
