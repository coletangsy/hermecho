"""
OpenRouter fallbacks when Gemini 3.1 preview models error or return no body.

Hermecho tries the user-selected model first, then a single OpenAI GPT
backup on OpenRouter for known Gemini 3.1 Pro / Flash family slugs.
"""
from __future__ import annotations

from typing import List, Optional

# Match OpenRouter model ids the project defaults to for Pro vs Flash roles.
# Pro + audio (multimodal transcribe): GPT Audio; Pro + text-only paths: 5.4.
_FALLBACK_GEMINI_31_PRO_MULTIMODAL = "openai/gpt-audio"
_FALLBACK_GEMINI_31_PRO = "openai/gpt-5.4"
_FALLBACK_GEMINI_31_FLASH = "openai/gpt-5.4-mini"


def _normalized(model_id: str) -> str:
    """Lowercase trimmed id for substring checks."""
    return model_id.strip().lower()


def openrouter_fallback_model(
    model_id: str,
    *,
    multimodal_transcription: bool = False,
) -> Optional[str]:
    """
    Return a backup OpenRouter model for Gemini 3.1 slugs, else None.

    Flash-family ids (including ``flash-lite``) map to ``openai/gpt-5.4-mini``.
    Pro-family ids map to ``openai/gpt-audio`` when ``multimodal_transcription``
    is True (audio in / JSON segments out), else ``openai/gpt-5.4`` for
    text-only API use. Other models get no fallback.

    Args:
        model_id: OpenRouter model slug (e.g. ``google/gemini-3.1-pro-preview``).
        multimodal_transcription: Use audio-capable Pro fallback (transcribe).

    Returns:
        Fallback slug, or ``None`` when none is configured for this primary.
    """
    m = _normalized(model_id)
    if "gemini-3.1" not in m:
        return None
    if "flash" in m:
        return _FALLBACK_GEMINI_31_FLASH
    if "pro" in m:
        if multimodal_transcription:
            return _FALLBACK_GEMINI_31_PRO_MULTIMODAL
        return _FALLBACK_GEMINI_31_PRO
    return None


def openrouter_models_with_fallback(
    model_id: str,
    *,
    multimodal_transcription: bool = False,
) -> List[str]:
    """
    Build an ordered list: primary model, then optional fallback (deduped).

    Args:
        model_id: User- or default-selected OpenRouter model slug.
        multimodal_transcription: Forwarded to ``openrouter_fallback_model``.

    Returns:
        One or two model ids to try in order until one succeeds.
    """
    primary = model_id.strip()
    fb = openrouter_fallback_model(
        primary,
        multimodal_transcription=multimodal_transcription,
    )
    if not fb or fb == primary:
        return [primary]
    return [primary, fb]
