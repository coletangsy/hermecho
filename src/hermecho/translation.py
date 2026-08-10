"""
This module contains functions for translating text using OpenRouter.
"""
import hashlib
import inspect
import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from .progress import emit_progress
from .prompts import build_translation_prompt


# Constants for the sliding window approach
# Approximate character threshold for a single translation request.
TOKEN_THRESHOLD = 128000  # Max characters to send in a single prompt
CHUNK_SIZE = 200          # Number of segments per chunk, increased for better performance
OVERLAP_SIZE = 3         # Number of segments to overlap

_MAX_TRANSLATION_RETRIES = 2


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_PROVIDER_ROUTING = {
    "order": ["alibaba", "atlas-cloud/fp8"],
    "allow_fallbacks": True,
    "require_parameters": True,
}


def translation_prompt_fingerprint() -> str:
    """Return a fingerprint that changes whenever the translation prompt changes."""
    try:
        source = inspect.getsource(build_translation_prompt)
    except (OSError, TypeError):
        code = build_translation_prompt.__code__
        source = repr((code.co_code, code.co_consts))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


class _JSONObject(dict):
    """JSON object that retains duplicate keys for Translation Gate validation."""

    def __init__(self, pairs: List[Tuple[str, Any]]) -> None:
        super().__init__()
        self.duplicate_keys: List[str] = []
        for key, value in pairs:
            if key in self:
                self.duplicate_keys.append(key)
            self[key] = value


def _translation_retry_delay(attempt: int) -> float:
    delay = min(2.5 * (2 ** attempt), 120.0)
    return max(0.0, delay * (1.0 + random.uniform(-0.25, 0.25)))


def _make_openrouter_client() -> Any:
    """Create an OpenRouter client using the OPENROUTER_API_KEY environment variable."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "`openai` is required for translation. Install project dependencies "
            "with `python -m pip install -e .`."
        ) from exc
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def _merge_api_usage_tokens(
    totals: Dict[str, int],
    usage: Optional[Dict[str, Any]],
) -> None:
    """Accumulate token usage counts into totals."""
    if not usage or not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = usage.get(key)
        if val is not None:
            totals[key] = totals.get(key, 0) + int(val)


def _log_translation_api_tokens(
    label: str,
    usage: Optional[Dict[str, Any]],
) -> None:
    """Print token usage from an OpenAI-compatible chat completion response."""
    if not usage:
        print(f"{label}: (no usage metadata)")
        return
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    tt = usage.get("total_tokens")
    parts = []
    if pt is not None:
        parts.append(f"prompt_tokens={pt}")
    if ct is not None:
        parts.append(f"completion_tokens={ct}")
    if tt is not None:
        parts.append(f"total_tokens={tt}")
    if parts:
        print(f"{label}: " + ", ".join(parts))
    else:
        print(f"{label}: usage={usage!r}")


def _read_usage_field(usage: Any, key: str) -> Optional[int]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    return int(value) if value is not None else None


def _usage_from_openai_response(response: Any) -> Optional[Dict[str, Any]]:
    """
    Extract token usage from an OpenAI-compatible chat completion response.

    Maps OpenAI usage fields to a canonical dict with prompt_tokens,
    completion_tokens, and total_tokens.
    """
    if response is None:
        return None
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    out: Dict[str, Any] = {}
    pt = _read_usage_field(usage, "prompt_tokens")
    ct = _read_usage_field(usage, "completion_tokens")
    tt = _read_usage_field(usage, "total_tokens")
    if pt is not None:
        out["prompt_tokens"] = pt
    if ct is not None:
        out["completion_tokens"] = ct
    if tt is not None:
        out["total_tokens"] = tt
    return out if out else None


def _message_content_from_openai_response(response: Any) -> str:
    """Return the first assistant message content from a chat completion."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("OpenRouter response did not include any choices.")
    choice = choices[0]
    if isinstance(choice, dict):
        message = choice.get("message")
    else:
        message = getattr(choice, "message", None)
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if not isinstance(content, str):
        raise ValueError("OpenRouter response message did not include text content.")
    return content


def _translate_chunk(
    chunk_segments: List[Dict],
    target_language: str,
    translation_model: str,
    reference_material: Optional[str],
    context: Dict[str, str],
    locked_terms: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Request one untrusted JSON translation response for a chunk."""
    prompt_text = build_translation_prompt(
        chunk_segments=chunk_segments,
        target_language=target_language,
        reference_material=reference_material,
        context=context,
        locked_terms=locked_terms,
    )

    try:
        client = _make_openrouter_client()
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return None, None

    response_text = ""
    usage: Optional[Dict[str, Any]] = None
    try:
        response = client.chat.completions.create(
            model=translation_model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
            response_format={"type": "json_object"},
            extra_body={"provider": OPENROUTER_PROVIDER_ROUTING},
        )
        usage = _usage_from_openai_response(response)
        _log_translation_api_tokens("Translation API tokens", usage)
        response_text = _message_content_from_openai_response(response)
        return json.loads(response_text, object_pairs_hook=_JSONObject), usage
    except json.JSONDecodeError as exc:
        print(
            f"Warning: Failed to decode JSON from the model's response: {exc}. "
            f"Preview: {response_text[:200]!r}"
        )
        return None, usage
    except Exception as exc:
        print(f"An unexpected error occurred during chunk translation: {exc}")
        return None, usage


def _translation_id(segment: Dict, index: int) -> str:
    return str(segment.get("_translation_id", index))


def _validate_translation_response(
    response_json: Any,
    requested_segments: List[Dict],
    locked_terms: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Accept only exact, non-empty, Locked-Term-compliant response entries."""
    requested_by_id = {
        _translation_id(segment, index): segment
        for index, segment in enumerate(requested_segments)
    }
    requested_ids = set(requested_by_id)
    if (
        not isinstance(response_json, dict)
        or set(response_json) != {"translations"}
        or not isinstance(response_json["translations"], dict)
    ):
        return {}, {
            translation_id: ["malformed_response"]
            for translation_id in requested_by_id
        }

    duplicate_keys = getattr(response_json, "duplicate_keys", [])
    if duplicate_keys:
        rule = f"duplicate_key({', '.join(sorted(duplicate_keys))})"
        return {}, {
            translation_id: [rule]
            for translation_id in requested_by_id
        }

    translations = response_json["translations"]
    duplicate_ids = getattr(translations, "duplicate_keys", [])
    if duplicate_ids:
        rule = f"duplicate_id({', '.join(sorted(duplicate_ids))})"
        return {}, {
            translation_id: [rule]
            for translation_id in requested_by_id
        }

    if any(not isinstance(translation_id, str) for translation_id in translations):
        return {}, {
            translation_id: ["malformed_id"]
            for translation_id in requested_by_id
        }

    extra_ids = set(translations) - requested_ids
    if extra_ids:
        rule = f"unexpected_id({', '.join(sorted(extra_ids))})"
        return {}, {
            translation_id: [rule]
            for translation_id in requested_by_id
        }

    accepted: Dict[str, str] = {}
    defects: Dict[str, List[str]] = {}
    for translation_id, source_segment in requested_by_id.items():
        if translation_id not in translations:
            defects[translation_id] = ["missing_id"]
            continue

        translated_text = translations[translation_id]
        if not isinstance(translated_text, str):
            defects[translation_id] = ["malformed_value"]
            continue

        translated_text = translated_text.strip()
        if not translated_text:
            defects[translation_id] = ["empty_translation"]
            continue

        source_text = source_segment.get("text", "")
        failed_terms = [
            f"locked_term({source})"
            for source, target in locked_terms.items()
            if source in source_text and target not in translated_text
        ]
        if failed_terms:
            defects[translation_id] = failed_terms
            continue
        accepted[translation_id] = translated_text

    return accepted, defects


def _format_translation_gate_defects(defects: Dict[str, List[str]]) -> str:
    return "; ".join(
        f"{translation_id}: {', '.join(rules)}"
        for translation_id, rules in sorted(defects.items())
    )


def _report_translation_gate_failure(defects: Dict[str, List[str]]) -> None:
    detail = _format_translation_gate_defects(defects)
    print(f"Translation Gate failed: {detail}")
    emit_progress(
        "translation_gate",
        "error",
        "Translation Gate failed",
        detail=detail,
    )


def _context_for_retry(
    chunk_segments: List[Dict],
    context: Dict[str, str],
) -> Dict[str, str]:
    """Preserve outer context and add the original chunk in source order."""
    return {
        "prev": context.get("prev", ""),
        "next": context.get("next", ""),
        "full_chunk": "\n".join(
            f"{_translation_id(segment, index)}: {segment['text']}"
            for index, segment in enumerate(chunk_segments)
        ),
    }


def _translate_chunk_with_gate(
    chunk_segments: List[Dict],
    target_language: str,
    translation_model: str,
    reference_material: Optional[str],
    context: Dict[str, str],
    locked_terms: Dict[str, str],
) -> Tuple[Optional[Dict[str, str]], Dict[str, int], Dict[str, List[str]]]:
    """Translate one chunk, retrying only entries rejected by the gate."""
    accepted: Dict[str, str] = {}
    pending_segments = chunk_segments
    usage_totals: Dict[str, int] = {}
    defects: Dict[str, List[str]] = {}

    for retry in range(_MAX_TRANSLATION_RETRIES + 1):
        if retry:
            pending_ids = ", ".join(
                _translation_id(segment, index)
                for index, segment in enumerate(pending_segments)
            )
            delay = _translation_retry_delay(retry - 1)
            print(
                f"Translation Gate: retry {retry}/{_MAX_TRANSLATION_RETRIES} "
                f"for IDs {pending_ids} in {delay:.1f}s..."
            )
            time.sleep(delay)

        retry_context = (
            context
            if retry == 0
            else _context_for_retry(chunk_segments, context)
        )
        response_json, usage = _translate_chunk(
            pending_segments,
            target_language,
            translation_model,
            reference_material,
            retry_context,
            locked_terms,
        )
        _merge_api_usage_tokens(usage_totals, usage)
        newly_accepted, defects = _validate_translation_response(
            response_json,
            pending_segments,
            locked_terms,
        )
        accepted.update(newly_accepted)
        if not defects:
            return accepted, usage_totals, {}
        pending_segments = [
            segment
            for index, segment in enumerate(pending_segments)
            if _translation_id(segment, index) in defects
        ]

    return None, usage_totals, defects


def translate_segments(
    segments: List[Dict],
    target_language: str,
    translation_model: str,
    reference_material: Optional[str],
    preserve_punctuation: bool = False,
    locked_terms: Optional[Dict[str, str]] = None,
    accepted_chunk_loader: Optional[
        Callable[[int, List[Dict]], Optional[Dict[str, str]]]
    ] = None,
    accepted_chunk_saver: Optional[
        Callable[[int, List[Dict], Dict[str, str]], None]
    ] = None,
) -> Optional[List[Dict]]:
    """
    Translates transcribed text segments using an optimized, two-layer strategy.

    For shorter texts that fit within the token threshold, it translates the entire
    content in a single batch for maximum speed. For longer texts, it uses a sliding
    window approach with a Translation Gate for every model response.

    Args:
        segments: A list of transcribed segments.
        target_language: The target language for the translation.
        translation_model: The OpenRouter model slug to use for translation.
        reference_material: Optional reference text for context-aware translation.
        locked_terms: Optional source-to-target terms enforced by the gate.
        accepted_chunk_loader: Optional source of previously accepted chunks.
        accepted_chunk_saver: Optional destination for newly accepted chunks.

    Returns:
        A list of translated segments, or None if a critical error occurs.
    """
    # Kept for API compatibility; accepted Translation Sentences always keep punctuation.
    _ = preserve_punctuation
    print("Translating text using an optimized strategy...")
    locked_terms = locked_terms or {}
    translation_segments = [
        {**segment, "_translation_id": str(index)}
        for index, segment in enumerate(segments)
        if segment.get("text", "").strip() != "[no speech]"
    ]
    num_segments = len(translation_segments)

    if not translation_segments:
        print("No speech segments require translation.")
        emit_progress(
            "translation",
            "complete",
            "No speech segments require translation",
            current=0,
            total=0,
            pct=100,
        )
        return []

    translated_segments_text: Dict[str, str] = {}

    def load_accepted_chunk(chunk_index: int, chunk: List[Dict]) -> Optional[Dict[str, str]]:
        if accepted_chunk_loader is None:
            return None
        cached = accepted_chunk_loader(chunk_index, chunk)
        if cached is None:
            return None
        accepted, defects = _validate_translation_response(
            {"translations": cached},
            chunk,
            locked_terms,
        )
        return accepted if not defects else None

    # Calculate the total length to decide on the translation strategy
    full_text = "\n".join([seg["text"] for seg in translation_segments])
    # A rough estimation of the overhead from the prompt template and reference material
    prompt_overhead = len(reference_material or "") + 1000
    total_length = len(full_text) + prompt_overhead

    try:
        use_sliding_window = False
        usage_totals: Dict[str, int] = {}

        if total_length < TOKEN_THRESHOLD:
            print(
                "Text is short enough. Attempting to translate in a "
                "single batch."
            )
            emit_progress(
                "translation_strategy",
                "running",
                "Using single batch translation",
                total=1,
            )
            with tqdm(total=1, desc="Translating (single batch)", unit="batch") as pbar:
                emit_progress(
                    "translation",
                    "running",
                    "Translating single batch",
                )
                translated_segments_text = load_accepted_chunk(0, translation_segments)
                chunk_usage: Dict[str, int] = {}
                defects: Dict[str, List[str]] = {}
                if translated_segments_text is None:
                    translated_segments_text, chunk_usage, defects = _translate_chunk_with_gate(
                        translation_segments,
                        target_language,
                        translation_model,
                        reference_material,
                        context={},
                        locked_terms=locked_terms,
                    )
                    if translated_segments_text is not None and accepted_chunk_saver is not None:
                        accepted_chunk_saver(0, translation_segments, translated_segments_text)
                pbar.update(1)
            _merge_api_usage_tokens(usage_totals, chunk_usage)

            if translated_segments_text is None:
                _report_translation_gate_failure(defects)
                return None
        else:
            print("Text is too long, using sliding window translation.")
            use_sliding_window = True

        # Strategy 2: Sliding Window (Chunks)
        if use_sliding_window:
            translated_segments_text = {}
            num_chunks = (num_segments + CHUNK_SIZE - 1) // CHUNK_SIZE
            emit_progress(
                "translation_strategy",
                "running",
                "Using sliding window translation",
                total=num_chunks,
            )

            for i in tqdm(
                range(num_chunks),
                desc="Translating in chunks",
                unit="chunk",
            ):
                start_index = i * CHUNK_SIZE
                end_index = min(start_index + CHUNK_SIZE, num_segments)
                chunk = translation_segments[start_index:end_index]

                # Define context
                prev_start = max(0, start_index - OVERLAP_SIZE)
                prev_context_segments = translation_segments[prev_start:start_index]
                next_start = end_index
                next_end = min(next_start + OVERLAP_SIZE, num_segments)
                next_context_segments = translation_segments[next_start:next_end]

                context = {
                    'prev': "\n".join([seg["text"] for seg in prev_context_segments]),
                    'next': "\n".join([seg["text"] for seg in next_context_segments])
                }

                emit_progress(
                    "translation",
                    "running",
                    f"Translating chunk {i + 1}/{num_chunks}",
                    current=i + 1,
                    total=num_chunks,
                )
                translated_chunk = load_accepted_chunk(i, chunk)
                u: Dict[str, int] = {}
                defects: Dict[str, List[str]] = {}
                if translated_chunk is None:
                    translated_chunk, u, defects = _translate_chunk_with_gate(
                        chunk,
                        target_language,
                        translation_model,
                        reference_material,
                        context,
                        locked_terms,
                    )
                    if translated_chunk is not None and accepted_chunk_saver is not None:
                        accepted_chunk_saver(i, chunk, translated_chunk)
                _merge_api_usage_tokens(usage_totals, u)

                if translated_chunk is None:
                    _report_translation_gate_failure(defects)
                    return None

                if translated_chunk:
                    translated_segments_text.update(translated_chunk)
                    emit_progress(
                        "translation",
                        "complete",
                        f"Translated chunk {i + 1}/{num_chunks}",
                        current=i + 1,
                        total=num_chunks,
                    )

        _log_translation_api_tokens(
            "Translation API tokens — cumulative (reported chunks)",
            usage_totals,
        )

        final_segments = []
        for i, segment in enumerate(segments):
            original_text = segment.get("text", "").strip()
            if original_text == "[no speech]":
                continue
            translated_text = translated_segments_text.get(str(i))
            if translated_text is None:
                _report_translation_gate_failure({str(i): ["missing_id"]})
                return None
            translated_segment = segment.copy()
            translated_segment["source_text"] = original_text
            translated_segment["text"] = translated_text
            final_segments.append(translated_segment)

        print("Text translated successfully.")
        emit_progress(
            "translation",
            "complete",
            "Text translated successfully",
            current=len(final_segments),
            total=num_segments,
            pct=100,
        )
        return final_segments

    except Exception as e:
        print(f"An error occurred during text translation: {e}")
        emit_progress(
            "translation",
            "error",
            "An error occurred during text translation",
            detail=str(e),
        )
        return None
