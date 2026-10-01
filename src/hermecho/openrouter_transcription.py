"""Resumable, word-timestamped OpenRouter transcription.

The production transcription path deliberately keeps the remote backend in a
small, dependency-free module.  The evaluator owns the common ffmpeg,
ffprobe, normalisation, and atomic JSON helpers; they are imported lazily here
because the evaluator imports :mod:`hermecho.transcription`.
"""
from __future__ import annotations

import base64
import http.client
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .checkpoints import (
    _is_finite_number as _finite_float,
    _source_content_signature as _content_signature,
    fingerprint_data as _fingerprint,
    fingerprint_file,
)
from .progress import emit_progress

TRANSCRIPTION_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
DEFAULT_OPENROUTER_TRANSCRIPTION_MODEL = "microsoft/mai-transcribe-2"
TRANSCRIPTION_ASSEMBLY_RULES = "openrouter-word-chunks-v3"
CHUNK_SECONDS = 60.0
OVERLAP_SECONDS = 1.0
OVERLAP_TIMESTAMP_TOLERANCE = 0.25
BOUNDARY_CONTEXT_SECONDS = 10.0
CHECKPOINT_VERSION = 1
REQUEST_TIMEOUT_SECONDS = 120
_MAX_WORD_TIMESTAMP_ATTEMPTS = 3


class OpenRouterRequestError(RuntimeError):
    """A retryable OpenRouter transport or transient HTTP request failure."""


class OpenRouterWordTimestampError(RuntimeError):
    """A response contains invalid word-timestamp evidence."""

    def __init__(
        self,
        message: str,
        *,
        attempts: list[dict[str, Any]] | None = None,
        diagnostics: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = [dict(attempt) for attempt in attempts or []]
        self.diagnostics = list(diagnostics or [])


def _extract_audio(
    source: Path,
    destination: Path,
    start: float = 0,
    duration: float | None = None,
) -> None:
    """Lazily reuse the evaluator's deterministic MP3 extraction helper."""
    from .openrouter_asr_evaluation import _extract_audio as evaluator_extract_audio

    evaluator_extract_audio(source, destination, start, duration)


def _ffprobe_duration(path: Path) -> float:
    """Lazily reuse the evaluator's positive finite duration check."""
    from .openrouter_asr_evaluation import _ffprobe_duration as evaluator_ffprobe_duration

    return evaluator_ffprobe_duration(path)


def _normalise_words(response: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    """Lazily reuse the evaluator's word timestamp normalisation."""
    from .openrouter_asr_evaluation import _normalise_words as evaluator_normalise_words

    return evaluator_normalise_words(response, duration)


def _write_json(path: Path, value: Any) -> None:
    """Lazily reuse the evaluator's atomic JSON writer."""
    from .openrouter_asr_evaluation import _write_json as evaluator_write_json

    evaluator_write_json(path, value)


def _audio_fingerprint(path: Path) -> str:
    try:
        return fingerprint_file(str(path))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Unable to fingerprint transcription audio: {path}") from error


def _request_error_detail(error: HTTPError) -> str:
    try:
        detail = error.read(1000).decode("utf-8", errors="replace")
    except (OSError, UnicodeError):
        detail = ""
    return detail.strip()


def _header(response: Any, *names: str) -> Optional[str]:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    for name in names:
        try:
            value = headers.get(name)
        except (AttributeError, TypeError):
            value = None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _actual_text_value(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _actual_provider_value(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        for key in ("name", "id", "slug"):
            candidate = _actual_text_value(value.get(key))
            if candidate is not None:
                return candidate
        return None
    return _actual_text_value(value)


def _request_transcription(
    audio_path: Path,
    model: str,
    language: Optional[str],
    temperature: float,
    api_key: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Request one audio chunk and return ``(body, metadata)``.

    Transport failures are the only errors represented by
    :class:`OpenRouterRequestError`.  Once an HTTP response arrives, malformed
    JSON and invalid transcript evidence are data errors and remain ordinary
    ``RuntimeError`` instances.
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter transcription.")

    try:
        audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"Unable to read extracted transcription audio: {audio_path}") from error
    if not audio_data:
        raise RuntimeError(f"Extracted transcription audio is empty: {audio_path}")

    payload: dict[str, Any] = {
        "model": model,
        "input_audio": {"data": audio_data, "format": "mp3"},
        "response_format": "verbose_json",
        "timestamp_granularities": ["word"],
        "temperature": temperature,
    }
    if language is not None and language.strip():
        payload["language"] = language
    if model == DEFAULT_OPENROUTER_TRANSCRIPTION_MODEL:
        payload["provider"] = {
            "options": {
                "azure": {
                    "enhancedMode": {
                        "modelOptions": {"transcribeStyle": "verbatim"}
                    }
                }
            }
        }

    request = Request(
        TRANSCRIPTION_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            raw_body = response.read()
            response_model = _header(response, "X-Model", "x-model")
            response_provider = _header(response, "X-Provider", "x-provider")
            generation_id = _header(response, "X-Generation-Id", "x-generation-id")
    except HTTPError as error:
        detail = _request_error_detail(error)
        message = f"OpenRouter HTTP {error.code}"
        if detail:
            message = f"{message}: {detail}"
        if error.code in {408, 429} or error.code >= 500:
            raise OpenRouterRequestError(message) from error
        raise RuntimeError(message) from error
    except (URLError, TimeoutError, OSError, http.client.HTTPException) as error:
        raise OpenRouterRequestError(f"OpenRouter connection failed: {error}") from error

    elapsed = time.monotonic() - started
    try:
        body = json.loads(raw_body)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as error:
        raise RuntimeError("OpenRouter returned malformed JSON.") from error
    if not isinstance(body, dict):
        raise RuntimeError("OpenRouter returned a non-object response.")

    usage = body.get("usage")
    cost: float | None = None
    if isinstance(usage, dict):
        raw_cost = usage.get("cost")
        if _finite_float(raw_cost) and float(raw_cost) >= 0:
            cost = float(raw_cost)
    actual_model = _actual_text_value(body.get("model")) or response_model
    actual_provider = _actual_provider_value(body.get("provider")) or response_provider
    if generation_id is None:
        generation_id = _actual_text_value(body.get("generation_id")) or _actual_text_value(
            body.get("id")
        )
    metadata = {
        "cost_usd": cost,
        "model": actual_model,
        "provider": actual_provider,
        "generation_id": generation_id,
        "elapsed_seconds": round(elapsed, 3),
    }
    return body, metadata


def _response_words(response: dict[str, Any], duration: float) -> tuple[str, list[dict[str, Any]]]:
    """Validate response evidence, allowing an explicitly silent chunk."""
    text = response.get("text")
    if not isinstance(text, str):
        raise RuntimeError("OpenRouter response has no valid transcript text.")

    segments = response.get("segments")
    nested_words: list[Any] = []
    nested_texts: list[str] = []
    if segments is not None:
        if not isinstance(segments, list):
            raise RuntimeError("OpenRouter response has malformed segments.")
        for segment in segments:
            if not isinstance(segment, dict):
                raise RuntimeError("OpenRouter response has malformed segments.")
            if "text" in segment and not isinstance(segment["text"], str):
                raise RuntimeError("OpenRouter response has malformed segment text.")
            if isinstance(segment.get("text"), str):
                nested_texts.append(segment["text"])
            segment_words = segment.get("words")
            if segment_words is not None and not isinstance(segment_words, list):
                raise RuntimeError("OpenRouter response has malformed segment words.")
            if isinstance(segment_words, list):
                nested_words.extend(segment_words)

    raw_words = response.get("words")
    if raw_words is None and segments is not None:
        raw_words = nested_words
    if text.strip() == "" and any(segment_text.strip() for segment_text in nested_texts):
        raise RuntimeError("OpenRouter response has transcript text without top-level word timestamps.")
    if text.strip() == "" and raw_words in (None, []):
        return text, []
    if not isinstance(raw_words, list) or not raw_words:
        raise RuntimeError("OpenRouter response has no word-level timestamps.")
    try:
        words = _normalise_words(response, duration)
    except (TypeError, ValueError, KeyError, OverflowError) as error:
        raise OpenRouterWordTimestampError(
            f"OpenRouter response has invalid word timestamps: {error}"
        ) from error
    if not words:
        raise RuntimeError("OpenRouter response has no word-level timestamps.")
    if any(word["end"] > duration for word in words):
        raise OpenRouterWordTimestampError(
            "OpenRouter response has word timestamps beyond the chunk duration."
        )
    word_text = "".join(word["word"] for word in words)
    if not _content_signature(text) or _content_signature(text) != _content_signature(word_text):
        raise RuntimeError("OpenRouter response text is not fully covered by its word timestamps.")
    return text, words


def _metadata_after_timestamp_retry(
    attempts: list[dict[str, Any]], diagnostics: list[str]
) -> dict[str, Any]:
    """Aggregate charged request metadata without retaining response bodies."""
    if len(attempts) == 1:
        return attempts[0]

    metadata = dict(attempts[-1])
    costs = [attempt.get("cost_usd") for attempt in attempts]
    elapsed = [attempt.get("elapsed_seconds") for attempt in attempts]
    metadata["cost_usd"] = (
        sum(float(value) for value in costs)
        if costs and all(_finite_float(value) and float(value) >= 0 for value in costs)
        else None
    )
    metadata["elapsed_seconds"] = (
        round(sum(float(value) for value in elapsed), 3)
        if elapsed and all(_finite_float(value) and float(value) >= 0 for value in elapsed)
        else None
    )
    metadata["attempt_count"] = len(attempts)
    metadata["retry_diagnostics"] = list(diagnostics)
    metadata["attempts"] = []
    for index, attempt in enumerate(attempts):
        record = dict(attempt)
        if index < len(diagnostics):
            record["validation_error"] = diagnostics[index]
        metadata["attempts"].append(record)
    return metadata


def _record_timestamp_retry_failure(
    state: dict[str, Any],
    *,
    scope: str,
    identifier: str,
    error: BaseException,
    status: str = "exhausted",
) -> None:
    """Persist charged retry metadata without retaining invalid response bodies."""
    attempts = getattr(error, "attempts", None)
    if not isinstance(attempts, list):
        attempts = getattr(error, "timestamp_retry_attempts", [])
    diagnostics = getattr(error, "diagnostics", None)
    if not isinstance(diagnostics, list):
        diagnostics = getattr(error, "timestamp_retry_diagnostics", [])
    history = state.get("retry_history")
    if history is None:
        history = []
        state["retry_history"] = history
    if not isinstance(history, list):
        raise RuntimeError("Invalid OpenRouter retry history checkpoint.")
    history.append(
        {
            "status": status,
            "scope": scope,
            "identifier": identifier,
            "attempt_count": len(attempts),
            "attempts": [dict(attempt) for attempt in attempts if isinstance(attempt, dict)],
            "diagnostics": list(diagnostics) or [str(error)],
        }
    )


def _attach_timestamp_retry_context(
    error: BaseException,
    attempts: list[dict[str, Any]],
    diagnostics: list[str],
) -> None:
    if attempts:
        setattr(error, "timestamp_retry_attempts", [dict(attempt) for attempt in attempts])
        setattr(error, "timestamp_retry_diagnostics", list(diagnostics))


def _transcribe_validated_words(
    audio_path: Path,
    model: str,
    language: Optional[str],
    temperature: float,
    api_key: str,
    duration: float,
    *,
    label: str,
    current: int | None = None,
    total: int | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Request one chunk, retrying only invalid word-timestamp responses."""
    attempts: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for attempt_number in range(1, _MAX_WORD_TIMESTAMP_ATTEMPTS + 1):
        if attempt_number > 1:
            fields: dict[str, Any] = {
                "detail": {
                    "kind": "invalid_word_timestamps",
                    "attempt": attempt_number,
                    "max_attempts": _MAX_WORD_TIMESTAMP_ATTEMPTS,
                },
            }
            if current is not None:
                fields["current"] = current
            if total is not None:
                fields["total"] = total
            emit_progress(
                "transcription",
                "running",
                f"Retrying {label} after invalid word timestamps "
                f"(attempt {attempt_number}/{_MAX_WORD_TIMESTAMP_ATTEMPTS})",
                **fields,
            )

        try:
            body, metadata = _request_transcription(
                audio_path, model, language, temperature, api_key
            )
        except RuntimeError as error:
            _attach_timestamp_retry_context(error, attempts, diagnostics)
            raise
        attempts.append(metadata if isinstance(metadata, dict) else {})
        try:
            text, words = _response_words(body, duration)
        except OpenRouterWordTimestampError as error:
            diagnostics.append(str(error))
            if attempt_number < _MAX_WORD_TIMESTAMP_ATTEMPTS:
                continue
            diagnostic_text = "; ".join(
                f"attempt {index}: {message}"
                for index, message in enumerate(diagnostics, start=1)
            )
            raise OpenRouterWordTimestampError(
                f"{diagnostics[-1]} (after {attempt_number} attempts; "
                f"diagnostics: {diagnostic_text})",
                attempts=attempts,
                diagnostics=diagnostics,
            ) from error
        except RuntimeError as error:
            _attach_timestamp_retry_context(error, attempts, diagnostics)
            raise
        return text, words, _metadata_after_timestamp_retry(attempts, diagnostics)

    raise AssertionError("OpenRouter timestamp retry loop did not return or raise")


def _validate_cached_words(words: Any, duration: float, text: Any) -> bool:
    if not isinstance(text, str) or not isinstance(words, list):
        return False
    if not words:
        return text.strip() == ""
    try:
        normalised = _normalise_words({"words": words}, duration)
    except (TypeError, ValueError, KeyError, OverflowError):
        return False
    if normalised != words or not text.strip():
        return False
    if any(word["end"] > duration for word in words):
        return False
    return _content_signature(text) == _content_signature("".join(word["word"] for word in words))


def _chunk_specs(duration: float) -> list[dict[str, float]]:
    specs: list[dict[str, float]] = []
    index = 0
    core_start = 0.0
    while core_start < duration:
        core_end = min(core_start + CHUNK_SECONDS, duration)
        # Each request has one second of context on either side, while the
        # non-overlapping core owns the returned words.
        extract_start = max(0.0, core_start - OVERLAP_SECONDS)
        extract_end = min(duration, core_end + OVERLAP_SECONDS)
        specs.append(
            {
                "index": float(index),
                "start": extract_start,
                "end": extract_end,
                "core_start": core_start,
                "core_end": core_end,
            }
        )
        if core_end >= duration:
            break
        core_start = core_end
        index += 1
    return specs


def _chunk_fingerprint(config_fingerprint: str, spec: dict[str, float]) -> str:
    return _fingerprint(
        {
            "config": config_fingerprint,
            "index": int(spec["index"]),
            "start": spec["start"],
            "end": spec["end"],
            "core_start": spec["core_start"],
            "core_end": spec["core_end"],
        }
    )


def _new_checkpoint(config: dict[str, Any], fingerprint: str, duration: float) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "status": "partial",
        "fingerprint": fingerprint,
        "config": config,
        "duration_seconds": duration,
        "chunks": {},
    }


def _load_checkpoint(path: Path, config_fingerprint: str, config: dict[str, Any], duration: float) -> dict[str, Any]:
    if not path.exists():
        return _new_checkpoint(config, config_fingerprint, duration)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        raise RuntimeError(f"Unable to read OpenRouter transcription checkpoint: {path}") from error
    if not isinstance(raw, dict):
        raise RuntimeError("OpenRouter transcription checkpoint is not a JSON object.")
    if raw.get("fingerprint") != config_fingerprint:
        return _new_checkpoint(config, config_fingerprint, duration)
    if raw.get("version") != CHECKPOINT_VERSION or not isinstance(raw.get("chunks"), dict):
        raise RuntimeError("OpenRouter transcription checkpoint has an invalid structure.")
    return raw


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, state)
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"Unable to write OpenRouter transcription checkpoint: {path}") from error


def _cached_chunk(
    state: dict[str, Any],
    spec: dict[str, float],
    config_fingerprint: str,
) -> Optional[dict[str, Any]]:
    chunks = state.get("chunks")
    if not isinstance(chunks, dict):
        return None
    record = chunks.get(str(int(spec["index"])))
    if not isinstance(record, dict):
        return None
    if (
        record.get("status") != "complete"
        or record.get("fingerprint") != _chunk_fingerprint(config_fingerprint, spec)
        or record.get("start") != spec["start"]
        or record.get("end") != spec["end"]
        or record.get("core_start") != spec["core_start"]
        or record.get("core_end") != spec["core_end"]
    ):
        return None
    duration = spec["end"] - spec["start"]
    if not _validate_cached_words(record.get("words"), duration, record.get("text")):
        return None
    return record


def _words_for_output(
    chunks: list[tuple[dict[str, float], dict[str, Any]]],
    duration: float,
    *,
    boundary_resolver: Callable[..., list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Reconcile shared word evidence before joining adjacent audio chunks."""
    kept: list[dict[str, Any]] = []
    previous_spec: Optional[dict[str, float]] = None
    for spec, record in chunks:
        offset = spec["start"]
        absolute_words = [
            {
                "word": word["word"],
                "start": float(word["start"]) + offset,
                "end": float(word["end"]) + offset,
            }
            for word in record["words"]
        ]
        if previous_spec is not None:
            overlap_start = spec["start"]
            overlap_end = previous_spec["end"]

            def in_overlap(word: dict[str, Any]) -> bool:
                return (
                    word["end"] >= overlap_start
                    or math.isclose(word["end"], overlap_start, abs_tol=1e-9, rel_tol=0)
                ) and (
                    word["start"] <= overlap_end
                    or math.isclose(word["start"], overlap_end, abs_tol=1e-9, rel_tol=0)
                )

            left = [word for word in kept if in_overlap(word)]
            right = [word for word in absolute_words if in_overlap(word)]
            if len(left) != len(right) or any(
                _content_signature(first["word"]).casefold()
                != _content_signature(second["word"]).casefold()
                or abs(first["start"] - second["start"]) > OVERLAP_TIMESTAMP_TOLERANCE
                or abs(first["end"] - second["end"]) > OVERLAP_TIMESTAMP_TOLERANCE
                for first, second in zip(left, right)
            ):
                if boundary_resolver is not None:
                    kept = boundary_resolver(spec["core_start"], kept, absolute_words)
                    previous_spec = spec
                    continue
                raise RuntimeError(
                    f"OpenRouter overlapping transcripts disagree near {spec['core_start']:g}s; "
                    "cannot reconcile complete Source Word evidence."
                )
            # Keep the earlier chunk's exact evidence once, regardless of which
            # side of the core boundary either response assigned to that word.
            absolute_words = [word for word in absolute_words if not in_overlap(word)]
        kept.extend(absolute_words)
        previous_spec = spec

    _validate_absolute_words(kept, duration)
    return kept


def _splice_boundary_words(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    bridge: list[dict[str, Any]],
    boundary: float,
) -> list[dict[str, Any]]:
    """Join at matching three-word/time anchors outside the disputed overlap.

    The middle comes from one new audio response. No conflicting words or
    timestamps are guessed, interpolated, or copied into that response.
    """
    def matches(first, second):
        return bool(_content_signature(first["word"])) and (
            _content_signature(first["word"]).casefold()
            == _content_signature(second["word"]).casefold()
            and abs(first["start"] - second["start"]) <= OVERLAP_TIMESTAMP_TOLERANCE
            and abs(first["end"] - second["end"]) <= OVERLAP_TIMESTAMP_TOLERANCE
        )

    def anchors(original, side):
        found = []
        for i in range(len(original) - 2):
            if side == "left" and original[i + 2]["end"] > boundary - OVERLAP_SECONDS:
                continue
            if side == "right" and original[i]["start"] < boundary + OVERLAP_SECONDS:
                continue
            for j in range(len(bridge) - 2):
                if all(matches(original[i + k], bridge[j + k]) for k in range(3)):
                    found.append((i, j))
        return found

    left_anchors, right_anchors = anchors(left, "left"), anchors(right, "right")
    if not left_anchors or not right_anchors:
        raise RuntimeError(f"OpenRouter boundary near {boundary:g}s has no matching word/time anchors.")
    li, lb = left_anchors[-1]
    ri, rb = right_anchors[0]
    if (sum(i == li or j == lb for i, j in left_anchors) != 1
            or sum(i == ri or j == rb for i, j in right_anchors) != 1):
        raise RuntimeError(f"OpenRouter boundary near {boundary:g}s has ambiguous anchors.")
    if lb + 3 > rb:
        raise RuntimeError(f"OpenRouter boundary near {boundary:g}s has inconsistent anchors.")
    joined = left[:li + 3] + bridge[lb + 3:rb] + right[ri:]
    if any(a["start"] > b["start"] or a["end"] > b["end"] for a, b in zip(joined, joined[1:])):
        raise RuntimeError(f"OpenRouter boundary near {boundary:g}s has unordered word evidence.")
    return joined


def _segments_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not words:
        return []
    return [
        {
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": " ".join(word["word"].strip() for word in words),
            "words": words,
        }
    ]


def _validate_absolute_words(words: list[dict[str, Any]], duration: float) -> None:
    if not words:
        return
    try:
        normalised = _normalise_words({"words": words}, duration)
    except (TypeError, ValueError, KeyError, OverflowError) as error:
        raise RuntimeError(f"OpenRouter transcription has invalid absolute word timing: {error}") from error
    if normalised != words or any(word["end"] > duration for word in words):
        raise RuntimeError("OpenRouter transcription has invalid absolute word timing.")


def transcribe_openrouter(
    audio_path: str,
    model: str = DEFAULT_OPENROUTER_TRANSCRIPTION_MODEL,
    language: Optional[str] = None,
    temperature: float = 0.0,
    checkpoint_path: str | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Transcribe an audio/video file through resumable OpenRouter chunks.

    A successful chunk is flushed to ``checkpoint_path`` before the next chunk
    is requested.  The returned segments contain absolute word timestamps and
    are composed entirely from this remote run; this function never mixes
    another transcription backend into its result.
    """
    if not isinstance(model, str) or not model.strip():
        raise RuntimeError("OpenRouter transcription model must be a nonempty string.")
    if language is not None and not isinstance(language, str):
        raise RuntimeError("OpenRouter transcription language must be a string or None.")
    if not _finite_float(temperature) or not 0 <= float(temperature) <= 1:
        raise RuntimeError("OpenRouter transcription temperature must be finite and between 0 and 1.")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter transcription.")

    source_path = Path(audio_path)
    try:
        duration = _ffprobe_duration(source_path)
    except (OSError, ValueError, TypeError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Unable to probe transcription audio: {source_path}") from error
    if not _finite_float(duration) or float(duration) <= 0:
        raise RuntimeError(f"Unable to determine a positive transcription duration: {source_path}")

    audio_hash = _audio_fingerprint(source_path)
    config = {
        "audio_sha256": audio_hash,
        "model": model,
        "language": language,
        "temperature": float(temperature),
        "chunk_seconds": CHUNK_SECONDS,
        "overlap_seconds": OVERLAP_SECONDS,
    }
    config_fingerprint = _fingerprint(config)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    state = (
        _new_checkpoint(config, config_fingerprint, float(duration))
        if checkpoint is None or force
        else _load_checkpoint(checkpoint, config_fingerprint, config, float(duration))
    )
    # Acquisition evidence remains reusable when only assembly policy changes.
    # Accepted pipeline transcripts separately fingerprint the assembly rules.
    state["assembly_policy"] = {
        "rules": TRANSCRIPTION_ASSEMBLY_RULES,
        "timestamp_tolerance": OVERLAP_TIMESTAMP_TOLERANCE,
        "boundary_context_seconds": BOUNDARY_CONTEXT_SECONDS,
    }
    state["assembly_fingerprint"] = _fingerprint(state["assembly_policy"])
    state["status"] = "partial"
    if checkpoint is not None:
        _save_checkpoint(checkpoint, state)

    specs = _chunk_specs(float(duration))
    completed: list[tuple[dict[str, float], dict[str, Any]]] = []
    used_repairs: list[tuple[dict[str, float], dict[str, Any]]] = []

    def repair_boundary(boundary, left, right):
        start = max(0.0, boundary - BOUNDARY_CONTEXT_SECONDS)
        end = min(float(duration), boundary + BOUNDARY_CONTEXT_SECONDS)
        if (sum(start <= word["start"] and word["end"] <= boundary - OVERLAP_SECONDS
                for word in left) < 3
                or sum(boundary + OVERLAP_SECONDS <= word["start"] and word["end"] <= end
                       for word in right) < 3):
            raise RuntimeError(
                f"OpenRouter overlapping transcripts disagree near {boundary:g}s; "
                "insufficient word/time anchors for boundary repair."
            )
        repair_fingerprint = _fingerprint({
            "config": config_fingerprint, "boundary": boundary, "start": start, "end": end,
            "assembly": state["assembly_fingerprint"],
        })
        repairs = state.setdefault("boundary_repairs", {})
        if not isinstance(repairs, dict):
            raise RuntimeError("Invalid OpenRouter boundary checkpoint.")
        record = repairs.get(str(boundary))
        emit_progress(
            "transcription",
            "running",
            f"Reconciling audio boundary near {boundary:g}s",
            current=len(completed),
            total=len(specs),
        )
        if not (isinstance(record, dict) and record.get("fingerprint") == repair_fingerprint
                and record.get("start") == start and record.get("end") == end
                and _validate_cached_words(record.get("words"), end - start, record.get("text"))):
            try:
                with tempfile.TemporaryDirectory(prefix="hermecho-openrouter-boundary-") as temporary_dir:
                    audio = Path(temporary_dir) / "boundary.mp3"
                    _extract_audio(source_path, audio, start, end - start)
                    text, words, metadata = _transcribe_validated_words(
                        audio,
                        model,
                        language,
                        float(temperature),
                        api_key,
                        end - start,
                        label=f"boundary audio near {boundary:g}s",
                        current=len(completed),
                        total=len(specs),
                    )
            except OpenRouterWordTimestampError as error:
                _record_timestamp_retry_failure(
                    state,
                    scope="boundary",
                    identifier=str(boundary),
                    error=error,
                )
                if checkpoint is not None:
                    _save_checkpoint(checkpoint, state)
                # A repair must never change the selected transcription source.
                raise RuntimeError(f"OpenRouter boundary repair near {boundary:g}s failed: {error}") from error
            except RuntimeError as error:
                if getattr(error, "timestamp_retry_attempts", None):
                    _record_timestamp_retry_failure(
                        state,
                        scope="boundary",
                        identifier=str(boundary),
                        error=error,
                        status="interrupted",
                    )
                    if checkpoint is not None:
                        _save_checkpoint(checkpoint, state)
                # A repair must never change the selected transcription source.
                raise RuntimeError(f"OpenRouter boundary repair near {boundary:g}s failed: {error}") from error
            except (OSError, ValueError, TypeError, KeyError, OverflowError,
                    subprocess.CalledProcessError) as error:
                # A repair must never change the selected transcription source.
                raise RuntimeError(f"OpenRouter boundary repair near {boundary:g}s failed: {error}") from error
            record = {"fingerprint": repair_fingerprint, "start": start, "end": end,
                      "text": text, "words": words, "metadata": metadata}
            repairs[str(boundary)] = record
            if checkpoint is not None:
                _save_checkpoint(checkpoint, state)
        bridge = [{"word": word["word"], "start": word["start"] + start,
                   "end": word["end"] + start} for word in record["words"]]
        result = _splice_boundary_words(left, right, bridge, boundary)
        used_repairs.append(({}, record))
        return result

    with tempfile.TemporaryDirectory(prefix="hermecho-openrouter-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        for spec in specs:
            cached = None if force else _cached_chunk(state, spec, config_fingerprint)
            if cached is not None:
                completed.append((spec, cached))
                emit_progress("transcription", "running", "Reusing transcription chunk",
                              current=len(completed), total=len(specs))
                continue

            chunk_path = temporary_root / f"chunk-{int(spec['index']):04d}.mp3"
            chunk_duration = spec["end"] - spec["start"]
            emit_progress("transcription", "running", "Transcribing audio chunk",
                          current=len(completed) + 1, total=len(specs))
            try:
                _extract_audio(source_path, chunk_path, spec["start"], chunk_duration)
            except (OSError, ValueError, TypeError, subprocess.CalledProcessError) as error:
                raise RuntimeError(
                    f"Unable to extract OpenRouter transcription chunk {int(spec['index'])}."
                ) from error
            try:
                text, words, metadata = _transcribe_validated_words(
                    chunk_path,
                    model,
                    language,
                    float(temperature),
                    api_key,
                    chunk_duration,
                    label=f"audio chunk {int(spec['index'])}",
                    current=len(completed) + 1,
                    total=len(specs),
                )
            except OpenRouterWordTimestampError as error:
                _record_timestamp_retry_failure(
                    state,
                    scope="chunk",
                    identifier=str(int(spec["index"])),
                    error=error,
                )
                if checkpoint is not None:
                    _save_checkpoint(checkpoint, state)
                raise
            except RuntimeError as error:
                if getattr(error, "timestamp_retry_attempts", None):
                    _record_timestamp_retry_failure(
                        state,
                        scope="chunk",
                        identifier=str(int(spec["index"])),
                        error=error,
                        status="interrupted",
                    )
                    if checkpoint is not None:
                        _save_checkpoint(checkpoint, state)
                raise
            except (OSError, TypeError, ValueError, KeyError, OverflowError) as error:
                raise RuntimeError(
                    f"OpenRouter transcription chunk {int(spec['index'])} returned invalid data."
                ) from error

            record = {
                "status": "complete",
                "fingerprint": _chunk_fingerprint(config_fingerprint, spec),
                "start": spec["start"],
                "end": spec["end"],
                "core_start": spec["core_start"],
                "core_end": spec["core_end"],
                "text": text,
                "words": words,
                "metadata": metadata,
            }
            state["chunks"][str(int(spec["index"]))] = record
            state["status"] = "partial"
            if checkpoint is not None:
                _save_checkpoint(checkpoint, state)
            completed.append((spec, record))

    words = _words_for_output(completed, float(duration), boundary_resolver=repair_boundary)
    segments = _segments_from_words(words)
    state["status"] = "complete"
    state["metadata"] = _aggregate_metadata(
        [*completed, *used_repairs], retry_history=state.get("retry_history")
    )
    if checkpoint is not None:
        _save_checkpoint(checkpoint, state)
    return segments


def _aggregate_metadata(
    chunks: list[tuple[dict[str, float], dict[str, Any]]],
    *,
    retry_history: Any = None,
) -> dict[str, Any]:
    metadata = [
        record["metadata"] if isinstance(record.get("metadata"), dict) else {}
        for _spec, record in chunks
    ]
    charged_history: list[dict[str, Any]] = []
    if isinstance(retry_history, list):
        for history in retry_history:
            if not isinstance(history, dict):
                continue
            attempts = history.get("attempts")
            if isinstance(attempts, list):
                charged_history.extend(
                    dict(attempt) for attempt in attempts if isinstance(attempt, dict)
                )
    all_metadata = [*metadata, *charged_history]
    costs = [item.get("cost_usd") for item in all_metadata]
    elapsed = [item.get("elapsed_seconds") for item in all_metadata]
    cost = (
        sum(float(value) for value in costs)
        if costs and all(_finite_float(value) and value >= 0 for value in costs)
        else None
    )
    elapsed_seconds = (
        round(sum(float(value) for value in elapsed), 3)
        if elapsed and all(_finite_float(value) and value >= 0 for value in elapsed)
        else None
    )
    return {
        "cost_usd": cost,
        "model": [item.get("model") for item in all_metadata],
        "provider": [item.get("provider") for item in all_metadata],
        "generation_id": [item.get("generation_id") for item in all_metadata],
        "elapsed_seconds": elapsed_seconds,
    }
