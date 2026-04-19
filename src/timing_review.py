"""
Multimodal subtitle timing review via OpenRouter (audio + JSON cues).

Uses Korean source text plus translated text for context; only ``start`` and
``end`` are updated from the model. Translation strings are re-applied from
the input segments so the model cannot change subtitle wording.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import requests

from openrouter_fallback import openrouter_models_with_fallback
from transcription import (
    OPENROUTER_CHAT_URL,
    get_openrouter_http_referer,
    _audio_duration_seconds,
    _extract_openrouter_choice_message,
    _ffmpeg_extract_chunk,
    _infer_openrouter_audio_format,
    _json_object_response_format_for_model,
    _log_openrouter_token_usage,
    _merge_openrouter_usage,
    _openrouter_message_content_to_text,
    _parse_json_object_from_model_text,
)

# Default max seconds per review request (smaller than full transcribe chunks).
DEFAULT_TIMING_REVIEW_CHUNK_SECONDS = 120.0

# Monotonic repair after review: same spirit as multimodal repair, but we keep
# list order so cue text never swaps with a neighbor (translation order is
# authoritative). Input segments are expected to be roughly time-ordered.
_MIN_REVIEW_SEG_SEC = 0.05
_MIN_REVIEW_GAP_SEC = 0.015


def _clamp_and_repair_subtitles_in_order(
    segments: List[Dict[str, Any]],
    clip_end: float,
) -> List[Dict[str, Any]]:
    """
    Clamp start/end to ``[0, clip_end]`` and remove overlaps in timeline order.

    Unlike ``_repair_multimodal_segment_times``, this does **not** sort by
    start time, so each dict keeps its pairing of ``text`` / ``source_text``
    with adjusted times.

    Args:
        segments: Segment dicts with numeric ``start`` / ``end``.
        clip_end: Audio duration upper bound in seconds.

    Returns:
        New list with repaired ``start`` / ``end`` only.
    """
    if clip_end <= 0 or not segments:
        return [dict(s) for s in segments]

    out: List[Dict[str, Any]] = []
    for raw in segments:
        s = dict(raw)
        try:
            start = float(s["start"])
            end = float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        start = max(0.0, min(start, clip_end))
        end = max(0.0, min(end, clip_end))
        if end <= start:
            end = min(clip_end, start + _MIN_REVIEW_SEG_SEC)
        if end - start < _MIN_REVIEW_SEG_SEC:
            end = min(clip_end, start + _MIN_REVIEW_SEG_SEC)
        s["start"], s["end"] = start, end
        out.append(s)

    for i in range(1, len(out)):
        prev = out[i - 1]
        cur = out[i]
        p_start, p_end = float(prev["start"]), float(prev["end"])
        c_start = float(cur["start"])
        min_prev_end = p_start + _MIN_REVIEW_SEG_SEC
        if c_start < p_end + _MIN_REVIEW_GAP_SEC:
            new_p_end = c_start - _MIN_REVIEW_GAP_SEC
            if new_p_end < min_prev_end:
                new_p_end = min_prev_end
            prev["end"] = new_p_end
            if c_start < float(prev["end"]) + _MIN_REVIEW_GAP_SEC:
                cur["start"] = float(prev["end"]) + _MIN_REVIEW_GAP_SEC
            if float(cur["end"]) <= float(cur["start"]):
                cur["end"] = min(
                    clip_end,
                    float(cur["start"]) + _MIN_REVIEW_SEG_SEC,
                )

    return out


def slice_segments_for_window(
    segments: List[Dict[str, Any]],
    window_start: float,
    window_end: float,
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Return (global_index, segment) pairs for cues overlapping the window.

    Overlap uses half-open [window_start, window_end) against segment
    [start, end] with standard interval overlap.

    Args:
        segments: Full timeline segment dicts with numeric start/end.
        window_start: Inclusive window start in seconds.
        window_end: Exclusive window end in seconds.

    Returns:
        List of index and segment dict in timeline order.
    """
    out: List[Tuple[int, Dict[str, Any]]] = []
    for i, seg in enumerate(segments):
        try:
            s = float(seg["start"])
            e = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e <= window_start or s >= window_end:
            continue
        out.append((i, seg))
    return out


def build_review_payload_cues(
    indexed_segments: List[Tuple[int, Dict[str, Any]]],
    window_start: float,
    window_span: float,
) -> List[Dict[str, Any]]:
    """
    Build JSON-serializable cues with clip-relative start/end.

    Args:
        indexed_segments: Pairs of global index and segment dict. Indices
            are not sent; local ``id`` is 0..n-1 in list order.
        window_start: Absolute seconds where the extracted audio chunk starts.
        window_span: Length of the chunk in seconds (clip is [0, span)).

    Returns:
        List of dicts: id, start, end (relative), text (source language),
        translation (subtitle language).
    """
    cues: List[Dict[str, Any]] = []
    span = max(window_span, 1e-6)
    for local_id, (_gidx, seg) in enumerate(indexed_segments):
        try:
            abs_s = float(seg["start"])
            abs_e = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        rel_s = max(0.0, min(abs_s - window_start, span))
        rel_e = max(rel_s, min(abs_e - window_start, span))
        source = str(seg.get("source_text", "")).strip()
        translation = str(seg.get("text", "")).strip()
        cues.append(
            {
                "id": local_id,
                "start": round(rel_s, 3),
                "end": round(rel_e, 3),
                "text": source,
                "translation": translation,
            }
        )
    return cues


def parse_review_response(
    raw: str,
    expected_count: int,
) -> Optional[List[Dict[str, Any]]]:
    """
    Parse model JSON into a list of timing rows for one chunk.

    Expects ``{"segments":[{"id":int,"start":float,"end":float}, ...]}``.
    Translation fields in the response are ignored.

    Args:
        raw: Model message content string.
        expected_count: Number of cues in the request (local ids 0..n-1).

    Returns:
        Sorted list of dicts with id, start, end, or None if invalid.
    """
    if expected_count <= 0:
        return []
    parsed_obj = _parse_json_object_from_model_text(raw)
    if not parsed_obj:
        return None
    raw_list = parsed_obj.get("segments")
    if not isinstance(raw_list, list):
        return None
    rows: List[Dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        try:
            sid = int(item["id"])
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({"id": sid, "start": start, "end": end})
    rows.sort(key=lambda r: r["id"])
    if len(rows) != expected_count:
        print(
            f"Warning: timing review expected {expected_count} segments, "
            f"got {len(rows)}."
        )
        return None
    seen = {int(r["id"]) for r in rows}
    if seen != set(range(expected_count)):
        print("Warning: timing review response has non-contiguous ids.")
        return None
    return rows


def _build_timing_review_prompt(
    source_language: str,
    target_language: str,
    chunk_duration_sec: float,
) -> str:
    """
    Build user instructions for clip-relative timing correction.

    Audio is in the source language; subtitles may differ. Timing must follow
    speech in the audio, not reading speed of the translation.
    """
    return "\n".join(
        [
            "You are a professional subtitle timing editor.",
            f"The spoken audio is primarily in language: {source_language!r}.",
            f"Each cue includes ``translation``: subtitle text in "
            f"{target_language!r} for on-screen display.",
            "The ``text`` field is the source-language transcript line "
            "(read-only for alignment).",
            "",
            "Task: listen to THIS audio clip and adjust ONLY ``start`` and "
            "``end`` for each cue so they match when the corresponding "
            "source-language speech is audible in the clip.",
            "",
            "Rules:",
            "- Times in your output MUST be in seconds, relative to the start "
            "of THIS clip (0 = clip start, end <= clip length).",
            "- Do NOT change ``translation`` or ``text``; they are not part of "
            "your output schema.",
            "- Do NOT add or remove cues; output exactly one row per input id.",
            "- Align ``start`` to the earliest clear speech onset for that "
            "cue (first audible syllable/phoneme). If unsure, bias earlier "
            "rather than later — late subtitles are worse than a slightly "
            "early start.",
            "- Fix late subtitles aggressively: if speech clearly begins before "
            "the given start time, move start earlier; do not leave a long "
            "silent lead-in before the line appears.",
            "- Fix lingering ends: if the source phrase has ended in audio "
            "before the given end, move end earlier.",
            "- Keep pauses between consecutive cues natural: if two lines are "
            "back-to-back in speech with only a brief breath, use a short gap "
            "(roughly 0.05–0.2s) between end and the next start — avoid "
            "artificially large blank gaps between related sentences.",
            "- Do NOT time cues from how long it takes to read the "
            "translation; boundaries come from Korean/source speech only.",
            "",
            "Negative example: if speech is clearly audible at 0.2s but "
            "start is 1.2s, move start toward 0.2s unless it would overlap "
            "the previous cue.",
            "",
            f"This clip is about {chunk_duration_sec:.2f} seconds long.",
            "",
            "Return ONLY valid JSON (no markdown) with this exact shape:",
            '{"segments":[{"id":0,"start":0.0,"end":1.2}, ...]}',
            "Each segment: id (integer, 0..n-1), start, end (numbers).",
        ]
    )


def review_timing_chunk(
    chunk_audio_path: str,
    payload_cues: List[Dict[str, Any]],
    review_model: str,
    source_language: str,
    target_language: str,
    chunk_duration_sec: float,
    usage_log_label: str,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """
    Call OpenRouter for one audio chunk and a list of timing cues.

    Tries ``review_model`` first, then the configured OpenAI GPT fallback when
    the primary is a Gemini 3.1 Pro/Flash family slug.

    Args:
        chunk_audio_path: Path to the extracted chunk (e.g. MP3).
        payload_cues: Cues with clip-relative start/end and text fields.
        review_model: OpenRouter model id.
        source_language: Spoken language label for the prompt.
        target_language: Subtitle language label for the prompt.
        chunk_duration_sec: Clip length for the prompt text.
        usage_log_label: Label for token usage logging.

    Returns:
        (parsed timing rows, usage) or (None, usage) on failure.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY is not set.")
        return None, None

    try:
        with open(chunk_audio_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("ascii")
    except OSError as exc:
        print(f"Error: could not read chunk audio: {exc}")
        return None, None

    audio_format = _infer_openrouter_audio_format(chunk_audio_path)
    prompt_intro = _build_timing_review_prompt(
        source_language,
        target_language,
        chunk_duration_sec,
    )
    cues_json = json.dumps(
        {"cues": payload_cues},
        ensure_ascii=False,
    )
    user_text = f"{prompt_intro}\n\nInput cues (clip-relative times):\n{cues_json}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": get_openrouter_http_referer(),
        "X-Title": "Hermecho Timing Review",
    }

    models = openrouter_models_with_fallback(review_model)
    last_usage: Optional[Dict[str, Any]] = None

    for attempt, model_id in enumerate(models):
        if attempt > 0:
            print(
                f"Timing review: primary model failed for this chunk; "
                f"retrying with fallback ({model_id})..."
            )

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": b64,
                                "format": audio_format,
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
        }
        fmt = _json_object_response_format_for_model(model_id)
        if fmt is not None:
            payload["response_format"] = fmt

        try:
            resp = requests.post(
                OPENROUTER_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=600,
            )
        except requests.RequestException as exc:
            print(f"Error: OpenRouter request failed: {exc}")
            if attempt + 1 < len(models):
                continue
            return None, last_usage

        if resp.status_code != 200:
            print(
                "Error: OpenRouter returned "
                f"{resp.status_code}: {resp.text[:500]}"
            )
            if attempt + 1 < len(models):
                continue
            return None, last_usage

        try:
            body = resp.json()
        except json.JSONDecodeError as exc:
            print(f"Error: invalid JSON from OpenRouter: {exc}")
            if attempt + 1 < len(models):
                continue
            return None, last_usage

        usage: Optional[Dict[str, Any]] = None
        if isinstance(body, dict):
            raw_usage = body.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage
                last_usage = usage

        message, shape_err = _extract_openrouter_choice_message(body)
        if shape_err is not None or message is None:
            print(f"Error: {shape_err}")
            _log_openrouter_token_usage(usage_log_label, usage)
            if attempt + 1 < len(models):
                continue
            return None, usage

        _log_openrouter_token_usage(usage_log_label, usage)

        text = _openrouter_message_content_to_text(message.get("content"))
        if text is None:
            print(
                "Error: OpenRouter returned no assistant text in "
                "choices[0].message for timing review."
            )
            if attempt + 1 < len(models):
                continue
            return None, usage

        parsed = parse_review_response(text, len(payload_cues))
        if parsed is not None:
            return parsed, usage
        if attempt + 1 < len(models):
            continue
        return None, usage

    return None, last_usage


def _apply_chunk_times(
    working: List[Dict[str, Any]],
    index_map: List[int],
    window_start: float,
    parsed_rows: List[Dict[str, Any]],
) -> None:
    """Write absolute start/end from clip-relative rows into working."""
    for row in parsed_rows:
        local_id = int(row["id"])
        if local_id < 0 or local_id >= len(index_map):
            continue
        gidx = index_map[local_id]
        rel_s = float(row["start"])
        rel_e = float(row["end"])
        if rel_e < rel_s:
            rel_s, rel_e = rel_e, rel_s
        working[gidx]["start"] = window_start + rel_s
        working[gidx]["end"] = window_start + rel_e


def review_subtitle_timing(
    audio_path: str,
    segments: List[Dict[str, Any]],
    chunk_seconds: float,
    review_model: str,
    source_language: str,
    target_language: str,
) -> Optional[List[Dict[str, Any]]]:
    """
    Refine subtitle start/end times using chunked multimodal review.

    On per-chunk parse or HTTP failure, that chunk's cues are left unchanged.
    Returns None only on hard errors (missing file, unreadable duration).

    Args:
        audio_path: Extracted audio path (same timeline as segment times).
        segments: Translated segments with ``text``, ``start``, ``end``, and
            ``source_text`` (Korean or source transcript).
        chunk_seconds: Max seconds per API request.
        review_model: OpenRouter model id.
        source_language: Spoken language (prompt).
        target_language: Subtitle language (prompt).

    Returns:
        New list of segments with adjusted times, or None on hard failure.
    """
    if not os.path.exists(audio_path):
        print(f"Error: audio not found for timing review: {audio_path}")
        return None
    duration = _audio_duration_seconds(audio_path)
    if duration is None or duration <= 0:
        print("Error: could not read duration for timing review.")
        return None

    span_limit = max(float(chunk_seconds), 1.0)
    working = [dict(s) for s in segments]
    tmpdir = tempfile.mkdtemp(prefix="hermecho_timing_review_")
    usage_totals: Dict[str, int] = {}

    try:
        window_start = 0.0
        chunk_idx = 0
        while window_start < duration:
            span = min(span_limit, duration - window_start)
            window_end = window_start + span
            chunk_path = os.path.join(tmpdir, f"tr_{chunk_idx:04d}.mp3")
            if not _ffmpeg_extract_chunk(
                audio_path,
                chunk_path,
                window_start,
                span,
            ):
                print(
                    "Warning: ffmpeg chunk failed for timing review; "
                    "skipping this window."
                )
                window_start = window_end
                chunk_idx += 1
                continue

            indexed = slice_segments_for_window(
                working,
                window_start,
                window_end,
            )
            if not indexed:
                window_start = window_end
                chunk_idx += 1
                continue

            index_map = [i for i, _s in indexed]
            payload = build_review_payload_cues(
                indexed,
                window_start,
                span,
            )
            if not payload:
                window_start = window_end
                chunk_idx += 1
                continue

            label = f"Timing review chunk {chunk_idx + 1} API tokens"
            parsed, usage = review_timing_chunk(
                chunk_path,
                payload,
                review_model,
                source_language,
                target_language,
                span,
                label,
            )
            _merge_openrouter_usage(usage_totals, usage)
            if parsed is not None:
                _apply_chunk_times(working, index_map, window_start, parsed)
            else:
                print(
                    f"Warning: timing review chunk {chunk_idx + 1} failed; "
                    "keeping original times for cues in this window."
                )

            window_start = window_end
            chunk_idx += 1

        repaired = _clamp_and_repair_subtitles_in_order(working, duration)
        _log_openrouter_token_usage(
            "Timing review API tokens — cumulative (all chunks)",
            usage_totals if usage_totals else None,
        )
        print("Timing review finished.")
        return repaired
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
