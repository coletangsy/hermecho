"""
Subtitle timing review via Google Generative AI (audio + JSON cues).

Uses Korean source text plus translated text for context; only ``start`` and
``end`` are updated from the model. Translation strings are re-applied from
the input segments so the model cannot change subtitle wording.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from google import genai
from google.genai import types

from models import TimingReviewResponse, seconds_to_srt, srt_to_seconds
from retry import compute_backoff
from transcription import (
    _audio_duration_seconds,
    _ffmpeg_extract_chunk,
    _infer_gemini_inline_audio_mime_type,
)

DEFAULT_TIMING_REVIEW_CHUNK_SECONDS = 120.0

_MIN_REVIEW_SEG_SEC = 0.05
_MIN_REVIEW_GAP_SEC = 0.015
_MAX_REVIEW_ATTEMPTS = 3


def _make_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def _log_review_usage(label: str, response: Any) -> None:
    um = getattr(response, "usage_metadata", None)
    if um is None:
        print(f"{label}: (no usage metadata)")
        return
    parts = []
    pt = getattr(um, "prompt_token_count", None)
    ct = getattr(um, "candidates_token_count", None)
    tt = getattr(um, "total_token_count", None)
    if pt is not None:
        parts.append(f"promptTokenCount={pt}")
    if ct is not None:
        parts.append(f"candidatesTokenCount={ct}")
    if tt is not None:
        parts.append(f"totalTokenCount={tt}")
    if parts:
        print(f"{label}: " + ", ".join(parts))
    else:
        print(f"{label}: (empty usage metadata)")


def _usage_dict_from_response(response: Any) -> Optional[Dict[str, Any]]:
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return None
    out: Dict[str, Any] = {}
    pt = getattr(um, "prompt_token_count", None)
    ct = getattr(um, "candidates_token_count", None)
    tt = getattr(um, "total_token_count", None)
    if pt is not None:
        out["prompt_tokens"] = int(pt)
    if ct is not None:
        out["completion_tokens"] = int(ct)
    if tt is not None:
        out["total_tokens"] = int(tt)
    return out if out else None


def _merge_usage(totals: Dict[str, int], usage: Optional[Dict[str, Any]]) -> None:
    if not usage:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = usage.get(key)
        if val is not None:
            totals[key] = totals.get(key, 0) + int(val)


def _clamp_and_repair_subtitles_in_order(
    segments: List[Dict[str, Any]],
    clip_end: float,
) -> List[Dict[str, Any]]:
    """
    Clamp start/end to ``[0, clip_end]`` and remove overlaps in timeline order.

    Unlike ``_repair_multimodal_segment_times``, this does **not** sort by
    start time, so each dict keeps its pairing of ``text`` / ``source_text``
    with adjusted times.
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
    """Return (global_index, segment) pairs for cues overlapping the window."""
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
    """Build JSON-serializable cues with clip-relative SRT timestamps."""
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
                "start": seconds_to_srt(rel_s),
                "end": seconds_to_srt(rel_e),
                "text": source,
                "translation": translation,
            }
        )
    return cues


def parse_review_response(
    reviewed: TimingReviewResponse,
    expected_count: int,
) -> Optional[List[Dict[str, Any]]]:
    """Convert a validated TimingReviewResponse into timing rows (seconds)."""
    if expected_count <= 0:
        return []
    rows: List[Dict[str, Any]] = []
    for item in reviewed.segments:
        try:
            start = srt_to_seconds(item.start)
            end = srt_to_seconds(item.end)
        except ValueError as exc:
            print(f"  Warning: skipping review segment with bad timestamp: {exc}")
            continue
        rows.append({"id": item.id, "start": start, "end": end})
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
            "- Timestamps in your output MUST be SRT format strings "
            "(HH:MM:SS,mmm), relative to the start of THIS clip "
            "(00:00:00,000 = clip start). Example: \"00:00:02,500\" = 2.5s in.",
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
            "Negative example: if speech is clearly audible at 00:00:00,200 "
            "but start is 00:00:01,200, move start toward 00:00:00,200 unless "
            "it would overlap the previous cue.",
            "",
            f"This clip is about {chunk_duration_sec:.2f} seconds long "
            f"(= {seconds_to_srt(chunk_duration_sec)}).",
            "",
            "Return ONLY valid JSON (no markdown) with this exact shape:",
            '{"segments":[{"id":0,"start":"00:00:00,000","end":"00:00:01,200"}, ...]}',
            "Each segment: id (integer, 0..n-1), start and end as SRT strings.",
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
    Call Gemini for one audio chunk and a list of timing cues.

    Args:
        chunk_audio_path: Path to the extracted chunk (e.g. MP3).
        payload_cues: Cues with clip-relative start/end and text fields.
        review_model: Gemini model id.
        source_language: Spoken language label for the prompt.
        target_language: Subtitle language label for the prompt.
        chunk_duration_sec: Clip length for the prompt text.
        usage_log_label: Label for token usage logging.

    Returns:
        (parsed timing rows, usage) or (None, usage) on failure.
    """
    try:
        client = _make_gemini_client()
    except ValueError as exc:
        print(f"Error: {exc}")
        return None, None

    try:
        with open(chunk_audio_path, "rb") as f:
            audio_bytes = f.read()
    except OSError as exc:
        print(f"Error: could not read chunk audio: {exc}")
        return None, None

    mime_type = _infer_gemini_inline_audio_mime_type(chunk_audio_path)
    prompt_intro = _build_timing_review_prompt(
        source_language, target_language, chunk_duration_sec
    )
    cues_json = json.dumps({"cues": payload_cues}, ensure_ascii=False)
    user_text = (
        f"{prompt_intro}\n\nInput cues (clip-relative SRT timestamps):\n{cues_json}"
    )

    last_usage: Optional[Dict[str, Any]] = None

    for attempt in range(_MAX_REVIEW_ATTEMPTS):
        if attempt > 0:
            delay = compute_backoff(attempt - 1)
            print(
                f"Timing review: attempt {attempt + 1}/{_MAX_REVIEW_ATTEMPTS} "
                f"retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

        uploaded_file = None
        try:
            uploaded_file = client.files.upload(
                file=chunk_audio_path,
                config=types.UploadFileConfig(mime_type=mime_type),
            )
            contents = [
                types.Part.from_text(text=user_text),
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=mime_type,
                ),
            ]
        except Exception as upload_exc:
            print(
                f"Warning: File API upload failed ({upload_exc}); "
                "falling back to inline audio."
            )
            contents = [
                types.Part.from_text(text=user_text),
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ]
            uploaded_file = None

        try:
            response = client.models.generate_content(
                model=review_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=TimingReviewResponse,
                ),
            )
            usage = _usage_dict_from_response(response)
            last_usage = usage
            _log_review_usage(usage_log_label, response)

            reviewed: Optional[TimingReviewResponse] = response.parsed
            if reviewed is None:
                print(
                    f"Warning: timing review SDK returned no parsed response "
                    f"(attempt {attempt + 1}/{_MAX_REVIEW_ATTEMPTS})."
                )
                if attempt + 1 < _MAX_REVIEW_ATTEMPTS:
                    continue
                return None, usage

            parsed = parse_review_response(reviewed, len(payload_cues))
            if parsed is not None:
                return parsed, usage

            if attempt + 1 < _MAX_REVIEW_ATTEMPTS:
                continue
            return None, usage

        except Exception as exc:
            print(f"Error: Gemini timing review request failed: {exc}")
            if attempt + 1 < _MAX_REVIEW_ATTEMPTS:
                continue
            return None, last_usage
        finally:
            if uploaded_file is not None:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

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
    Refine subtitle start/end times using chunked multimodal review via Gemini.

    On per-chunk parse or API failure, that chunk's cues are left unchanged.
    Returns None only on hard errors (missing file, unreadable duration).
    """
    if not os.path.exists(audio_path):
        print(f"Error: audio not found for timing review: {audio_path}")
        return None
    duration = _audio_duration_seconds(audio_path)
    if duration is None or duration <= 0:
        print("Error: could not read duration for timing review.")
        return None

    span_limit = max(float(chunk_seconds), 1.0)
    n_chunks = int(math.ceil(duration / span_limit))
    working = [dict(s) for s in segments]
    tmpdir = tempfile.mkdtemp(prefix="hermecho_timing_review_")
    usage_totals: Dict[str, int] = {}

    try:
        window_start = 0.0
        chunk_idx = 0
        with tqdm(total=n_chunks, desc="Reviewing subtitle timing", unit="chunk") as pbar:
            while window_start < duration:
                span = min(span_limit, duration - window_start)
                window_end = window_start + span
                chunk_path = os.path.join(tmpdir, f"tr_{chunk_idx:04d}.mp3")

                pbar.set_postfix({"chunk": f"{chunk_idx + 1}/{n_chunks}", "time": f"{window_start:.0f}s–{window_end:.0f}s"})

                if not _ffmpeg_extract_chunk(audio_path, chunk_path, window_start, span):
                    tqdm.write(
                        "Warning: ffmpeg chunk failed for timing review; "
                        "skipping this window."
                    )
                    window_start = window_end
                    chunk_idx += 1
                    pbar.update(1)
                    continue

                indexed = slice_segments_for_window(working, window_start, window_end)
                if not indexed:
                    window_start = window_end
                    chunk_idx += 1
                    pbar.update(1)
                    continue

                index_map = [i for i, _s in indexed]
                payload = build_review_payload_cues(indexed, window_start, span)
                if not payload:
                    window_start = window_end
                    chunk_idx += 1
                    pbar.update(1)
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
                _merge_usage(usage_totals, usage)
                if parsed is not None:
                    _apply_chunk_times(working, index_map, window_start, parsed)
                else:
                    tqdm.write(
                        f"Warning: timing review chunk {chunk_idx + 1} failed; "
                        "keeping original times for cues in this window."
                    )

                pbar.update(1)
                window_start = window_end
                chunk_idx += 1

        repaired = _clamp_and_repair_subtitles_in_order(working, duration)
        pt = usage_totals.get("prompt_tokens")
        ct = usage_totals.get("completion_tokens")
        tt = usage_totals.get("total_tokens")
        parts = []
        if pt is not None:
            parts.append(f"prompt_tokens={pt}")
        if ct is not None:
            parts.append(f"completion_tokens={ct}")
        if tt is not None:
            parts.append(f"total_tokens={tt}")
        label = "Timing review API tokens — cumulative (all chunks)"
        if parts:
            print(f"{label}: " + ", ".join(parts))
        else:
            print(f"{label}: (no usage data)")
        print("Timing review finished.")
        return repaired
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
