"""
This module contains functions for transcribing audio to text.
"""
import base64
import json
import math
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from openrouter_fallback import openrouter_models_with_fallback

# Default multimodal model: Pro for better segment timing (override via CLI).
# Keep in sync with --multimodal-model default in main.py.
DEFAULT_MULTIMODAL_MODEL = "google/gemini-3.1-pro-preview"
# Long files must be split: full-file base64 payloads often hit 502 / 400
# limits on OpenRouter and upstream providers.
DEFAULT_MULTIMODAL_CHUNK_SECONDS = 600.0
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
# OpenRouter uses Referer for attribution; README clone URL as default.
_DEFAULT_OPENROUTER_REFERER = "https://github.com/coletangsy/hermecho"


def get_openrouter_http_referer() -> str:
    """
    Return the HTTP-Referer URL for OpenRouter API requests.

    Override with the OPENROUTER_HTTP_REFERER environment variable when
    routing traffic from a different site or fork.

    Returns:
        A valid http(s) URL string.
    """
    return os.environ.get(
        "OPENROUTER_HTTP_REFERER",
        _DEFAULT_OPENROUTER_REFERER,
    )


def _merge_openrouter_usage(
    totals: Dict[str, int],
    usage: Optional[Dict[str, Any]],
) -> None:
    """Add OpenRouter-style usage dict into running totals."""
    if not usage or not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = usage.get(key)
        if val is not None:
            totals[key] = totals.get(key, 0) + int(val)


def _log_openrouter_token_usage(
    label: str,
    usage: Optional[Dict[str, Any]],
) -> None:
    """Print token usage from an OpenRouter chat completion response."""
    if not usage:
        print(f"{label}: (no usage in API response)")
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


def transcribe_audio(
    audio_path: str,
    model: str,
    language: str,
    initial_prompt: Optional[str] = None,
    temperature: float = 0.0,
) -> Optional[List[Dict]]:
    """
    Transcribes audio using the local OpenAI Whisper model.

    Args:
        audio_path: Path to the audio file.
        model: Whisper model size name (e.g. 'large').
        language: Source language code for Whisper.
        initial_prompt: Optional context string.
        temperature: Whisper sampling temperature.

    Returns:
        Segment dicts with timestamps, or None on failure.
    """
    try:
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            return None

        import whisper  # type: ignore

        print(f"Loading local Whisper model ({model})...")
        model = whisper.load_model(model)

        print(f"Transcribing audio locally (language: {language})...")
        if initial_prompt:
            print(f"Using initial prompt: {initial_prompt}")

        result = model.transcribe(  # type: ignore
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=True,
            fp16=False,
            initial_prompt=initial_prompt,
            carry_initial_prompt=True,
            temperature=temperature,
            condition_on_previous_text=False,
            no_speech_threshold=0.85,
            compression_ratio_threshold=1.7,
        )

        if not result["segments"]:
            detected_language = result.get("language", "unknown")
            print("Warning: Whisper model returned no transcription segments.")
            print(f"  - Detected language: {detected_language}")
            print(
                "  - This could be due to no speech, or the language "
                f"'({language})' being incorrect."
            )
            return []

        print("Audio transcribed successfully")
        print("Transcription: local Whisper (no API token usage).")
        return result["segments"]

    except (FileNotFoundError, RuntimeError) as e:
        print(f"An error occurred during local audio transcription: {e}")
        return None


def _infer_openrouter_audio_format(audio_path: str) -> str:
    """
    Map a file extension to OpenRouter input_audio format string.

    Unknown extensions default to mp3 (matches typical ffmpeg extract).
    """
    ext = os.path.splitext(audio_path)[1].lower().lstrip(".")
    if ext in ("wav", "mp3", "m4a", "aac", "flac", "ogg", "opus"):
        return ext if ext != "aac" else "m4a"
    return "mp3"


def _json_object_response_format_for_model(
    model_id: str,
) -> Optional[Dict[str, str]]:
    """
    Return OpenAI-style ``response_format`` for ``json_object`` mode, or None.

    Some OpenRouter audio routes (notably ``openai/gpt-audio``) return bodies
    without a standard ``choices`` array when ``response_format`` is set, so
    we omit it for those models and rely on the prompt for JSON-only output.
    """
    m = model_id.strip().lower()
    if m == "openai/gpt-audio":
        return None
    return {"type": "json_object"}


def _extract_openrouter_choice_message(
    body: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extract ``choices[0]['message']`` from an OpenRouter chat completion body.

    Args:
        body: Parsed JSON from the chat completions endpoint.

    Returns:
        ``(message_dict, None)`` on success, or ``(None, error_detail)``.
    """
    if not isinstance(body, dict):
        return None, "OpenRouter response was not a JSON object"
    err = body.get("error")
    if err is not None:
        if isinstance(err, dict):
            msg = err.get("message", str(err))
            code = err.get("code")
            if code is not None:
                return None, f"OpenRouter error ({code}): {msg}"
            return None, f"OpenRouter error: {msg}"
        return None, f"OpenRouter error: {err}"
    choices = body.get("choices")
    if choices is None:
        keys = ", ".join(sorted(body.keys()))
        return None, (
            "OpenRouter response missing 'choices' "
            f"(top-level keys: {keys or 'none'})"
        )
    if not isinstance(choices, list) or len(choices) == 0:
        return None, (
            "OpenRouter 'choices' was not a non-empty list "
            f"(got {type(choices).__name__})"
        )
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return None, "OpenRouter choices[0] was not an object"
    message = choice0.get("message")
    if not isinstance(message, dict):
        return None, "OpenRouter choices[0] missing a message object"
    return message, None


def _openrouter_message_content_to_text(content: Any) -> Optional[str]:
    """
    Normalize chat ``message["content"]`` to a single string for JSON parse.

    OpenRouter and some upstream APIs return either a string, a list of
    typed content parts (e.g. ``{"type": "text", "text": "..."}``), or
    ``None`` when the model omits text (refusal, error, or tool-only).

    Args:
        content: Raw ``message["content"]`` from the API JSON.

    Returns:
        Non-empty stripped text, or ``None`` if nothing usable is present.
    """
    if content is None:
        return None
    if isinstance(content, str):
        stripped = content.strip()
        return stripped if stripped else None
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                fragment = item.get("text")
                if isinstance(fragment, str) and fragment:
                    parts.append(fragment)
            elif isinstance(item, str) and item:
                parts.append(item)
        joined = "".join(parts).strip()
        return joined if joined else None
    return None


def _parse_json_object_from_model_text(
    raw: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """
    Parse a JSON object from model output, tolerating markdown fences.

    Args:
        raw: Model output string, or ``None`` / non-string (returns ``None``).
    """
    if raw is None or not isinstance(raw, str):
        return None
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _normalize_multimodal_segments(
    parsed: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Sort and validate segment dicts from multimodal JSON output.

    Drops entries with non-numeric times or empty text. Fixes inverted
    start/end when needed (phrase-level timing does not need word keys).
    """
    raw_list = parsed.get("segments")
    if not isinstance(raw_list, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        cleaned.append({"text": text, "start": start, "end": end})

    cleaned.sort(key=lambda s: (s["start"], s["end"]))
    return cleaned


# Multimodal models often return overlapping or out-of-span times; these
# bounds keep SRT output monotonic and within the decoded audio window.
_MIN_MULTIMODAL_SEGMENT_SEC = 0.05
_MIN_MULTIMODAL_GAP_SEC = 0.02


def _repair_multimodal_segment_times(
    segments: List[Dict[str, Any]],
    clip_end: float,
) -> List[Dict[str, Any]]:
    """
    Clamp segment times to [0, clip_end] and remove timeline overlaps.

    LLM-produced timestamps are approximate; overlaps confuse downstream
    subtitle adjustment. This pass is lossy on time only, not on text.

    Args:
        segments: Clip-relative or already-offset segments (caller sets
            clip_end to the valid window end in the same time basis).
        clip_end: Exclusive upper bound for end times (audio duration).

    Returns:
        New list of segment dicts with repaired start/end.
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
            end = min(clip_end, start + _MIN_MULTIMODAL_SEGMENT_SEC)
        if end - start < _MIN_MULTIMODAL_SEGMENT_SEC:
            end = min(clip_end, start + _MIN_MULTIMODAL_SEGMENT_SEC)
        s["start"], s["end"] = start, end
        out.append(s)

    out.sort(key=lambda x: (float(x["start"]), float(x["end"])))

    for i in range(1, len(out)):
        prev = out[i - 1]
        cur = out[i]
        p_start, p_end = float(prev["start"]), float(prev["end"])
        c_start = float(cur["start"])
        min_prev_end = p_start + _MIN_MULTIMODAL_SEGMENT_SEC
        if c_start < p_end + _MIN_MULTIMODAL_GAP_SEC:
            new_p_end = c_start - _MIN_MULTIMODAL_GAP_SEC
            if new_p_end < min_prev_end:
                new_p_end = min_prev_end
            prev["end"] = new_p_end
            if c_start < float(prev["end"]) + _MIN_MULTIMODAL_GAP_SEC:
                cur["start"] = float(prev["end"]) + _MIN_MULTIMODAL_GAP_SEC
            if float(cur["end"]) <= float(cur["start"]):
                cur["end"] = min(
                    clip_end,
                    float(cur["start"]) + _MIN_MULTIMODAL_SEGMENT_SEC,
                )

    return out


def _finalize_multimodal_segments_for_audio(
    segments: List[Dict[str, Any]],
    audio_path: str,
) -> List[Dict[str, Any]]:
    """
    Repair multimodal segments using ffprobe duration of the given file.

    If duration cannot be read, returns a shallow copy of the input list.
    """
    if not segments:
        return []
    clip_end = _audio_duration_seconds(audio_path)
    if clip_end is None or clip_end <= 0:
        return [dict(s) for s in segments]
    return _repair_multimodal_segment_times(segments, clip_end)


def _audio_duration_seconds(audio_path: str) -> Optional[float]:
    """
    Return media duration in seconds using ffprobe (no full decode).
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        out = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return float(out.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        print("Error: ffprobe failed or returned invalid duration.")
        return None


def _ffmpeg_extract_chunk(
    source_path: str,
    output_path: str,
    start_sec: float,
    duration_sec: float,
) -> bool:
    """
    Write one contiguous audio chunk with ffmpeg (re-encode for accuracy).
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        source_path,
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-acodec",
        "libmp3lame",
        "-q:a",
        "2",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Error: ffmpeg chunk extract failed: {e}")
        return False


def _build_multimodal_prompt(language: str, initial_prompt: Optional[str]) -> str:
    """
    Build user instructions for JSON transcript with clip-relative times.

    Segmentation is driven by how speech is delivered (prosody, pauses),
    not by clock targets: short, readable cues aligned to audible phases.
    """
    parts = [
        "You are a professional transcription assistant.",
        f"The primary language is {language!r}.",
        "Transcribe all intelligible speech.",
        "Return ONLY valid JSON (no markdown) with this shape:",
        '{"segments":[{"start":0.0,"end":1.5,"text":"phrase here"}, ...]}',
        "Rules:",
        "- start and end are seconds from the beginning of THIS audio clip.",
        "- Segment by how the speaker delivers content (prosodic phrases, "
        "clauses, breaths, short bursts), not by a time quota or clock.",
        "- Prefer several short segments over one long line: each segment "
        "should carry one coherent spoken chunk.",
        "- If one grammatical sentence is spoken in multiple audible phases "
        "(clear pause, breath, or clause boundary between them), use one "
        "JSON segment per phase; each segment's text is only the words "
        "spoken in that phase.",
        "- Do not merge distinct speech phases into one cue just because "
        "they belong to the same written sentence.",
        "- Do not split a single uninterrupted spoken phrase only to shorten "
        "lines; a new segment needs a real audible boundary in the audio.",
        "- Use phrase-level segments (not word-by-word).",
        "- Times must be non-overlapping and in ascending order.",
        "- start and end must stay within 0 and the length of this clip.",
        "- Omit non-speech; do not invent content.",
        "Timing quality (derive times from the audio, not from reading "
        "speed):",
        "- start: first instant that phrase is clearly audible; skip "
        "leading silence.",
        "- end: last instant that phrase is still fully audible; never "
        "end inside a word or syllable.",
        "- Put breaks only at audible clause or breath pauses; silence "
        "after a phrase belongs to that phrase's end time, not the next "
        "start.",
        "- Do not pad: a segment's window must match when that text is "
        "spoken; no extra seconds covering silence before or after.",
        "- If two boundaries seem plausible, prefer the split at the "
        "stronger pause or clearer consonant release.",
    ]
    if initial_prompt:
        parts.append(f"Additional context from the user:\n{initial_prompt}")
    return "\n".join(parts)


def _transcribe_clip_openrouter(
    audio_path: str,
    multimodal_model: str,
    language: str,
    initial_prompt: Optional[str],
    usage_log_label: str = "Transcription (multimodal) API tokens",
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """
    One OpenRouter multimodal request for a single audio file path.

    Returns:
        (segments, usage) where usage is the API's usage object if present,
        or (None, None) / (None, usage) on failure.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY is not set.")
        return None, None

    try:
        with open(audio_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("ascii")
    except OSError as e:
        print(f"Error: could not read audio file: {e}")
        return None, None

    audio_format = _infer_openrouter_audio_format(audio_path)
    user_text = _build_multimodal_prompt(language, initial_prompt)

    payload: Dict[str, Any] = {
        "model": multimodal_model,
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
        # Deterministic decoding for stable wording and timing JSON.
        "temperature": 0,
    }
    fmt = _json_object_response_format_for_model(multimodal_model)
    if fmt is not None:
        payload["response_format"] = fmt

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": get_openrouter_http_referer(),
        "X-Title": "Hermecho Multimodal Transcribe",
    }

    max_http_attempts = 3
    backoff_sec = 2.5
    resp: Optional[requests.Response] = None
    for http_try in range(max_http_attempts):
        try:
            resp = requests.post(
                OPENROUTER_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=900,
            )
        except requests.RequestException as e:
            print(f"Error: OpenRouter request failed: {e}")
            return None, None

        if resp.status_code == 200:
            break

        transient = resp.status_code in (502, 503, 504)
        if transient and http_try + 1 < max_http_attempts:
            print(
                "Warning: OpenRouter returned "
                f"{resp.status_code}; retrying in {backoff_sec:.0f}s "
                f"({http_try + 2}/{max_http_attempts})..."
            )
            time.sleep(backoff_sec)
            continue

        print(
            "Error: OpenRouter returned "
            f"{resp.status_code}: {resp.text[:500]}"
        )
        return None, None

    if resp is None or resp.status_code != 200:
        return None, None

    try:
        body = resp.json()
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON from OpenRouter: {e}")
        return None, None

    usage: Optional[Dict[str, Any]] = None
    if isinstance(body, dict):
        raw_usage = body.get("usage")
        if isinstance(raw_usage, dict):
            usage = raw_usage

    message, shape_err = _extract_openrouter_choice_message(body)
    if shape_err is not None or message is None:
        print(f"Error: {shape_err}")
        _log_openrouter_token_usage(usage_log_label, usage)
        return None, usage

    choices = body.get("choices")
    choice0 = choices[0] if isinstance(choices, list) and choices else {}

    _log_openrouter_token_usage(usage_log_label, usage)

    content = _openrouter_message_content_to_text(message.get("content"))
    if content is None:
        finish_reason = (
            choice0.get("finish_reason")
            if isinstance(choice0, dict)
            else None
        )
        refusal = (
            message.get("refusal")
            if isinstance(message, dict)
            else None
        )
        print(
            "Error: OpenRouter returned no assistant text in "
            f"choices[0].message (finish_reason={finish_reason!r}, "
            f"refusal={refusal!r}). The model may have refused the request, "
            "hit a provider error, or returned an unsupported content shape."
        )
        return None, usage

    parsed = _parse_json_object_from_model_text(content)
    if not parsed:
        print("Error: model output was not valid JSON with a segments list.")
        return None, usage

    segments = _normalize_multimodal_segments(parsed)
    if not segments:
        print("Warning: multimodal model returned no usable segments.")
        return [], usage
    segments = _finalize_multimodal_segments_for_audio(segments, audio_path)
    return segments, usage


def _transcribe_multimodal_by_chunks(
    audio_path: str,
    model_id: str,
    language: str,
    initial_prompt: Optional[str],
    duration_sec: float,
    chunk_seconds: float,
) -> Optional[List[Dict[str, Any]]]:
    """
    Transcribe long audio via overlapping time windows and ffmpeg chunks.

    Each chunk is sent as its own OpenRouter request (smaller payload than
    one base64 blob for the entire file). Segment times from the model are
    clip-relative per chunk; this function offsets them to the full timeline.

    Args:
        audio_path: Full source audio path (used for ffmpeg input and final
            duration repair).
        model_id: OpenRouter model slug for this attempt.
        language: Source language label for the prompt.
        initial_prompt: Optional user context for the prompt.
        duration_sec: Total media length in seconds (from ffprobe).
        chunk_seconds: Window length in seconds (at least 60; caller clamps).

    Returns:
        Merged segment dicts for the full file, or None if any chunk fails.
    """
    if chunk_seconds < 60.0:
        return None

    n_chunks = int(math.ceil(duration_sec / chunk_seconds))
    merged: List[Dict[str, Any]] = []

    print(
        f"Long audio ({duration_sec:.0f}s): transcribing in up to "
        f"{n_chunks} chunk(s) of ~{chunk_seconds:.0f}s (API size limits)."
    )

    with tempfile.TemporaryDirectory(prefix="hermecho_mm_") as tmpdir:
        offset = 0.0
        idx = 0
        while offset < duration_sec:
            idx += 1
            span = min(chunk_seconds, duration_sec - offset)
            chunk_path = os.path.join(tmpdir, f"chunk_{idx:04d}.mp3")
            if not _ffmpeg_extract_chunk(audio_path, chunk_path, offset, span):
                return None
            t_end = offset + span
            print(
                f"  Multimodal chunk {idx}/{n_chunks}: "
                f"{offset:.0f}s–{t_end:.0f}s..."
            )
            segs, _usage = _transcribe_clip_openrouter(
                chunk_path,
                model_id,
                language,
                initial_prompt,
            )
            if segs is None:
                return None
            for raw in segs:
                row = dict(raw)
                try:
                    s0 = float(row["start"]) + offset
                    s1 = float(row["end"]) + offset
                except (KeyError, TypeError, ValueError):
                    continue
                row["start"] = s0
                row["end"] = s1
                merged.append(row)
            offset = t_end

    merged = _finalize_multimodal_segments_for_audio(merged, audio_path)
    return merged


def transcribe_audio_multimodal(
    audio_path: str,
    language: str,
    multimodal_model: str = DEFAULT_MULTIMODAL_MODEL,
    initial_prompt: Optional[str] = None,
    chunk_seconds: float = DEFAULT_MULTIMODAL_CHUNK_SECONDS,
) -> Optional[List[Dict[str, Any]]]:
    """
    Transcribe audio via OpenRouter using a multimodal model (e.g. Gemini).

    Segments are prompted as short, phase-aligned cues (natural pauses and
    prosody), not fixed-duration slices.

    Long files are split with ffmpeg into multiple requests (default chunk
    length ``DEFAULT_MULTIMODAL_CHUNK_SECONDS``) so payloads stay within
    gateway and provider limits. Pass ``chunk_seconds=0`` to force a single
    request for the entire file (not recommended for long media).

    Args:
        audio_path: Path to audio (e.g. ffmpeg-extracted MP3).
        language: BCP-47 or short label (e.g. 'ko') for prompting.
        multimodal_model: OpenRouter model id.
        initial_prompt: Optional domain context (names, mixed languages).
        chunk_seconds: Max seconds per API request; ``0`` disables chunking.

    Returns:
        List of segment dicts with start, end, text; None on hard errors.
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return None

    duration = _audio_duration_seconds(audio_path)
    if duration is None:
        return None

    eff_chunk = 0.0
    if chunk_seconds > 0:
        eff_chunk = max(60.0, float(chunk_seconds))
    use_chunks = eff_chunk > 0 and duration > eff_chunk

    models = openrouter_models_with_fallback(
        multimodal_model,
        multimodal_transcription=True,
    )
    for attempt, model_id in enumerate(models):
        if attempt == 0:
            mode = (
                f"~{eff_chunk:.0f}s chunks"
                if use_chunks
                else "single request"
            )
            print(
                f"Transcribing with multimodal model ({model_id}), "
                f"{duration:.0f}s audio ({mode})..."
            )
        else:
            print(
                f"Primary multimodal model failed; retrying with fallback "
                f"({model_id})..."
            )

        if use_chunks:
            segs = _transcribe_multimodal_by_chunks(
                audio_path,
                model_id,
                language,
                initial_prompt,
                duration,
                eff_chunk,
            )
        else:
            segs, _usage = _transcribe_clip_openrouter(
                audio_path,
                model_id,
                language,
                initial_prompt,
            )

        if segs is not None:
            return segs
    return None
