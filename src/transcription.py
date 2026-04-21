"""
This module contains functions for transcribing audio to text.
"""
import math
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from google import genai
from google.genai import types

from models import TranscriptResponse, seconds_to_srt, srt_to_seconds
from retry import compute_backoff


DEFAULT_MULTIMODAL_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_MULTIMODAL_CHUNK_SECONDS = 5.0 * 60.0

_CHUNK_AUDIO_BITRATE = "32k"
_CHUNK_AUDIO_CHANNELS = "1"
_CHUNK_MAX_RETRIES = 3


def _make_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def _log_gemini_token_usage(label: str, response: Any) -> None:
    """Print token counts from a Gemini SDK response's usage_metadata."""
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


def _infer_gemini_inline_audio_mime_type(audio_path: str) -> str:
    """Map a file extension to a MIME type for Gemini audio parts."""
    ext = os.path.splitext(audio_path)[1].lower().lstrip(".")
    if ext in ("mp3", "mpeg"):
        return "audio/mpeg"
    if ext == "wav":
        return "audio/wav"
    if ext in ("m4a", "aac", "mp4"):
        return "audio/mp4"
    if ext == "flac":
        return "audio/flac"
    if ext in ("ogg", "opus"):
        return "audio/ogg"
    return "audio/mpeg"


def _strip_utf8_bom(text: str) -> str:
    if text.startswith("\ufeff"):
        return text[1:]
    return text


def _strip_outer_markdown_json_fence(text: str) -> str:
    """If ``text`` is wrapped in one markdown fence, unwrap it."""
    stripped = text.strip()
    fence = re.match(
        r"^```(?:json)?\s*([\s\S]*?)\s*```$",
        stripped,
        re.IGNORECASE,
    )
    if fence:
        return fence.group(1).strip()
    return stripped


def _first_balanced_json_object_slice(text: str) -> Optional[str]:
    """Return the substring of the first top-level ``{...}`` balanced for JSON."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _coerce_parsed_transcript_root(
    parsed: Any,
) -> Optional[Dict[str, Any]]:
    """Normalize decoded JSON to ``{"segments": [...]}`` when possible."""
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        if not parsed:
            return {"segments": []}
        if all(isinstance(x, dict) for x in parsed):
            return {"segments": parsed}
    return None


def _parse_json_object_from_model_text(
    raw: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """
    Parse transcript JSON from model output with several recovery strategies.

    Handles UTF-8 BOM, markdown fences, leading/trailing prose, and a top-level
    segment array instead of an object.
    """
    if raw is None or not isinstance(raw, str):
        return None
    text = _strip_utf8_bom(raw.strip())
    text = _strip_outer_markdown_json_fence(text)

    candidates: List[str] = []
    if text:
        candidates.append(text)
    inner = _first_balanced_json_object_slice(text)
    if inner and inner not in candidates:
        candidates.append(inner)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            sliced = _first_balanced_json_object_slice(candidate)
            if sliced is None or sliced == candidate:
                continue
            try:
                parsed = json.loads(sliced)
            except json.JSONDecodeError:
                continue
        coerced = _coerce_parsed_transcript_root(parsed)
        if isinstance(coerced, dict):
            return coerced
    return None


def _normalize_multimodal_segments(
    parsed: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Sort and validate segment dicts from multimodal JSON output."""
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


_MIN_MULTIMODAL_SEGMENT_SEC = 0.05
_MIN_MULTIMODAL_GAP_SEC = 0.02


def _repair_multimodal_segment_times(
    segments: List[Dict[str, Any]],
    clip_end: float,
) -> List[Dict[str, Any]]:
    """Clamp segment times to [0, clip_end] and remove timeline overlaps."""
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
    """Repair multimodal segments using ffprobe duration of the given file."""
    if not segments:
        return []
    clip_end = _audio_duration_seconds(audio_path)
    if clip_end is None or clip_end <= 0:
        return [dict(s) for s in segments]
    return _repair_multimodal_segment_times(segments, clip_end)


def _audio_duration_seconds(audio_path: str) -> Optional[float]:
    """Return media duration in seconds using ffprobe (no full decode)."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
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
    """Write one contiguous audio chunk with ffmpeg at speech-optimised quality."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", source_path,
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-acodec", "libmp3lame",
        "-b:a", _CHUNK_AUDIO_BITRATE,
        "-ac", _CHUNK_AUDIO_CHANNELS,
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Error: ffmpeg chunk extract failed: {e}")
        return False


def _build_multimodal_prompt(language: str, initial_prompt: Optional[str]) -> str:
    """Build user instructions for JSON transcript with SRT-format timestamps."""
    parts = [
        "You are a professional transcription assistant.",
        f"The primary language is {language!r}.",
        "Transcribe all intelligible speech.",
        "Output format (strict):",
        "- Reply with exactly ONE JSON object and nothing else.",
        "- No markdown code fences (no triple backticks), no labels, and no "
        "commentary before or after the JSON.",
        "- Use standard JSON: double-quoted keys and strings; escape internal "
        'double quotes in transcript text as \\".',
        '- Top-level object must be: {"segments":[...]}',
        "- Each segment has three fields: \"start\", \"end\" (SRT timestamp "
        "strings in HH:MM:SS,mmm format, relative to the start of THIS clip), "
        'and "text" (the spoken words).',
        "Example (structure only — times are relative to this clip's start):",
        '{"segments":['
        '{"start":"00:00:00,000","end":"00:00:01,200","text":"short phrase"},'
        '{"start":"00:00:01,500","end":"00:00:03,100","text":"another phrase"}'
        "]}",
        "Rules:",
        "- start and end MUST be SRT timestamp strings in EXACTLY this format: "
        "HH:MM:SS,mmm — two-digit hours, two-digit minutes, two-digit seconds, "
        "a COMMA (not a colon or dot), then exactly three-digit milliseconds. "
        "Always include all four parts. CORRECT: \"00:02:41,000\". "
        "WRONG: \"02:41:000\" (missing hours, colon before ms). "
        "WRONG: \"02:41:00\" (missing milliseconds). "
        "WRONG: \"00:02:41.000\" (dot instead of comma).",
        "- Timestamps must stay within 00:00:00,000 and the length of this clip.",
        "- Hard limit: every segment MUST be 15 seconds or shorter. "
        "No exceptions. If a speaker talks continuously for longer than "
        "15 seconds without stopping, split at the nearest clause boundary "
        "or breath pause and open a new segment.",
        "- Target 5–12 seconds per segment. Never produce a single segment "
        "covering a multi-sentence passage or a long unbroken monologue.",
        "- Segment by how the speaker delivers content (prosodic phrases, "
        "clauses, breaths, short bursts), not by a time quota or clock.",
        "- Korean speech split cues: conjunctions and discourse markers "
        "(그래서, 근데, 왜냐면, 그니까, 그러다가, 그래가지고, 해서, 하지만), "
        "sentence-final endings (-요, -죠, -어요, -거든요, -습니다, -이에요), "
        "topic/subject particles that open a new clause (은/는 after a pause), "
        "and any audible pause or in-breath in the recording.",
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
        "- Timestamps must be non-overlapping and in ascending order.",
        "- Omit non-speech; do not invent content.",
        "Timing quality (derive times from the audio, not from reading speed):",
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


def transcribe_audio(
    audio_path: str,
    model: str,
    language: str,
    initial_prompt: Optional[str] = None,
    temperature: float = 0.0,
) -> Optional[List[Dict]]:
    """
    Transcribes audio using the local OpenAI Whisper model.
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


def _transcribe_clip_gemini_sdk(
    audio_path: str,
    gemini_model: str,
    language: str,
    initial_prompt: Optional[str],
    usage_log_label: str = "Transcription (Gemini) API tokens",
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """
    One Google AI Studio ``generateContent`` call via the google-genai SDK.

    Uses pydantic ``response_schema`` (TranscriptResponse) so the SDK enforces
    the JSON shape and returns a validated ``response.parsed`` object, avoiding
    manual JSON parsing and thought_signature contamination issues.
    """
    try:
        client = _make_gemini_client()
    except ValueError as exc:
        print(f"Error: {exc}")
        return None, None

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return None, None

    mime_type = _infer_gemini_inline_audio_mime_type(audio_path)
    user_text = _build_multimodal_prompt(language, initial_prompt)

    max_attempts = 4
    for attempt in range(max_attempts):
        if attempt > 0:
            delay = compute_backoff(attempt - 1)
            print(
                f"  Gemini transcription: retrying in {delay:.1f}s "
                f"({attempt + 1}/{max_attempts})..."
            )
            time.sleep(delay)

        uploaded_file = None
        try:
            try:
                uploaded_file = client.files.upload(
                    file=audio_path,
                    config=types.UploadFileConfig(mime_type=mime_type),
                )
                audio_part = types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=mime_type,
                )
                raw_mb = os.path.getsize(audio_path) / (1024 * 1024)
                print(f"  File API upload OK ({raw_mb:.1f} MB) → {uploaded_file.uri}")
            except Exception as upload_exc:
                raw_mb = os.path.getsize(audio_path) / (1024 * 1024)
                print(
                    f"  Warning: File API upload failed ({upload_exc}) for "
                    f"{raw_mb:.1f} MB chunk; falling back to inline bytes."
                )
                with open(audio_path, "rb") as fh:
                    audio_bytes = fh.read()
                audio_part = types.Part.from_bytes(
                    data=audio_bytes, mime_type=mime_type
                )
                uploaded_file = None

            response = client.models.generate_content(
                model=gemini_model,
                contents=[
                    types.Part.from_text(text=user_text),
                    audio_part,
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=TranscriptResponse,
                ),
            )

            _log_gemini_token_usage(usage_log_label, response)

            transcript: Optional[TranscriptResponse] = response.parsed
            if transcript is None:
                print(
                    "Error: SDK returned no parsed transcript (response_schema "
                    "validation failed). Raw text preview: "
                    + repr((response.text or "")[:400])
                )
                if attempt + 1 < max_attempts:
                    continue
                return None, None

            segments: List[Dict[str, Any]] = []
            for seg in transcript.segments:
                try:
                    start = srt_to_seconds(seg.start)
                    end = srt_to_seconds(seg.end)
                except ValueError as ts_err:
                    print(f"  Warning: skipping segment with bad timestamp: {ts_err}")
                    continue
                text = seg.text.strip()
                if not text:
                    continue
                if end < start:
                    start, end = end, start
                segments.append({"start": start, "end": end, "text": text})

            if not segments:
                print("Warning: Gemini model returned no usable segments.")
                return [], None

            segments = _finalize_multimodal_segments_for_audio(segments, audio_path)
            return segments, None

        except Exception as exc:
            print(f"Error: Gemini generateContent failed: {exc}")
            if attempt + 1 < max_attempts:
                continue
            return None, None
        finally:
            if uploaded_file is not None:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

    return None, None


_ClipFn = Callable[
    [str, str, str, Optional[str]],
    Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]],
]


def _transcribe_multimodal_by_chunks(
    audio_path: str,
    model_id: str,
    language: str,
    initial_prompt: Optional[str],
    duration_sec: float,
    chunk_seconds: float,
    transcribe_clip: "_ClipFn",
    inter_chunk_sleep: float = 2.0,
) -> Optional[List[Dict[str, Any]]]:
    """
    Transcribe long audio via time windows and ffmpeg chunks.

    Each chunk is its own API request. Per-chunk failures are retried up to
    ``_CHUNK_MAX_RETRIES`` times with exponential backoff; if a chunk still
    fails after all retries it is skipped with a warning rather than aborting
    the entire job.
    """
    if chunk_seconds < 60.0:
        return None

    n_chunks = int(math.ceil(duration_sec / chunk_seconds))
    merged: List[Dict[str, Any]] = []
    failed_chunks: List[int] = []

    print(
        f"Long audio ({duration_sec:.0f}s): transcribing in up to "
        f"{n_chunks} chunk(s) of ~{chunk_seconds:.0f}s."
    )

    with tempfile.TemporaryDirectory(prefix="hermecho_mm_") as tmpdir:
        offset = 0.0
        idx = 0
        with tqdm(total=n_chunks, desc="Transcribing chunks", unit="chunk") as pbar:
            while offset < duration_sec:
                idx += 1
                span = min(chunk_seconds, duration_sec - offset)
                chunk_path = os.path.join(tmpdir, f"chunk_{idx:04d}.mp3")
                t_end = offset + span

                pbar.set_postfix({"chunk": f"{idx}/{n_chunks}", "time": f"{offset:.0f}s–{t_end:.0f}s"})

                if not _ffmpeg_extract_chunk(audio_path, chunk_path, offset, span):
                    tqdm.write(
                        f"  Warning: ffmpeg failed for chunk {idx}/{n_chunks} "
                        f"({offset:.0f}s–{t_end:.0f}s); skipping."
                    )
                    failed_chunks.append(idx)
                    offset = t_end
                    pbar.update(1)
                    continue

                try:
                    chunk_mb = os.path.getsize(chunk_path) / (1024 * 1024)
                except OSError:
                    chunk_mb = 0.0

                segs: Optional[List[Dict[str, Any]]] = None
                for retry in range(_CHUNK_MAX_RETRIES):
                    suffix = f" (retry {retry}/{_CHUNK_MAX_RETRIES - 1})" if retry else ""
                    tqdm.write(
                        f"  Multimodal chunk {idx}/{n_chunks}: "
                        f"{offset:.0f}s–{t_end:.0f}s ({chunk_mb:.1f} MB){suffix}..."
                    )
                    segs, _usage = transcribe_clip(
                        chunk_path, model_id, language, initial_prompt
                    )
                    if segs is not None:
                        break
                    if retry + 1 < _CHUNK_MAX_RETRIES:
                        tqdm.write(
                            f"  Chunk {idx} failed; waiting 60s before retry "
                            f"(rate-limit cooldown)..."
                        )
                        time.sleep(60.0)

                if segs is None:
                    tqdm.write(
                        f"  Warning: chunk {idx}/{n_chunks} failed after all attempts; "
                        f"skipping this window ({offset:.0f}s–{t_end:.0f}s)."
                    )
                    failed_chunks.append(idx)
                else:
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
                    if idx < n_chunks and inter_chunk_sleep > 0:
                        tqdm.write(
                            f"  Waiting {inter_chunk_sleep:.0f}s before next chunk "
                            "(rate-limit window)..."
                        )
                        time.sleep(inter_chunk_sleep)

                pbar.update(1)
                offset = t_end

    if failed_chunks:
        pct = 100 * len(failed_chunks) / n_chunks
        print(
            f"Warning: {len(failed_chunks)}/{n_chunks} chunk(s) failed "
            f"({pct:.0f}% of audio). Chunks: {failed_chunks}. "
            "Transcription may have gaps in those windows."
        )

    if not merged and failed_chunks:
        print("Error: all transcription chunks failed; returning None.")
        return None

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
    Transcribe audio with a Gemini multimodal model via Google AI Studio.

    Uses the google-genai SDK with the File API for audio upload. Long files
    are split with ffmpeg into multiple requests (default chunk length
    ``DEFAULT_MULTIMODAL_CHUNK_SECONDS``) so payloads stay within limits.
    Pass ``chunk_seconds=0`` to force a single request for the entire file
    (not recommended for long media).

    Args:
        audio_path: Path to audio (e.g. ffmpeg-extracted MP3).
        language: BCP-47 or short label (e.g. 'ko') for prompting.
        multimodal_model: Gemini model id (e.g. 'gemini-3.1-flash-lite-preview').
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

    mode = f"~{eff_chunk:.0f}s chunks" if use_chunks else "single request"
    print(
        f"Transcribing with multimodal model ({multimodal_model}) via "
        f"Google AI Studio, {duration:.0f}s audio ({mode})..."
    )

    if use_chunks:
        return _transcribe_multimodal_by_chunks(
            audio_path,
            multimodal_model,
            language,
            initial_prompt,
            duration,
            eff_chunk,
            _transcribe_clip_gemini_sdk,
            inter_chunk_sleep=60.0,
        )
    else:
        segs, _usage = _transcribe_clip_gemini_sdk(
            audio_path, multimodal_model, language, initial_prompt
        )
        return segs
