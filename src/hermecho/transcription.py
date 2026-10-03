"""
Whisper transcription with optional MLX and explicit OpenRouter backends.
"""
import copy
import importlib.util
import math
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

from .asr_comparison import DEFAULT_EVIDENCE_DIR, evidence_allows_mlx
from .openrouter_transcription import DEFAULT_OPENROUTER_TRANSCRIPTION_MODEL
from .progress import emit_progress


MLX_LARGE_V3_MODEL = "mlx-community/whisper-large-v3-mlx"
MLX_MODEL_NAMES = {
    "large": MLX_LARGE_V3_MODEL,
    "large-v3": MLX_LARGE_V3_MODEL,
}


def _mlx_model_path(model: str) -> str:
    """Prefer the existing local Hugging Face snapshot over a network lookup."""
    model_repo = MLX_MODEL_NAMES[model]
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return model_repo
    cached_config = try_to_load_from_cache(
        repo_id=model_repo,
        filename="config.json",
    )
    if isinstance(cached_config, str):
        snapshot_dir = Path(cached_config).parent
        if any(
            (snapshot_dir / filename).exists()
            for filename in ("weights.safetensors", "weights.npz")
        ):
            return str(snapshot_dir)
    return model_repo


def validate_mlx_backend(model: str) -> Optional[str]:
    """Return an actionable error when MLX cannot run with this model."""
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "arm64e"}:
        return (
            "MLX Whisper requires Apple Silicon. "
            "Use --transcription-backend whisper on this machine."
        )
    if model not in MLX_MODEL_NAMES:
        return (
            "MLX Whisper supports only large-v3. "
            "Use --model large or --model large-v3."
        )
    if importlib.util.find_spec("mlx_whisper") is None:
        return 'MLX Whisper is not installed. Install it with `python -m pip install -e ".[mlx]"`.'
    return None


def resolve_transcription_backend(
    requested_backend: str,
    model: str,
    comparison_evidence_dir: str | Path | None = None,
) -> str:
    """Resolve ``auto`` only after supported MLX has approved faster evidence."""
    if requested_backend != "auto":
        return requested_backend
    evidence_dir = DEFAULT_EVIDENCE_DIR if comparison_evidence_dir is None else comparison_evidence_dir
    if validate_mlx_backend(model) is None and evidence_allows_mlx(evidence_dir, model=model):
        return "mlx"
    return "whisper"


def _normalise_mlx_result(result: Any) -> tuple[str, List[Dict]]:
    """Return MLX segments in the existing Whisper segment and word schema."""
    if not isinstance(result, dict) or not isinstance(result.get("segments"), list):
        raise RuntimeError("MLX Whisper returned an invalid transcription result.")

    detected_language = result.get("language", "unknown")
    if not isinstance(detected_language, str):
        raise RuntimeError("MLX Whisper returned an invalid detected language.")

    normalised_segments: List[Dict] = []
    excluded_segments = 0
    excluded_words = 0
    for segment in result["segments"]:
        text = segment.get("text") if isinstance(segment, dict) else None
        words = segment.get("words") if isinstance(segment, dict) else None
        if (
            not isinstance(segment, dict)
            or not isinstance(text, str)
            or not isinstance(words, list)
        ):
            raise RuntimeError("MLX Whisper returned an invalid transcription segment.")
        if not text.strip() and not words:
            continue
        if not all(
            isinstance(word, dict) and isinstance(word.get("word"), str)
            for word in words
        ):
            raise RuntimeError("MLX Whisper returned an invalid Source Word timestamp.")
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("MLX Whisper returned invalid segment timestamps.") from error
        if not math.isfinite(start) or not math.isfinite(end) or start > end:
            excluded_segments += 1
            continue

        normalised_words = []
        invalid_word_timestamp = False
        for word in words:
            try:
                word_start = float(word["start"])
                word_end = float(word["end"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("MLX Whisper returned invalid Source Word timestamp.") from error
            if (
                not math.isfinite(word_start)
                or not math.isfinite(word_end)
                or word_start > word_end
            ):
                excluded_words += 1
                invalid_word_timestamp = True
                break
            normalised_words.append(
                {
                    "word": word["word"],
                    "start": word_start,
                    "end": word_end,
                }
            )
        if invalid_word_timestamp:
            excluded_segments += 1
            continue

        normalised_segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "words": normalised_words,
            }
        )

    if excluded_segments or excluded_words:
        print(
            "Warning: MLX Whisper excluded "
            f"{excluded_segments} segment(s) and {excluded_words} word(s) "
            "with non-finite or reversed timestamps."
        )

    return detected_language, normalised_segments


def _transcribe_with_mlx(
    audio_path: str,
    model: str,
    language: Optional[str],
    temperature: float,
    *,
    clip_timestamps: Optional[str] = None,
) -> List[Dict]:
    error = validate_mlx_backend(model)
    if error:
        raise RuntimeError(error)

    import mlx_whisper  # type: ignore

    mlx_model = _mlx_model_path(model)

    print(f"Loading MLX Whisper model from {mlx_model}...")
    clip_options = {"clip_timestamps": clip_timestamps} if clip_timestamps is not None else {}
    result = mlx_whisper.transcribe(  # type: ignore
        audio_path,
        path_or_hf_repo=mlx_model,
        language=language,
        word_timestamps=True,
        verbose=True,
        temperature=temperature,
        condition_on_previous_text=False,
        no_speech_threshold=0.85,
        compression_ratio_threshold=1.7,
        **clip_options,
    )
    detected_language, segments = _normalise_mlx_result(result)
    if not segments:
        print("Warning: MLX Whisper returned no transcription segments.")
        print(f"  - Detected language: {detected_language}")
        return []

    print(f"MLX Whisper detected language: {detected_language}")
    print("Audio transcribed successfully")
    print("Transcription: MLX Whisper (no API token usage).")
    return segments


def repair_mlx_word_timing(
    audio_path: str,
    segments: List[Dict],
    model: str,
    language: Optional[str],
    temperature: float,
    *,
    audit_path: Optional[str] = None,
) -> List[Dict]:
    """Acquire fresh local evidence around overlaps, joining only at stable anchors."""
    from .openrouter_transcription import (
        _ffprobe_duration, _splice_boundary_words,
        _validate_absolute_words, _write_json,
    )
    from .checkpoints import fingerprint_file

    words = [copy.deepcopy(word) for segment in segments for word in segment.get("words", [])]
    boundaries = []
    for left, right in zip(words, words[1:]):
        if right["start"] < left["end"] and not math.isclose(
            right["start"], left["end"], rel_tol=0, abs_tol=1e-9
        ):
            boundaries.append((left["end"] + right["start"]) / 2)
    if not boundaries:
        return segments
    if len(boundaries) > 8:
        raise RuntimeError("MLX timing recovery requires manual review: more than 8 overlapping word boundaries.")

    duration = _ffprobe_duration(Path(audio_path))
    audit = {
        "version": 1, "status": "partial", "backend": "mlx", "model": model,
        "language": language, "temperature": temperature,
        "audio_sha256": fingerprint_file(audio_path),
        "original_segments": copy.deepcopy(segments), "windows": [],
    }

    def save_audit():
        if audit_path:
            _write_json(Path(audit_path), audit)

    save_audit()
    for boundary in boundaries:
        # An earlier recovery window may already have covered a nearby overlap.
        if not any(left["end"] > right["start"] and
                   abs((left["end"] + right["start"]) / 2 - boundary) < 1e-6
                   for left, right in zip(words, words[1:])):
            continue
        start = max(0.0, math.floor(boundary - 10))
        end = min(duration, start + 24)
        emit_progress("transcription", "running", f"Re-transcribing MLX timing near {boundary:g}s")
        # Native clipping retains the full audio's time origin and avoids
        # introducing another MP3 encode and a fractional timestamp offset.
        fresh = _transcribe_with_mlx(
            audio_path, model, language, temperature, clip_timestamps=f"{start},{end}",
        )
        bridge = [
            copy.deepcopy(word)
            for segment in fresh for word in segment["words"]
        ]
        record = {"start": start, "end": end, "boundary": boundary, "segments": fresh}
        audit["windows"].append(record)
        save_audit()
        try:
            _validate_absolute_words(bridge, duration)
            if any(word["start"] < start or word["end"] > end for word in bridge):
                raise RuntimeError("MLX timing recovery returned words outside the requested window.")
            left = [word for word in words if start <= word["start"] < boundary]
            right = [word for word in words if boundary <= word["start"] <= end]
            acquired = _splice_boundary_words(left, right, bridge, boundary, backend_label="MLX")
            words = (
                [word for word in words if word["start"] < start]
                + acquired
                + [word for word in words if word["start"] > end]
            )
        except RuntimeError as error:
            record["error"] = str(error)
            save_audit()
            raise RuntimeError(f"MLX timing recovery near {boundary:g}s failed: {error}") from error
    _validate_absolute_words(words, duration)
    # Retain every unaffected segment and its exact text/timing; only the
    # acquired interval is reconstructed from its new Source Words.
    result = []
    position = 0
    for segment in segments:
        original = segment.get("words", [])
        if original and words[position:position + len(original)] == original:
            result.append(copy.deepcopy(segment))
            position += len(original)
            continue
        if original and original[0] in words[position:]:
            next_position = words.index(original[0], position)
            if next_position > position:
                acquired = words[position:next_position]
                result.append({"start": acquired[0]["start"], "end": acquired[-1]["end"],
                               "text": " ".join(w["word"].strip() for w in acquired), "words": acquired})
                position = next_position
            if words[position:position + len(original)] == original:
                result.append(copy.deepcopy(segment))
                position += len(original)
    if position < len(words):
        acquired = words[position:]
        result.append({"start": acquired[0]["start"], "end": acquired[-1]["end"],
                       "text": " ".join(w["word"].strip() for w in acquired), "words": acquired})
    from .sentence_first import build_source_sentences

    build_source_sentences(result)
    audit["status"] = "complete"
    save_audit()
    return result


def transcribe_audio(
    audio_path: str,
    model: str,
    language: Optional[str],
    temperature: float = 0.0,
    backend: str = "auto",
    transcription_model: str = DEFAULT_OPENROUTER_TRANSCRIPTION_MODEL,
    checkpoint_path: Optional[str] = None,
    force: bool = False,
) -> Optional[List[Dict]]:
    """
    Transcribe audio; remote errors propagate so the pipeline can classify them.
    """
    if backend == "openrouter":
        from .openrouter_transcription import transcribe_openrouter

        return transcribe_openrouter(
            audio_path, transcription_model, language, temperature,
            checkpoint_path=checkpoint_path, force=force,
        )
    try:
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            return None

        selected_backend = resolve_transcription_backend(backend, model)
        if selected_backend == "mlx":
            return _transcribe_with_mlx(audio_path, model, language, temperature)
        if selected_backend != "whisper":
            print(
                "Error: Unknown transcription backend "
                f"'{backend}'. Choose auto, whisper, or mlx."
            )
            return None

        import whisper  # type: ignore

        print(f"Loading local Whisper model ({model})...")
        whisper_model = whisper.load_model(model)

        print(f"Transcribing audio locally (language: {language or 'auto'})...")
        result = whisper_model.transcribe(  # type: ignore
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=True,
            fp16=False,
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
                f"'({language or 'auto'})' being incorrect."
            )
            return []

        print("Audio transcribed successfully")
        print("Transcription: local Whisper (no API token usage).")
        return result["segments"]

    except (FileNotFoundError, RuntimeError) as e:
        print(f"An error occurred during local audio transcription: {e}")
        return None
