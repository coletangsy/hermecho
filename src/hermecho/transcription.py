"""
Local Whisper transcription with an optional MLX backend.
"""
import os
import platform
from typing import Any, Dict, List, Optional


MLX_LARGE_V3_MODEL = "mlx-community/whisper-large-v3-mlx"
MLX_MODEL_NAMES = {
    "large": MLX_LARGE_V3_MODEL,
    "large-v3": MLX_LARGE_V3_MODEL,
}


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
    try:
        import mlx_whisper  # type: ignore
    except ImportError:
        return 'MLX Whisper is not installed. Install it with `python -m pip install -e ".[mlx]"`.'
    return None


def _normalise_mlx_result(result: Any) -> tuple[str, List[Dict]]:
    """Return MLX segments in the existing Whisper segment and word schema."""
    if not isinstance(result, dict) or not isinstance(result.get("segments"), list):
        raise RuntimeError("MLX Whisper returned an invalid transcription result.")

    detected_language = result.get("language", "unknown")
    if not isinstance(detected_language, str):
        raise RuntimeError("MLX Whisper returned an invalid detected language.")

    normalised_segments: List[Dict] = []
    for segment in result["segments"]:
        if (
            not isinstance(segment, dict)
            or not isinstance(segment.get("text"), str)
            or not isinstance(segment.get("words"), list)
        ):
            raise RuntimeError("MLX Whisper returned an invalid transcription segment.")
        if not all(
            isinstance(word, dict) and isinstance(word.get("word"), str)
            for word in segment["words"]
        ):
            raise RuntimeError("MLX Whisper returned an invalid Source Word timestamp.")
        try:
            normalised_segment = dict(segment)
            normalised_segment["start"] = float(segment["start"])
            normalised_segment["end"] = float(segment["end"])
            normalised_segment["text"] = segment["text"]
            normalised_segment["words"] = [
                {
                    **word,
                    "word": word["word"],
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                }
                for word in segment["words"]
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("MLX Whisper returned invalid segment timestamps.") from error

        normalised_segments.append(normalised_segment)

    return detected_language, normalised_segments


def _transcribe_with_mlx(
    audio_path: str,
    model: str,
    language: Optional[str],
    temperature: float,
) -> List[Dict]:
    error = validate_mlx_backend(model)
    if error:
        raise RuntimeError(error)

    import mlx_whisper  # type: ignore

    mlx_model = MLX_MODEL_NAMES[model]

    print(f"Loading MLX Whisper model ({mlx_model})...")
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


def transcribe_audio(
    audio_path: str,
    model: str,
    language: Optional[str],
    temperature: float = 0.0,
    backend: str = "auto",
) -> Optional[List[Dict]]:
    """
    Transcribes audio using the selected local Whisper backend.
    """
    try:
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            return None

        selected_backend = "whisper" if backend == "auto" else backend
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
