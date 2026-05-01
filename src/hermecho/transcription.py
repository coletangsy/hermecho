"""
Local Whisper transcription.
"""
import os
from typing import Dict, List, Optional


def transcribe_audio(
    audio_path: str,
    model: str,
    language: str,
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
        whisper_model = whisper.load_model(model)

        print(f"Transcribing audio locally (language: {language})...")
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
                f"'({language})' being incorrect."
            )
            return []

        print("Audio transcribed successfully")
        print("Transcription: local Whisper (no API token usage).")
        return result["segments"]

    except (FileNotFoundError, RuntimeError) as e:
        print(f"An error occurred during local audio transcription: {e}")
        return None
