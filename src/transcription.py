"""
This module contains functions for transcribing audio to text.
"""
import os
import whisper # type: ignore
from typing import Optional, List, Dict


def transcribe_audio(audio_path: str, model: str, language: str, initial_prompt: Optional[str] = None, temperature: float = 0.0) -> Optional[List[Dict]]:
    """
    Transcribes audio to text using the local OpenAI Whisper model, returning segments with timestamps.

    Args:
        audio_path: The path to the audio file.
        model: The name of the Whisper model to use.
        language: The language of the audio.
        initial_prompt: Optional text to provide context to the model (helps with mixed languages).
        temperature: Sampling temperature. 0.0 is deterministic.

    Returns:
        A list of transcription segments with timestamps, or None if an error occurs.
    """
    try:
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            return None

        print(f"Loading local Whisper model ({model})...")
        model = whisper.load_model(model)

        print(f"Transcribing audio locally (language: {language})...")
        if initial_prompt:
            print(f"Using initial prompt: {initial_prompt}")

        result = model.transcribe( # type: ignore
            audio_path, 
            language=language, 
            word_timestamps=True, 
            verbose=True, 
            fp16=False,
            initial_prompt=initial_prompt,
            temperature=temperature,
            condition_on_previous_text=False,
            no_speech_threshold=0.85,
            compression_ratio_threshold=2.4
        )

        # Save the full transcription to a file for review
        # with open("transcription.txt", "w", encoding="utf-8") as f:
        #     f.write(result["text"])

        # Check if any segments were transcribed
        if not result["segments"]:
            detected_language = result.get('language', 'unknown')
            print("Warning: Whisper model returned no transcription segments.")
            print(f"  - Detected language: {detected_language}")
            print(f"  - This could be due to the audio containing no speech, or the specified language '({language})' being incorrect.")
            return [] # Return an empty list to prevent downstream errors

        print("Audio transcribed successfully")
        return result["segments"]

    except (FileNotFoundError, RuntimeError) as e:
        print(f"An error occurred during local audio transcription: {e}")
        return None
