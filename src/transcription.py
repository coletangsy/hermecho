"""
This module contains functions for transcribing audio to text.
"""
import os
import whisper
from typing import Optional, List, Dict


def transcribe_audio(audio_path: str, model: str, language: str) -> Optional[List[Dict]]:
    """
    Transcribes audio to text using the local OpenAI Whisper model, returning segments with timestamps.

    Args:
        audio_path: The path to the audio file.
        model: The name of the Whisper model to use.
        language: The language of the audio.

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
        result = model.transcribe(
            audio_path, language=language, word_timestamps=True, verbose=True, fp16=False)

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
