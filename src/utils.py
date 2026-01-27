"""
This module contains utility functions for the video translator.
"""
import os
from typing import Optional


def load_reference_material(file_path: str) -> Optional[str]:
    """
    Loads reference material from a file.

    Args:
        file_path: The path to the reference file.

    Returns:
        The content of the file as a string, or None if the file doesn't exist.
    """
    if not file_path or not os.path.exists(file_path):
        if file_path:
            print(f"Warning: Reference file not found at {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, IOError) as e:
        print(f"An error occurred while reading the reference file: {e}")
        return None


def _print_segments(title: str, segments: list[dict]):
    """
    Prints a formatted list of subtitle segments to the console.

    Args:
        title: The title to display before printing the segments.
        segments: A list of segment dictionaries to print, each containing 'start', 'end', and 'text'.
    """
    print(f"\n{title}:\n---")
    for seg in segments:
        print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")
