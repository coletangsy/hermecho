"""
This module contains utility functions for the video translator.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple


class _LockedTermsJSONObject(dict):
    """JSON object that retains duplicate source keys while loading terms."""

    def __init__(self, pairs: List[Tuple[str, Any]]) -> None:
        super().__init__()
        self.duplicate_keys: List[str] = []
        for key, value in pairs:
            if key in self:
                self.duplicate_keys.append(key)
            self[key] = value


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


def load_locked_terms(file_path: str) -> Optional[Dict[str, str]]:
    """Load the required source-to-target Locked Terms mapping."""
    if not file_path:
        print("Error: --locked-terms-file path is required.")
        return None
    if not os.path.exists(file_path):
        print(f"Error: --locked-terms-file not found at {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            locked_terms = json.load(f, object_pairs_hook=_LockedTermsJSONObject)
    except (FileNotFoundError, IOError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"Error: Could not read --locked-terms-file at {file_path}: {exc}")
        return None

    if not isinstance(locked_terms, _LockedTermsJSONObject):
        print(
            f"Error: --locked-terms-file at {file_path} must be a JSON object "
            "of non-empty string pairs."
        )
        return None
    if locked_terms.duplicate_keys:
        duplicate_keys = ", ".join(sorted(locked_terms.duplicate_keys))
        print(
            f"Error: --locked-terms-file at {file_path} contains duplicate "
            f"source keys: {duplicate_keys}"
        )
        return None

    normalized_terms: Dict[str, str] = {}
    for source, target in locked_terms.items():
        if not isinstance(source, str) or not isinstance(target, str):
            print(
                f"Error: --locked-terms-file at {file_path} must be a JSON object "
                "of non-empty string pairs."
            )
            return None

        source = source.strip()
        target = target.strip()
        if not source or not target:
            print(
                f"Error: --locked-terms-file at {file_path} must be a JSON object "
                "of non-empty string pairs."
            )
            return None
        if source in normalized_terms:
            print(
                f"Error: --locked-terms-file at {file_path} contains a source-key "
                f"collision after whitespace normalization: {source!r}"
            )
            return None
        normalized_terms[source] = target

    return normalized_terms


def _print_segments(title: str, segments: List[Dict]) -> None:
    """
    Prints a formatted list of subtitle segments to the console.

    Args:
        title: The title to display before printing the segments.
        segments: A list of segment dictionaries to print, each containing 'start', 'end', and 'text'.
    """
    print(f"\n{title}:\n---")
    for seg in segments:
        print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")
