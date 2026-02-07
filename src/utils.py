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


import re

def extract_keywords_for_whisper(file_path: str, max_tokens: int = 200) -> str:
    """
    Extracts Korean keywords (General Info, Key Terms, Members) from the reference file.
    
    Args:
        file_path: Path to the markdown file.
        max_tokens: Approximate token limit (soft limit based on word count).
        
    Returns:
        A comma-separated string of Korean keywords.
    """
    content = load_reference_material(file_path)
    if not content:
        return ""

    keywords = []
    
    lines = content.split('\n')
    current_section = None

    for line in lines:
        line = line.strip()
        
        # Detect sections
        if line.startswith("## 1. General Info"):
            current_section = "general"
        elif line.startswith("## 2. Key Terms"):
            current_section = "terms"
        elif line.startswith("## 3. Members"):
            current_section = "members"
        elif line.startswith("##"):
            current_section = None

        # Parse General Info Table
        # | English | Korean (Official) | Notes |
        if current_section == "general" and line.startswith("|") and "---" not in line and "English" not in line:
            parts = line.split('|')
            if len(parts) >= 3:
                # Korean is in the 2nd column (index 2)
                korean_col = parts[2]
                extracted = re.findall(r'[가-힣]+', korean_col)
                keywords.extend(extracted)

        # Parse Key Terms
        # *   **Objekt (오브젝트)**: ...
        elif current_section == "terms" and line.startswith("*"):
            # Extract text inside parentheses that contains Korean
            matches = re.findall(r'\(([가-힣\s]+)\)', line)
            for match in matches:
                # Clean up whitespace
                keywords.extend(match.split())

        # Parse Members Table
        # | **S1** | Yoon SeoYeon | **윤서연** |
        elif current_section == "members" and '| **S' in line:
            parts = line.split('|')
            if len(parts) >= 4:
                # Korean name is in the 3rd column (index 3)
                korean_col = parts[3]
                extracted = re.findall(r'[가-힣]+', korean_col)
                keywords.extend(extracted)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for word in keywords:
        if word not in seen:
            seen.add(word)
            unique_keywords.append(word)
            
    # Limit to max_tokens (approximate) to avoid Whisper truncation
    # Whisper limit is ~224 tokens. 
    # Let's be safe and take the top ~80 words.
    return ", ".join(unique_keywords[:80])


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
