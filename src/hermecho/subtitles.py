"""
This module contains functions for generating, adjusting, and cleaning subtitles.
"""
import logging
import math
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional

PORTRAIT_SUBTITLE_PUNCTUATION = frozenset("，。！？；：、")
DELIVERY_BREAK_PUNCTUATION = PORTRAIT_SUBTITLE_PUNCTUATION | frozenset(",.!?;:")
HALF_WIDTH_WORD_CONNECTORS = frozenset("_-'./:@?&=%+#~’")


@dataclass(frozen=True)
class DeliveryProfile:
    name: str
    warning_line_cells: float
    repair_line_cells: float
    warning_cue_cells: float
    repair_cue_cells: float
    warning_cps: float = 8.0
    repair_cps: float = 12.0
    warning_min_duration: float = 1.0
    warning_max_duration: float = 7.0
    repair_min_duration: float = 0.5
    repair_max_duration: float = 10.0


PORTRAIT_DELIVERY_PROFILE = DeliveryProfile(
    name="portrait",
    warning_line_cells=10,
    repair_line_cells=12,
    warning_cue_cells=20,
    repair_cue_cells=24,
)
LANDSCAPE_DELIVERY_PROFILE = DeliveryProfile(
    name="landscape",
    warning_line_cells=16,
    repair_line_cells=20,
    warning_cue_cells=32,
    repair_cue_cells=40,
)


@dataclass(frozen=True)
class DeliveryDiagnostic:
    severity: str
    code: str
    cue_index: int
    message: str


@dataclass
class DeliveryGateResult:
    cues: List[Dict]
    diagnostics: List[DeliveryDiagnostic]

    @property
    def blocked(self) -> bool:
        return any(
            diagnostic.severity == "Structural Defect"
            for diagnostic in self.diagnostics
        )


def delivery_profile_for_orientation(is_portrait: bool) -> DeliveryProfile:
    """Return the deterministic profile for the displayed video orientation."""
    return PORTRAIT_DELIVERY_PROFILE if is_portrait else LANDSCAPE_DELIVERY_PROFILE


def visual_cell_count(text: str) -> float:
    """Return the display width of subtitle text in Visual Cells."""
    return sum(
        1.0 if unicodedata.east_asian_width(character) in {"F", "W"} else 0.5
        for character in text
        if character != "\n" and not unicodedata.category(character).startswith("M")
    )


def _presentation_diagnostic(
    diagnostics: List[DeliveryDiagnostic],
    cue_index: int,
    code: str,
    value: float,
    warning_limit: float,
    repair_limit: float,
    unit: str,
) -> None:
    if value > repair_limit:
        severity = "Repair Limit"
        limit = repair_limit
    elif value > warning_limit:
        severity = "Warning"
        limit = warning_limit
    else:
        return
    diagnostics.append(
        DeliveryDiagnostic(
            severity,
            code,
            cue_index,
            f"{value:g} {unit} exceeds {limit:g}",
        )
    )


def _structural_diagnostic(
    diagnostics: List[DeliveryDiagnostic],
    cue_index: int,
    code: str,
    message: str,
) -> None:
    diagnostics.append(DeliveryDiagnostic("Structural Defect", code, cue_index, message))


def _validate_source_words(
    segment: Dict,
    cue_index: int,
    diagnostics: List[DeliveryDiagnostic],
) -> None:
    words = segment.get("source_words")
    if words is None:
        words = segment.get("words")
    indices = segment.get("source_word_indices")
    if words is None:
        if indices is not None:
            _structural_diagnostic(
                diagnostics,
                cue_index,
                "missing_source_word_timing",
                "mapped cue has no Source Word timing",
            )
        return
    if not isinstance(words, list) or not words:
        _structural_diagnostic(
            diagnostics,
            cue_index,
            "missing_source_word_timing",
            "cue Source Words are missing",
        )
        return

    previous_end = None
    for word in words:
        try:
            word_start = float(word["start"])
            word_end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            _structural_diagnostic(
                diagnostics,
                cue_index,
                "missing_source_word_timing",
                "a Source Word has no usable timing",
            )
            return
        if not math.isfinite(word_start) or not math.isfinite(word_end):
            _structural_diagnostic(
                diagnostics,
                cue_index,
                "invalid_source_word_timing",
                "a Source Word has non-finite timing",
            )
            return
        if word_start < 0 or word_end < 0:
            _structural_diagnostic(
                diagnostics,
                cue_index,
                "invalid_source_word_timing",
                "a Source Word has a negative timestamp",
            )
            return
        if word_end <= word_start:
            _structural_diagnostic(
                diagnostics,
                cue_index,
                "invalid_source_word_timing",
                "a Source Word has non-positive timing",
            )
            return
        if previous_end is not None and word_start < previous_end:
            _structural_diagnostic(
                diagnostics,
                cue_index,
                "overlap_timing",
                "Source Word timings overlap",
            )
            return
        previous_end = word_end

    if indices is not None:
        valid_indices = (
            isinstance(indices, list)
            and len(indices) == len(words)
            and all(type(index) is int and index >= 0 for index in indices)
            and all(right == left + 1 for left, right in zip(indices, indices[1:]))
        )
        if not valid_indices:
            _structural_diagnostic(
                diagnostics,
                cue_index,
                "invalid_source_coverage",
                "Source Word indices must cover one continuous range",
            )


def _is_half_width_word_character(character: str) -> bool:
    return (
        unicodedata.east_asian_width(character) not in {"F", "W"}
        and (
            character.isalnum()
            or character in HALF_WIDTH_WORD_CONNECTORS
            or bool(unicodedata.combining(character))
        )
    )


def _wrap_delivery_text(text: str, profile: DeliveryProfile) -> str:
    """Wrap text into at most two deterministic lines without splitting words."""
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    if visual_cell_count(text) <= profile.repair_line_cells:
        return text

    candidates = []
    for index in range(1, len(text)):
        if (
            _is_half_width_word_character(text[index - 1])
            and _is_half_width_word_character(text[index])
        ):
            continue
        left = text[:index]
        right = text[index:]
        if not left or not right:
            continue
        left_cells = visual_cell_count(left)
        right_cells = visual_cell_count(right)
        if text[index - 1] in DELIVERY_BREAK_PUNCTUATION:
            boundary_kind = 0
        elif text[index - 1].isspace() or text[index].isspace():
            boundary_kind = 1
        else:
            boundary_kind = 2
        candidates.append((left, right, left_cells, right_cells, boundary_kind, index))

    if not candidates:
        return text

    fitting = [
        candidate
        for candidate in candidates
        if max(candidate[2], candidate[3]) <= profile.repair_line_cells
    ]
    if fitting:
        left, right, *_ = min(
            fitting,
            key=lambda candidate: (
                abs(candidate[2] - candidate[3]),
                candidate[4],
                candidate[5],
            ),
        )
    else:
        left, right, *_ = min(
            candidates,
            key=lambda candidate: (
                max(candidate[2], candidate[3]) - profile.repair_line_cells,
                candidate[4],
                abs(candidate[2] - candidate[3]),
                candidate[5],
            ),
        )
    return f"{left}\n{right}"


def apply_delivery_profile(
    segments: List[Dict],
    profile: DeliveryProfile,
) -> DeliveryGateResult:
    """Evaluate subtitle cues against a Delivery Profile without blocking warnings."""
    cues: List[Dict] = []
    diagnostics: List[DeliveryDiagnostic] = []
    previous_end = None
    cue_index = 0

    for segment in segments:
        text = segment.get("text")
        if segment.get("source_text") == "[no speech]" and (
            not isinstance(text, str) or not text.strip()
        ):
            continue
        cue_index += 1
        if not isinstance(text, str) or not text.strip():
            _structural_diagnostic(
                diagnostics, cue_index, "empty_piece", "cue text is empty"
            )
            continue
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            _structural_diagnostic(
                diagnostics,
                cue_index,
                "invalid_timing",
                "cue start and end must be numeric",
            )
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            _structural_diagnostic(
                diagnostics,
                cue_index,
                "invalid_timing",
                "cue start and end must be finite",
            )
            continue
        negative_timestamp = start < 0 or end < 0
        if negative_timestamp:
            _structural_diagnostic(
                diagnostics,
                cue_index,
                "invalid_timing",
                "cue start and end must not be negative",
            )
        if end < start:
            _structural_diagnostic(
                diagnostics,
                cue_index,
                "reversed_timing",
                "cue end precedes its start",
            )
            continue
        if end == start:
            _structural_diagnostic(
                diagnostics,
                cue_index,
                "non_positive_duration",
                "cue duration is zero",
            )
            continue
        if negative_timestamp:
            continue
        if previous_end is not None and start < previous_end:
            _structural_diagnostic(
                diagnostics,
                cue_index,
                "overlap_timing",
                "cue overlaps the previous cue",
            )
        previous_end = max(previous_end, end) if previous_end is not None else end
        _validate_source_words(segment, cue_index, diagnostics)

        cue = segment.copy()
        cue["text"] = _wrap_delivery_text(text, profile)
        cues.append(cue)
        duration = end - start
        cells = visual_cell_count(cue["text"])
        _presentation_diagnostic(
            diagnostics,
            cue_index,
            "cps",
            cells / duration,
            profile.warning_cps,
            profile.repair_cps,
            "cells/s",
        )
        _presentation_diagnostic(
            diagnostics,
            cue_index,
            "cue_cells",
            cells,
            profile.warning_cue_cells,
            profile.repair_cue_cells,
            "cells",
        )
        for line in cue["text"].splitlines() or [cue["text"]]:
            _presentation_diagnostic(
                diagnostics,
                cue_index,
                "line_cells",
                visual_cell_count(line),
                profile.warning_line_cells,
                profile.repair_line_cells,
                "cells",
            )
        if duration < profile.repair_min_duration or duration > profile.repair_max_duration:
            severity, limit = "Repair Limit", (
                profile.repair_min_duration
                if duration < profile.repair_min_duration
                else profile.repair_max_duration
            )
        elif duration < profile.warning_min_duration or duration > profile.warning_max_duration:
            severity, limit = "Warning", (
                profile.warning_min_duration
                if duration < profile.warning_min_duration
                else profile.warning_max_duration
            )
        else:
            continue
        diagnostics.append(
            DeliveryDiagnostic(
                severity,
                "duration",
                cue_index,
                f"{duration:g}s is outside the {limit:g}s limit",
            )
        )

    return DeliveryGateResult(cues, diagnostics)


def delivery_gate_report(result: DeliveryGateResult, profile: DeliveryProfile) -> str:
    """Return a compact, reviewable Delivery Gate report."""
    severities = ("Warning", "Repair Limit", "Structural Defect")
    lines = [f"Delivery Gate ({profile.name})"]
    for severity in severities:
        count = sum(
            diagnostic.severity == severity for diagnostic in result.diagnostics
        )
        label = {
            "Warning": "Warnings",
            "Repair Limit": "Repair Limits",
            "Structural Defect": "Structural Defects",
        }[severity]
        lines.append(f"{label}: {count}")
    lines.extend(
        f"{diagnostic.severity} cue {diagnostic.cue_index}: "
        f"{diagnostic.code} ({diagnostic.message})"
        for diagnostic in result.diagnostics
    )
    return "\n".join(lines)


def _split_no_words(seg: Dict, max_chars: int, max_duration: float) -> List[Dict]:
    """
    Proportional time split for segments without word-level timestamps.
    Time is distributed linearly by character count as a speech-rate proxy.
    """
    text = seg.get("text", "").strip()
    start = float(seg["start"])
    end = float(seg["end"])
    duration = end - start

    if not text or (len(text) <= max_chars and duration <= max_duration):
        return [seg]

    n_by_duration = math.ceil(duration / max_duration) if max_duration > 0 else 1
    n_by_chars = math.ceil(len(text) / max_chars) if max_chars > 0 else 1
    n_splits = max(n_by_duration, n_by_chars, 2)
    target_chars = math.ceil(len(text) / n_splits)

    tokens = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for token in tokens:
        token_len = len(token) + (1 if current else 0)  # +1 for joining space
        current.append(token)
        current_len += token_len
        if current_len >= target_chars:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

    if current:
        chunks.append(" ".join(current))

    if not chunks:
        return [seg]

    total_chars = max(1, len(text))
    result: List[Dict] = []
    char_pos = 0

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.strip()
        if not chunk_text:
            continue
        chunk_start = start + (char_pos / total_chars) * duration
        char_pos += len(chunk)
        chunk_end = end if i == len(chunks) - 1 else start + (char_pos / total_chars) * duration
        result.append({
            "text": chunk_text,
            "start": round(chunk_start, 3),
            "end": round(chunk_end, 3),
        })

    return result if result else [seg]


def split_long_segments(segments: List[Dict], max_chars: int = 40, max_duration: float = 7.0) -> List[Dict]:
    """
    Splits segments that exceed ``max_chars`` or ``max_duration``.

    Whisper segments (with per-word timestamps) are split at word boundaries.
    Multimodal segments (no ``words`` key) fall back to proportional text split
    via :func:`_split_no_words`.
    """
    split_segments = []

    for seg in segments:
        text = seg.get("text", "").strip()
        start = seg["start"]
        end = seg["end"]
        duration = end - start
        words = seg.get("words", [])

        if len(text) <= max_chars and duration <= max_duration:
            split_segments.append(seg)
            continue

        if not words:
            split_segments.extend(_split_no_words(seg, max_chars, max_duration))
            continue
            
        # Split logic: Try to split into chunks that fit constraints
        current_chunk_words = []
        current_chunk_start = words[0]["start"]
        
        for i, word_info in enumerate(words):
            current_chunk_words.append(word_info)
            
            # Check if adding this word exceeds limits relative to chunk start
            chunk_text = "".join([w["word"] for w in current_chunk_words]).strip()
            chunk_duration = word_info["end"] - current_chunk_start
            
            # Look ahead to see if next word would break the limit
            next_word_breaks = False
            if i + 1 < len(words):
                next_word = words[i+1]
                next_text_len = len(chunk_text) + len(next_word["word"])
                next_duration = next_word["end"] - current_chunk_start
                if next_text_len > max_chars or next_duration > max_duration:
                    next_word_breaks = True
            
            # If we need to split here (either current is long enough, or next breaks it)
            # But ensure we have at least something in the chunk
            if next_word_breaks or i == len(words) - 1:
                split_segments.append({
                    "text": chunk_text,
                    "start": current_chunk_start,
                    "end": word_info["end"],
                    "words": current_chunk_words
                })
                # Reset for next chunk
                if i + 1 < len(words):
                    current_chunk_start = words[i+1]["start"]
                    current_chunk_words = []
                    
    return split_segments


def fill_transcription_gaps(
    transcribed_segments: List[Dict],
    gap_threshold: float = 5.0,
    placeholder: str = "[no speech]",
) -> List[Dict]:
    """
    Identifies and fills significant time gaps in a transcription with placeholder text.

    This function iterates through the transcribed segments and checks the time difference
    between the end of one segment and the start of the next. If the gap exceeds the
    specified threshold, a new placeholder segment is inserted.

    Args:
        transcribed_segments: The list of transcription segments from Whisper.
        gap_threshold: The minimum duration (in seconds) of a gap to be filled.
        placeholder: The text to insert for the gap.

    Returns:
        A new list of segments with gaps filled.
    """
    if not transcribed_segments:
        return []

    filled_segments = [transcribed_segments[0]]
    for i in range(len(transcribed_segments) - 1):
        current_seg = transcribed_segments[i]
        next_seg = transcribed_segments[i + 1]

        gap = next_seg["start"] - current_seg["end"]

        if gap > gap_threshold:
            logging.warning(
                f"Gap of {gap:.2f}s detected. Inserting placeholder."
            )
            filled_segments.append({
                "text": placeholder,
                "start": current_seg["end"],
                "end": next_seg["start"]
            })
        
        filled_segments.append(next_seg)
    
    return filled_segments


def adjust_subtitle_timing(
    segments: List[Dict],
    time_buffer: float,
    silence_boundaries: Optional[List[float]] = None,
) -> List[Dict]:
    """
    Adjusts subtitle timings to fill gaps and ensures a consistent reading pace.

    This function extends the duration of each subtitle to meet the start of the next one,
    minus a small buffer. It helps prevent subtitles from flashing on the screen too quickly.

    Args:
        segments: A list of subtitle segments (can be transcribed or translated).
        time_buffer: The buffer time (in seconds) to maintain between subtitles.
        silence_boundaries: Silence-start times that subtitle cues must not cross.

    Returns:
        The adjusted list of segments.
    """
    time_buffer = max(0, time_buffer)

    if not segments:
        return []

    adjusted_segments = [seg.copy() for seg in segments]
    silence_boundaries = sorted(silence_boundaries or [])

    for i in range(len(adjusted_segments) - 1):
        current_segment = adjusted_segments[i]
        next_segment = adjusted_segments[i + 1]

        # The ideal end time for the current segment is the start of the next one minus the buffer.
        new_end_time = next_segment['start'] - time_buffer
        silence_start = next(
            (
                boundary
                for boundary in silence_boundaries
                if current_segment['start'] < boundary < next_segment['start']
            ),
            None,
        )
        if silence_start is not None:
            new_end_time = min(new_end_time, silence_start)

        # Update the end time. This extends shorter segments and shortens longer ones.
        current_segment['end'] = new_end_time

        # Ensure that the new end time does not precede the start time.
        if current_segment['end'] < current_segment['start']:
            # This can happen if the gap between segments is smaller than the time_buffer.
            # To avoid a negative or zero duration, we set the end time to be same as the start time,
            # which will make the subtitle appear as a flash. This is a safe fallback.
            current_segment['end'] = current_segment['start']

    # The last segment's end time is not modified as there's no next segment to overlap with.

    return adjusted_segments


def generate_srt(
    segments: List[Dict],
    output_path: str,
) -> None:
    """
    Generates an SRT subtitle file from translated segments.

    Args:
        segments: A list of segments with text, start, and end times.
        output_path: The path to save the .srt file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            start_time = seg["start"]
            end_time = seg["end"]
            text = seg["text"]

            # SRT time format: HH:MM:SS,ms
            start_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}"
            end_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}"

            f.write(f"{i + 1}\n")
            f.write(f"{start_srt} --> {end_srt}\n")
            f.write(f"{text}\n\n")
    print(f"SRT file generated at {output_path}")
