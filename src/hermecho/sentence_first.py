"""Sentence-first source grouping and target-language delivery."""
from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional

from .subtitles import (
    DeliveryDiagnostic,
    DeliveryGateResult,
    DeliveryProfile,
    PORTRAIT_DELIVERY_PROFILE,
    apply_delivery_profile,
)


TERMINAL_PUNCTUATION = frozenset("。！？!?；;…．.")
DEFAULT_PAUSE_THRESHOLD = 0.8
DEFAULT_SAFETY_DURATION = 20.0
REVIEW_CHECKS = (
    "translation completeness",
    "meaning boundaries",
    "timing",
    "readability",
    "locked terms",
    "punctuation",
    "presentation warnings",
)


class SentenceFirstError(ValueError):
    """Raised when immutable Source Word evidence is not usable."""


def resolve_subtitle_delivery(requested: str) -> str:
    """Resolve the promoted default delivery mode."""
    if requested == "sentence-first":
        return requested
    if requested == "legacy":
        return requested
    if requested == "auto":
        return "sentence-first"
    raise ValueError(
        f"Unknown subtitle delivery '{requested}'. Choose auto, legacy, or sentence-first."
    )


def _timestamp(value: Any, label: str) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as error:
        raise SentenceFirstError(
            f"Sentence-first delivery requires valid Source Word timestamps ({label})."
        ) from error
    if not math.isfinite(timestamp) or timestamp < 0:
        raise SentenceFirstError(
            f"Sentence-first delivery requires valid Source Word timestamps ({label})."
        )
    return timestamp


def _word_entries(segments: List[Dict]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    previous_end: Optional[float] = None
    for segment_index, segment in enumerate(segments):
        text = segment.get("text", "")
        if not isinstance(text, str):
            raise SentenceFirstError("Sentence-first delivery requires source text.")
        if not text.strip() or text.strip() == "[no speech]":
            continue
        words = segment.get("words")
        if not isinstance(words, list) or not words:
            raise SentenceFirstError(
                "Sentence-first delivery requires Source Word timestamps."
            )
        for word_index, word in enumerate(words):
            if not isinstance(word, dict) or not isinstance(word.get("word"), str):
                raise SentenceFirstError(
                    "Sentence-first delivery requires Source Word timestamps."
                )
            word_text = word["word"]
            if not word_text.strip():
                raise SentenceFirstError(
                    "Sentence-first delivery requires non-empty Source Words."
                )
            start = _timestamp(word.get("start"), "start")
            end = _timestamp(word.get("end"), "end")
            if end < start and not math.isclose(end, start, abs_tol=1e-9):
                raise SentenceFirstError(
                    "Sentence-first delivery requires non-negative Source Word timing."
                )
            if (
                previous_end is not None
                and start < previous_end
                and not math.isclose(start, previous_end, abs_tol=1e-9)
            ):
                raise SentenceFirstError(
                    "Sentence-first delivery requires ordered, non-overlapping Source Word timing."
                )
            previous_end = end
            entries.append(
                {
                    "word": copy.deepcopy(word),
                    "segment_text": text,
                    "segment_index": segment_index,
                    "word_index": word_index,
                    "segment_word_count": len(words),
                    "segment_last": word_index == len(words) - 1,
                    "start": start,
                    "end": end,
                }
            )
    return entries


def _ends_with_terminal_punctuation(entry: Dict[str, Any]) -> bool:
    word_text = entry["word"]["word"].rstrip()
    segment_text = entry["segment_text"].rstrip()
    return bool(
        (word_text and word_text[-1] in TERMINAL_PUNCTUATION)
        or (entry["segment_last"] and segment_text and segment_text[-1] in TERMINAL_PUNCTUATION)
    )


def _best_safety_split(
    pending: List[Dict[str, Any]],
    next_entry: Dict[str, Any],
    safety_duration: float,
) -> int:
    target = pending[0]["start"] + safety_duration
    candidates = []
    for index, left in enumerate(pending):
        right = pending[index + 1] if index + 1 < len(pending) else next_entry
        gap = max(0.0, right["start"] - left["end"])
        candidates.append((round(gap, 6), -abs(left["end"] - target), -(index + 1), index + 1))
    return max(candidates)[3]


def _source_sentence(entries: List[Dict[str, Any]], indices: List[int]) -> Dict:
    source_words = [copy.deepcopy(entry["word"]) for entry in entries]
    text_parts: List[str] = []
    segment_entry_counts: Dict[int, int] = {}
    for entry in entries:
        segment_entry_counts[entry["segment_index"]] = (
            segment_entry_counts.get(entry["segment_index"], 0) + 1
        )
    for entry in entries:
        if (
            entry["word_index"] == 0
            and entry["segment_word_count"]
            == segment_entry_counts[entry["segment_index"]]
        ):
            text_parts.append(entry["segment_text"].strip())
        elif not text_parts or text_parts[-1] != entry["segment_text"].strip():
            text_parts.append(entry["word"]["word"])
    text = ""
    for part in text_parts:
        if not part:
            continue
        if (
            text
            and text[-1].isalnum()
            and part[0].isalnum()
            and text[-1].isascii()
            and part[0].isascii()
        ):
            text += " "
        text += part
    text = text.strip()
    if not text:
        text = "".join(word["word"] for word in source_words).strip()
    return {
        "start": entries[0]["start"],
        "end": entries[-1]["end"],
        "text": text,
        "source_words": source_words,
        "source_word_indices": indices,
    }


def build_source_sentences(
    segments: List[Dict],
    *,
    pause_threshold: float = DEFAULT_PAUSE_THRESHOLD,
    safety_duration: float = DEFAULT_SAFETY_DURATION,
) -> List[Dict]:
    """Group immutable Source Words into deterministic Source Sentences."""
    if pause_threshold < 0 or safety_duration <= 0:
        raise ValueError("Sentence boundary thresholds must be positive.")
    entries = _word_entries(segments)
    sentences: List[Dict] = []
    pending: List[Dict[str, Any]] = []
    next_source_index = 0

    def flush(values: List[Dict[str, Any]]) -> None:
        nonlocal next_source_index
        if not values:
            return
        indices = list(range(next_source_index, next_source_index + len(values)))
        sentence = _source_sentence(values, indices)
        if sentence["text"]:
            sentences.append(sentence)
        next_source_index += len(values)

    for entry in entries:
        while pending:
            gap = entry["start"] - pending[-1]["end"]
            if gap >= pause_threshold:
                flush(pending)
                pending = []
                break
            elapsed = entry["start"] - pending[0]["start"]
            if elapsed < safety_duration:
                break
            split_at = _best_safety_split(pending, entry, safety_duration)
            if split_at >= len(pending):
                flush(pending)
                pending = []
                break
            flush(pending[:split_at])
            pending = pending[split_at:]

        pending.append(entry)
        if _ends_with_terminal_punctuation(entry):
            flush(pending)
            pending = []

    flush(pending)
    return sentences


def _sentence_words(sentence: Dict, sentence_index: int) -> tuple[List[Dict], List[int]]:
    words = sentence.get("source_words")
    indices = sentence.get("source_word_indices")
    if not isinstance(words, list) or not words:
        raise SentenceFirstError(
            f"Source Sentence {sentence_index} has no Source Word timestamps."
        )
    if not isinstance(indices, list) or len(indices) != len(words):
        raise SentenceFirstError(
            f"Source Sentence {sentence_index} has invalid Source Word coverage."
        )
    if any(type(index) is not int or index < 0 for index in indices) or any(
        right != left + 1 for left, right in zip(indices, indices[1:])
    ):
        raise SentenceFirstError(
            f"Source Sentence {sentence_index} has non-continuous Source Word coverage."
        )
    for word in words:
        if not isinstance(word, dict):
            raise SentenceFirstError(
                f"Source Sentence {sentence_index} has invalid Source Word timing."
            )
        start = _timestamp(word.get("start"), "start")
        end = _timestamp(word.get("end"), "end")
        if end < start and not math.isclose(end, start, abs_tol=1e-9):
            raise SentenceFirstError(
                f"Source Sentence {sentence_index} has invalid Source Word timing."
            )
    return copy.deepcopy(words), list(indices)


def _delivery_cue(
    sentence: Dict,
    text: str,
    words: List[Dict],
    indices: List[int],
) -> Dict:
    return {
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "text": text,
        "source_text": sentence.get("source_text", sentence.get("text", "")),
        "source_words": copy.deepcopy(words),
        "source_word_indices": list(indices),
    }


def _has_repair_limit(result: DeliveryGateResult) -> bool:
    return any(diagnostic.severity == "Repair Limit" for diagnostic in result.diagnostics)


def _has_cps_repair_limit(result: DeliveryGateResult) -> bool:
    return any(
        diagnostic.severity == "Repair Limit" and diagnostic.code == "cps"
        for diagnostic in result.diagnostics
    )


def _aligned_cues(
    sentence: Dict,
    text: str,
    words: List[Dict],
    indices: List[int],
    pieces: object,
) -> List[Dict]:
    if not isinstance(pieces, list) or not pieces:
        raise ValueError("Alignment returned no pieces.")
    cues: List[Dict] = []
    previous_end = indices[0] - 1
    consumed_text = ""
    for piece in pieces:
        if not isinstance(piece, dict) or not isinstance(piece.get("text"), str):
            raise ValueError("Alignment returned an invalid target piece.")
        piece_text = piece["text"]
        if not piece_text:
            raise ValueError("Alignment returned an empty target piece.")
        raw_end_index = piece.get("end_source_word_index")
        if raw_end_index is None:
            raw_end_index = piece.get("source_word_end_index")
        if raw_end_index is None:
            raw_end_index = piece.get("end_word_index")
        if raw_end_index is None:
            raw_end_index = piece.get("end_index")
        if type(raw_end_index) is not int:
            raise ValueError("Alignment Source Word ranges are not ordered.")
        if raw_end_index in indices:
            end_index = raw_end_index
        elif 0 <= raw_end_index < len(indices):
            end_index = indices[raw_end_index]
        else:
            raise ValueError("Alignment Source Word ranges are outside the sentence.")
        if end_index <= previous_end:
            raise ValueError("Alignment Source Word ranges are not ordered.")
        start_position = previous_end + 1 - indices[0]
        end_position = end_index - indices[0]
        if start_position < 0 or end_position >= len(words) or end_position < start_position:
            raise ValueError("Alignment Source Word ranges are outside the sentence.")
        piece_indices = indices[start_position : end_position + 1]
        if piece_indices != list(range(piece_indices[0], piece_indices[-1] + 1)):
            raise ValueError("Alignment Source Word ranges are not continuous.")
        cues.append(
            _delivery_cue(
                sentence,
                piece_text,
                words[start_position : end_position + 1],
                piece_indices,
            )
        )
        consumed_text += piece_text
        previous_end = end_index

    if previous_end != indices[-1] or consumed_text != text:
        raise ValueError("Alignment must cover Source Words and concatenate exactly.")

    merged_cues: List[Dict] = []
    for cue in cues:
        point_timed = math.isclose(cue["start"], cue["end"], abs_tol=1e-9)
        previous_point_timed = bool(merged_cues) and math.isclose(
            merged_cues[-1]["start"], merged_cues[-1]["end"], abs_tol=1e-9
        )
        if merged_cues and (point_timed or previous_point_timed):
            previous = merged_cues[-1]
            previous["text"] += cue["text"]
            previous["end"] = cue["end"]
            previous["source_words"].extend(cue["source_words"])
            previous["source_word_indices"].extend(cue["source_word_indices"])
        else:
            merged_cues.append(cue)
    return merged_cues


def build_delivery_cues(
    translated_sentences: List[Dict],
    profile: Optional[DeliveryProfile] = None,
    *,
    fit_repair: Optional[Callable[[Dict, DeliveryProfile], Optional[str]]] = None,
    align: Optional[Callable[[Dict], Optional[List[Dict]]]] = None,
    time_buffer: float = 0.1,
) -> DeliveryGateResult:
    """Turn accepted Translation Sentences into timed Delivery Cues."""
    profile = profile or PORTRAIT_DELIVERY_PROFILE
    if time_buffer < 0:
        raise ValueError("Delivery time buffer must not be negative.")
    cues: List[Dict] = []
    diagnostics: List[DeliveryDiagnostic] = []
    for sentence_index, sentence in enumerate(translated_sentences, start=1):
        try:
            words, indices = _sentence_words(sentence, sentence_index)
        except SentenceFirstError as error:
            diagnostics.append(
                DeliveryDiagnostic(
                    "Structural Defect",
                    "missing_source_word_timing",
                    sentence_index,
                    str(error),
                )
            )
            continue
        text = sentence.get("text")
        if not isinstance(text, str) or not text.strip():
            diagnostics.append(
                DeliveryDiagnostic(
                    "Structural Defect",
                    "empty_piece",
                    sentence_index,
                    "Translation Sentence is empty",
                )
            )
            continue
        cue = _delivery_cue(sentence, text, words, indices)
        initial_result = apply_delivery_profile([cue], profile)
        accepted_text = text
        if _has_cps_repair_limit(initial_result) and fit_repair is not None:
            for _ in range(2):
                repaired_text = fit_repair(
                    {**sentence, "text": accepted_text},
                    profile,
                )
                if not isinstance(repaired_text, str) or not repaired_text.strip():
                    continue
                repaired_cue = _delivery_cue(sentence, repaired_text, words, indices)
                if not apply_delivery_profile([repaired_cue], profile).blocked:
                    accepted_text = repaired_text
                    break

        cue = _delivery_cue(sentence, accepted_text, words, indices)
        cue_result = apply_delivery_profile([cue], profile)
        if _has_repair_limit(cue_result) and align is not None:
            alignment_succeeded = False
            for _ in range(2):
                try:
                    aligned = _aligned_cues(
                        sentence,
                        accepted_text,
                        words,
                        indices,
                        align({**sentence, "text": accepted_text}),
                    )
                except (TypeError, ValueError):
                    continue
                cues.extend(aligned)
                alignment_succeeded = True
                break
            if alignment_succeeded:
                continue
        cues.append(cue)

    for index in range(len(cues) - 1):
        next_start = cues[index + 1]["start"]
        if next_start > cues[index]["end"]:
            cues[index]["end"] = min(next_start, cues[index]["end"] + time_buffer)

    result = apply_delivery_profile(cues, profile)
    diagnostics.extend(result.diagnostics)
    return DeliveryGateResult(result.cues, diagnostics)
