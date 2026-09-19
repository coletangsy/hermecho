"""Sentence-first source grouping and target-language delivery."""
from __future__ import annotations

import copy
import math
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .subtitles import (
    DeliveryDiagnostic,
    DeliveryGateResult,
    DeliveryProfile,
    PORTRAIT_DELIVERY_PROFILE,
    apply_delivery_profile,
    delivery_candidate_score,
)


TERMINAL_PUNCTUATION = frozenset("。！？!?；;…．.")
DEFAULT_PAUSE_THRESHOLD = 0.8
DEFAULT_SAFETY_DURATION = 20.0
DEFAULT_REVIEW_MAX_WORDS = 4
DEFAULT_REVIEW_MAX_DURATION = 2.5
DEFAULT_SUSPICIOUS_SOURCE_WORD_DURATION = 5.0


@dataclass(frozen=True)
class SourceGroupingDiagnostic:
    """A non-blocking finding from ambiguous Source Sentence review."""

    code: str
    start: float
    end: float
    message: str
    outcome: str = "fallback"


@dataclass(frozen=True)
class SourceTimingDiagnostic:
    """A non-blocking finding about suspicious Source Word timing."""

    code: str
    start: float
    end: float
    message: str


@dataclass
class SourceSentenceReviewResult:
    """Reviewed Source Sentences plus diagnostics and cache eligibility."""

    sentences: List[Dict]
    diagnostics: List[SourceGroupingDiagnostic]
    cacheable: bool = True


class SentenceFirstError(ValueError):
    """Raised when immutable Source Word evidence is not usable."""


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


def _assemble_source_text(words: Sequence[Dict]) -> str:
    """Join Source Words while respecting observed spaces and script boundaries."""
    text = ""
    for word in words:
        part = str(word.get("word", ""))
        if not part:
            continue
        if text and not text[-1].isspace() and part[0].isspace():
            text += " "
        stripped = part.strip()
        if not stripped:
            continue
        if (
            text
            and text[-1].isalnum()
            and stripped[0].isalnum()
            and text[-1].isascii()
            and stripped[0].isascii()
        ):
            text += " "
        text += stripped
    return text.strip()


def _source_content_signature(text: str) -> str:
    """Remove presentation spacing and punctuation for Source Word validation."""
    return "".join(
        character
        for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _source_text_matches_words(text: str, words: Sequence[Dict]) -> bool:
    expected = _source_content_signature(_assemble_source_text(words))
    actual = _source_content_signature(text)
    return bool(expected) and actual == expected


def _merged_source_sentence(
    sentences: Sequence[Dict],
    *,
    text: Optional[str] = None,
) -> Dict:
    """Merge adjacent Source Sentences without changing Source Word evidence."""
    if not sentences:
        raise SentenceFirstError("Cannot merge an empty Source Sentence group.")
    words: List[Dict] = []
    indices: List[int] = []
    for sentence in sentences:
        sentence_words, sentence_indices = _sentence_words(sentence, 0)
        if indices and sentence_indices[0] != indices[-1] + 1:
            raise SentenceFirstError("Source Sentence groups must be contiguous.")
        words.extend(sentence_words)
        indices.extend(sentence_indices)
    merged_text = text.strip() if isinstance(text, str) else _assemble_source_text(words)
    if not _source_text_matches_words(merged_text, words):
        raise SentenceFirstError("Source Sentence review rewrote Source Words.")
    return {
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "text": merged_text,
        "source_words": copy.deepcopy(words),
        "source_word_indices": indices,
    }


def _sentence_has_terminal_punctuation(sentence: Dict) -> bool:
    text = str(sentence.get("text", "")).rstrip()
    return bool(text and text[-1] in TERMINAL_PUNCTUATION)


def ambiguous_source_boundaries(
    sentences: Sequence[Dict],
    *,
    max_words: int = DEFAULT_REVIEW_MAX_WORDS,
    max_duration: float = DEFAULT_REVIEW_MAX_DURATION,
) -> List[Dict]:
    """Select only short, nearby, or apparently incomplete boundaries for review."""
    candidates: List[Dict] = []
    for boundary_index, (left, right) in enumerate(zip(sentences, sentences[1:])):
        left_words = left.get("source_words")
        right_words = right.get("source_words")
        if not isinstance(left_words, list) or not isinstance(right_words, list):
            continue
        if not left_words or not right_words:
            continue
        try:
            gap = max(0.0, float(right["start"]) - float(left["end"]))
            left_duration = float(left["end"]) - float(left["start"])
            start = float(left["start"])
            end = float(right["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (gap, left_duration, start, end)):
            continue
        short_left = len(left_words) <= max_words or left_duration <= max_duration
        nearby = gap < DEFAULT_PAUSE_THRESHOLD
        incomplete = not _sentence_has_terminal_punctuation(left)
        if not (
            (short_left and (nearby or (incomplete and gap <= max_duration)))
            or (incomplete and nearby)
        ):
            continue
        reasons = []
        if short_left:
            reasons.append("short preceding sentence")
        if incomplete:
            reasons.append("apparently incomplete preceding sentence")
        if nearby:
            reasons.append("short pause")
        candidates.append(
            {
                "boundary_index": boundary_index,
                "start": start,
                "end": end,
                "gap": gap,
                "reason": "; ".join(reasons),
                "left": {
                    "text": str(left.get("text", "")),
                    "start": float(left["start"]),
                    "end": float(left["end"]),
                    "source_word_indices": list(left.get("source_word_indices", [])),
                },
                "right": {
                    "text": str(right.get("text", "")),
                    "start": float(right["start"]),
                    "end": float(right["end"]),
                    "source_word_indices": list(right.get("source_word_indices", [])),
                },
            }
        )
    return candidates


def _review_decision_map(response: Any, candidate_indices: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Parse the small, strict decision shape used by the boundary review API."""
    if not isinstance(response, dict):
        raise ValueError("Source Sentence review did not return a JSON object.")
    raw_decisions = response.get("decisions")
    decisions: Dict[int, Dict[str, Any]] = {}
    if isinstance(raw_decisions, list):
        for decision in raw_decisions:
            if not isinstance(decision, dict):
                raise ValueError("Source Sentence review contained an invalid decision.")
            raw_index = decision.get("boundary_index", decision.get("index"))
            if type(raw_index) is not int:
                raise ValueError("Source Sentence review contained an invalid boundary index.")
            if raw_index in decisions:
                raise ValueError("Source Sentence review repeated a boundary index.")
            merge = decision.get("merge")
            if merge is None:
                action = decision.get("action")
                merge = action == "merge" if isinstance(action, str) else None
            if type(merge) is not bool:
                raise ValueError("Source Sentence review contained an invalid merge action.")
            decisions[raw_index] = {"merge": merge, "text": decision.get("text")}
    elif isinstance(raw_decisions, dict):
        for raw_index, decision in raw_decisions.items():
            try:
                boundary_index = int(raw_index)
            except (TypeError, ValueError) as error:
                raise ValueError("Source Sentence review contained an invalid boundary index.") from error
            if boundary_index in decisions:
                raise ValueError("Source Sentence review repeated a boundary index.")
            if isinstance(decision, bool):
                merge = decision
                text = None
            elif isinstance(decision, str) and decision in {"merge", "keep"}:
                merge = decision == "merge"
                text = None
            elif isinstance(decision, dict):
                merge = decision.get("merge")
                if merge is None and isinstance(decision.get("action"), str):
                    merge = decision["action"] == "merge"
                text = decision.get("text")
            else:
                raise ValueError("Source Sentence review contained an invalid decision.")
            if type(merge) is not bool:
                raise ValueError("Source Sentence review contained an invalid merge action.")
            decisions[boundary_index] = {"merge": merge, "text": text}
    elif isinstance(response.get("merge_boundaries"), list):
        merge_boundaries = response["merge_boundaries"]
        if any(type(index) is not int for index in merge_boundaries):
            raise ValueError("Source Sentence review contained an invalid boundary index.")
        if any(index not in candidate_indices for index in merge_boundaries):
            raise ValueError("Source Sentence review referenced a clear boundary.")
        decisions = {
            index: {"merge": index in merge_boundaries, "text": None}
            for index in candidate_indices
        }
    else:
        raise ValueError("Source Sentence review did not include decisions.")

    expected = set(candidate_indices)
    if set(decisions) != expected:
        raise ValueError("Source Sentence review did not cover every candidate boundary.")
    return decisions


def apply_source_boundary_decisions(
    sentences: Sequence[Dict],
    response: Any,
    candidates: Optional[Sequence[Dict]] = None,
) -> List[Dict]:
    """Validate and apply an accepted boundary-review response."""
    candidates = list(candidates or ambiguous_source_boundaries(sentences))
    decisions = _review_decision_map(
        response,
        [int(candidate["boundary_index"]) for candidate in candidates],
    )
    result: List[Dict] = []
    group_start = 0
    for boundary_index in range(len(sentences) - 1):
        decision = decisions.get(boundary_index)
        if decision is None or not decision["merge"]:
            result.append(_merged_source_sentence(sentences[group_start : boundary_index + 1]))
            group_start = boundary_index + 1
    if group_start < len(sentences):
        group = sentences[group_start:]
        text = None
        if len(group) == 2:
            decision = decisions.get(group_start)
            if decision and isinstance(decision.get("text"), str):
                text = decision["text"]
        result.append(_merged_source_sentence(group, text=text))
    return result


def review_ambiguous_source_boundaries(
    sentences: Sequence[Dict],
    review: Optional[Callable[[List[Dict]], Any]] = None,
) -> SourceSentenceReviewResult:
    """Review selected boundaries once and safely fall back on invalid output."""
    source_sentences = [copy.deepcopy(sentence) for sentence in sentences]
    candidates = ambiguous_source_boundaries(source_sentences)
    if not candidates or review is None:
        return SourceSentenceReviewResult(source_sentences, [], True)
    try:
        reviewed = apply_source_boundary_decisions(
            source_sentences,
            review(candidates),
            candidates,
        )
    except Exception as error:
        first = candidates[0]
        diagnostic = SourceGroupingDiagnostic(
            "source_boundary_review_failed",
            float(first["start"]),
            float(first["end"]),
            f"{error}; retained deterministic Source Sentence grouping",
        )
        return SourceSentenceReviewResult(source_sentences, [diagnostic], False)
    return SourceSentenceReviewResult(reviewed, [], True)


def diagnose_source_word_timing(
    segments: Sequence[Dict],
    *,
    max_duration: float = DEFAULT_SUSPICIOUS_SOURCE_WORD_DURATION,
) -> List[SourceTimingDiagnostic]:
    """Report unusually long Source Word spans without inventing replacement times."""
    diagnostics: List[SourceTimingDiagnostic] = []
    for segment in segments:
        if not isinstance(segment, dict) or segment.get("text", "").strip() == "[no speech]":
            continue
        words = segment.get("words")
        if not isinstance(words, list):
            continue
        for word in words:
            if not isinstance(word, dict):
                continue
            try:
                start = float(word["start"])
                end = float(word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (start, end)) or end <= start:
                continue
            if end - start > max_duration:
                diagnostics.append(
                    SourceTimingDiagnostic(
                        "long_source_word",
                        start,
                        end,
                        f"Source Word {word.get('word', '')!r} spans {end - start:g}s",
                    )
                )
    return diagnostics


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


_TEXT_REPAIR_CODES = frozenset({"cps", "cue_cells", "line_cells", "rendered_lines"})


def _has_text_repair_limit(result: DeliveryGateResult) -> bool:
    return any(
        diagnostic.severity == "Repair Limit"
        and diagnostic.code in _TEXT_REPAIR_CODES
        for diagnostic in result.diagnostics
    )


def _should_fit_repair(result: DeliveryGateResult) -> bool:
    return any(
        diagnostic.code == "rendered_lines"
        or (
            diagnostic.severity == "Repair Limit"
            and diagnostic.code in {"cps", "cue_cells", "line_cells"}
        )
        for diagnostic in result.diagnostics
    )


def _has_alignment_repair_limit(
    result: DeliveryGateResult,
    *,
    can_split_duration: bool,
    profile: DeliveryProfile,
) -> bool:
    if _has_text_repair_limit(result) or any(
        diagnostic.code == "rendered_lines"
        and diagnostic.severity in {"Warning", "Repair Limit"}
        for diagnostic in result.diagnostics
    ):
        return True
    if not can_split_duration:
        return False
    for diagnostic in result.diagnostics:
        if diagnostic.code != "duration" or diagnostic.start is None or diagnostic.end is None:
            continue
        duration = diagnostic.end - diagnostic.start
        if duration > profile.warning_max_duration:
            return True
        if diagnostic.severity == "Repair Limit" and "outside" in diagnostic.message:
            return True
    return False


def _delivery_feedback(result: DeliveryGateResult) -> List[str]:
    return [
        f"{diagnostic.code}: {diagnostic.message}"
        for diagnostic in result.diagnostics
        if diagnostic.severity in {"Warning", "Repair Limit"}
    ]


def _combine_cues(left: Dict, right: Dict) -> Dict:
    combined = copy.deepcopy(left)
    combined["end"] = right["end"]
    combined["text"] = f"{left.get('text', '')}{right.get('text', '')}"
    combined["source_words"] = copy.deepcopy(left.get("source_words", []))
    combined["source_words"].extend(copy.deepcopy(right.get("source_words", [])))
    combined["source_word_indices"] = list(left.get("source_word_indices", []))
    combined["source_word_indices"].extend(right.get("source_word_indices", []))
    return combined


def _merge_adjacent_short_cues(
    cues: List[Dict],
    profile: DeliveryProfile,
) -> List[Dict]:
    """Merge short adjacent alignment pieces when the merged candidate remains usable."""
    merged: List[Dict] = []
    for cue in cues:
        if not merged:
            merged.append(cue)
            continue
        previous = merged[-1]
        try:
            duration = float(cue["end"]) - float(cue["start"])
            previous_duration = float(previous["end"]) - float(previous["start"])
        except (KeyError, TypeError, ValueError):
            merged.append(cue)
            continue
        if (
            duration >= profile.warning_min_duration
            and previous_duration >= profile.warning_min_duration
        ):
            merged.append(cue)
            continue
        combined = _combine_cues(previous, cue)
        separate_result = apply_delivery_profile([previous, cue], profile)
        combined_result = apply_delivery_profile([combined], profile)
        if (
            not combined_result.blocked
            and not _has_text_repair_limit(combined_result)
            and delivery_candidate_score(combined_result, profile)
            <= delivery_candidate_score(separate_result, profile)
        ):
            merged[-1] = combined
        else:
            merged.append(cue)
    return merged


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
) -> DeliveryGateResult:
    """Turn accepted Translation Sentences into timed Delivery Cues."""
    profile = profile or PORTRAIT_DELIVERY_PROFILE
    cues: List[Dict] = []
    diagnostics: List[DeliveryDiagnostic] = []
    for sentence_index, sentence in enumerate(translated_sentences, start=1):
        try:
            words, indices = _sentence_words(sentence, sentence_index)
        except SentenceFirstError as error:
            start = None
            end = None
            try:
                start = float(sentence["start"])
                end = float(sentence["end"])
            except (KeyError, TypeError, ValueError):
                pass
            diagnostics.append(
                DeliveryDiagnostic(
                    "Structural Defect",
                    "missing_source_word_timing",
                    sentence_index,
                    str(error),
                    start=start,
                    end=end,
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
                    start=float(words[0]["start"]),
                    end=float(words[-1]["end"]),
                )
            )
            continue
        cue = _delivery_cue(sentence, text, words, indices)
        initial_result = apply_delivery_profile([cue], profile)
        accepted_text = text
        fit_repair_attempted = False
        fit_repair_succeeded = False
        if _should_fit_repair(initial_result) and fit_repair is not None:
            fit_repair_attempted = True
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
                    fit_repair_succeeded = True
                    break

        cue = _delivery_cue(sentence, accepted_text, words, indices)
        cue_result = apply_delivery_profile([cue], profile)
        if (
            _has_alignment_repair_limit(
                cue_result,
                can_split_duration=len(words) > 1,
                profile=profile,
            )
            and align is not None
        ):
            best_cues: Optional[List[Dict]] = None
            best_result: Optional[DeliveryGateResult] = None
            last_feedback = _delivery_feedback(cue_result)
            last_error = "alignment did not return an accepted candidate"
            for _ in range(2):
                try:
                    aligned = _aligned_cues(
                        sentence,
                        accepted_text,
                        words,
                        indices,
                        align(
                            {
                                **sentence,
                                "text": accepted_text,
                                "_delivery_profile": profile,
                                "_delivery_feedback": last_feedback,
                            }
                        ),
                    )
                    aligned = _merge_adjacent_short_cues(aligned, profile)
                    candidate_result = apply_delivery_profile(aligned, profile)
                    if candidate_result.blocked:
                        raise ValueError("alignment candidate has a Structural Defect")
                except (TypeError, ValueError) as error:
                    last_error = str(error)
                    last_feedback = [last_error]
                    continue
                if (
                    best_result is None
                    or delivery_candidate_score(candidate_result, profile)
                    < delivery_candidate_score(best_result, profile)
                ):
                    best_cues = aligned
                    best_result = candidate_result
                last_feedback = _delivery_feedback(candidate_result)
                if not _has_alignment_repair_limit(
                    candidate_result,
                    can_split_duration=len(words) > 1,
                    profile=profile,
                ):
                    break
            if best_cues is not None:
                cues.extend(best_cues)
                if _has_alignment_repair_limit(
                    best_result or cue_result,
                    can_split_duration=len(words) > 1,
                    profile=profile,
                ):
                    diagnostics.append(
                        DeliveryDiagnostic(
                            "Repair Limit",
                            "alignment_unresolved",
                            len(cues),
                            "Best accepted Alignment candidate still exceeds a "
                            "Delivery Profile limit",
                            start=float(words[0]["start"]),
                            end=float(words[-1]["end"]),
                            outcome="candidate retained for Best-effort Delivery",
                        )
                    )
                continue
            if fit_repair_attempted and not fit_repair_succeeded:
                diagnostics.append(
                    DeliveryDiagnostic(
                        "Repair Limit",
                        "fit_repair_failed",
                        len(cues) + 1,
                        "Fit Repair did not return a Translation-Gate-accepted sentence",
                        start=float(words[0]["start"]),
                        end=float(words[-1]["end"]),
                        outcome="Alignment fallback also failed",
                    )
                )
            diagnostics.append(
                DeliveryDiagnostic(
                    "Repair Limit",
                    "alignment_failed",
                    len(cues) + 1,
                    last_error,
                    start=float(words[0]["start"]),
                    end=float(words[-1]["end"]),
                    outcome=(
                        "accepted Fit Repair retained"
                        if fit_repair_succeeded
                        else "full Translation Sentence retained"
                    ),
                )
            )
        if fit_repair_attempted and not fit_repair_succeeded and not align:
            diagnostics.append(
                DeliveryDiagnostic(
                    "Repair Limit",
                    "fit_repair_failed",
                    len(cues) + 1,
                    "Fit Repair did not return a Translation-Gate-accepted sentence",
                    start=float(words[0]["start"]),
                    end=float(words[-1]["end"]),
                    outcome="full Translation Sentence retained",
                )
            )
        cues.append(cue)

    result = apply_delivery_profile(cues, profile)
    diagnostics.extend(result.diagnostics)
    return DeliveryGateResult(result.cues, diagnostics)
