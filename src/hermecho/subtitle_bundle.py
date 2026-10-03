"""Versioned same-time subtitles and a separate, non-mutating render plan."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from typing import Any

from .subtitles import DeliveryProfile, apply_delivery_profile
from .sentence_first import SourceTimingDiagnostic
from .translation import translation_request_counts

GROUPING_POLICY = "source-sentence-rules-v3"
TIMING_POLICY = "source-sentence-preserved-v1"


_TIMESTAMP_FRAGMENT = r"-?\d+:\d{2}:\d{2}[,.]\d{3}"
_TIMESTAMP_RE = re.compile(
    r"(?P<sign>-?)(?P<hours>\d+):(?P<minutes>\d{2}):"
    r"(?P<seconds>\d{2})[,.](?P<millis>\d{3})"
)
_TIMING_LINE_RE = re.compile(
    rf"^\s*(?P<start>{_TIMESTAMP_FRAGMENT})\s*-->\s*"
    rf"(?P<end>{_TIMESTAMP_FRAGMENT})(?:\s+.*)?\s*$"
)


def _parse_timestamp(value: str, block_number: int) -> float:
    match = _TIMESTAMP_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid source SRT time in block {block_number}")
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid source SRT time in block {block_number}")
    milliseconds = (
        int(match.group("hours")) * 3_600_000
        + minutes * 60_000
        + seconds * 1_000
        + int(match.group("millis"))
    )
    return (-1 if match.group("sign") else 1) * milliseconds / 1000


def read_source_srt(path: str) -> list[dict[str, Any]]:
    """Read an SRT while retaining every syntactically valid source cue.

    Imported studio SRTs may use arbitrary identifiers, omit identifiers, use a
    comma or dot millisecond separator, and append WebVTT-style settings to the
    timing line.  The identifier is deliberately not copied into the legacy
    Hermecho segment shape; Anamnesis assigns its own stable ID at import time.
    Timing and text are retained even when their range is negative, zero-length,
    reversed, or overlaps another cue.
    """
    with open(path, encoding="utf-8-sig") as handle:
        content = handle.read().replace("\r\n", "\n").replace("\r", "\n")
    content = content.lstrip("\ufeff")
    normalized = content.strip("\n")
    blocks = re.split(r"\n[ \t]*\n", normalized) if normalized else []
    result: list[dict[str, Any]] = []
    for number, block in enumerate(blocks, 1):
        lines = block.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if len(lines) < 2:
            raise ValueError(f"Invalid source SRT block {number}")

        timing_index = 0
        match = _TIMING_LINE_RE.fullmatch(lines[timing_index].strip())
        if match is None and len(lines) >= 2:
            identifier = lines[0].strip()
            if not identifier:
                raise ValueError(f"Invalid source SRT block {number}")
            timing_index = 1
            match = _TIMING_LINE_RE.fullmatch(lines[timing_index].strip())
        if match is None:
            raise ValueError(f"Invalid source SRT block {number}")

        text_lines = lines[timing_index + 1 :]
        if not text_lines or not "\n".join(text_lines).strip():
            raise ValueError(f"Invalid source SRT block {number}: missing text")
        result.append(
            {
                "start": _parse_timestamp(match.group("start"), number),
                "end": _parse_timestamp(match.group("end"), number),
                "text": "\n".join(text_lines),
            }
        )
    if not result:
        raise ValueError("Source SRT contains no cues")
    return result


def preserve_source_translation(
    source: list[dict[str, Any]], translated: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(source) != len(translated):
        raise ValueError("Translation is incomplete: source and translation counts differ")
    cues = []
    for original, translation in zip(source, translated):
        if not isinstance(translation.get("text"), str) or not translation["text"].strip():
            raise ValueError("Translation contains an empty cue")
        start, end = float(original["start"]), float(original["end"])
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Source timing must be finite")
        cues.append({**original, "source_text": original["text"], "text": translation["text"], "start": start, "end": end})
    return cues


def render_plan(
    cues: list[dict[str, Any]], duration: float | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    visible: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    for index, cue in enumerate(cues):
        start, end = float(cue["start"]), float(cue["end"])
        visible_start, visible_end = max(0, start), min(end, duration) if duration is not None else end
        reason = "non_positive_duration" if end <= start else "outside_video" if visible_end <= visible_start else None
        if reason:
            omitted.append({"cue_id": f"translation-{index}", "start_ms": round(start * 1000), "end_ms": round(end * 1000), "reason": reason})
        else:
            visible.append({**cue, "start": visible_start, "end": visible_end})
    return visible, omitted


def write_bundle(
    path: str,
    source: list[dict[str, Any]],
    translated: list[dict[str, Any]],
    *,
    video_fingerprint: str,
    source_language: str | None,
    target_language: str,
    profile: DeliveryProfile,
    duration: float | None,
    source_timing_diagnostics: list[SourceTimingDiagnostic] | None = None,
) -> dict[str, Any]:
    # The old profile is used for diagnostics only. It must never change saved cues.
    diagnostics = [dict(asdict(d), severity="Warning") for d in apply_delivery_profile(translated, profile).diagnostics]
    _, omitted = render_plan(translated, duration)
    source_cues, translation_cues = [], []
    for index, (original, cue) in enumerate(zip(source, translated)):
        timing = {"start_ms": round(float(original["start"]) * 1000), "end_ms": round(float(original["end"]) * 1000)}
        sid = f"source-{index}"
        source_cues.append({"cue_id": sid, "order": index, "text": original["text"], **timing, "source_word_indices": original.get("source_word_indices", []), "source_words": original.get("source_words", [])})
        translation_cues.append({"cue_id": f"translation-{index}", "order": index, "text": cue["text"], **timing, "source_cue_ids": [sid]})
    bundle = {"bundle_version": 1, "grouping_policy_version": GROUPING_POLICY, "timing_policy_version": TIMING_POLICY, "source_fingerprint": video_fingerprint, "source_language": source_language, "target_language": target_language, "source_cues": source_cues, "translation_cues": translation_cues, "diagnostics": diagnostics, "omitted": omitted, "llm_request_counts": {"boundary_review": 0, "alignment": 0, "fit_repair": 0}}
    bundle["source_timing_diagnostics"] = [
        dict(asdict(diagnostic), severity="Warning") for diagnostic in source_timing_diagnostics or []
    ]
    bundle["translation_sdk_requests"] = translation_request_counts()
    bundle["request_count_scope"] = "application SDK calls; SDK transport retries excluded"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, allow_nan=False, indent=2)
    return bundle
