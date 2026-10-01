"""Versioned same-time subtitles and a separate, non-mutating render plan."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict

from .subtitles import apply_delivery_profile
from .translation import translation_request_counts

GROUPING_POLICY = "source-sentence-rules-v3"
TIMING_POLICY = "source-sentence-preserved-v1"


def read_source_srt(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as handle:
        content = handle.read().replace("\r\n", "\n").replace("\r", "\n").strip()
    blocks = re.split(r"\n[ \t]*\n", content) if content else []
    pattern = re.compile(r"^(-?\d+:\d{2}:\d{2},\d{3}) --> (-?\d+:\d{2}:\d{2},\d{3})$")
    result = []
    for number, block in enumerate(blocks, 1):
        lines = block.splitlines()
        match = pattern.fullmatch(lines[1]) if len(lines) >= 3 and lines[0].isdigit() else None
        if not match:
            raise ValueError(f"Invalid source SRT block {number}")
        def seconds(value):
            sign = -1 if value.startswith("-") else 1
            v = list(map(int, re.split(r"[:,]", value.lstrip("-"))))
            if v[1] >= 60 or v[2] >= 60:
                raise ValueError(f"Invalid source SRT time in block {number}")
            return sign * (v[0] * 3600 + v[1] * 60 + v[2] + v[3] / 1000)
        result.append({"start": seconds(match.group(1)), "end": seconds(match.group(2)), "text": "\n".join(lines[2:])})
    if not result:
        raise ValueError("Source SRT contains no cues")
    return result


def preserve_source_translation(source: list[dict], translated: list[dict]) -> list[dict]:
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


def render_plan(cues: list[dict], duration: float | None) -> tuple[list[dict], list[dict]]:
    visible, omitted = [], []
    for index, cue in enumerate(cues):
        start, end = float(cue["start"]), float(cue["end"])
        visible_start, visible_end = max(0, start), min(end, duration) if duration is not None else end
        reason = "non_positive_duration" if end <= start else "outside_video" if visible_end <= visible_start else None
        if reason:
            omitted.append({"cue_id": f"translation-{index}", "start_ms": round(start * 1000), "end_ms": round(end * 1000), "reason": reason})
        else:
            visible.append({**cue, "start": visible_start, "end": visible_end})
    return visible, omitted


def write_bundle(path: str, source: list[dict], translated: list[dict], *, video_fingerprint: str, source_language: str | None, target_language: str, profile, duration: float | None) -> dict:
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
    bundle["translation_sdk_requests"] = translation_request_counts()
    bundle["request_count_scope"] = "application SDK calls; SDK transport retries excluded"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, allow_nan=False, indent=2)
    return bundle
