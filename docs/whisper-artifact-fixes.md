# Whisper Transcription Artifacts — Bugs and Fixes

**Date:** 2025-03-15  
**Status:** Implemented  
**Scope:** Single-speaker pipeline; cleanup of Whisper output before translation and subtitles.

This document records the Whisper-related bugs observed on real runs and the corresponding fixes added for review.

---

## 1. Overview

When running the pipeline on single-speaker videos, Whisper sometimes produced:

- Duplicate or zero-duration segments
- Speaker-name hallucination (repeated names)
- Periodic repetition of disclaimer-like text over long silent sections

The fixes are implemented as a **Guardrail 0** step: **artifact cleaning** right after transcription, plus a **prompt hint** to reduce speaker-related hallucination at transcribe time.

**Pipeline order (relevant part):**

1. Transcribe (Whisper)
2. **Clean artifacts** (new) — see Section 3
3. Split long segments
4. Fill gaps
5. Translate, adjust timing, burn subtitles

**Code locations:**

- **`src/main.py`** — Single-speaker prompt prefix; call to `clean_transcription_artifacts()` after transcription.
- **`src/subtitles.py`** — `clean_transcription_artifacts()`, `_is_speaker_hallucination()`, `_drop_periodic_silence_hallucination()`.

---

## 2. Bugs and Fixes (Detail)

### Bug 1: Zero-duration segments

**Observed:**

Segments with identical start and end time, e.g.:

```text
[17:12.260 --> 17:12.260]  Q. 꿈에서 어떤 말을 했는지 기억하고 있는 사람?
[17:18.280 --> 17:18.280]  Q. 꿈에서 어떤 말을 했는지 기억하고 있는 사람?
[17:18.400 --> 17:18.400]  Q. 꿈에서 어떤 말을 했는지 기억하고 있는 사람?
```

**Cause:** Whisper can emit segments with `start == end`, which are invalid for subtitles and add noise.

**Fix:** In `clean_transcription_artifacts()`, drop any segment where `end <= start` before other cleanup steps.

---

### Bug 2: Consecutive duplicate segments (same text repeated)

**Observed:**

The same line repeated many times with adjacent or overlapping timestamps:

```text
[17:00.860 --> 17:09.840]  Q. 꿈에서 어떤 말을 했는지 기억하고 있는 사람?
[17:10.160 --> 17:12.260]  Q. 꿈에서 어떤 말을 했는지 기억하고 있는 사람?
[17:12.260 --> 17:18.280]  Q. 꿈에서 어떤 말을 했는지 기억하고 있는 사람?
...
[17:27.080 --> 17:30.740]  Q. 꿈에서 어떤 말을 했는지 기억하고 있는 사람?
```

**Cause:** With `condition_on_previous_text=False` (used to avoid other repetition loops), Whisper may output the same phrase for multiple overlapping windows, producing many segments with identical text.

**Fix:** After other drops, merge **consecutive** segments that have the same trimmed text into a single segment: keep the first segment’s `start`, set its `end` to the last segment’s `end`, and merge `words` if present. Result: one segment spanning the full time range instead of many duplicates.

---

### Bug 3: Speaker-name hallucination

**Observed:**

A long segment filled with repeated names (e.g. in single-speaker video):

```text
[17:37.160 --> 18:00.780]  제이홉, 린, 지연, 린, 지연, 린, 지연, 린, 지연, ...
```

**Cause:** Known Whisper behaviour: when audio is unclear (music, noise, silence), the model may emit comma-separated “speaker” or name-like tokens instead of real speech.

**Fix:**

- **At transcribe time:** The Whisper initial prompt is prefixed with:  
  `"Single speaker. No speaker names or labels. "`  
  so the model is nudged not to output speaker labels or names.
- **At cleanup time:** In `clean_transcription_artifacts()`, segments are classified by `_is_speaker_hallucination(text)`. The heuristic: text split by commas/spaces has few unique tokens (e.g. ≤4) repeated many times (e.g. ≥8 tokens, or ≥2× unique count), or 1–2 unique tokens repeated ≥6 times. Such segments are **dropped** (no replacement); downstream gap-fill may insert `[no speech]` if the gap is large enough.

---

### Bug 4: Periodic silence hallucination (long empty intro)

**Observed:**

Videos that start with long silence produced the same disclaimer-like line at regular intervals (e.g. every ~30 seconds):

```text
[00:28.880 --> 00:29.680]  자막 제공 및 자막 제공 및 광고를 포함하고 있습니다.
[00:58.880 --> 00:59.680]  자막 제공 및 자막 제공 및 광고를 포함하고 있습니다.
[01:28.880 --> 01:29.680]  자막 제공 및 자막 제공 및 광고를 포함하고 있습니다.
...
[07:58.880 --> 07:59.680]  자막 제공 및 자막 제공 및 광고를 포함하고 있습니다.
```

**Cause:** On silence or near-silence, Whisper often outputs common boilerplate (e.g. Korean subtitle disclaimer text) repeatedly at fixed intervals instead of leaving silence empty.

**Fix:** In `clean_transcription_artifacts()`, `_drop_periodic_silence_hallucination()` runs before other content-based drops. It groups segments by **exact normalized text**. For any group where:

- the same text appears in **≥ 3** segments, and  
- the time span from the first segment’s start to the last segment’s end is **≥ 120 seconds**,

all segments in that group are **dropped**. This removes the repeated disclaimer over the long silent intro. Short-span repetition (e.g. same text a few times over &lt;2 minutes) is not dropped; consecutive identical segments are still merged by the existing merge step.

---

## 3. Cleanup step order (`clean_transcription_artifacts`)

Applied in this order:

1. **Drop zero-duration segments** — `end > start` only.
2. **Drop periodic silence hallucination** — same text in ≥3 segments spanning ≥120s.
3. **Drop speaker-hallucination segments** — heuristic on comma-separated repeated tokens.
4. **Merge consecutive identical text** — one segment per run of identical content.

All drops are **removals** (no placeholder text inserted by this step). `fill_transcription_gaps` later may insert `[no speech]` for gaps above its threshold.

---

## 4. Prompt change (single-speaker)

**Location:** `src/main.py` (construction of `full_prompt` for Whisper).

**Change:** The initial prompt passed to Whisper is prefixed with:

```text
Single speaker. No speaker names or labels. 
```

So the full prompt becomes:

```text
Single speaker. No speaker names or labels. <--initial_prompt> Context: <keywords if any>
```

This is applied for all runs (single-speaker pipeline); it helps reduce speaker-name and label hallucination without changing the rest of the pipeline.

---

## 5. Reduce-hallucination follow-up (Direction C: A1, A2 placeholder, B1, B2)

- **A1 — Stronger prompt:** The single-speaker prefix now also includes: “Do not output lists of names or speaker labels. Output only the spoken words you hear.”
- **A2 — VAD (optional, future):** `--use-vad` is accepted; when set, a notice is printed and full-audio transcription is used. VAD-based “transcribe only non-silent chunks” is not yet implemented.
- **B1 — Stricter heuristic:** `_is_speaker_hallucination()` now requires ≥12 tokens (was 8), repeat ratio ≥3× unique count (was 2×), and for 1–2 unique tokens ≥8 repeats (was 6).
- **B2 — Duration gate:** Speaker-hallucination segments are dropped only if duration > `speaker_hallucination_min_duration` (default 15s). Configurable via `--speaker-hallucination-min-duration`.

---

## 6. Tuning (for review)

- **Periodic hallucination:** Thresholds are `min_repeats=3` and `min_span_seconds=120.0`. If shorter spans (e.g. 60s) or different repeat counts are needed, they can be made configurable.
- **Speaker hallucination:** Heuristics are in `_is_speaker_hallucination()`. If new patterns appear (e.g. different languages or name formats), the conditions can be extended.
- **Single-speaker prompt:** If a future multi-speaker mode is added, this prefix should be conditional (e.g. only when not in multi-speaker mode).
