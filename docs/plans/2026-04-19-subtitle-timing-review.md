# Subtitle timing review (multimodal) — implementation plan

> **For agentic workers:** Implement task-by-task with tests after each unit of behavior. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional pipeline stage after translation that sends **chunked audio** plus **per-cue timing metadata** to a multimodal LLM with **`temperature: 0`**, returning **adjusted `start`/`end` only** (same segment count), using **Korean audio** cues while subtitles may be **Chinese** (or any target language). The stage must explicitly target **audio–subtitle mismatch**, including the common case where **subtitles appear slower than the audio** (late `start`, overly late `end`, or cumulative drift relative to when the Korean phrase is audible).

**Architecture:** Reuse the existing pattern in `transcription.py`: `ffprobe` duration → `ffmpeg` extract time-bounded MP3 chunks → OpenRouter `chat/completions` with `input_audio` + strict JSON `response_format`. Input JSON lists each cue with **absolute** timeline `start`/`end`, **Korean source `text`** (for audio alignment), and **translated `translation` text** (for display context only; model must not rewrite it). Output is the same list with refined times; merge chunks back to a full timeline and run existing `_repair_multimodal_segment_times` (or a thin wrapper) for monotonic bounds. Wire the stage in `main.py` behind a CLI flag (default off for cost). Set **`temperature=0`** on translation `ChatOpenAI` for consistency with deterministic JSON passes.

**Desync / “subs slower than audio” (product intent):** Treat current `start`/`end` as a **hypothesis** from upstream (Whisper, gap-fill, multimodal transcript, etc.). The reviewer’s job is to **pull cues toward the audible Korean** when the hypothesis is late or long: e.g. if speech clearly begins before the given `start`, **move `start` earlier** (within clip and neighbor constraints); if the Korean phrase has already ended in audio but the cue still runs, **move `end` earlier**. Do **not** extend `end` based on how long it takes to read the Chinese line—reading speed is out of scope; bounds come from **speech activity** (and natural pauses) in the Korean clip. After review, the Whisper path may still run `adjust_subtitle_timing` (extends ends to next `start` minus buffer); document that this can **reintroduce** longer on-screen time—if lag fixes are visually undone, consider running timing review **after** `adjust_subtitle_timing` for Whisper, or add a flag to skip buffer extension when `--timing-review` was used (decide at implementation time; note in Task 5).

**Tech stack:** Python 3, `requests`, OpenRouter API (same as `transcription._transcribe_clip_openrouter`), `ffmpeg`/`ffprobe` (already required), `unittest` + `unittest.mock` (existing test style).

**Cross-lingual constraint (prompt contract):** The system message / user text must state explicitly: (1) spoken audio is **Korean**; (2) each cue’s `translation` is **not** Korean and is **read-only**; (3) use the **waveform / speech activity in the audio** and the **Korean `text`** to place **boundaries**—especially to correct **late subtitles** (when the subtitle appears to trail the speaker); (4) **never** change `translation` strings; (5) **never** change segment count or order; (6) only output adjusted floats for `start`/`end` per id; (7) **do not** align cues to Chinese reading duration—Chinese is only to know which Korean-aligned window this cue belongs to; timing comes from **Korean speech onset/offset** in the clip.

---

## File map

| File | Role |
|------|------|
| `src/timing_review.py` (new) | Chunking, prompt builder, OpenRouter call, parse/validate, merge absolute times. |
| `src/transcription.py` (modify) | Optionally export reuse: either import `_ffmpeg_extract_chunk`, `_audio_duration_seconds`, `_infer_openrouter_audio_format`, `OPENROUTER_CHAT_URL` from here into timing_review, **or** move shared “OpenRouter audio POST” to a tiny `src/openrouter_audio.py` — **YAGNI:** prefer importing the existing private helpers from `transcription` first to avoid churn; if that feels wrong, duplicate the minimal ffmpeg POST block once in `timing_review.py` with a comment “keep in sync with transcription” (only if circular import appears). |
| `src/main.py` (modify) | Flags: `--timing-review`, `--timing-review-model`; call reviewer after `translate_segments`, before `adjust_subtitle_timing` / SRT; pass `audio_path` — **important:** today audio is deleted in `finally`; timing review needs the file **before** removal, so run review **inside `try`** while `audio_path` exists (or copy WAV to a temp path early — simplest: run review before `finally` cleanup, same as transcription). |
| `src/translation.py` (modify) | `ChatOpenAI(..., temperature=0)` for deterministic translations (user requirement alignment). |
| `tests/test_timing_review.py` (new) | Unit tests for JSON normalization, merge, idempotent segment count, mocked HTTP. |

---

### Task 1: Prompt + JSON schema (document in code)

**Files:**
- Create: `src/timing_review.py`
- Test: `tests/test_timing_review.py`

- [ ] **Step 1: Write failing tests for pure helpers**

Add functions in `timing_review.py` (names illustrative):

- `build_review_payload_segments(segments: list[dict]) -> list[dict]` — each item: `id`, `start`, `end`, `text` (Korean source), `translation` (target). Skip or fail on missing keys (define behavior: require both texts for review, or allow empty translation with warning).

- `parse_review_response(raw: str, expected_ids: list[int]) -> list[dict] | None` — reuse `_parse_json_object_from_model_text` from `transcription` if imported, or copy the fence-tolerant parser into `timing_review` to avoid import cycles.

Example test (minimal):

```python
import unittest

from timing_review import build_review_payload_segments, parse_review_response


class TestBuildReviewPayloadSegments(unittest.TestCase):
    def test_builds_ordered_ids(self) -> None:
        segs = [
            {"start": 1.0, "end": 2.0, "text": "안녕", "translation": "你好"},
            {"start": 2.5, "end": 4.0, "text": "세계", "translation": "世界"},
        ]
        out = build_review_payload_segments(segs)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["id"], 0)
        self.assertEqual(out[0]["text"], "안녕")
        self.assertEqual(out[0]["translation"], "你好")
        self.assertEqual(out[1]["id"], 1)


class TestParseReviewResponse(unittest.TestCase):
    def test_parses_valid_minimal(self) -> None:
        raw = '{"segments":[{"id":0,"start":1.1,"end":2.0},{"id":1,"start":2.5,"end":4.0}]}'
        parsed = parse_review_response(raw, expected_ids=[0, 1])
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["start"], 1.1)
```

Run: `python -m pytest tests/test_timing_review.py -v`  
Expected: **FAIL** (module/functions missing).

- [ ] **Step 2: Implement `build_review_payload_segments` and `parse_review_response`**

Validation rules for `parse_review_response`:

- Top-level JSON: `{"segments": [...]}`.
- Each segment has `id`, `start`, `end` (numeric); **no** `translation` rewrite in output (ignore if model sends extra keys).
- After parse, sort by `id`; require exact set of ids `0..n-1` matching input length; else return `None` and log.

- [ ] **Step 3: Run tests — expect PASS**

Run: `python -m pytest tests/test_timing_review.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/timing_review.py tests/test_timing_review.py
git commit -m "feat(timing-review): add segment payload builder and JSON parser"
```

---

### Task 2: Relative-time windowing + merge back to absolute

**Files:**
- Modify: `src/timing_review.py`
- Modify: `tests/test_timing_review.py`

- [ ] **Step 1: Write failing tests for window merge**

Functions:

- `slice_segments_for_window(segments, window_start, window_end) -> tuple[list[dict], list[int]]`  
  Return segments overlapping `[window_start, window_end)` and their original indices.

- `apply_window_time_adjustments(full_segments, window_start, adjustments_by_id) -> None`  
  Mutate copies: for each adjusted cue, set `start = window_start + rel_start`, `end = window_start + rel_end`, clamped to `[window_start, window_end)` intersecting original segment span (define clamp policy in docstring: clamp to window bounds then global repair pass).

Test idea: two segments in window `[10, 20)`; model returns times relative to 0–10s clip; merged absolute starts should be `10 + t`.

```python
def test_merge_relative_to_absolute(self) -> None:
    from timing_review import apply_window_adjustments_to_full_list

    full = [
        {"start": 10.0, "end": 12.0, "text": "a", "translation": "A"},
        {"start": 12.0, "end": 15.0, "text": "b", "translation": "B"},
    ]
    # Model returns clip-relative 0..duration for extracted chunk matching [10,15)
    rel = [
        {"id": 0, "start": 0.0, "end": 2.0},
        {"id": 1, "start": 2.0, "end": 5.0},
    ]
    out = apply_window_adjustments_to_full_list(
        full_segments=full,
        window_start=10.0,
        window_duration=5.0,
        relative_segments=rel,
        index_map=[0, 1],
    )
    self.assertAlmostEqual(out[0]["start"], 10.0)
    self.assertAlmostEqual(out[0]["end"], 12.0)
```

Adjust names/signatures to your implementation; the invariant to test is **offset add** + **index mapping**.

- [ ] **Step 2: Implement merge helpers**

- [ ] **Step 3: pytest PASS**

- [ ] **Step 4: Commit** — `feat(timing-review): merge clip-relative times to absolute timeline`

---

### Task 3: OpenRouter multimodal call (temperature 0)

**Files:**
- Modify: `src/timing_review.py`

- [ ] **Step 1: Write test with `requests.post` mocked**

Patch `timing_review.requests.post` to return a fake `200` JSON body with `choices[0].message.content` holding valid JSON string.

Assert `json.loads` of the **request** `kwargs['json']` contains `"temperature": 0` and `input_audio`.

- [ ] **Step 2: Implement `review_timing_chunk`**

Signature sketch:

```python
def review_timing_chunk(
    chunk_audio_path: str,
    window_start: float,
    segments_in_window: list[dict],
    review_model: str,
    source_language: str,
    target_language: str,
) -> tuple[list[dict] | None, dict | None]:
    ...
```

Implementation details:

- Build user text: rules from **Architecture** + JSON shape example; include one short **negative example** in prose (“if the Korean line is already audible at 0.2s but `start` is 1.2s, move `start` toward 0.2s unless the next cue forbids overlap”).
- Payload mirrors `transcription._transcribe_clip_openrouter`: `model`, `messages` with text + `input_audio`, `response_format: json_object`, **`temperature`: 0**.
- Log token usage like `_log_openrouter_token_usage` (import from `transcription` or duplicate small logger).

- [ ] **Step 3: pytest PASS**

- [ ] **Step 4: Commit** — `feat(timing-review): OpenRouter chunk review with temperature 0`

---

### Task 4: Orchestrator `review_subtitle_timing`

**Files:**
- Modify: `src/timing_review.py`
- Modify: `tests/test_timing_review.py` (mock ffmpeg or chunk function)

- [ ] **Step 1: Implement `review_subtitle_timing(audio_path, segments, chunk_seconds, review_model, source_language, target_language) -> list[dict] | None`**

Algorithm:

1. `duration = _audio_duration_seconds(audio_path)` — import from `transcription`.
2. Loop `window_start` in steps of `chunk_seconds` (same pattern as `transcribe_audio_multimodal`).
3. For each window, `_ffmpeg_extract_chunk(audio_path, tmp_path, window_start, span)` — import from `transcription`.
4. `slice_segments_for_window(...)`.
5. If no segments in window, skip.
6. Build JSON payload with **clip-relative** `start`/`end` for model input (`seg["start"] - window_start`) so the model’s task matches the audio file length (document this in prompt: “times in this JSON are relative to this clip”).
7. Call `review_timing_chunk`; on failure return `None` or fall back to original segments (choose one policy; **recommend:** on failure, log and keep original times for that window).
8. Merge adjustments into a working copy of `segments`.
9. After all windows, `merged = _repair_multimodal_segment_times(segments, clip_end=duration)` from `transcription` to fix overlaps / bounds.

- [ ] **Step 2: Unit test orchestrator with mocks** — patch `_ffmpeg_extract_chunk` to write empty file and `review_timing_chunk` to echo adjusted JSON.

- [ ] **Step 3: pytest full file**

Run: `python -m pytest tests/test_timing_review.py -v`

- [ ] **Step 4: Commit** — `feat(timing-review): full audio-chunk orchestration and global repair`

---

### Task 5: CLI + `main.py` integration

**Files:**
- Modify: `src/main.py`
- Modify: `src/translation.py`

- [ ] **Step 1: Add `ChatOpenAI(..., temperature=0)`** in `translation.py` for `_translate_chunk`.

- [ ] **Step 2: argparse**

- `--timing-review` — `action="store_true"`, help text explains extra OpenRouter cost, Korean audio + translated subtitles.
- `--timing-review-model` — default e.g. same as multimodal or `google/gemini-3.1-flash-lite-preview` (match project defaults).

- [ ] **Step 3: Call site**

After `translate_segments` succeeds; then choose **one** ordering (document in code comments): **(A)** `adjust_subtitle_timing` → timing review → SRT (review sees buffer-adjusted spans; may still pull late `start` earlier); **(B)** timing review → `adjust_subtitle_timing` (buffer pass may **lengthen** `end` after review—can partly undo “end too late” fixes); **(C)** timing review → skip `adjust_subtitle_timing` when `--timing-review` (tightest sync; tradeoff: shorter flashes). Default recommendation: **(A)** or **(C)** depending on whether you prioritize gap-filling or strict lip-sync. Then `_print_segments` / SRT:

```python
if args.timing_review:
    from timing_review import review_subtitle_timing

    reviewed = review_subtitle_timing(
        audio_path,
        translated_segments,
        chunk_seconds=min(args.chunk_seconds, 120.0),  # or dedicated --timing-review-chunk-seconds
        review_model=args.timing_review_model,
        source_language=args.language,
        target_language=args.target_language,
    )
    if reviewed is not None:
        translated_segments = reviewed
```

Use a dedicated `--timing-review-chunk-seconds` default **120** if full `--chunk-seconds` (300) is too large for review payloads with long Chinese strings.

- [ ] **Step 4: Manual smoke test** (document in commit message): short clip with `--timing-review`, confirm SRT times change slightly and JSON errors don’t crash pipeline; on a clip where subs were **visibly late** vs Korean speech, confirm **`start` moves earlier** (or `end` tightens) in the output SRT after review.

- [ ] **Step 5: Commit** — `feat(cli): optional timing review after translation; translation temperature 0`

---

### Task 6: Verification checklist (no placeholder)

- [ ] Run `python -m pytest tests/test_transcription_multimodal.py tests/test_timing_review.py -v` — all green.
- [ ] `ruff` / `flake8` / project linter if configured — clean on touched files.
- [ ] Grep confirms `temperature` is `0` for: multimodal transcribe (already), translation `ChatOpenAI`, timing review POST body.

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Model changes Chinese text despite instructions | Parse only `start`/`end` by `id`; **re-attach** `translation` from input segment when building output list; never trust model for subtitle text. |
| Chunk boundary splits a Korean phrase | Overlap windows (e.g. 2s) like translation’s `OVERLAP_SIZE` concept, or only apply adjustments for segments whose midpoint lies inside window center 80%; **MVP:** no overlap, document boundary weakness; **v2:** overlap + merge policy. |
| Token / size limits | Smaller `--timing-review-chunk-seconds`; trim long `translation` in payload with ellipsis **only if** you also pass full Korean `text` (timing driven by audio + Korean). |
| Audio file deleted too early | Keep review inside `try` before `finally` removes `audio_path`. |
| `adjust_subtitle_timing` extends each cue’s `end` toward the next `start` | Can make subs feel “slow” again after review; mitigate by ordering (review after adjust), skipping adjust when review ran, or lowering `time_buffer` when using review—capture the chosen behavior in Task 5. |
| Model pulls `start` so early it collides with previous cue | Rely on `_repair_multimodal_segment_times` + minimum gap constants; optional cap: new `start` not more than `X` s before previous `end` (only if empirical issues). |

---

## Out of scope (YAGNI)

- Rewriting translation strings or detecting mistranslation.
- Word-level alignment between Korean and Chinese.
- Training a separate alignment model offline.

---

## Summary

Deliver a **new module** with **tested** JSON + merge logic, an **OpenRouter multimodal** loop with **`temperature: 0`**, **CLI wiring** in `main.py`, and **explicit prompts** that Korean audio drives boundaries (including **fixing late / lagging cues**) while Chinese (target) text is display-only and immutable in output. Resolve interaction with **`adjust_subtitle_timing`** so lag fixes are not accidentally undone. Set **translation** LLM **`temperature=0`** for consistency.
