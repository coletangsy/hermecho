# VAD-Based Transcription (A2) Implementation Plan

**Goal:** When `--use-vad` is set, run Voice Activity Detection on the extracted audio, transcribe only speech segments, and map segment timestamps back to the original video timeline so subtitles remain accurate.

**Architecture:** Optional path in `transcribe_audio()`: if `use_vad=True`, load audio once, run VAD to get speech (start, end) in original time, transcribe each chunk via Whisper on a slice of the audio array, add chunk offset to every segment’s `start`/`end` (and to `words[].start`/`end` if present), merge and return. Downstream pipeline is unchanged; it still receives segments in “original video time.”

**Tech stack:** Python, existing `openai-whisper` (torch), Silero VAD via `torch.hub` (no new pip dependency), `whisper.load_audio` for 16 kHz mono float32, NumPy for slicing.

**Design context:** `docs/plans/2025-03-15-reduce-name-hallucination-brainstorm.md` (Direction A2); timestamp mapping described in prior “add back to video” discussion.

---

## Overview: Data flow with VAD

1. **Extract audio** (unchanged) → e.g. `video.mp4` → `video.mp3`.
2. **Load audio once** → `whisper.load_audio(audio_path)` → float32 mono 16 kHz (same as Whisper’s internal format).
3. **Run VAD** on that array → list of `(start_sec, end_sec)` in **original** time (relative to full audio = video timeline).
4. **For each speech segment:**  
   - Slice array: `audio[start_sample:end_sample]`.  
   - `model.transcribe(segment_audio, ...)` → segments with `start`/`end` **relative to this chunk**.  
   - Add chunk offset: `start_orig = chunk_start_sec + seg["start"]`, same for `end` and for each `words[i].start`/`end`.  
   - Append to combined list (optionally sort by `start_orig`).
5. **Return** combined segments (all in original video time). Rest of pipeline (artifact cleaning, split, gap-fill, translate, burn) is unchanged.

**Guarantee:** Every segment’s `start` and `end` (and word timestamps) are in seconds from the start of the **original** audio/video, so SRT and burning subtitles into the video stay correct.

---

## Task 1: Add VAD dependency (Silero via torch.hub)

**Scope:** Use Silero VAD loaded with `torch.hub` so we don’t add a new pip package (torch is already a dependency of openai-whisper). If the project prefers an explicit package later, we can switch to e.g. `silero-vad` or another VAD.

**Files:**
- New: `src/vad.py` (or add to `transcription.py`; recommended: separate module for clarity and testing).

**Steps:**

1. **Create `src/vad.py`** with:
   - A function `get_speech_segments(audio_path: str, ...) -> list[tuple[float, float]]` that:
     - Loads audio in Whisper-compatible form: use `whisper.load_audio(audio_path)` (16 kHz mono float32).
     - Calls the array-based helper (below) and returns its result.
   - A function `get_speech_segments_from_array(audio_array: np.ndarray, sample_rate: int = 16000) -> list[tuple[float, float]]` that:
     - Loads Silero VAD once: e.g. `torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)` (and the utility `get_speech_timestamps` or equivalent).
     - Runs VAD on the array (Silero expects 16 kHz mono; our array is already that).
     - Converts sample indices to seconds: `start_sec = start_sample / sample_rate`, `end_sec = end_sample / sample_rate`.
     - Returns a list of `(start_sec, end_sec)` in ascending order, with optional merging of very close segments (e.g. merge if gap &lt; 0.3 s) to avoid tiny chunks.
   - `get_speech_segments(audio_path)` loads with `whisper.load_audio(audio_path)` then calls `get_speech_segments_from_array(audio, 16000)`. This allows `transcribe_audio` to load audio once and call the array API to avoid reading the file twice (see Task 2).
   - Document sample rate (16000) and that returned times are in seconds relative to the start of the file (original video timeline).
   - Handle edge cases: no speech → return `[]`; entire file speech → return `[(0.0, duration_sec)]` or equivalent.

2. **Optional parameters** for `get_speech_segments` (can be added later):  
   `min_speech_duration_ms`, `min_silence_duration_ms`, `speech_pad_ms`, `threshold`, etc., to match Silero’s API. Start with defaults that work for Korean/English conversation.

3. **Verification:**  
   - Unit test or one-off script: load a short audio file, call `get_speech_segments(audio_path)`, assert list of (float, float) and that values are non-negative and ordered.  
   - Manual run on a video with a known silent intro: assert first segment start is &gt; 0 (or that long silence is not in the first segment).

**Deliverable:** `src/vad.py` with `get_speech_segments(audio_path)` returning `list[tuple[float, float]]` in original-time seconds.

---

## Task 2: Implement chunk-wise transcription with timestamp mapping

**Files:**
- Modify: `src/transcription.py`.

**Steps:**

1. **Add a helper** (e.g. `_transcribe_audio_slice(audio_array: np.ndarray, model, language, initial_prompt, temperature, ...) -> list[dict]`) that:
   - Calls `model.transcribe(audio_array, language=..., initial_prompt=..., ...)` with the same kwargs as today (word_timestamps=True, etc.).
   - Returns `result["segments"]` or `[]` if missing, without changing timestamps (they are relative to the slice).

2. **Add a helper** (e.g. `_segments_to_original_time(segments: list[dict], offset_sec: float) -> list[dict]`) that:
   - Deep-copies or builds new segment dicts.
   - For each segment: `seg["start"] = seg["start"] + offset_sec`, `seg["end"] = seg["end"] + offset_sec`.
   - If segment has `"words"` with `start`/`end`, add `offset_sec` to each word’s `start` and `end`.
   - Returns the new list.

3. **Implement the VAD branch** inside `transcribe_audio()` when `use_vad=True`:
   - Remove the “not yet implemented” print; in its place:
     - Load audio: `audio = whisper.load_audio(audio_path)`.
     - Import and call `get_speech_segments(audio_path)` (or pass `audio` and sample rate if you add a signature that accepts array to avoid loading twice). Prefer a single load and pass the array into VAD if the VAD API accepts it, so we don’t read the file twice.
     - If speech segments list is empty: either return `[]` (no speech) or fall back to full-audio transcription; document the choice (recommend: return `[]` and let upstream handle “no segments” as today).
     - For each `(chunk_start_sec, chunk_end_sec)`:
       - Compute sample indices: `chunk_start_sec * 16000`, `chunk_end_sec * 16000` (ensure int and within bounds).
       - Slice: `segment_audio = audio[start_idx:end_idx]`.
       - If slice is too short (e.g. &lt; 0.1 s), skip or merge; optional.
       - Call `_transcribe_audio_slice(segment_audio, model, ...)` (need to load model once before the loop).
       - Call `_segments_to_original_time(segments, offset_sec=chunk_start_sec)`.
       - Append to a combined list.
     - After the loop: concatenate all lists and **sort by `start`** (in case of overlap or out-of-order chunks).
     - Return the combined list.
   - Ensure the Whisper model is loaded **once** before the loop (same as current flow) and reused for each chunk.

4. **Audio loading:**  
   For VAD we need the array. Options: (a) `get_speech_segments(audio_path)` loads internally and we load again in `transcribe_audio` for slicing, or (b) load once in `transcribe_audio`, pass array into a new `get_speech_segments_from_array(audio_array, sample_rate=16000)` and use the same array for slicing. Prefer (b) to avoid two file reads and to keep one source of truth for sample rate.

**Deliverable:** When `use_vad=True`, transcription uses VAD + per-chunk Whisper + offset mapping; returned segments are in original video time. When `use_vad=False`, behavior is unchanged.

---

## Task 3: Wire VAD in transcribe_audio and document

**Files:**
- Modify: `src/transcription.py`.
- Optionally: `src/main.py` (help text), `docs/whisper-artifact-fixes.md` or README.

**Steps:**

1. In `transcribe_audio()`:
   - When `use_vad=True`, follow the flow from Task 2 (load audio once → VAD → chunk loop → map timestamps → merge + sort).
   - When `use_vad=False`, keep current full-audio `model.transcribe(audio_path, ...)` path.
   - Remove the placeholder print (“VAD-based transcription is not yet implemented”).

2. Update the docstring of `transcribe_audio()`: describe that when `use_vad=True`, speech segments are detected, each is transcribed separately, and timestamps are mapped back to the original audio/video timeline so downstream subtitle burning remains correct.

3. Optional: In `main.py`, update the help for `--use-vad` to say that it uses VAD to transcribe only speech regions and that timestamps are in original video time.

4. Optional: Add one or two sentences in `docs/whisper-artifact-fixes.md` (or the brainstorm doc) stating that A2 is implemented and that segment timestamps are always in original video time.

**Deliverable:** CLI `--use-vad` enables real VAD-based transcription; docs and help reflect the behavior.

---

## Task 4: Edge cases and robustness

**Files:**
- `src/vad.py`, `src/transcription.py`.

**Steps:**

1. **No speech detected:**  
   If `get_speech_segments` returns `[]`, decide: return `[]` from `transcribe_audio` (recommended) so the rest of the pipeline sees “no segments” and can warn or exit as it does today when Whisper returns no segments.

2. **Very short chunks:**  
   Optionally skip chunks shorter than e.g. 0.2 s to avoid Whisper instability; or merge consecutive speech segments that are very close (e.g. gap &lt; 0.3 s) in the VAD step so we don’t send tiny slices.

3. **Empty segment list from Whisper for a chunk:**  
   Some chunks may return no segments (e.g. only noise). Don’t add offset to nothing; just skip and continue. Combined list may have “gaps” in time; that’s acceptable and gap-fill (or other guardrails) can still run.

4. **Sample rate consistency:**  
   Use 16000 everywhere (Whisper’s `load_audio` is 16 kHz; Silero VAD is typically 8k or 16k; use 16k so no resampling is needed).

5. **Sorting:**  
   After merging segments from all chunks, sort by `seg["start"]` so the order is strictly increasing. This keeps downstream logic (split long segments, gap-fill, etc.) consistent.

**Deliverable:** No crash on no-speech or empty chunks; timestamps remain correct; one place (e.g. 16000) for sample rate.

---

## Task 5: Verification and testing

**Steps:**

1. **Regression:** Run the pipeline **without** `--use-vad` on an existing test video; compare segment count and a few timestamps (or SRT) to a previous run. Expect no change.

2. **VAD run:** Run **with** `--use-vad` on the same video (and optionally one with long silent intro).  
   - Check that subtitles still align with speech (play video and spot-check).  
   - Check that first segment start time is &gt; 0 if the video has a long silent intro.  
   - Check that no segment has `start` or `end` negative or beyond video duration (if you have duration available, optional assert).

3. **Logging:** Optionally log number of VAD segments and total duration of speech vs file duration so users can see that VAD is active (e.g. “VAD: 12 speech segments, 4.2 min speech in 10.0 min audio”).

**Deliverable:** Manual (or automated) checks that (1) no-VAD behavior is unchanged, (2) with-VAD subtitles are in sync and timestamps are in original time.

---

## Summary checklist

| Item | Status |
|------|--------|
| VAD module: `get_speech_segments` (or from array) returning (start_sec, end_sec) in original time | Task 1 |
| Chunk-wise transcribe + `_segments_to_original_time` (segments + words) | Task 2 |
| `transcribe_audio(use_vad=True)` uses VAD path; no placeholder message | Task 3 |
| No speech / short chunks / empty Whisper output handled | Task 4 |
| Regression (no VAD) + VAD run (sync and timestamps) verified | Task 5 |

**Dependencies:** Silero VAD via `torch.hub` (no new pip package). If the environment has no internet at runtime, consider bundling the Silero model or adding an optional dependency and document it.

**Reference:** Timestamp mapping and “add back to video” accuracy are preserved because every segment’s `start`/`end` (and word timestamps) are converted to original video time before leaving `transcribe_audio()`.
