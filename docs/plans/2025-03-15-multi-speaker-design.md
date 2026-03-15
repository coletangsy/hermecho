# Multi-Speaker Conversation Support — Design

**Date:** 2025-03-15  
**Status:** Design (pending approval)  
**Goal:** Improve Hermecho for multi-person conversations (transcription, translation, optional speaker context) while keeping single-person behavior unchanged. Prefer explicit flag first; later, allow the pipeline to decide mode. Use speaker labels mainly for better translation.

---

## 1. Scope

- **In scope**
  - Explicit `--multi-speaker` flag to enable multi-speaker–aware prompts (Whisper + translation).
  - Prompt-only Phase 1: no new dependencies; single-person remains default and unchanged when the flag is not set.
  - Future: optional auto-detect (e.g. via diarization) so the pipeline can set multi-speaker mode.
  - Future: speaker labels used primarily to improve translation (who said what); display of labels in subtitles as a separate option.

- **Out of scope for Phase 1**
  - Speaker diarization (no new pipeline step yet).
  - Displaying speaker labels in final SRT (can be added in a later phase when we have speaker IDs).

---

## 2. User-Facing Behavior

- **Default (no flag):** Current behavior. Single-speaker–oriented prompts; no change for existing use.
- **With `--multi-speaker`:** Whisper and translation use multi-speaker–aware instructions (turn-taking, pronouns/references, dialogue flow). Same pipeline; only prompts and any minor formatting differ.
- **Later:** Pipeline may auto-detect multi-speaker (e.g. after adding diarization) and set this mode when appropriate; the flag remains for override.

---

## 3. Phase 1: Explicit Flag + Prompts

### 3.1 CLI

- Add optional boolean flag: `--multi-speaker`.
  - Default: `False` (current behavior).
  - When `True`: use multi-speaker prompts and any multi-speaker–specific formatting (e.g. instructing the model that input may be dialogue).

### 3.2 Whisper (transcription)

- **Current:** `initial_prompt` is built from user `--initial_prompt` plus optional context keywords (e.g. from reference file).
- **When `--multi-speaker` is True:** Append (or inject) a short clause so Whisper knows the audio may be a conversation, for example:
  - “This may be a conversation between multiple speakers. Preserve natural turn-taking; avoid merging different speakers’ words into one segment when there is a clear pause or change of speaker.”
- **When `--multi-speaker` is False:** Do not add this clause; keep existing behavior.

### 3.3 Translation

- **Current:** System prompt describes Korean → target language, JSON output, reference material, previous/next context, names, etc. No mention of multiple speakers.
- **When `--multi-speaker` is True:** Add a dedicated short block to the translation prompt, for example:
  - State that the input may be **multi-speaker dialogue**.
  - Instruct to preserve who said what and to resolve pronouns and references (“he”, “she”, “that person”, etc.) to the correct speaker using context.
  - Instruct to maintain natural dialogue flow and tone per line.
- **When `--multi-speaker` is False:** Do not add this block; keep existing prompt.

### 3.4 Data flow

- No change to segment structure or count. Translation still receives a JSON array of segment texts; output remains a 1:1 list of translated strings. No speaker IDs in Phase 1.

### 3.5 Backward compatibility

- Default remains single-speaker. No new required args. Existing scripts and commands behave as today unless `--multi-speaker` is passed.

---

## 4. Implementation Notes

- **Where to pass the flag:** From `main.py` (argparse) into `transcribe_audio()` and `translate_segments()`. Both need a parameter (e.g. `multi_speaker: bool = False`).
- **Transcription:** In `main.py`, when building `full_prompt` for Whisper, if `args.multi_speaker` then append the multi-speaker clause to `full_prompt` before calling `transcribe_audio()`.
- **Translation:** In `translation.py`, `_translate_chunk()` (and thus `translate_segments()`) should accept something like `multi_speaker: bool = False`. When `True`, the prompt template adds the multi-speaker block; when `False`, the template is unchanged.
- **Testing:** Run on a single-speaker video without the flag (expect same behavior as before) and with the flag (expect no regression, possibly slightly different wording). Run on a multi-speaker video with the flag and check that pronouns and references in the translation are more consistent.

---

## 5. Future: Auto-Detect and Speaker Labels

- **Auto-detect:** When diarization is added, the pipeline could run it first (or in parallel with Whisper), count distinct speakers, and set multi-speaker mode when count > 1. The explicit `--multi-speaker` flag would still override (e.g. force on for 1-speaker content that is “conversational” in style).
- **Speaker labels for translation:** In a later phase, segments can carry a `speaker` field (e.g. from diarization). When sending to the LLM, format as “Speaker A: …”, “Speaker B: …” so translation benefits from knowing who said what. Speaker labels are primarily for **better translation**, not necessarily for display.
- **Display of labels:** A separate option (e.g. `--speaker-labels-in-subtitles`) can control whether the final SRT includes “A:” / “B:” in the subtitle text or strips them for a clean look.

---

## 6. Summary

| Item                         | Phase 1                          | Later (optional)                    |
|-----------------------------|-----------------------------------|-------------------------------------|
| Explicit `--multi-speaker`  | Yes                               | Keep; allow pipeline to set mode    |
| Whisper prompt              | Multi-speaker clause when flag on | —                                   |
| Translation prompt          | Multi-speaker block when flag on  | Add speaker-prefixed input          |
| Auto-detect multi-speaker   | No                                | Via diarization (speaker count)     |
| Speaker labels              | No                                | For translation; display optional   |

Once this design is approved, the next step is to create a short implementation plan (tasks and order) and then implement Phase 1 only.
