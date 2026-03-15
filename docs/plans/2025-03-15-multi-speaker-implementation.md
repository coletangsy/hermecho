# Multi-Speaker (Phase 1) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an explicit `--multi-speaker` flag and multi-speaker–aware Whisper and translation prompts so multi-person content improves while single-person behavior stays unchanged.

**Architecture:** CLI flag in `main.py`; when set, append a clause to the Whisper initial prompt and add a block to the translation system prompt. No new dependencies or segment structure changes.

**Tech Stack:** Python, argparse, existing `transcription.py` / `translation.py` / LangChain prompt template.

**Design reference:** `docs/plans/2025-03-15-multi-speaker-design.md`

---

## Task 1: Add `--multi-speaker` CLI flag

**Files:**
- Modify: `src/main.py` (`_parse_arguments`)

**Step 1: Add the argument**

In `_parse_arguments()`, after the `--box_background` argument (around line 56), add:

```python
parser.add_argument("--multi_speaker", action="store_true",
                    help="Enable multi-speaker–aware prompts for conversation (Whisper + translation). Default: off.")
```

**Step 2: Verify the flag exists**

Run from repo root:

```bash
python src/main.py --help
```

Expected: Help output includes `--multi_speaker` and the description above.

**Step 3: Commit**

```bash
git add src/main.py
git commit -m "feat(cli): add --multi_speaker flag"
```

---

## Task 2: Append Whisper multi-speaker clause when flag is set

**Files:**
- Modify: `src/main.py` (prompt building in `_process_video`, ~lines 84–89)

**Step 1: Add clause when `args.multi_speaker` is True**

Current block:

```python
# Generate context-aware prompt for Whisper
keywords = extract_keywords_for_whisper(args.reference_file)
full_prompt = args.initial_prompt
if keywords:
    full_prompt = f"{full_prompt} Context: {keywords}"
print(f"Generated Whisper Prompt: {full_prompt}")
```

Change to:

```python
# Generate context-aware prompt for Whisper
keywords = extract_keywords_for_whisper(args.reference_file)
full_prompt = args.initial_prompt
if keywords:
    full_prompt = f"{full_prompt} Context: {keywords}"
if getattr(args, "multi_speaker", False):
    full_prompt = (
        f"{full_prompt} This may be a conversation between multiple speakers. "
        "Preserve natural turn-taking; avoid merging different speakers' words "
        "into one segment when there is a clear pause or change of speaker."
    )
print(f"Generated Whisper Prompt: {full_prompt}")
```

**Step 2: Verify**

Run with a dummy invocation (no need for real video) to ensure the branch is reachable, e.g. run `python src/main.py --multi_speaker nonexistent.mp4` from repo root and confirm it fails later (e.g. file not found) after printing a Whisper prompt that contains “multiple speakers”. Optional: add a one-line script or test that builds the prompt with `multi_speaker=True` and asserts the string "multiple speakers" in the result.

**Step 3: Commit**

```bash
git add src/main.py
git commit -m "feat(whisper): add multi-speaker clause to initial prompt when --multi_speaker"
```

---

## Task 3: Add `multi_speaker` to translation and prompt block

**Files:**
- Modify: `src/translation.py` (`_translate_chunk` and `translate_segments`)

**Step 1: Add parameter and conditional block in `_translate_chunk`**

- In `_translate_chunk`, add parameter `multi_speaker: bool = False` to the function signature (after `context: Dict[str, str]`).
- After the existing `prompt_template` assignment (the block ending with rule "6. **Preserve Original Tone**..."), add:

```python
if multi_speaker:
    prompt_template += (
        "\n\nMulti-Speaker Dialogue:\n"
        "The text may be a multi-speaker conversation. Preserve who said what; "
        "translate pronouns and references (e.g. 'he', 'she', 'that person') "
        "so they clearly refer to the correct speaker in context. "
        "Maintain natural dialogue flow and tone for each line.\n"
    )
```

**Step 2: Thread `multi_speaker` through `translate_segments`**

- Add parameter `multi_speaker: bool = False` to `translate_segments`.
- In the single-batch call to `_translate_chunk`, pass `multi_speaker=multi_speaker`.
- In the sliding-window loop, in both the primary `_translate_chunk` call and in the fallback sub-chunk/mini-chunk `_translate_chunk` calls, pass `multi_speaker=multi_speaker`.

**Step 3: Verify**

Run:

```bash
python -c "
from translation import _translate_chunk
# Build minimal prompt by calling with multi_speaker=True and inspect (or mock) – or just import and run translate_segments with multi_speaker=True on 1 segment
from langchain_core.prompts import ChatPromptTemplate
# Quick sanity: ensure translate_segments accepts multi_speaker
from translation import translate_segments
# translate_segments([], 'Traditional Chinese (Taiwan)', 'google/gemini-3.1-flash-lite-preview', None, multi_speaker=True) would need segments; at least check signature
import inspect
sig = inspect.signature(translate_segments)
assert 'multi_speaker' in sig.parameters
print('OK')
"
```

Expected: `OK` printed.

**Step 4: Commit**

```bash
git add src/translation.py
git commit -m "feat(translation): add multi_speaker prompt block for dialogue"
```

---

## Task 4: Pass `multi_speaker` from main to translation

**Files:**
- Modify: `src/main.py` (call to `translate_segments`, ~line 113)

**Step 1: Add keyword argument**

Find:

```python
translated_segments = translate_segments(
    transcribed_segments,
    target_language=args.target_language,
    translation_model=args.translation_model,
    reference_material=reference_material
)
```

Change to:

```python
translated_segments = translate_segments(
    transcribed_segments,
    target_language=args.target_language,
    translation_model=args.translation_model,
    reference_material=reference_material,
    multi_speaker=getattr(args, "multi_speaker", False),
)
```

**Step 2: Manual verification**

- Run on a **single-speaker** video **without** `--multi_speaker`: behavior and output should match previous runs (no regression).
- Run on the same or a **multi-speaker** video **with** `--multi_speaker`: pipeline should complete; translation prompt will include the multi-speaker block (can verify by adding a temporary print of the prompt or by checking translation quality).

**Step 3: Commit**

```bash
git add src/main.py
git commit -m "feat(main): pass --multi_speaker to translate_segments"
```

---

## Task 5: Update README and design doc status

**Files:**
- Modify: `README.md` (CLI arguments section)
- Modify: `docs/plans/2025-03-15-multi-speaker-design.md` (Status line)

**Step 1: Document the flag in README**

In the "Command-Line Arguments" section, add an entry:

- `--multi_speaker`: Enable multi-speaker–aware prompts (Whisper + translation) for conversations. Default: off. Use for videos with multiple people speaking.

**Step 2: Mark design as implemented for Phase 1**

In `docs/plans/2025-03-15-multi-speaker-design.md`, change the Status line to:

`**Status:** Phase 1 implemented (2025-03-15).`

**Step 3: Commit**

```bash
git add README.md docs/plans/2025-03-15-multi-speaker-design.md
git commit -m "docs: document --multi_speaker and mark Phase 1 implemented"
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Add `--multi_speaker` to argparse |
| 2 | Whisper: append multi-speaker clause when flag set |
| 3 | Translation: `multi_speaker` param + prompt block |
| 4 | Main: pass `args.multi_speaker` to `translate_segments` |
| 5 | README + design doc update |

No new dependencies. Single-speaker default unchanged. Manual test: run with and without `--multi_speaker` on one single-speaker and one multi-speaker video to confirm no regression and improved multi-speaker behavior.
