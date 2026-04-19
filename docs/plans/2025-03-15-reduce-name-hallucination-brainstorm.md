# Reducing Name Hallucination and Content Loss — Brainstorm

**Date:** 2025-03-15  
**Status:** Implemented (Direction C: A1, optional A2 placeholder, B1, B2)  
**Goal:** Reduce Whisper name hallucination and avoid dropping real content when cleaning artifacts.

---

## 1. Problem

**Name hallucination** in single-speaker runs causes two issues:

1. **Whisper emits fake content** — On unclear audio (music, noise, silence), Whisper sometimes outputs comma-separated name-like tokens (e.g. "제이홉, 린, 지연, 린, 지연...") instead of leaving silence or transcribing real speech. That pollutes subtitles.

2. **Cleanup is losing real content** — The current speaker-hallucination heuristic in `_is_speaker_hallucination()` drops any segment that looks like “few unique tokens repeated many times.” That pattern can also match **real speech**, e.g.:
   - Someone actually listing names or short words
   - Repeated words in a sentence (e.g. “네, 네, 네” or “지연, 지연” as emphasis)
   - Short phrases with commas that happen to repeat

So we both want to **reduce how often Whisper hallucinates names** and **avoid dropping real content** when we clean. This brainstorm explores options for both; no code changes in this doc.

---

## 2. Direction A: Reduce hallucination at transcribe time (Whisper)

**Idea:** Give Whisper fewer chances to hallucinate and stronger instructions not to.

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A1. Stronger / more specific prompt** | Add explicit instructions, e.g. “Do not output lists of names or speaker labels. Output only the spoken words you hear.” (and keep “Single speaker. No speaker names or labels.”). | No new deps; easy to try. | Whisper may still ignore; prompt length is limited (~224 tokens). |
| **A2. Skip silence with VAD** | Run a Voice Activity Detector (VAD) first; only send non-silent chunks to Whisper. Less silence in → less hallucination on silence. | Targets root cause (silence → names). | New dependency and pipeline change; need to map chunk timings back to original timeline. |
| **A3. Tune Whisper params** | e.g. raise `no_speech_threshold` so more frames are classified as “no speech” and not decoded; or try `condition_on_previous_text=True` with care to avoid repetition loops. | Can reduce junk output without new pipeline steps. | May increase false “no speech” and drop quiet real speech; repetition can get worse with `condition_on_previous_text=True`. |
| **A4. Different model or fork** | Use a Whisper fork/version that mitigates hallucination (e.g. via repetition penalty or prompt filtering). | Might fix the issue at source. | Research and possibly new dependency; behaviour may differ by language. |

**Summary for A:** A1 is low-cost and worth doing. A2 is the most structural fix but requires design and implementation. A3 is a small change with trade-offs. A4 is a larger dependency/choice.

---

## 3. Direction B: Smarter post-processing (fewer false drops)

**Idea:** Keep a cleanup step but make it less aggressive so we don’t drop real content that looks like “names.”

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **B1. Stricter heuristic** | Only treat as hallucination when the pattern is **very** strong: e.g. more tokens (e.g. ≥12), higher repeat ratio (e.g. ≥3× unique count), or minimum segment length. | Fewer real segments dropped. | More true hallucination segments may slip through. |
| **B2. Use segment duration** | Name hallucination often appears in **long** segments (e.g. 20+ seconds of “린, 지연, 린, 지연...”). Require both the token pattern **and** long duration (e.g. >15s) before dropping. | Short real lists (a few names) are kept. | Short hallucination bursts remain. |
| **B3. Replace instead of drop** | Instead of dropping, try to **strip** only the repetitive name-like part (e.g. remove trailing “, A, B, A, B...” if the rest looks like normal speech) and keep the rest. | Preserves real content at the start of the segment. | Hard to do reliably; can leave broken or odd text. |
| **B4. Make it configurable** | Add a flag (e.g. `--aggressive-hallucination-filter`) or config; when off, skip speaker-hallucination dropping (or use a much looser heuristic). | User can choose “keep more content” vs “clean more.” | Default choice and docs matter; some users may see more junk. |
| **B5. Log only, don’t drop** | In a first iteration, only **log** segments that would be dropped (with a “would drop” marker) and still pass them through. Use logs to tune thresholds or to build a blocklist of exact phrases. | No content loss; data to tune heuristics or blocklists. | No automatic cleanup until later. |

**Summary for B:** B2 (duration gate) is a good next step: only drop when the segment is long **and** matches the repetition pattern. B1 (stricter thresholds) and B4 (configurable) are complementary. B5 helps tune without losing content.

---

## 4. Direction C: Combine source + post-processing

**Idea:** Do a bit of both so we rely less on any single lever.

- **Transcribe:** Stronger prompt (A1) and optionally VAD (A2) or param tweaks (A3) to reduce how often name hallucination appears.
- **Post-process:** Loosen the filter so it only fires on “obvious” cases: e.g. require **both** the current token-repetition pattern **and** (a) segment duration above a threshold (B2), and/or (b) stricter token counts (B1). Optionally make the filter configurable (B4) or first “log only” (B5).

This reduces content loss from over-aggressive dropping while still cleaning the worst hallucination.

---

## 5. Recommended next steps (for implementation later)

1. **Quick win (no new deps):**  
   - Strengthen the Whisper prompt (A1).  
   - Add a **duration gate** to speaker-hallucination (B2): e.g. only drop if segment duration > 15s (or configurable) **and** the current heuristic says “hallucination.” Keep short segments even if they match the token pattern.

2. **Tuning / safety:**  
   - Add a “log only” mode (B5) or a config flag to disable speaker-hallucination dropping (B4), so you can run on real content and see what would be dropped before committing to a default.

3. **Larger improvement (if needed):**  
   - Design VAD-based skipping of silence (A2) so Whisper sees less silence and hallucinates less at the source.

---

## 6. Open questions (for review)

- What segment duration (e.g. 15s, 20s) would you treat as “obviously hallucination” vs “could be real list”?
- Do you prefer a CLI flag to turn off speaker-hallucination dropping, or a config file, or both?
- Are you okay adding a VAD dependency later to reduce silence sent to Whisper?

No code changes in this document; this is for review and to inform a future implementation plan.
