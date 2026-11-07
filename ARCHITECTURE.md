# Hermecho Architecture

## Purpose and trust boundary

Hermecho turns Korean-audio video or explicitly imported source subtitles into Traditional Chinese (Taiwan) subtitles and, optionally, a burned-in MP4. It is a local CLI with external model and media-tool boundaries. The normal transcription backend is local Whisper; uploading audio to OpenRouter requires the explicit OpenRouter backend flag and credentials.

## Pipeline

```text
video or --source-srt
        ↓
audio extraction / source-subtitle import
        ↓
selected transcription backend
  ├─ local Whisper / approved MLX path
  └─ explicit OpenRouter word-timestamp path
        ↓
Source Words → sentence-first grouping
        ↓
OpenRouter Translation Gate + Locked Terms
        ↓
Delivery Gate and subtitle bundle
        ├─ source/translated SRT
        └─ optional visible-interval SRT → ffmpeg burn-in MP4
```

`src/hermecho/cli.py` and `src/main.py` expose the command. `transcription.py`, `openrouter_transcription.py`, and `asr_comparison.py` own acquisition and comparison. `sentence_first.py`, `subtitles.py`, and `subtitle_bundle.py` preserve cue identity, millisecond timing, source references, and diagnostics. `translation.py` and `prompts.py` perform the model translation and response checks. `pipeline.py`, `checkpoints.py`, `progress.py`, and `video_processing.py` coordinate resumable work, progress, and media rendering.

Saved SRT output preserves every source cue and its timing, including zero-duration cues. Rendering uses a separate visible-interval view and may omit invalid/outside-video cues with reasons; it does not rewrite or shorten the saved subtitle files. Versioned subtitle bundles retain the source fingerprint, policy, diagnostics, and render omissions.

## Evaluation and operational boundaries

- ASR comparison and OpenRouter evaluation produce review artifacts; they do not silently change the normal backend or approve a model.
- Invalid word timings, missing anchors, malformed responses, authentication failures, and missing credentials block the relevant path. Do not invent a local fallback for a request that explicitly requires remote evidence.
- `ffmpeg` with subtitle filter support is required for burn-in. Keep `.env`, model weights, media, and generated output local.
- The proportionate checks are `pytest -q` and focused test modules under `tests/`; inspect bundle and gate reports even when the CLI exits successfully.
