# Hermecho

Hermecho translates videos with Korean audio into Traditional Chinese (Taiwan) subtitles. It uses local Whisper for transcription, OpenRouter for translation, writes timestamped SRT files, and can hard-burn subtitles into a translated MP4.

## Features

- Local Whisper transcription with no transcription API usage.
- OpenRouter translation with reference-file context for names and terms.
- Translation Gate rejects incomplete model responses and enforces JSON Locked Terms.
- Segment guardrails for long subtitles, transcription gaps, and post-translation timing buffers.
- SRT-only, transcribe-only, and full burn-in modes.
- Subtitle styling controls for font, size, background box, margins, and ASS alignment.
- `ffmpeg` subtitle-filter detection before burn-in.
- Deterministic portrait and landscape Delivery Profiles measure subtitle width in Visual Cells and wrap to at most two lines. Every translated run writes a Delivery Gate report; presentation limits use Best-effort Delivery, while structural timing defects block final output.

The current pipeline does not include multimodal transcription, transcription prompts, keyword extraction, or timing-review stages.

## Installation

Prerequisites:

- Python 3.11+
- `ffmpeg` with the `subtitles` filter (`libass` support)

Create an environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

To update the existing Conda environment used for this project:

```bash
conda install -n hermecho python=3.11
conda run -n hermecho python -m pip install -e ".[dev]"
conda run -n hermecho python --version
```

On Apple Silicon, install the optional MLX Whisper runtime to try the large-v3
candidate backend:

```bash
python -m pip install -e ".[mlx]"
```

MLX model weights download on first use. It is opt-in: `auto` continues to use
portable Whisper until the comparison run receives Human Approval.

`requirements.txt` is kept for compatibility and installs the editable package:

```bash
python -m pip install -r requirements.txt
```

Create `.env` in the repo root:

```text
OPENROUTER_API_KEY="your_openrouter_key"
```

Translation uses OpenRouter's OpenAI-compatible API. The default model is
`deepseek/deepseek-v4-pro`, and requests prefer Alibaba first, then
AtlasCloud FP8, with provider fallback enabled.

Check `ffmpeg` subtitle support:

```bash
ffmpeg -hide_banner -filters | rg subtitles
```

## Usage

Place input videos under `input/` or set `--input_dir`.

Supported entrypoints:

```bash
hermecho episode01.mp4
python src/main.py episode01.mp4
PYTHONPATH=src python -m hermecho.cli episode01.mp4
```

Common modes:

```bash
hermecho clip.mp4 --transcribe-only
hermecho clip.mp4 --srt-only
hermecho clip.mp4 --save-source-transcript
hermecho clip.mp4 --input_dir ./videos --output_dir ./exports
```

The full pipeline is:

```text
extract audio -> local Whisper transcription -> split/fill segments -> OpenRouter Translation Gate -> timing adjustment -> SRT -> optional MP4 burn-in
```

For translated runs, `--locked-terms-file` is required and defaults to
`references/locked_terms.json`. It is a machine-readable JSON source-to-target
mapping enforced by the Translation Gate; a missing or invalid mapping blocks
translation and final SRT/MP4 delivery. `--reference_file` remains separate
Markdown prompt context. Accepted translations preserve punctuation for both
landscape and portrait delivery; portrait processing may wrap or split cues but
does not remove accepted punctuation.

## Options

Run `hermecho --help` for the full list.

| Option | Purpose |
| --- | --- |
| `video_filename` | File name inside `--input_dir`. |
| `--model` | Whisper model size, default `large`. |
| `--transcription-backend` | `auto` (currently Whisper), `whisper`, or Apple-Silicon-only `mlx`. MLX supports `large` / `large-v3` as large-v3. |
| `--language` | Source audio language, auto-detected by default. |
| `--target_language` | Translation target, default `Traditional Chinese (Taiwan)`. |
| `--translation_model` | OpenRouter model slug, default `deepseek/deepseek-v4-pro`. |
| `--reference_file` | Translation reference material, default `references/tripleS.md`. |
| `--locked-terms-file` | Required JSON source-to-target mapping for translated runs; defaults to `references/locked_terms.json`. Missing or invalid mappings block translation and final SRT/MP4 delivery. |
| `--temperature` | Whisper sampling temperature, default `0.0`. |
| `--time_buffer` | Seconds between subtitle cues after timing adjustment. |
| `--transcribe-only` | Write source-language SRT and stop. |
| `--srt-only` | Write translated SRT and skip video burn-in. |
| `--save-source-transcript` | Also write source-language SRT during a translated run. |
| `--font_name`, `--font_size`, `--outline_width`, `--box_background` | Burn-in subtitle styling. Font defaults to `Heiti TC`. |
| `--fonts-dir` | Font directory for FFmpeg; defaults to the macOS MobileAsset font directory. |
| `--margin_v`, `--margin_h`, `--alignment` | Burn-in subtitle placement. |
| `--stage-cooldown` | Delay between stages, default `60`; use `0` to disable. |

Outputs are written under `output/<video_basename>/` with a `YYYYMMDD_HHMMSS` timestamp. Translated runs also write a matching `*_delivery_gate.txt` report with any presentation warnings, Repair Limits, or Structural Defects.

## Hermecho Cloud rollout

Before deploying Hermecho Cloud changes that accept portrait jobs, install the compatible Hermecho release on the processor Mac. The pipeline owns the orientation-specific Delivery Profile used by both SRT and burned-in MP4 output.

## Development

Project metadata lives in `pyproject.toml`. Runtime code is packaged under `src/hermecho/`; `src/main.py` is only a compatibility wrapper.

```text
src/
├── main.py
└── hermecho/
    ├── cli.py
    ├── pipeline.py
    ├── transcription.py
    ├── translation.py
    ├── prompts.py
    ├── subtitles.py
    ├── video_processing.py
    └── utils.py
```

Run tests:

```bash
conda run -n hermecho python -m pytest tests/ -q
```

Local design notes and implementation plans belong under `docs/`. That directory is intentionally ignored and not tracked in Git; keep any review-ready operational guidance in this `README.md` or `AGENTS.md` instead.
