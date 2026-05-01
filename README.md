
# Hermecho

> A fusion of "Hermes," the Greek messenger god symbolizing interpretation and communication, with "Echo," the nymph who repeats spoken words. This evokes the tool's transcription of Korean audio and its translation into Chinese subtitles, bridging languages like a divine echo.

This project is a high-performance, command-line tool for translating videos with Korean audio into videos with Traditional Chinese (Taiwan) subtitles. It leverages a sophisticated pipeline that includes local transcription, intelligent LLM-based translation, and automated subtitle generation.

## Key Features

*   **Automated End-to-End Pipeline:** From video input to a final, subtitled video output with minimal user intervention.
*   **High-Quality Transcription:** Defaults to **Google AI Studio (Gemini) multimodal** transcription (chunked audio) when you set `GEMINI_API_KEY`. Optional local **Whisper** via `--whisper` gives you an offline or API-free transcribe step.
*   **Intelligent, Context-Aware Translation:** Employs a Large Language Model (for example `gemini-3.1-flash-lite-preview`) with a highly optimized prompt that performs several advanced tasks:
    *   **Contextual Correction:** Uses a reference file to intelligently correct transcription errors in names and key terminology.
    *   **Smart Name Handling:** Translates names to official English stage names and preserves specific English terms.
*   **Optimized for Performance & Robustness:** The translation process is architected for speed and reliability:
    *   **Adaptive Batch Processing:** Automatically selects the best strategy (single batch vs. sliding window) based on text length.
    *   **Multi-Level Fallback:** If a translation chunk fails, it recursively splits into smaller batches (50 -> 10 segments) to isolate and resolve issues without failing the whole process.
*   **Enhanced Transcription:**
    *   **Gemini (AI Studio), OpenRouter multimodal, or Whisper:** By default, chunked audio uses a Gemini model via Google AI Studio (`GEMINI_API_KEY`), or OpenRouter if you pass a `provider/model` slug; use **`--whisper`** to transcribe locally instead.
    *   **Context-Aware Prompting:** Extracts keywords from reference materials to guide the transcription prompt, improving accuracy for specific terms and names.
    *   **Segment Optimization:** Automatically splits overly long segments for better subtitle readability.
    *   **Timing review (default):** After translation, a multimodal pass refines subtitle start/end times against the audio (extra API usage). Pass **`--no-timing-review`** to skip it.
*   **Customizable Subtitles:**
    *   **Styling Options:** Supports custom fonts, sizes, outlines, and background boxes for professional-looking subtitles.
    *   **Positioning Controls:** Fine-tune subtitle placement with vertical and horizontal margins plus ASS-style alignment presets (`--margin_v`, `--margin_h`, `--alignment`).
    *   **Versioning:** Output files are timestamped to prevent overwriting and allow easy version tracking.
*   **FFmpeg Capability Check:** Detects whether your local `ffmpeg` build includes the `subtitles` filter (`libass`) before the burn-in stage and prints a clear remediation hint if it is missing.
*   **Robust Error & Gap Handling:** The pipeline includes several guardrails for a more professional result:
    *   **Transcription Gap Filling:** Automatically detects and fills long periods of silence with a `[no speech]` placeholder.
    *   **Strict JSON Communication:** Enforces a strict JSON-based workflow with the LLM to prevent malformed outputs and conversational filler.
*   **Subtitle Generation:** Creates a standard `.srt` subtitle file and burns it directly into the final video.

## Demo
[![Watch the video](https://img.youtube.com/vi/zl5XqVPHbeE/maxresdefault.jpg)](https://www.youtube.com/watch?v=zl5XqVPHbeE)

*This video demonstrates the final output of the solution, featuring subtitles created by this project. 
[[Link to Original Video](https://www.youtube.com/watch?v=zHr59WHRkdA)] 


## How it Works

The video translation process is a multi-stage pipeline designed for quality and robustness:

1.  **Audio Extraction:** The audio is extracted from the input video using `ffmpeg`.
2.  **Transcription:** Audio is transcribed with **Google AI Studio (Gemini)** multimodal chunked requests by default (or **local Whisper** with `--whisper`), using an initial prompt plus keywords from the reference file when available.
3.  **Refinement:** Long segments are split for readability, and significant time gaps are filled with placeholders.
4.  **Intelligent Translation:** The text segments are sent to an LLM. The system employs a robust strategy:
    *   **Adaptive Batching:** Attempts single-batch translation for shorter texts.
    *   **Sliding Window with Fallback:** For longer texts or failed batches, it uses a sliding window approach. If a chunk fails, it recursively breaks it down into smaller sub-chunks to ensure completion.
5.  **Subtitle Generation:** Translated segments become a timestamped `.srt`. **Timing review** runs by default (skip with `--no-timing-review`) to realign cue times to the audio. On the **Whisper** path, `adjust_subtitle_timing` then applies `--time_buffer` between cues; the default multimodal path does not run that stretch and keeps the reviewed (or translated) segment times.
6.  **Video Finalization:** A new video file is created with the generated subtitles burned directly into it, applying custom styling (font, size, background).

## Getting Started

### Prerequisites

*   Python 3.9+
*   `ffmpeg`: This must be installed on your system. On macOS, you can use Homebrew: `brew install ffmpeg`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/coletangsy/hermecho.git
    cd hermecho
    ```

2.  **Create and use a Conda environment** (keeps dependencies isolated from your system Python):
    ```bash
    conda create -n hermecho python=3.11 -y
    conda activate hermecho
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file** in the root directory. The current pipeline uses **Google AI Studio** for multimodal transcription, translation, and timing review, so set **`GEMINI_API_KEY`**. With **`--whisper`**, only the transcription stage becomes local; translation and timing review still use Gemini unless you stop earlier with `--transcribe-only` or skip timing review with `--no-timing-review`.
    ```
    GEMINI_API_KEY="your_google_ai_studio_key"
    ```

### FFmpeg note

Hard-burned subtitles require an `ffmpeg` build with the `subtitles` filter (that means `libass` support is available). You can verify your local binary with:

```bash
ffmpeg -hide_banner -filters | rg subtitles
```

If that command does not show `subtitles`, reinstall `ffmpeg` with subtitle support before running the final burn-in stage.

### Running tests

Unit tests live under `tests/`. For development, install `requirements-dev.txt` in an isolated environment, then run from the repository root with `src` on `PYTHONPATH` so imports match the CLI:

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src python -m pytest tests/ -v
```

Pytest writes cache under `.pytest_cache/`; that directory is listed in `.gitignore` alongside typical local virtualenv folders (`.venv/`, `venv/`, etc.) if you use them.

Local design notes and implementation plans belong under `docs/`. That directory is intentionally ignored and not tracked in Git; keep any review-ready operational guidance in this `README.md` or `AGENTS.md` instead.

### Experimental VibeVoice-ASR environment

VibeVoice-ASR is intentionally kept out of the normal runtime requirements because it pulls heavy local ML dependencies and model downloads. Use a separate environment when comparing Gemini transcription against local VibeVoice:

```bash
python -m venv .venv-vibevoice
source .venv-vibevoice/bin/activate
pip install -r requirements-vibevoice.txt
PYTHONPATH=src python -m asr_comparison --skip-gemini --video Wsp9Z6-S0LA.mp4
```

The strongest official Transformers-compatible checkpoint is `microsoft/VibeVoice-ASR-HF`. It is an 8B BF16 model, so Apple Silicon machines with limited unified memory may fail or run slowly. The comparison runner records local hardware preflight data in `comparison_report.md`; if local inference cannot complete, rerun on a CUDA GPU before falling back to quantized community variants.

Useful VibeVoice knobs:

```bash
PYTHONPATH=src python -m asr_comparison \
  --skip-gemini \
  --video Wsp9Z6-S0LA.mp4 \
  --vibevoice-model microsoft/VibeVoice-ASR-HF \
  --vibevoice-device auto \
  --vibevoice-dtype auto \
  --vibevoice-tokenizer-chunk-size 64000
```

For source-transcript-only probing before paying for translation and burn-in:

```bash
PYTHONPATH=src python -m asr_comparison --skip-gemini --vibevoice-transcribe-only --video Wsp9Z6-S0LA.mp4
```

## Usage

Place your source file under `input/` (or set `--input_dir`), then run the CLI from the **repository root** so imports resolve correctly:

```bash
python src/main.py my_video.mp4
```

That runs the **full pipeline**: extract audio → transcribe → (optional source SRT) → translate → subtitle timing → write SRT → burn subtitles into a new MP4 under `output/<video_basename>/`.

### Quick examples

**Default (Gemini multimodal transcribe, translate, burn-in)**

```bash
python src/main.py episode01.mp4
```

**Transcription only (Korean SRT, no translation or video)**

```bash
python src/main.py clip.mp4 --transcribe-only
```

**Local Whisper instead of multimodal** (same translation and burn steps unless `--transcribe-only`)

```bash
python src/main.py clip.mp4 --whisper
```

**Override multimodal model**

```bash
python src/main.py clip.mp4 \
  --multimodal-model gemini-3.1-flash-lite-preview
```

**Write the translated SRT but skip MP4 burn-in**

```bash
python src/main.py clip.mp4 --srt-only
```

**Keep a source-language SRT alongside the translated run**

```bash
python src/main.py clip.mp4 --save-source-transcript
```

**Timing review options** (runs by default after translation; tune model or chunk size)

```bash
python src/main.py clip.mp4 \
  --timing-review-model google/gemini-3.1-flash-lite-preview \
  --timing-review-chunk-seconds 120
```

**Skip timing review** (faster, fewer API calls)

```bash
python src/main.py clip.mp4 --no-timing-review
```

**Custom input/output folders**

```bash
python src/main.py myfile.mp4 --input_dir ./videos --output_dir ./exports
```

### Command-line arguments

Run `python src/main.py --help` for the full list. Common options:

| Option | Purpose |
|--------|---------|
| `video_filename` | Name of the file inside `--input_dir` (not a full path unless that matches how you set `input_dir`). |
| `--whisper` | Use local Whisper for transcription instead of the default OpenRouter multimodal path. |
| `--model` | Whisper size when `--whisper` is set (`tiny` … `large`; default `large`). |
| `--language` | Source audio language for transcription (default `ko`). |
| `--multimodal-model` | Gemini model id for multimodal transcription via **Google AI Studio** (`GEMINI_API_KEY`). Default comes from `transcription.DEFAULT_MULTIMODAL_MODEL`. |
| `--multimodal-chunk-seconds` | Max seconds of audio per Gemini multimodal transcription request (default from `transcription.DEFAULT_MULTIMODAL_CHUNK_SECONDS`; minimum 60 when chunking, `0` = one request for the whole file). |
| `--transcribe-only` | Stop after source-language SRT; no translation or burn-in. |
| `--srt-only` | Run transcription, translation, and timing review, then stop after writing the translated SRT. |
| `--save-source-transcript` | With the full pipeline, also writes `*_transcript_source.srt` before translation. |
| `--target_language` | Translation target (default `Traditional Chinese (Taiwan)`). |
| `--translation_model` | Gemini model id for translation via Google AI Studio (default `gemini-3.1-flash-lite-preview`). |
| `--timing-review` / `--no-timing-review` | Timing refinement after translation is **on** by default (`--no-timing-review` to skip). |
| `--timing-review-model` | Gemini model id for timing review via Google AI Studio (default `gemini-3.1-flash-lite-preview`). |
| `--timing-review-chunk-seconds` | Max seconds of audio per timing-review request (default `120`). |
| `--reference_file` | Markdown (or other) reference for translation context and Whisper/multimodal prompt keywords (default `references/tripleS.md`). |
| `--initial_prompt` | Base prompt prepended to transcription (default mentions Korean and English). |
| `--temperature` | Whisper sampling temperature when using `--whisper` (default `0.0`). |
| `--time_buffer` | Seconds between subtitle cues after `adjust_subtitle_timing` on the **`--whisper`** path (default `0.1`). Default multimodal transcription skips that pass. |
| `--input_dir` / `--output_dir` | Override `input` and `output` directories. |
| `--font_name`, `--font_size`, `--outline_width`, `--box_background` | Subtitle burn-in styling (defaults: PingFang TC, 12, no outline, box background on). |
| `--margin_v` | Vertical margin in pixels from the frame edge (default `20`). |
| `--margin_h` | Horizontal margin in pixels for left/right padding (default `10`). |
| `--alignment` | Subtitle alignment using ASS numpad layout: `1`=bottom-left, `2`=bottom-center (default), `3`=bottom-right, `4`=mid-left, `5`=mid-center, `6`=mid-right, `7`=top-left, `8`=top-center, `9`=top-right. |
| `--stage-cooldown` | Seconds to wait between pipeline stages to avoid API 503 errors (default `60`; set to `0` to disable). |

### Output layout

Under `output/<video_basename>/`, filenames include a timestamp (`YYYYMMDD_HHMMSS`) to avoid overwrites:

*   **Full run:** `<name>_<timestamp>_subtitles.srt`, `<name>_<timestamp>_translated.mp4`, and if `--save-source-transcript` was set, `<name>_<timestamp>_transcript_source.srt`.
*   **`--transcribe-only`:** `<name>_<timestamp>_transcript.srt` only.


## Project Structure

The project is organized into a modular structure to separate concerns and improve maintainability.

```
/
├── .env                  # Stores API keys and other secrets.
├── .gitignore            # Specifies files for Git to ignore.
├── DESIGN.md             # The detailed technical design document for the project.
├── README.md             # This file.
├── requirements.txt      # Lists all Python dependencies for easy installation.
│
├── input/                # Directory for placing your source video files.
│
├── output/               # Directory where all generated files are saved.
│   └── {video_name}/     # Each video gets its own subdirectory.
│       ├── subtitles.srt
│       └── translated.mp4
│
├── references/           # Directory for context-aware translation files.
│   └── tripleS.md        # Example reference file containing names or terms.
│
├── tests/                # Unit tests (run with pytest; see “Running tests”).
│
└── src/                  # Contains all the core application logic.
    ├── main.py             # CLI entry point and pipeline orchestration.
    ├── video_processing.py # ffmpeg: audio extract, subtitle burn-in, filter checks.
    ├── transcription.py    # Whisper and Gemini multimodal transcription.
    ├── translation.py      # LLM translation via Gemini.
    ├── subtitles.py        # SRT generation, gaps, segment splits, timing buffers.
    ├── timing_review.py    # Default post-translation multimodal pass to refine cue times.
    └── utils.py            # Reference loading and Whisper keyword extraction.

```
