# Hermecho

Hermecho translates videos with Korean audio into Traditional Chinese (Taiwan) subtitles. It uses local Whisper or explicitly selected OpenRouter ASR for transcription, OpenRouter for translation, writes timestamped SRT files, and can hard-burn subtitles into a translated MP4.

## Features

- Local Whisper transcription in the normal pipeline with no transcription API usage.
- Optional OpenRouter word-timestamp transcription with resumable audio chunks.
- OpenRouter translation with reference-file context for names and terms.
- Translation Gate rejects incomplete model responses and enforces JSON Locked Terms.
- Deterministic Source Sentence grouping with complete same-time translations.
- SRT-only, transcribe-only, and full burn-in modes.
- Subtitle styling controls for font, size, background box, margins, and ASS alignment.
- `ffmpeg` subtitle-filter detection before burn-in.
- Source and translated SRT preserve every cue and identical millisecond timing, including zero-duration cues.
- Source Boundary Review, Alignment, and Fit Repair are absent from production generation. LLM requests are limited to translation and its necessary response retries.
- Every translated run writes a versioned `*_subtitle_bundle.json` with explicit source references, grouping/timing policy, source fingerprint, diagnostics, and render omissions.
- Rendering uses a separate visible-interval SRT. Zero/reversed/outside-video cues are omitted with reasons; overlaps and long translations remain. Saved SRT is never shortened or retimed.
- `--source-srt captions.srt` translates explicitly imported source subtitles without ASR or invented Source Words.

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

MLX model weights download on first use, then reuse the local Hugging Face
snapshot on later runs. `auto` keeps portable Whisper unless
local Comparison Run evidence records a faster MLX median and explicit Human
Approval with no Candidate-only regression.

Create that evidence with the fixed ten-minute review range:

```bash
python -m hermecho.asr_comparison input/20251231_w-yGSP1c3bg.mp4 --language ko
```

It writes manifests, timings, a source-transcript diff, a shared-audio Review
Composite, and `output/asr-comparison/review.md`. A reviewer must complete its
checklist and mark the decision `approved` before `auto` can select MLX.

For a separate OpenRouter ASR Evaluation, run:

```bash
python -m hermecho.openrouter_asr_evaluation input/Wsp9Z6-S0LA.mp4 \
  --output-dir output/openrouter-asr-evaluation-20260925 --max-cost-usd 10
```

This tool compares local Whisper `large` with `google/gemini-3.5-transcribe`
and `microsoft/mai-transcribe-2`. It fixes Korean (`ko`) and uses no vocabulary
prompt. Each OpenRouter model first receives a 30-second probe; a missing or
invalid word-timestamp response disqualifies it from the full-video run. Eligible
models receive the same 60-second audio ranges with one second of overlap.
The tool reconciles adjacent overlap word sequences and timing before joining
the transcripts, preserving one original copy of each shared word. Conflicting
overlap evidence blocks that candidate and remains available for review.
Elapsed time sums the full-video API requests for remote models and local
transcription for Whisper; audio extraction and the remote probes are separate.

The output directory must be empty. It contains `manifest.json`,
`comparison.json`, per-model responses, `review.md`, and five audio clips for
human listening. The report lists zero-duration and over-three-second word
timestamps for review. `progress.json` records completed calls if a run stops early.
The budget check stops new requests when reported cumulative OpenRouter cost
reaches `--max-cost-usd`; a single request can take the total past that value,
and an interrupted or malformed response may have unreported cost. There is no
automatic retry. The prior source SRT and local Whisper output are comparison
material, not a manually verified answer key, so the tool does not report a
word error rate or approve a model. This evaluation does not change the normal
Hermecho transcription backend or MLX approval evidence.

`requirements.txt` is kept for compatibility and installs the editable package:

```bash
python -m pip install -r requirements.txt
```

Create `.env` in the repo root:

```text
OPENROUTER_API_KEY="your_openrouter_key"
```

Translation uses OpenRouter's OpenAI-compatible API. The default model is
`deepseek/deepseek-v4.1-flash`, and requests prefer Alibaba first, then
AtlasCloud FP8, with provider fallback enabled.

Check `ffmpeg` subtitle support:

```bash
ffmpeg -hide_banner -filters | rg subtitles
```

The standard Homebrew `ffmpeg` formula does not include libass. To replace it
with the libass-enabled `homebrew-ffmpeg` build, run:

```bash
brew uninstall ffmpeg
brew trust homebrew-ffmpeg/ffmpeg
brew tap homebrew-ffmpeg/ffmpeg
brew install homebrew-ffmpeg/ffmpeg/ffmpeg
```

This trusts a third-party tap and replaces Homebrew's core `ffmpeg`. It is
required for hard-burned subtitles and Review Composites.

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

For agent-operated jobs, use [run-hermecho-job](.agents/skills/run-hermecho-job/SKILL.md).
Monitoring an existing session does not launch another process. Transcribe-only
runs using a local backend do not need translation credentials or Locked Terms; translated and MLX
comparison runs validate both before starting. Verify the expected artifacts
and gate reports even when the CLI exits with status zero.

The full pipeline is:

```text
extract audio -> selected transcription backend -> Source Sentences -> OpenRouter Translation Gate -> Delivery Gate -> SRT -> optional MP4 burn-in
```

Sentence-first delivery is the supported translated-subtitle path after the approved Phase 3 review:

```bash
hermecho clip.mp4
```

Select OpenRouter transcription explicitly for source SRT or translated output:

```bash
hermecho clip.mp4 --transcription-backend openrouter --transcribe-only
hermecho clip.mp4 --transcription-backend openrouter \
  --transcription-model microsoft/mai-transcribe-2 --language ko --srt-only
```

`--transcription-backend openrouter` authorizes uploading audio and requires
`OPENROUTER_API_KEY`. The default remote model is `microsoft/mai-transcribe-2`;
`--model` still selects the local Whisper model used for fallback. `auto` stays
local even when an API key exists. Omit `--language` for remote auto-detection.
MAI uses verbatim transcription without diarization or keyword biasing, following
the [OpenRouter speech-to-text API](https://openrouter.ai/docs/guides/overview/multimodal/stt).
Other model slugs must return complete word timestamps to work in this pipeline.

Remote audio is converted to mono 16 kHz MP3 and sent in 60-second core ranges
with one second of context on each side. Adjacent overlapping word sequences
must agree in content and order, with timestamp drift of at most 0.25 seconds.
The earlier chunk's original words and timestamps are retained once, so drift
across a core boundary cannot duplicate or drop a matched word. If overlap
evidence conflicts, the production pipeline requests a 20-second audio window
around that boundary using the same remote model. Three consecutive words with
matching text and timestamps on each side must anchor this new evidence outside
the disputed overlap. Only the span between those anchors is replaced, keeping
its original returned word timestamps. Missing anchors, invalid timing, or a
failed repair request block transcription without local fallback. Repairs incur
an additional transcription request per conflicting boundary and are cached
alongside the original chunks; retries reuse both. ASR Evaluation retains its
strict comparison policy and does not request repairs. Chunk and repair progress
is emitted through `HERMECHO_PROGRESS`. Timestamps are offset back to the full
audio. Completed chunks, boundary evidence, and
reported cost, provider, model, generation ID, and request latency are retained
in `output/<video>/.openrouter-transcription.json`; unavailable metadata stays
unknown. The checkpoint records raw acquisition and assembly fingerprints
separately: assembly-only changes reuse matching raw chunks while rebuilding
the joined transcript under the current policy. Repeating the same command
resumes matching chunks. Audio, model,
language, or temperature changes invalidate them; `--force` recomputes them.

Network failures, timeouts, HTTP 408/429, and server errors trigger local Whisper
transcription of the entire audio. Successful remote chunks remain available for
a later retry, but the delivered Source Transcript contains only Whisper words.
The local result is cached with the local backend fingerprint; another explicit
OpenRouter run retries the remote path. Missing credentials, invalid parameters,
authentication errors, malformed responses, and missing word timing block
transcription without fallback. Invalid word-timestamp responses receive
up to two additional requests with the same backend and model; these bounded
requests may incur charges. Each rejected response is retained only as request
metadata plus validation diagnostics in the checkpoint; a successful response
is retained as validated Source Words. Transport and other request failures do
not receive automatic paid retries. When a chunk at least 40 seconds long
still fails after those retries, the remote path makes one bounded recovery
attempt using two overlapping windows around the midpoint (10 seconds of
context on each side). Each child window is validated with up to the same two
additional requests and cannot start another recovery tree. The two child
responses must splice through unique three-word/time anchors and pass absolute
word validation; otherwise the run fails closed without a local fallback.
Original and child request metadata, charges, validation diagnostics, and
subwindow provenance are retained once in the chunk checkpoint. A recovered
chunk is reused on resume after its words validate, so a retry
does not issue another request.

Local MLX transcription applies the same evidence boundary when adjacent
words overlap: it uses native `clip_timestamps` to re-transcribe a 24-second
window starting ten seconds before the disputed boundary (rounded down to a
whole second), with the same MLX model and the original audio time origin.
It then splices only at unique three-word/time
anchors. It bounds a run to eight original overlap repairs, records the
original segments and fresh windows in `.mlx-timing-repairs.json`, and resets
downstream grouping and translation after an accepted repair. It never clamps,
sorts, or dictionary-deduplicates source words; missing or ambiguous anchors
block delivery. Nonempty original or fresh MLX segments without Source Word
timestamps also block recovery rather than dropping their text.

For translated runs, `--locked-terms-file` is required and defaults to
`references/locked_terms.json`. It is a machine-readable JSON source-to-target
mapping enforced by the Translation Gate; a missing or invalid mapping blocks
translation and final SRT/MP4 delivery. `--reference_file` remains separate
Markdown prompt context. The Translation Gate accepts nonempty translations
even when they omit source sentence terminal punctuation. Every accepted translation
is attached to its original Source Sentence without splitting, merging, wrapping,
or timing adjustment. Quality findings are warnings. Missing translations or
unusable Source Word evidence remain operation failures.

`--source-srt` keeps the imported segmentation and timing; imported cues do not
claim Source Word evidence. `--transcribe-only` retains its separate transcript
workflow. The generation grouping policy is `source-sentence-rules-v3`; old
reviewed grouping is not reused under this policy, but compatible ASR is reused.
Translation checkpoints depend on the source grouping and translation inputs,
so a pure render/style change reuses accepted translations.

## Options

Run `hermecho --help` for the full list.

| Option | Purpose |
| --- | --- |
| `video_filename` | File name inside `--input_dir`. |
| `--model` | Whisper model size, default `large`. |
| `--transcription-backend` | `auto`, `whisper`, Apple-Silicon-only `mlx`, or explicit `openrouter`. `auto` selects MLX only with approved faster local comparison evidence; otherwise it uses Whisper. MLX supports `large` / `large-v3` as large-v3. |
| `--transcription-model` | OpenRouter ASR model slug, default `microsoft/mai-transcribe-2`; only used with the `openrouter` backend. |
| `--language` | Source audio language, auto-detected by default. |
| `--target_language` | Translation target, default `Traditional Chinese (Taiwan)`. |
| `--translation_model` | OpenRouter model slug, default `deepseek/deepseek-v4.1-flash`. |
| `--reference_file` | Translation reference material, default `references/tripleS.md`. |
| `--locked-terms-file` | Required JSON source-to-target mapping for translated runs; defaults to `references/locked_terms.json`. Missing or invalid mappings block translation and final SRT/MP4 delivery. |
| `--temperature` | Transcription sampling temperature, default `0.0`. |
| `--transcribe-only` | Write source-language SRT and stop. |
| `--source-srt` | Translate a source SRT without ASR, preserving all cues and times. |
| `--srt-only` | Write translated SRT and skip video burn-in. |
| `--save-source-transcript` | Also write source-language SRT during a translated run. |
| `--font_name`, `--font_size`, `--outline_width`, `--box_background` | Burn-in subtitle styling. Font defaults to `Heiti TC`. |
| `--fonts-dir` | Font directory for FFmpeg; defaults to the macOS MobileAsset font directory. |
| `--margin_v`, `--margin_h`, `--alignment` | Burn-in subtitle placement. |
| `--stage-cooldown` | Delay between stages, default `60`; use `0` to disable. |
| `--force` | Recompute all stages instead of reusing completed checkpoints. |

Outputs are written under `output/<video_basename>/` with a `YYYYMMDD_HHMMSS` timestamp. Each video also keeps one versioned, atomic `.hermecho-checkpoint.json`: matching completed transcription, accepted Source Sentence grouping, and Translation-Gate-approved chunks resume automatically; changing the grouping fingerprint invalidates downstream translation chunks; `--force` bypasses it. MLX transcription skips segments with non-finite or reversed segment or word timestamps before checkpointing and reports the exclusions as a warning. Translated runs also write a matching `*_subtitle_bundle.json` and `*_delivery_gate.txt` report. Bundles separate complete saved cues from render omissions and record source fingerprints and policy versions. The `source_timing_diagnostics` field preserves suspicious Source Word timing as warnings without rewriting words or timestamps; the delivery report includes those warnings.

## Hermecho Cloud rollout

Before deploying Hermecho Cloud changes that accept portrait jobs, install the compatible Hermecho release on the processor Mac. The pipeline preserves source timing in saved SRT; orientation-specific quality diagnostics and render plans do not rewrite saved captions.

## Development

Project metadata lives in `pyproject.toml`. Runtime code is packaged under `src/hermecho/`; `src/main.py` is only a compatibility wrapper.

```text
src/
├── main.py
└── hermecho/
    ├── cli.py
    ├── pipeline.py
    ├── sentence_first.py
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
