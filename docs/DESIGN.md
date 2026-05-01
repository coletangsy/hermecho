# Hermecho Design

Hermecho is a command-line video translation pipeline. It transcribes source audio with local Whisper, translates subtitle segments with Gemini, writes SRT files, and can hard-burn the translated subtitles into a new MP4.

## Pipeline

1. `hermecho.video_processing.extract_audio` extracts an MP3 track from the input video with `ffmpeg`.
2. `hermecho.transcription.transcribe_audio` runs local Whisper with word timestamps.
3. `hermecho.subtitles.split_long_segments` and `fill_transcription_gaps` prepare readable source subtitle segments.
4. `hermecho.translation.translate_segments` translates segments with Gemini. Prompt construction lives in `hermecho.prompts`.
5. `hermecho.subtitles.adjust_subtitle_timing` applies the configured gap buffer, then `generate_srt` writes the translated SRT.
6. `hermecho.video_processing.burn_subtitles_into_video` optionally burns subtitles into an MP4 using the `ffmpeg` subtitles filter.

There is no Gemini multimodal transcription stage, transcription prompt injection, keyword extraction stage, or timing-review stage in the current pipeline.

## Package Layout

```text
src/
├── main.py                  # Compatibility wrapper for python src/main.py
└── hermecho/
    ├── cli.py               # argparse, console entrypoint, config mapping
    ├── pipeline.py          # PipelineConfig and process_video orchestration
    ├── transcription.py     # Local Whisper transcription
    ├── translation.py       # Gemini translation and chunk fallback strategy
    ├── prompts.py           # Translation prompt construction
    ├── subtitles.py         # Segment splitting, gap filling, timing, SRT
    ├── video_processing.py  # ffmpeg audio extraction and subtitle burn-in
    ├── gemini_sdk.py        # Lazy Google GenAI SDK import
    ├── retry.py             # Backoff helpers
    └── utils.py             # Reference loading and segment printing
```

`pyproject.toml` is the primary project metadata source. `requirements.txt` is kept as a compatibility installer that points to the editable package.

## Public Interfaces

Supported command entrypoints:

```bash
hermecho clip.mp4
python src/main.py clip.mp4
PYTHONPATH=src python -m hermecho.cli clip.mp4
```

The internal orchestration API is:

```python
from hermecho.pipeline import PipelineConfig, process_video

process_video(PipelineConfig(video_filename="clip.mp4"))
```
