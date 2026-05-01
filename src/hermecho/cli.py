"""Command-line interface for Hermecho."""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from dotenv import load_dotenv

from .pipeline import PipelineConfig, process_video
from .video_processing import is_ffmpeg_installed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse Hermecho command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe a video with local Whisper, translate with Gemini, "
            "then burn subtitles. Omit --transcribe-only for the full "
            "translate + burn pipeline."
        ),
    )
    parser.add_argument("video_filename", help="The filename of the video in the input directory.")
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Stop after source-language subtitles: write SRT only, skip translation and burn-in.",
    )
    parser.add_argument(
        "--srt-only",
        action="store_true",
        help="Run transcription and translation, then stop after writing the SRT file; skip burning subtitles into the video.",
    )
    parser.add_argument(
        "--save-source-transcript",
        action="store_true",
        help="With the full pipeline, also write a source-language SRT before translation.",
    )
    parser.add_argument("--model", default="large", help="The Whisper model for transcription.")
    parser.add_argument("--language", default="ko", help="The language of the audio for transcription.")
    parser.add_argument(
        "--target_language",
        default="Traditional Chinese (Taiwan)",
        help="The target language for translation.",
    )
    parser.add_argument(
        "--translation_model",
        default="gemini-3.1-flash-lite-preview",
        help="Gemini model id for translation via Google AI Studio.",
    )
    parser.add_argument("--time_buffer", type=float, default=0.1, help="Buffer time between subtitles in seconds.")
    parser.add_argument("--input_dir", default="input", help="The directory where the input video is located.")
    parser.add_argument("--output_dir", default="output", help="The directory where the output files will be saved.")
    parser.add_argument("--reference_file", default="references/tripleS.md", help="Optional path to a reference file.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Whisper sampling temperature.")
    parser.add_argument("--font_name", default="PingFang TC", help="Font name for subtitles.")
    parser.add_argument("--font_size", type=int, default=12, help="Font size for subtitles.")
    parser.add_argument("--outline_width", type=int, default=0, help="Subtitle outline width.")
    parser.add_argument(
        "--box_background",
        action="store_true",
        default=True,
        help="Use a black box background for subtitles instead of an outline.",
    )
    parser.add_argument("--margin_v", type=int, default=20, help="Vertical margin for subtitles in pixels.")
    parser.add_argument("--margin_h", type=int, default=10, help="Horizontal margin for subtitles in pixels.")
    parser.add_argument(
        "--alignment",
        type=int,
        default=2,
        choices=list(range(1, 10)),
        help="Subtitle alignment using ASS numpad layout.",
    )
    parser.add_argument(
        "--stage-cooldown",
        type=int,
        default=60,
        help="Seconds to wait between pipeline stages to avoid API 503 errors.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Convert parsed CLI arguments into a pipeline config."""
    return PipelineConfig(**vars(args))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint."""
    load_dotenv()
    if not is_ffmpeg_installed():
        print("Error: ffmpeg is not installed. Please install it to proceed.")
        print("On macOS, you can use Homebrew: brew install ffmpeg")
        return

    process_video(config_from_args(parse_args(argv)))


if __name__ == "__main__":
    main()

