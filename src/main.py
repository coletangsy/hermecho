"""
This script provides a command-line interface to translate a video from one language to another.
"""
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

from video_processing import extract_audio, burn_subtitles_into_video, is_ffmpeg_installed
from transcription import transcribe_audio
from translation import translate_segments
from subtitles import fill_transcription_gaps, adjust_subtitle_timing, generate_srt, split_long_segments
from utils import load_reference_material, _print_segments, extract_keywords_for_whisper

load_dotenv()


def _parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the video translation script.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Translate a Korean video with subtitles.")
    parser.add_argument(
        "video_filename", help="The filename of the video in the input directory.")
    parser.add_argument("--model", default="large",
                        help="The Whisper model for transcription (e.g., 'tiny', 'base', 'small', 'medium', 'large').")
    parser.add_argument("--language", default="ko",
                        help="The language of the audio for transcription.")
    parser.add_argument("--target_language", default="Traditional Chinese (Taiwan)",
                        help="The target language for translation.")
    parser.add_argument("--translation_model",
                        default="google/gemini-2.5-pro", help="The model for translation.")
    parser.add_argument("--time_buffer", type=float, default=0.1,
                        help="Buffer time between subtitles in seconds.")
    parser.add_argument("--input_dir", default="input",
                        help="The directory where the input video is located.")
    parser.add_argument("--output_dir", default="output",
                        help="The directory where the output files will be saved.")
    parser.add_argument("--reference_file",  default="references/tripleS.md",
                        help="Optional path to a reference file for translation context.")
    parser.add_argument("--initial_prompt", default="This is a conversation in Korean and English.",
                        help="Initial prompt to guide Whisper (e.g., 'This is a Korean video with some English').")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Whisper sampling temperature (0.0 is deterministic, higher is more creative).")
    parser.add_argument("--font_name", default="PingFang TC",
                        help="Font name for subtitles (default: PingFang TC).")
    parser.add_argument("--font_size", type=int, default=12,
                        help="Font size for subtitles (default: 12).")
    parser.add_argument("--outline_width", type=int, default=0,
                        help="Subtitle outline width (0 for no outline).")
    parser.add_argument("--box_background", action="store_true", default=True,
                        help="Use a black box background for subtitles instead of an outline (default: True).")
    return parser.parse_args()


def _process_video(args: argparse.Namespace):
    """
    Orchestrates the entire video processing workflow.

    This function takes the parsed command-line arguments and manages the video processing pipeline, including:
    1. Audio extraction.
    2. Audio transcription.
    3. Text translation.
    4. Subtitle generation.
    5. Burning subtitles into the video.

    It ensures that temporary files are cleaned up after the process.

    Args:
        args: An argparse.Namespace object with all the required script arguments.
    """
    video_path = os.path.abspath(os.path.join(
        args.input_dir, args.video_filename))
    audio_path = extract_audio(video_path)
    if not audio_path:
        return

    try:
        # Generate context-aware prompt for Whisper
        keywords = extract_keywords_for_whisper(args.reference_file)
        full_prompt = args.initial_prompt
        if keywords:
            full_prompt = f"{full_prompt} Context: {keywords}"
            print(f"Generated Whisper Prompt: {full_prompt}")

        transcribed_segments = transcribe_audio(
            audio_path, 
            model=args.model, 
            language=args.language,
            initial_prompt=full_prompt,
            temperature=args.temperature
        )
        if not transcribed_segments:
            return

        _print_segments(f"Original Transcription ({args.language})", transcribed_segments)

        # Guardrail 0: Split long segments
        transcribed_segments = split_long_segments(transcribed_segments)
        _print_segments("Transcription after Splitting", transcribed_segments)

        # Guardrail 1: Fill any significant gaps in the transcription
        transcribed_segments = fill_transcription_gaps(transcribed_segments)
        _print_segments("Transcription after Gap-Filling", transcribed_segments)

        # Load reference material if a path is provided
        reference_material = load_reference_material(args.reference_file)

        translated_segments = translate_segments(
            transcribed_segments,
            target_language=args.target_language,
            translation_model=args.translation_model,
            reference_material=reference_material
        )

        if translated_segments:
            _print_segments(
                f"Translation ({args.target_language})", translated_segments)

            adjusted_segments = adjust_subtitle_timing(
                translated_segments, args.time_buffer)
            _print_segments(f"Adjusted Subtitles", adjusted_segments)

            video_name = os.path.splitext(args.video_filename)[0]
            output_dir = os.path.join(args.output_dir, video_name)
            os.makedirs(output_dir, exist_ok=True)

            # Generate timestamp for versioning
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            srt_path = os.path.join(output_dir, f"{video_name}_{timestamp}_subtitles.srt")
            generate_srt(adjusted_segments, srt_path)

            # Burn subtitles into a new video file
            output_video_path = os.path.join(
                output_dir, f"{video_name}_{timestamp}_translated.mp4")
            burn_subtitles_into_video(
                video_path,
                os.path.abspath(srt_path),
                os.path.abspath(output_video_path),
                font_name=args.font_name,
                font_size=args.font_size,
                outline_width=args.outline_width,
                use_box_background=args.box_background
            )

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def main():
    """
    The main entry point of the script.

    It checks for dependencies, parses arguments, and starts the video processing.
    """
    if not is_ffmpeg_installed():
        print("Error: ffmpeg is not installed. Please install it to proceed.")
        print("On macOS, you can use Homebrew: brew install ffmpeg")
        return

    args = _parse_arguments()
    _process_video(args)


if __name__ == "__main__":
    main()
