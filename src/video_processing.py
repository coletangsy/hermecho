"""
This module contains functions for video and audio processing.
"""
import os
import subprocess
import threading
from typing import List, Optional

from tqdm import tqdm


def _video_duration_seconds(video_path: str) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(out.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def extract_audio(video_path: str) -> Optional[str]:
    """
    Extracts audio from a video file and saves it as an MP3 file.

    Args:
        video_path: The path to the video file.

    Returns:
        The path to the extracted audio file, or None if an error occurs.
    """
    print("Extracting audio from video...")
    try:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return None

        # Generate audio path based on video filename to allow concurrent processing
        base_name = os.path.splitext(video_path)[0]
        audio_path = f"{base_name}.mp3"

        # Construct the ffmpeg command. -y overwrites the output file if it exists.
        command = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            "-y",
            audio_path
        ]

        try:
            # Execute the command
            subprocess.run(command, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except FileNotFoundError:
            print("Error: ffmpeg is not installed. Please install it to proceed.")
            print("On macOS, you can use Homebrew: brew install ffmpeg")
            return None
        except subprocess.CalledProcessError as e:
            # This prints the error from ffmpeg if it fails
            print(
                f"An error occurred while running ffmpeg: {e.stderr.decode()}")
            return None
        print(f"Audio extracted successfully to {audio_path}")
        return audio_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"An error occurred during audio extraction: {e}")
        return None


def burn_subtitles_into_video(
    video_path: str,
    srt_path: str,
    output_video_path: str,
    font_name: str = "Helvetica",
    font_size: int = 24,
    outline_width: int = 0,
    use_box_background: bool = False,
    margin_v: int = 20,
    margin_h: int = 10,
    alignment: int = 2,
):
    """
    Burns subtitles from an SRT file into a video.

    Args:
        video_path: Path to the original video.
        srt_path: Path to the SRT subtitle file.
        output_video_path: Path for the new video with subtitles.
        font_name: The font to use for subtitles.
        font_size: The font size for subtitles.
        outline_width: The width of the text outline (0 for no outline).
        use_box_background: Whether to use a black box background for subtitles.
        margin_v: Vertical margin in pixels (distance from the frame edge).
        margin_h: Horizontal margin in pixels applied to both left and right.
        alignment: ASS numpad alignment (1–9); 2 = bottom-center (default).
    """
    print(f"Burning subtitles into video: {output_video_path}")

    # ffmpeg command to burn subtitles.
    # The srt_path needs to be escaped for ffmpeg's filtergraph syntax,
    # especially for Windows paths.
    escaped_srt_path = srt_path.replace('\\', '/').replace(':', '\\:')

    # Construct the subtitles filter with style options
    # BorderStyle=3 is an opaque box. BorderStyle=1 is outline.
    # BackColour=&H80000000 sets the background to semi-transparent black.
    # In BorderStyle=3, 'Outline' controls the padding of the box.
    if use_box_background:
        style_options = (
            f"FontName={font_name},FontSize={font_size},"
            f"Outline=3,Shadow=0,BorderStyle=3,BackColour=&H80000000,"
            f"MarginV={margin_v},MarginL={margin_h},MarginR={margin_h},"
            f"Alignment={alignment}"
        )
    else:
        style_options = (
            f"FontName={font_name},FontSize={font_size},"
            f"Outline={outline_width},Shadow=0,BorderStyle=1,"
            f"MarginV={margin_v},MarginL={margin_h},MarginR={margin_h},"
            f"Alignment={alignment}"
        )

    subtitles_filter = f"subtitles={escaped_srt_path}:force_style='{style_options}'"

    duration = _video_duration_seconds(video_path)

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", subtitles_filter,
        "-c:v", "libx264",  # H.264 codec for wide compatibility
        "-pix_fmt", "yuv420p",  # Pixel format for compatibility
        "-c:a", "aac",      # AAC audio codec for wide compatibility
        "-strict", "experimental",
        "-progress", "pipe:1",
        "-nostats",
        output_video_path,
        "-y",  # Overwrite output file if it exists
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stderr_lines: List[str] = []

        def _drain_stderr() -> None:
            if process.stderr:
                for line in process.stderr:
                    stderr_lines.append(line)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        total_sec = int(duration) if duration else None
        with tqdm(total=total_sec, desc="Burning subtitles", unit="s", dynamic_ncols=True) as pbar:
            last_sec = 0
            if process.stdout:
                for line in process.stdout:
                    if line.startswith("out_time_us="):
                        try:
                            us = int(line.split("=", 1)[1].strip())
                            if us >= 0:
                                current_sec = us // 1_000_000
                                delta = current_sec - last_sec
                                if delta > 0:
                                    pbar.update(delta)
                                    last_sec = current_sec
                        except (ValueError, IndexError):
                            pass

        process.wait()
        stderr_thread.join(timeout=5.0)

        if process.returncode != 0:
            stderr_output = "".join(stderr_lines)
            print("An error occurred while running ffmpeg to burn subtitles:")
            print(f"Command: {' '.join(command)}")
            print(f"FFmpeg stderr: {stderr_output}")
        else:
            print("Successfully burned subtitles into the video.")

    except FileNotFoundError:
        print("Error: ffmpeg is not installed. Please install it to proceed.")
        print("On macOS, you can use Homebrew: brew install ffmpeg")
    except Exception as e:
        print(f"An unexpected error occurred during subtitle burning: {e}")


def is_ffmpeg_installed() -> bool:
    """
    Checks if ffmpeg is installed and available in the system's PATH.

    Returns:
        True if ffmpeg is installed, False otherwise.
    """
    try:
        subprocess.run(["ffmpeg", "-version"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
