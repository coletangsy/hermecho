"""
This module contains functions for video and audio processing.
"""
import os
import subprocess
from typing import Optional


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


def burn_subtitles_into_video(video_path: str, srt_path: str, output_video_path: str, font_name: str = "Helvetica", font_size: int = 24, outline_width: int = 0, use_box_background: bool = False):
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
        # Box style: Opaque box (3), Padding (3), Semi-transparent black background
        style_options = f"FontName={font_name},FontSize={font_size},Outline=3,Shadow=0,BorderStyle=3,BackColour=&H80000000"
    else:
        # Text only style: Outline defined by user, No shadow, Standard border (1)
        style_options = f"FontName={font_name},FontSize={font_size},Outline={outline_width},Shadow=0,BorderStyle=1"

    subtitles_filter = f"subtitles={escaped_srt_path}:force_style='{style_options}'"

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", subtitles_filter,
        "-c:v", "libx264",  # H.264 codec for wide compatibility
        "-pix_fmt", "yuv420p",  # Pixel format for compatibility
        "-c:a", "aac",      # AAC audio codec for wide compatibility
        "-strict", "experimental",
        output_video_path,
        "-y"  # Overwrite output file if it exists
    ]

    try:
        subprocess.run(command, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print("Successfully burned subtitles into the video.")
    except FileNotFoundError:
        print("Error: ffmpeg is not installed. Please install it to proceed.")
        print("On macOS, you can use Homebrew: brew install ffmpeg")
    except subprocess.CalledProcessError as e:
        print("An error occurred while running ffmpeg to burn subtitles:")
        print(f"Command: {' '.join(command)}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
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
