"""End-to-end video translation pipeline orchestration."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime

from tqdm import trange

from .subtitles import (
    adjust_subtitle_timing,
    fill_transcription_gaps,
    generate_srt,
    split_long_segments,
)
from .transcription import transcribe_audio
from .translation import translate_segments
from .utils import _print_segments, load_reference_material
from .video_processing import burn_subtitles_into_video, extract_audio


@dataclass
class PipelineConfig:
    video_filename: str
    transcribe_only: bool = False
    srt_only: bool = False
    save_source_transcript: bool = False
    model: str = "large"
    language: str = "ko"
    target_language: str = "Traditional Chinese (Taiwan)"
    translation_model: str = "gemini-3.1-flash-lite-preview"
    time_buffer: float = 0.1
    input_dir: str = "input"
    output_dir: str = "output"
    reference_file: str = "references/tripleS.md"
    temperature: float = 0.0
    font_name: str = "PingFang TC"
    font_size: int = 12
    outline_width: int = 0
    box_background: bool = True
    margin_v: int = 20
    margin_h: int = 10
    alignment: int = 2
    stage_cooldown: int = 60


def _stage_banner(current: int, total: int, label: str) -> None:
    width = 60
    header = f"  Stage {current}/{total} ▸ {label}  "
    pad = max(0, width - len(header))
    print(f"\n{'━' * width}")
    print(f"{header}{' ' * pad}")
    print(f"{'━' * width}")


def _stage_cooldown(seconds: int) -> None:
    if seconds <= 0:
        return
    for _ in trange(seconds, desc="  API cooldown", unit="s", leave=False, ncols=60):
        time.sleep(1)


def process_video(config: PipelineConfig) -> None:
    """Run the configured Hermecho video translation pipeline."""
    total_stages = 3 if config.transcribe_only else 4
    if not config.transcribe_only and not config.srt_only:
        total_stages += 1

    stage = 0

    def next_stage(label: str) -> None:
        nonlocal stage
        if stage > 0:
            _stage_cooldown(config.stage_cooldown)
        stage += 1
        _stage_banner(stage, total_stages, label)

    next_stage("Extracting Audio")
    video_path = os.path.abspath(os.path.join(config.input_dir, config.video_filename))
    audio_path = extract_audio(video_path)
    if not audio_path:
        return

    try:
        next_stage("Transcribing Audio")
        transcribed_segments = transcribe_audio(
            audio_path,
            model=config.model,
            language=config.language,
            temperature=config.temperature,
        )
        if not transcribed_segments:
            return

        _print_segments(f"Original Transcription ({config.language})", transcribed_segments)

        transcribed_segments = split_long_segments(transcribed_segments)
        _print_segments("Transcription after Splitting", transcribed_segments)

        transcribed_segments = fill_transcription_gaps(transcribed_segments)
        _print_segments("Transcription after Gap-Filling", transcribed_segments)

        video_name = os.path.splitext(config.video_filename)[0]
        output_dir = os.path.join(config.output_dir, video_name)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if config.transcribe_only:
            next_stage("Writing Transcript SRT")
            srt_path = os.path.join(output_dir, f"{video_name}_{timestamp}_transcript.srt")
            generate_srt(transcribed_segments, srt_path)
            print("Transcribe-only mode: done (no translation or burn-in).")
            return

        reference_material = load_reference_material(config.reference_file)

        if config.save_source_transcript:
            source_srt = os.path.join(
                output_dir,
                f"{video_name}_{timestamp}_transcript_source.srt",
            )
            generate_srt(transcribed_segments, source_srt)

        next_stage(f"Translating to {config.target_language}")
        translated_segments = translate_segments(
            transcribed_segments,
            target_language=config.target_language,
            translation_model=config.translation_model,
            reference_material=reference_material,
        )

        if translated_segments:
            translation_label = f"Translation ({config.target_language})"
            _print_segments(translation_label, translated_segments)

            final_subtitle_segments = adjust_subtitle_timing(
                translated_segments,
                config.time_buffer,
            )
            _print_segments("Adjusted Subtitles", final_subtitle_segments)

            next_stage("Writing Subtitle SRT")
            srt_path = os.path.join(output_dir, f"{video_name}_{timestamp}_subtitles.srt")
            generate_srt(final_subtitle_segments, srt_path)

            if config.srt_only:
                print("SRT-only mode: subtitle file written, skipping video burn-in.")
            else:
                next_stage("Burning Subtitles into Video")
                output_video_path = os.path.join(
                    output_dir,
                    f"{video_name}_{timestamp}_translated.mp4",
                )
                burn_subtitles_into_video(
                    video_path,
                    os.path.abspath(srt_path),
                    os.path.abspath(output_video_path),
                    font_name=config.font_name,
                    font_size=config.font_size,
                    outline_width=config.outline_width,
                    use_box_background=config.box_background,
                    margin_v=config.margin_v,
                    margin_h=config.margin_h,
                    alignment=config.alignment,
                )

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

