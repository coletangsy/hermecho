"""End-to-end video translation pipeline orchestration."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from tqdm import trange

from .progress import emit_progress
from .subtitles import (
    adjust_subtitle_timing,
    fill_transcription_gaps,
    generate_srt,
    limit_portrait_subtitle_lines,
    split_long_segments,
)
from .transcription import transcribe_audio
from .translation import translate_segments
from .utils import _print_segments, load_locked_terms, load_reference_material
from .video_processing import burn_subtitles_into_video, extract_audio, is_portrait_video


@dataclass
class PipelineConfig:
    video_filename: str
    transcribe_only: bool = False
    srt_only: bool = False
    save_source_transcript: bool = False
    model: str = "large"
    language: Optional[str] = None
    target_language: str = "Traditional Chinese (Taiwan)"
    translation_model: str = "deepseek/deepseek-v4-pro"
    time_buffer: float = 0.1
    input_dir: str = "input"
    output_dir: str = "output"
    reference_file: str = "references/tripleS.md"
    locked_terms_file: str = "references/locked_terms.json"
    temperature: float = 0.0
    font_name: str = "Heiti TC"
    fonts_dir: Optional[str] = None
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
    emit_progress("audio_extraction", "running", "Extracting audio")
    audio_path = extract_audio(video_path)
    if not audio_path:
        emit_progress("audio_extraction", "error", "Audio extraction failed")
        return
    emit_progress("audio_extraction", "complete", "Audio extracted", detail=audio_path)

    try:
        next_stage("Transcribing Audio")
        emit_progress("transcription", "running", "Transcribing audio")
        transcribed_segments = transcribe_audio(
            audio_path,
            model=config.model,
            language=config.language,
            temperature=config.temperature,
        )
        if not transcribed_segments:
            emit_progress("transcription", "error", "Audio transcription failed")
            return
        emit_progress(
            "transcription",
            "complete",
            "Audio transcribed",
            current=len(transcribed_segments),
            total=len(transcribed_segments),
            pct=100,
        )

        _print_segments(
            f"Original Transcription ({config.language or 'auto'})",
            transcribed_segments,
        )

        transcribed_segments = split_long_segments(transcribed_segments)
        _print_segments("Transcription after Splitting", transcribed_segments)

        transcribed_segments = fill_transcription_gaps(transcribed_segments)
        silence_boundaries = [
            float(segment["start"])
            for segment in transcribed_segments
            if segment.get("text", "").strip() == "[no speech]"
        ]
        transcribed_segments = [
            segment
            for segment in transcribed_segments
            if segment.get("text", "").strip() != "[no speech]"
        ]
        _print_segments("Transcription after Gap-Filling", transcribed_segments)

        video_name = os.path.splitext(config.video_filename)[0]
        output_dir = os.path.join(config.output_dir, video_name)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if config.transcribe_only:
            next_stage("Writing Transcript SRT")
            srt_path = os.path.join(output_dir, f"{video_name}_{timestamp}_transcript.srt")
            emit_progress("source_srt_write", "running", "Writing transcript SRT")
            generate_srt(transcribed_segments, srt_path)
            emit_progress(
                "source_srt_write",
                "complete",
                "Transcript SRT written",
                detail=srt_path,
                pct=100,
            )
            print("Transcribe-only mode: done (no translation or burn-in).")
            emit_progress("completion", "complete", "Hermecho pipeline completed", pct=100)
            return

        is_portrait = is_portrait_video(video_path)
        reference_material = load_reference_material(config.reference_file)
        locked_terms = load_locked_terms(config.locked_terms_file)
        if locked_terms is None:
            emit_progress(
                "translation_gate",
                "error",
                "Locked Terms configuration is invalid",
            )
            return

        if config.save_source_transcript:
            source_srt = os.path.join(
                output_dir,
                f"{video_name}_{timestamp}_transcript_source.srt",
            )
            emit_progress("source_srt_write", "running", "Writing source transcript SRT")
            generate_srt(transcribed_segments, source_srt)
            emit_progress(
                "source_srt_write",
                "complete",
                "Source transcript SRT written",
                detail=source_srt,
                pct=100,
            )

        next_stage(f"Translating to {config.target_language}")
        emit_progress(
            "translation",
            "running",
            f"Translating to {config.target_language}",
        )
        translated_segments = translate_segments(
            transcribed_segments,
            target_language=config.target_language,
            translation_model=config.translation_model,
            reference_material=reference_material,
            locked_terms=locked_terms,
            preserve_punctuation=True,
        )

        if translated_segments is not None:
            emit_progress(
                "translation",
                "complete",
                "Translation completed",
                current=len(translated_segments),
                total=len(transcribed_segments),
                pct=100,
            )
            translation_label = f"Translation ({config.target_language})"
            _print_segments(translation_label, translated_segments)

            emit_progress(
                "subtitle_timing_adjustment",
                "running",
                "Adjusting subtitle timing",
            )
            final_subtitle_segments = adjust_subtitle_timing(
                translated_segments,
                config.time_buffer,
                silence_boundaries=silence_boundaries,
            )
            if is_portrait:
                final_subtitle_segments = limit_portrait_subtitle_lines(
                    final_subtitle_segments
                )
            emit_progress(
                "subtitle_timing_adjustment",
                "complete",
                "Subtitle timing adjusted",
                current=len(final_subtitle_segments),
                total=len(translated_segments),
                pct=100,
            )
            _print_segments("Adjusted Subtitles", final_subtitle_segments)

            next_stage("Writing Subtitle SRT")
            srt_path = os.path.join(output_dir, f"{video_name}_{timestamp}_subtitles.srt")
            emit_progress("translated_srt_write", "running", "Writing translated SRT")
            generate_srt(final_subtitle_segments, srt_path)
            emit_progress(
                "translated_srt_write",
                "complete",
                "Translated SRT written",
                detail=srt_path,
                pct=100,
            )

            if config.srt_only:
                print("SRT-only mode: subtitle file written, skipping video burn-in.")
                emit_progress("completion", "complete", "Hermecho pipeline completed", pct=100)
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
                    fonts_dir=config.fonts_dir,
                    font_size=config.font_size,
                    outline_width=config.outline_width,
                    use_box_background=config.box_background,
                    margin_v=config.margin_v,
                    margin_h=config.margin_h,
                    alignment=config.alignment,
                )
                emit_progress("completion", "complete", "Hermecho pipeline completed", pct=100)
        else:
            print("Translation Gate blocked final SRT/video delivery.")
            emit_progress(
                "translation",
                "error",
                "Translation Gate blocked final SRT/video delivery",
            )

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
