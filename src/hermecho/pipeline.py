"""End-to-end video translation pipeline orchestration."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4
from typing import Optional

from tqdm import trange

from .checkpoints import CheckpointStore, fingerprint_data, fingerprint_file
from .openrouter_transcription import (
    DEFAULT_OPENROUTER_TRANSCRIPTION_MODEL,
    TRANSCRIPTION_ASSEMBLY_RULES,
    OpenRouterRequestError,
)
from .progress import emit_progress
from .subtitles import (
    delivery_profile_for_orientation,
    generate_srt,
    split_long_segments,
)
from .sentence_first import (
    SentenceFirstError,
    build_source_sentences,
    diagnose_source_word_timing,
)
from .transcription import (
    resolve_transcription_backend,
    transcribe_audio,
    validate_mlx_backend,
)
from .translation import (
    translate_segments,
    track_translation_requests,
    translation_prompt_fingerprint,
)
from .utils import _print_segments, load_locked_terms, load_reference_material
from .video_processing import burn_subtitles_into_video, extract_audio, is_portrait_video, _video_duration_seconds
from .subtitle_bundle import GROUPING_POLICY, preserve_source_translation, read_source_srt, render_plan, write_bundle


@dataclass
class PipelineConfig:
    video_filename: str
    transcribe_only: bool = False
    srt_only: bool = False
    save_source_transcript: bool = False
    model: str = "large"
    transcription_backend: str = "auto"
    transcription_model: str = DEFAULT_OPENROUTER_TRANSCRIPTION_MODEL
    language: Optional[str] = None
    target_language: str = "Traditional Chinese (Taiwan)"
    translation_model: str = "deepseek/deepseek-v4.1-flash"
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
    force: bool = False
    source_srt: Optional[str] = None


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
    """Run one generation with isolated application request accounting."""
    with track_translation_requests():
        _process_video(config)


def _process_video(config: PipelineConfig) -> None:
    """Run the configured Hermecho video translation pipeline."""
    if config.source_srt:
        _process_source_srt(config)
        return
    comparison_evidence_dir = os.path.join(config.output_dir, "asr-comparison")
    transcription_backend = resolve_transcription_backend(
        config.transcription_backend,
        config.model,
        comparison_evidence_dir,
    )
    if transcription_backend == "mlx":
        error = validate_mlx_backend(config.model)
        if error:
            print(f"Error: {error}")
            emit_progress("transcription", "error", error)
            return
    if transcription_backend == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            error = "OPENROUTER_API_KEY is required for OpenRouter transcription."
            print(f"Error: {error}")
            emit_progress("transcription", "error", error)
            return
        if not config.transcription_model.strip():
            error = "--transcription-model must be a nonempty OpenRouter model slug."
            print(f"Error: {error}")
            emit_progress("transcription", "error", error)
            return
        if not math.isfinite(config.temperature) or not 0 <= config.temperature <= 1:
            error = "OpenRouter transcription temperature must be a finite number between 0 and 1."
            print(f"Error: {error}")
            emit_progress("transcription", "error", error)
            return

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
    video_name = os.path.splitext(config.video_filename)[0]
    output_dir = os.path.join(config.output_dir, video_name)
    checkpoint_store = CheckpointStore(
        os.path.join(output_dir, ".hermecho-checkpoint.json")
    )
    emit_progress("audio_extraction", "running", "Extracting audio")
    audio_path = extract_audio(video_path)
    if not audio_path:
        emit_progress("audio_extraction", "error", "Audio extraction failed")
        return
    emit_progress("audio_extraction", "complete", "Audio extracted", detail=audio_path)

    try:
        next_stage("Transcribing Audio")
        emit_progress("transcription", "running", "Transcribing audio")
        transcription_inputs = {
            "audio": fingerprint_file(audio_path),
            "backend": transcription_backend,
            "language": config.language,
            "model": config.transcription_model if transcription_backend == "openrouter" else config.model,
            "temperature": config.temperature,
        }
        if transcription_backend == "openrouter":
            transcription_inputs["rules"] = TRANSCRIPTION_ASSEMBLY_RULES
        transcription_fingerprint = fingerprint_data(transcription_inputs)
        transcription_segments = (
            None
            if config.force
            else checkpoint_store.load_transcription(
                transcription_fingerprint,
                require_words=transcription_backend == "openrouter",
            )
        )
        if transcription_segments is None:
            remote_options = {}
            if transcription_backend == "openrouter":
                os.makedirs(output_dir, exist_ok=True)
                remote_options = {
                    "transcription_model": config.transcription_model,
                    "checkpoint_path": os.path.join(output_dir, ".openrouter-transcription.json"),
                    "force": config.force,
                }
            try:
                transcription_segments = transcribe_audio(
                    audio_path,
                    model=config.model,
                    language=config.language,
                    temperature=config.temperature,
                    backend=transcription_backend,
                    **remote_options,
                )
            except OpenRouterRequestError as error:
                print(f"OpenRouter request failed: {error}")
                print("Re-transcribing the entire audio with local Whisper; remote chunks are not mixed into the Source Transcript.")
                emit_progress("transcription", "running", "Falling back to local Whisper for the entire audio")
                transcription_inputs.update(backend="whisper", model=config.model)
                transcription_backend = "whisper"
                transcription_inputs.pop("rules", None)
                transcription_fingerprint = fingerprint_data(transcription_inputs)
                transcription_segments = (
                    None if config.force else checkpoint_store.load_transcription(transcription_fingerprint)
                )
                if transcription_segments is None:
                    transcription_segments = transcribe_audio(
                        audio_path, model=config.model, language=config.language,
                        temperature=config.temperature, backend="whisper",
                    )
            except RuntimeError as error:
                print(f"Audio transcription blocked: {error}")
                emit_progress("transcription", "error", str(error))
                return
            if not transcription_segments:
                emit_progress("transcription", "error", "Audio transcription failed")
                return
            try:
                checkpoint_store.save_transcription(
                    transcription_fingerprint,
                    transcription_segments,
                    require_words=transcription_backend == "openrouter",
                )
            except ValueError as error:
                print(f"Audio transcription checkpoint blocked: {error}")
                emit_progress("transcription", "error", str(error))
                return
        else:
            print("Reusing completed transcription checkpoint.")
        emit_progress(
            "transcription",
            "complete",
            "Audio transcribed",
            current=len(transcription_segments),
            total=len(transcription_segments),
            pct=100,
        )

        _print_segments(
            f"Original Transcription ({config.language or 'auto'})",
            transcription_segments,
        )

        transcript_segments = []
        source_sentences = []
        locked_terms = {}
        reference_material = None
        if config.transcribe_only:
            transcript_segments = split_long_segments(transcription_segments)
            transcript_segments = [
                segment
                for segment in transcript_segments
                if segment.get("text", "").strip() != "[no speech]"
            ]
            _print_segments("Transcription after Splitting", transcript_segments)
        else:
            reference_material = load_reference_material(config.reference_file)
            locked_terms = load_locked_terms(config.locked_terms_file)
            if locked_terms is None:
                emit_progress(
                    "translation_gate",
                    "error",
                    "Locked Terms configuration is invalid",
                )
                return
            source_timing_diagnostics = diagnose_source_word_timing(
                transcription_segments
            )
            try:
                deterministic_source_sentences = build_source_sentences(
                    transcription_segments
                )
            except SentenceFirstError as error:
                print(f"Sentence-first delivery blocked: {error}")
                emit_progress("subtitle_delivery", "error", str(error))
                return
            source_grouping_fingerprint = fingerprint_data(
                {
                    "transcription": transcription_fingerprint,
                    "grouping_rules": GROUPING_POLICY,
                }
            )
            source_sentences = (
                None
                if config.force
                else checkpoint_store.load_source_sentences(source_grouping_fingerprint)
            )
            if source_sentences is None:
                source_sentences = deterministic_source_sentences
                try:
                    checkpoint_store.save_source_sentences(source_grouping_fingerprint, source_sentences)
                except ValueError as error:
                    print(f"Warning: Source Sentence checkpoint skipped: {error}")
            else:
                print("Reusing completed Source Sentence grouping checkpoint.")
            _print_segments("Source Sentences", source_sentences)

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:12]

        if config.transcribe_only:
            next_stage("Writing Transcript SRT")
            srt_path = os.path.join(output_dir, f"{video_name}_{timestamp}_transcript.srt")
            emit_progress("source_srt_write", "running", "Writing transcript SRT")
            generate_srt(transcript_segments, srt_path)
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

        if config.save_source_transcript:
            source_srt = os.path.join(
                output_dir,
                f"{video_name}_{timestamp}_transcript_source.srt",
            )
            emit_progress("source_srt_write", "running", "Writing source transcript SRT")
            generate_srt(source_sentences, source_srt)
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
        translation_fingerprint = fingerprint_data(
            {
                "locked_terms": locked_terms,
                "model": config.translation_model,
                "prompt": translation_prompt_fingerprint(),
                "reference": reference_material or "",
                "source": fingerprint_data(source_sentences),
                "target_language": config.target_language,
                # Preserve matching checkpoints created before legacy delivery retired.
                "subtitle_delivery": "sentence-first",
            }
        )
        checkpoint_store.discard_stale_translation(translation_fingerprint)

        def load_accepted_chunk(chunk_index: int, chunk: list[dict]) -> Optional[dict]:
            if config.force:
                return None
            expected_ids = [
                str(segment.get("_translation_id", index))
                for index, segment in enumerate(chunk)
            ]
            return checkpoint_store.load_accepted_translation_chunk(
                translation_fingerprint,
                chunk_index,
                fingerprint_data(chunk),
                expected_ids,
            )

        def save_accepted_chunk(
            chunk_index: int,
            chunk: list[dict],
            translations: dict[str, str],
        ) -> None:
            checkpoint_store.save_accepted_translation_chunk(
                translation_fingerprint,
                chunk_index,
                fingerprint_data(chunk),
                translations,
            )

        translated_sentences = translate_segments(
            source_sentences,
            target_language=config.target_language,
            translation_model=config.translation_model,
            reference_material=reference_material,
            locked_terms=locked_terms,
            accepted_chunk_loader=load_accepted_chunk,
            accepted_chunk_saver=save_accepted_chunk,
        )

        if translated_sentences is not None:
            emit_progress(
                "translation",
                "complete",
                "Translation completed",
                current=len(translated_sentences),
                total=len(source_sentences),
                pct=100,
            )
            translation_label = f"Translation ({config.target_language})"
            _print_segments(translation_label, translated_sentences)

            profile = delivery_profile_for_orientation(is_portrait)
            delivery_cues = preserve_source_translation(source_sentences, translated_sentences)
            duration = _video_duration_seconds(video_path)
            bundle_path = os.path.join(output_dir, f"{video_name}_{timestamp}_subtitle_bundle.json")
            bundle = write_bundle(bundle_path, source_sentences, delivery_cues,
                video_fingerprint=fingerprint_file(video_path), source_language=config.language,
                target_language=config.target_language, profile=profile, duration=duration)
            report_path = os.path.join(output_dir, f"{video_name}_{timestamp}_delivery_gate.txt")
            with open(report_path, "w", encoding="utf-8") as report_file:
                report_file.write("Subtitle timing preserved; quality findings are warnings.\n")
                report_file.write(json.dumps({"diagnostics": bundle["diagnostics"], "omitted": bundle["omitted"]}, ensure_ascii=False, indent=2))
            emit_progress("delivery_gate", "complete", "Source subtitle timing preserved", detail=report_path)

            next_stage("Writing Subtitle SRT")
            srt_path = os.path.join(output_dir, f"{video_name}_{timestamp}_subtitles.srt")
            emit_progress("translated_srt_write", "running", "Writing translated SRT")
            generate_srt(delivery_cues, srt_path)
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
                render_srt_path = os.path.join(output_dir, f"{video_name}_{timestamp}_render.srt")
                visible, _ = render_plan(delivery_cues, duration)
                generate_srt(visible, render_srt_path)
                burn_subtitles_into_video(
                    video_path,
                    os.path.abspath(render_srt_path),
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


def _process_source_srt(config: PipelineConfig) -> None:
    """Translate explicitly imported source captions without ASR or invented words."""
    if config.source_srt is None:
        raise ValueError("Source SRT path is required")
    source = read_source_srt(config.source_srt)
    reference = load_reference_material(config.reference_file)
    locked = load_locked_terms(config.locked_terms_file)
    if locked is None:
        raise ValueError("Locked Terms configuration is invalid")
    output_dir = os.path.join(config.output_dir, os.path.splitext(config.video_filename)[0])
    os.makedirs(output_dir, exist_ok=True)
    store = CheckpointStore(os.path.join(output_dir, ".hermecho-checkpoint.json"))
    key = fingerprint_data({"source": source, "model": config.translation_model,
        "target_language": config.target_language, "prompt": translation_prompt_fingerprint(),
        "reference": reference, "locked_terms": locked})
    store.discard_stale_translation(key)
    def load_chunk(index, chunk):
        if config.force:
            return None
        return store.load_accepted_translation_chunk(key, index, fingerprint_data(chunk),
            [str(s.get("_translation_id", i)) for i, s in enumerate(chunk)])
    def save_chunk(index, chunk, translations):
        store.save_accepted_translation_chunk(key, index, fingerprint_data(chunk), translations)
    translated = translate_segments(source, target_language=config.target_language,
        translation_model=config.translation_model, reference_material=reference, locked_terms=locked,
        accepted_chunk_loader=load_chunk, accepted_chunk_saver=save_chunk)
    if translated is None:
        raise ValueError("Translation Gate blocked final SRT/video delivery")
    translated = preserve_source_translation(source, translated)
    video_path = os.path.abspath(os.path.join(config.input_dir, config.video_filename))
    output_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:12]
    base = os.path.join(output_dir, f"{os.path.splitext(config.video_filename)[0]}_{output_id}")
    generate_srt(source, base + "_transcript_source.srt")
    generate_srt(translated, base + "_subtitles.srt")
    duration = _video_duration_seconds(video_path)
    write_bundle(base + "_subtitle_bundle.json", source, translated,
        video_fingerprint=fingerprint_file(video_path), source_language=config.language,
        target_language=config.target_language, profile=delivery_profile_for_orientation(is_portrait_video(video_path)), duration=duration)
    if not config.srt_only:
        visible, _ = render_plan(translated, duration)
        generate_srt(visible, base + "_render.srt")
        burn_subtitles_into_video(video_path, base + "_render.srt", base + "_translated.mp4",
            font_name=config.font_name, fonts_dir=config.fonts_dir, font_size=config.font_size,
            outline_width=config.outline_width, use_box_background=config.box_background,
            margin_v=config.margin_v, margin_h=config.margin_h, alignment=config.alignment)
    emit_progress("completion", "complete", "Hermecho pipeline completed", pct=100)
