"""Frozen-transcript comparison run for Phase 3 delivery promotion."""
from __future__ import annotations

import argparse
import difflib
import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv

from .asr_comparison import _extract_review_range, _write_review_composite
from .checkpoints import fingerprint_data
from .sentence_first import REVIEW_CHECKS


DEFAULT_START = "00:29:30.000"
DEFAULT_END = "00:39:30.000"
DEFAULT_EVIDENCE_DIR = Path("output/sentence-first-comparison")
REVIEW_MEDIA_FILENAME = "20251231_w-yGSP1c3bg.mp4"


@dataclass(frozen=True)
class ComparisonConfig:
    video_path: Path
    output_dir: Path = DEFAULT_EVIDENCE_DIR
    model: str = "large"
    transcription_backend: str = "auto"
    language: Optional[str] = "ko"
    temperature: float = 0.0
    target_language: str = "Traditional Chinese (Taiwan)"
    translation_model: str = "deepseek/deepseek-v4-pro"
    reference_file: Path = Path("references/tripleS.md")
    locked_terms_file: Path = Path("references/locked_terms.json")
    time_buffer: float = 0.1
    font_name: str = "Heiti TC"
    fonts_dir: Optional[str] = None
    font_size: int = 12
    outline_width: int = 0
    box_background: bool = True
    margin_v: int = 20
    margin_h: int = 10
    alignment: int = 2
    start: str = DEFAULT_START
    end: str = DEFAULT_END


def _relative_path(output_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(output_dir.resolve()))
    except ValueError as error:
        raise ValueError(f"Comparison artifact must be inside {output_dir}: {path}") from error


def _one_artifact(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {pattern} artifact under {directory}, found {len(matches)}."
        )
    return matches[0]


def _delivery_gate_status(path: Path) -> str:
    try:
        report = path.read_text(encoding="utf-8")
    except OSError:
        return "failed"
    return (
        "passed"
        if "Structural Defects: 0" in report and "Translation Gate blocked" not in report
        else "failed"
    )


def _write_review_artifacts(
    output_dir: Path,
    *,
    source_video: Path,
    range_path: Path,
    start: str,
    end: str,
    frozen_transcription: Path,
    frozen_source_transcript: Path,
    frozen_fingerprint: str,
    shared_options: dict[str, object],
    baseline_source: Path,
    candidate_source: Path,
    baseline_video: Path,
    candidate_video: Path,
    baseline_gate: Path,
    candidate_gate: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_stat = source_video.stat()
    diff_path = output_dir / "source_transcript.diff"
    diff = difflib.unified_diff(
        baseline_source.read_text(encoding="utf-8").splitlines(keepends=True),
        candidate_source.read_text(encoding="utf-8").splitlines(keepends=True),
        fromfile="legacy/source_transcript.srt",
        tofile="sentence-first/source_transcript.srt",
    )
    diff_path.write_text("".join(diff), encoding="utf-8")

    composite_path = output_dir / "review_composite.mp4"
    _write_review_composite(baseline_video, candidate_video, composite_path)

    artifacts = {
        "manifest": "manifest.json",
        "frozen_transcription": _relative_path(output_dir, frozen_transcription),
        "frozen_source_transcript": _relative_path(output_dir, frozen_source_transcript),
        "source_transcript_diff": diff_path.name,
        "review_composite": composite_path.name,
        "legacy_delivery_gate": _relative_path(output_dir, baseline_gate),
        "sentence_first_delivery_gate": _relative_path(output_dir, candidate_gate),
    }
    delivery_gates = {
        "baseline": _delivery_gate_status(baseline_gate),
        "candidate": _delivery_gate_status(candidate_gate),
    }
    manifest = {
        "comparison_variable": "subtitle_delivery",
        "media_range": {
            "source": str(source_video.resolve()),
            "source_name": source_video.name,
            "source_bytes": source_stat.st_size,
            "source_modified_at": datetime.fromtimestamp(
                source_stat.st_mtime,
                timezone.utc,
            ).isoformat(),
            "start": start,
            "end": end,
            "prepared_media": range_path.name,
            "shared_audio": "review_composite.mp4",
        },
        "frozen_transcription": {
            "path": artifacts["frozen_transcription"],
            "fingerprint": frozen_fingerprint,
        },
        "baseline": "legacy",
        "candidate": "sentence-first",
        "shared": shared_options,
        "delivery_gates": delivery_gates,
        "artifacts": artifacts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "comparison.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    checklist = [
        f"- {check.title()}: PENDING"
        for check in REVIEW_CHECKS
    ]
    regressions = [
        f"- Candidate-only {check}: PENDING"
        for check in REVIEW_CHECKS
    ]
    (output_dir / "review.md").write_text(
        "\n".join(
            [
                "# Sentence-first Delivery Comparison Review",
                "",
                "## Frozen comparison",
                f"- Media: `{source_video.name}` from `{start}` through `{end}`",
                f"- Frozen transcription: `{artifacts['frozen_transcription']}`",
                "- Baseline: legacy delivery",
                "- Candidate: sentence-first delivery",
                f"- Source transcript diff: `{artifacts['source_transcript_diff']}`",
                "- Review Composite: unscaled side-by-side video with one shared audio track",
                "",
                "## Review checklist",
                *checklist,
                "",
                "## Candidate-only regression checklist",
                *regressions,
                "",
                "## Timestamped Candidate-only regressions",
                "- PENDING: record `HH:MM:SS.mmm: description`, or replace with `none`.",
                "",
                "## Human Approval",
                "- Reviewer: PENDING",
                "- Date: PENDING",
                "- Decision: PENDING",
                "",
                "Do not mark this approved unless every Candidate-only checklist item "
                "is `no` and both Delivery Gates passed.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_comparison(config: ComparisonConfig) -> Path:
    """Freeze one transcript, run both delivery paths, and write review evidence."""
    if not config.video_path.is_file():
        raise FileNotFoundError(f"Comparison media not found: {config.video_path}")
    if config.video_path.name != REVIEW_MEDIA_FILENAME:
        raise ValueError(f"Phase 3 uses the fixed media {REVIEW_MEDIA_FILENAME}.")
    if (config.start, config.end) != (DEFAULT_START, DEFAULT_END):
        raise ValueError("Phase 3 uses the fixed 10-minute review range.")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"Comparison output must be empty: {config.output_dir}")

    from .video_processing import _ffmpeg_supports_subtitles_filter

    if not _ffmpeg_supports_subtitles_filter():
        raise RuntimeError(
            "Comparison requires an ffmpeg build with the `subtitles` filter "
            "(libass support) before it starts."
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    range_path = config.output_dir / "media_range.mp4"
    _extract_review_range(config.video_path, range_path, config.start, config.end)

    from .pipeline import PipelineConfig, process_video
    from .subtitles import generate_srt
    from .transcription import resolve_transcription_backend, transcribe_audio
    from .video_processing import extract_audio

    selected_backend = resolve_transcription_backend(
        config.transcription_backend,
        config.model,
    )
    audio_path = extract_audio(str(range_path))
    if not audio_path:
        raise RuntimeError("Could not extract comparison audio.")
    try:
        transcription = transcribe_audio(
            audio_path,
            model=config.model,
            language=config.language,
            temperature=config.temperature,
            backend=selected_backend,
        )
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    if not transcription:
        raise RuntimeError("Comparison transcription produced no Source Words.")

    frozen_path = config.output_dir / "frozen_transcription.json"
    frozen_fingerprint = fingerprint_data(transcription)
    frozen_path.write_text(
        json.dumps(
            {
                "version": 1,
                "source": range_path.name,
                "fingerprint": frozen_fingerprint,
                "segments": transcription,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    frozen_source_path = config.output_dir / "frozen_source_transcript.srt"
    generate_srt(
        [segment for segment in transcription if segment.get("text", "").strip() != "[no speech]"],
        str(frozen_source_path),
    )

    results: dict[str, dict[str, Path]] = {}
    for mode in ("legacy", "sentence-first"):
        mode_dir = config.output_dir / mode
        process_video(
            PipelineConfig(
                video_filename=range_path.name,
                input_dir=str(range_path.parent),
                output_dir=str(mode_dir),
                save_source_transcript=True,
                model=config.model,
                transcription_backend=selected_backend,
                subtitle_delivery=mode,
                language=config.language,
                target_language=config.target_language,
                translation_model=config.translation_model,
                time_buffer=config.time_buffer,
                reference_file=str(config.reference_file),
                locked_terms_file=str(config.locked_terms_file),
                font_name=config.font_name,
                fonts_dir=config.fonts_dir,
                font_size=config.font_size,
                outline_width=config.outline_width,
                box_background=config.box_background,
                margin_v=config.margin_v,
                margin_h=config.margin_h,
                alignment=config.alignment,
                stage_cooldown=0,
                transcription_artifact=str(frozen_path),
            )
        )
        try:
            results[mode] = {
                "source": _one_artifact(mode_dir, "*source*.srt"),
                "video": _one_artifact(mode_dir, "*translated.mp4"),
                "gate": _one_artifact(mode_dir, "*_delivery_gate.txt"),
            }
        except FileNotFoundError as error:
            raise RuntimeError(
                f"{mode} delivery did not finish; comparison stopped before review artifacts. "
                "Check the preceding pipeline error."
            ) from error
    _write_review_artifacts(
        config.output_dir,
        source_video=config.video_path,
        range_path=range_path,
        start=config.start,
        end=config.end,
        frozen_transcription=frozen_path,
        frozen_source_transcript=frozen_source_path,
        frozen_fingerprint=frozen_fingerprint,
        shared_options={
            "model_checkpoint": config.model,
            "machine": {
                "system": platform.system(),
                "machine": platform.machine(),
                "release": platform.release(),
            },
            "runtime_versions": {"python": platform.python_version()},
            "language": config.language,
            "temperature": config.temperature,
            "transcription_backend": selected_backend,
            "references": {
                "reference_file": str(config.reference_file),
                "locked_terms_file": str(config.locked_terms_file),
            },
            "translation": {
                "provider": "OpenRouter",
                "model": config.translation_model,
                "target_language": config.target_language,
            },
            "subtitle_style": {
                "font_name": config.font_name,
                "fonts_dir": config.fonts_dir,
                "font_size": config.font_size,
                "outline_width": config.outline_width,
                "box_background": config.box_background,
                "margin_v": config.margin_v,
                "margin_h": config.margin_h,
                "alignment": config.alignment,
                "time_buffer": config.time_buffer,
            },
            "effective_cli_options": {
                "video_filename": range_path.name,
                "input_dir": str(range_path.parent),
                "model": config.model,
                "language": config.language,
                "temperature": config.temperature,
                "target_language": config.target_language,
                "translation_model": config.translation_model,
                "reference_file": str(config.reference_file),
                "locked_terms_file": str(config.locked_terms_file),
                "time_buffer": config.time_buffer,
                "font_name": config.font_name,
                "fonts_dir": config.fonts_dir,
                "font_size": config.font_size,
                "outline_width": config.outline_width,
                "box_background": config.box_background,
                "margin_v": config.margin_v,
                "margin_h": config.margin_h,
                "alignment": config.alignment,
                "subtitle_delivery": "varies: legacy versus sentence-first",
            },
        },
        baseline_source=frozen_source_path,
        candidate_source=frozen_source_path,
        baseline_video=results["legacy"]["video"],
        candidate_video=results["sentence-first"]["video"],
        baseline_gate=results["legacy"]["gate"],
        candidate_gate=results["sentence-first"]["gate"],
    )
    return config.output_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> ComparisonConfig:
    parser = argparse.ArgumentParser(
        description="Run the fixed-transcript legacy versus sentence-first delivery comparison."
    )
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--model", default="large")
    parser.add_argument(
        "--transcription-backend",
        choices=("auto", "whisper", "mlx"),
        default="auto",
    )
    parser.add_argument("--language", default="ko")
    args = parser.parse_args(argv)
    return ComparisonConfig(**vars(args))


def main(argv: Optional[Sequence[str]] = None) -> None:
    load_dotenv()
    output_dir = run_comparison(parse_args(argv))
    print(f"Sentence-first comparison evidence written to {output_dir}")


if __name__ == "__main__":
    main()
