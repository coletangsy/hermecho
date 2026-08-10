"""Repeatable Whisper-versus-MLX Comparison Runs and approval evidence."""
from __future__ import annotations

import argparse
import difflib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence


BASELINE_BACKEND = "whisper"
CANDIDATE_BACKEND = "mlx"
DEFAULT_EVIDENCE_DIR = Path("output/asr-comparison")
DEFAULT_START = "00:29:30.000"
DEFAULT_END = "00:39:30.000"
REVIEW_MEDIA_FILENAME = "20251231_w-yGSP1c3bg.mp4"
RUNS_PER_BACKEND = 3
REVIEW_CHECKS = (
    "missing speech",
    "repetition",
    "hallucination",
    "name regression",
    "timing regression",
    "unreadable subtitle",
)


@dataclass(frozen=True)
class ComparisonConfig:
    """Inputs held constant for a Comparison Run."""

    video_path: Path
    output_dir: Path = DEFAULT_EVIDENCE_DIR
    model: str = "large"
    language: Optional[str] = None
    temperature: float = 0.0
    reference_file: Path = Path("references/tripleS.md")
    locked_terms_file: Path = Path("references/locked_terms.json")
    translation_model: str = "deepseek/deepseek-v4-pro"
    target_language: str = "Traditional Chinese (Taiwan)"
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    time_buffer: float = 0.1
    font_name: str = "Heiti TC"
    fonts_dir: Optional[str] = None
    font_size: int = 12
    outline_width: int = 0
    margin_v: int = 20
    margin_h: int = 10
    alignment: int = 2
    stage_cooldown: int = 0


def evidence_allows_mlx(evidence_dir: str | Path, *, model: str) -> bool:
    """Return whether completed evidence explicitly promotes MLX for ``model``."""
    directory = Path(evidence_dir)
    evidence = _read_json(directory / "comparison.json")
    manifest = _read_json(directory / "manifest.json")

    report_manifest = evidence.get("manifest")
    runs = evidence.get("runs")
    if (
        evidence.get("comparison_variable") != "transcription_backend"
        or not isinstance(report_manifest, dict)
        or report_manifest.get("model_checkpoint") != model
        or report_manifest.get("path") != "manifest.json"
        or not isinstance(runs, list)
        or not _manifest_allows_mlx(manifest, directory, model)
        or not _required_artifacts_exist(directory, evidence.get("artifacts"))
        or not _successful_warmups(evidence.get("warmups"))
    ):
        return False

    expected_order = [
        backend
        for _ in range(RUNS_PER_BACKEND)
        for backend in (BASELINE_BACKEND, CANDIDATE_BACKEND)
    ]
    if not all(isinstance(run, dict) for run in runs):
        return False
    if [run.get("backend") for run in runs] != expected_order:
        return False

    timings: dict[str, list[float]] = {BASELINE_BACKEND: [], CANDIDATE_BACKEND: []}
    for iteration, run in enumerate(runs):
        backend = expected_order[iteration]
        value = run.get("transcription_seconds")
        if (
            not _successful_process(
                run,
                backend,
                iteration=iteration // 2 + 1,
                warm_cache=False,
            )
            or not _valid_metric(value)
        ):
            return False
        timings[backend].append(float(value))

    if statistics.median(timings[CANDIDATE_BACKEND]) >= statistics.median(timings[BASELINE_BACKEND]):
        return False
    return _review_is_explicitly_approved(directory / "review.md")


def _manifest_allows_mlx(manifest: dict[str, Any], directory: Path, model: str) -> bool:
    if manifest.get("comparison_variable") != "transcription_backend":
        return False
    media_range = manifest.get("media_range")
    shared = manifest.get("shared")
    baseline = manifest.get("baseline")
    candidate = manifest.get("candidate")
    if not all(isinstance(value, dict) for value in (media_range, shared, baseline, candidate)):
        return False

    source = media_range.get("source")
    prepared_media = media_range.get("prepared_media")
    if (
        not isinstance(source, str)
        or Path(source).name != REVIEW_MEDIA_FILENAME
        or media_range.get("source_name") != REVIEW_MEDIA_FILENAME
        or media_range.get("start") != DEFAULT_START
        or media_range.get("end") != DEFAULT_END
        or not isinstance(prepared_media, str)
        or Path(prepared_media).resolve() != (directory / "media_range.mp4").resolve()
        or not Path(prepared_media).is_file()
    ):
        return False

    if not _shared_options_are_valid(shared, model):
        return False

    baseline_options = dict(baseline)
    candidate_options = dict(candidate)
    return (
        baseline_options.pop("transcription_backend", None) == BASELINE_BACKEND
        and candidate_options.pop("transcription_backend", None) == CANDIDATE_BACKEND
        and baseline_options == candidate_options
    )


def _shared_options_are_valid(shared: dict[str, Any], model: str) -> bool:
    required_shared = {
        "model_checkpoint",
        "machine",
        "runtime_versions",
        "language",
        "temperature",
        "prompt",
        "references",
        "translation",
        "subtitle_style",
        "effective_cli_options",
    }
    if not required_shared.issubset(shared):
        return False
    language = shared["language"]
    temperature = shared["temperature"]
    prompt = shared["prompt"]
    references = shared["references"]
    translation = shared["translation"]
    subtitle_style = shared["subtitle_style"]
    options = shared["effective_cli_options"]
    if (
        shared["model_checkpoint"] != model
        or not isinstance(shared["machine"], dict)
        or not isinstance(shared["runtime_versions"], dict)
        or not _optional_text(language)
        or not _finite_number(temperature)
        or not _optional_text(prompt)
        or not isinstance(references, dict)
        or not all(
            _nonempty_text(references.get(field))
            for field in ("reference_file", "locked_terms_file")
        )
        or not isinstance(translation, dict)
        or translation.get("provider") != "OpenRouter"
        or not all(
            _nonempty_text(translation.get(field))
            for field in ("model", "target_language")
        )
        or not _subtitle_style_is_valid(subtitle_style)
    ):
        return False
    return _effective_options_match_shared(
        options,
        model=model,
        language=language,
        temperature=temperature,
        references=references,
        translation=translation,
        subtitle_style=subtitle_style,
    )


def _effective_options_match_shared(
    options: Any,
    *,
    model: str,
    language: Optional[str],
    temperature: float,
    references: dict[str, Any],
    translation: dict[str, Any],
    subtitle_style: dict[str, Any],
) -> bool:
    if not isinstance(options, dict):
        return False
    if not all(
        _nonempty_text(options.get(field))
        for field in ("video_filename", "input_dir")
    ):
        return False
    if (
        options.get("model") != model
        or options.get("language") != language
        or options.get("temperature") != temperature
        or options.get("target_language") != translation["target_language"]
        or options.get("translation_model") != translation["model"]
        or options.get("reference_file") != references["reference_file"]
        or options.get("locked_terms_file") != references["locked_terms_file"]
        or options.get("save_source_transcript") is not True
        or not _nonnegative_integer(options.get("stage_cooldown"))
    ):
        return False
    return all(
        options.get(field) == subtitle_style[field]
        for field in (
            "font_name",
            "fonts_dir",
            "font_size",
            "outline_width",
            "box_background",
            "margin_v",
            "margin_h",
            "alignment",
            "time_buffer",
        )
    ) and options["box_background"] == _cli_defaults().box_background


def _subtitle_style_is_valid(style: Any) -> bool:
    if not isinstance(style, dict):
        return False
    return (
        _nonempty_text(style.get("font_name"))
        and _nonempty_text(style.get("fonts_dir"))
        and _positive_integer(style.get("font_size"))
        and _nonnegative_integer(style.get("outline_width"))
        and isinstance(style.get("box_background"), bool)
        and _nonnegative_integer(style.get("margin_v"))
        and _nonnegative_integer(style.get("margin_h"))
        and _alignment(style.get("alignment"))
        and _nonnegative_number(style.get("time_buffer"))
    )


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _optional_text(value: Any) -> bool:
    return value is None or _nonempty_text(value)


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _nonnegative_number(value: Any) -> bool:
    return _finite_number(value) and value >= 0


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _alignment(value: Any) -> bool:
    return _positive_integer(value) and value <= 9


def _required_artifacts_exist(directory: Path, artifacts: Any) -> bool:
    if not isinstance(artifacts, dict):
        return False
    for key in ("manifest", "source_transcript_diff", "review_composite"):
        value = artifacts.get(key)
        if not isinstance(value, str):
            return False
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not (directory / path).is_file():
            return False
    return True


def _successful_warmups(warmups: Any) -> bool:
    if not isinstance(warmups, list) or len(warmups) != 2:
        return False
    return all(
        _successful_process(warmup, backend, warm_cache=True)
        for warmup, backend in zip(warmups, (BASELINE_BACKEND, CANDIDATE_BACKEND))
    )


def _successful_process(
    record: Any,
    backend: str,
    *,
    warm_cache: bool,
    iteration: Optional[int] = None,
) -> bool:
    return (
        isinstance(record, dict)
        and record.get("backend") == backend
        and record.get("fresh_process") is True
        and record.get("warm_cache") is warm_cache
        and record.get("completed") is True
        and record.get("returncode") == 0
        and (iteration is None or record.get("iteration") == iteration)
        and _has_metrics(record)
    )


def _review_is_explicitly_approved(review_path: Path) -> bool:
    try:
        lines = review_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False

    if "## Human Approval" not in lines:
        return False
    fields = {}
    for line in lines:
        if line.startswith("- ") and ": " in line:
            key, value = line[2:].split(": ", 1)
            fields[key.casefold()] = value.strip().casefold()

    if fields.get("decision") != "approved":
        return False
    if fields.get("reviewer") in {None, "", "pending"}:
        return False
    if fields.get("date") in {None, "", "pending"}:
        return False
    try:
        date.fromisoformat(fields["date"])
    except ValueError:
        return False
    return all(fields.get(f"candidate-only {check}") == "no" for check in REVIEW_CHECKS)


def run_comparison(config: ComparisonConfig) -> Path:
    """Run cached warmups and three alternating fresh runs, then write evidence."""
    output_dir = config.output_dir
    if not config.video_path.is_file():
        raise FileNotFoundError(f"Comparison media not found: {config.video_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Comparison output must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    range_path = output_dir / "media_range.mp4"
    _extract_review_range(config.video_path, range_path, config.start, config.end)

    manifest = _manifest(config, range_path)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "baseline_manifest.json", {**manifest["shared"], "transcription_backend": BASELINE_BACKEND})
    _write_json(output_dir / "candidate_manifest.json", {**manifest["shared"], "transcription_backend": CANDIDATE_BACKEND})

    warmups: list[dict[str, Any]] = []
    for backend in (BASELINE_BACKEND, CANDIDATE_BACKEND):
        warm = _run_once(config, range_path, output_dir / "warm" / backend, backend, warm_cache=True)
        warmups.append(warm)
        if not warm["completed"]:
            _write_json(output_dir / "comparison.json", _comparison_report(config, [], warmups))
            raise RuntimeError(f"Could not warm {backend}; no comparison evidence was approved.")

    runs: list[dict[str, Any]] = []
    for iteration in range(1, RUNS_PER_BACKEND + 1):
        for backend in (BASELINE_BACKEND, CANDIDATE_BACKEND):
            run = _run_once(
                config,
                range_path,
                output_dir / "runs" / f"{iteration}-{backend}",
                backend,
                warm_cache=False,
            )
            run["iteration"] = iteration
            runs.append(run)

    report = _comparison_report(config, runs, warmups)
    _write_json(output_dir / "comparison.json", report)
    if not all(run["completed"] for run in runs):
        raise RuntimeError("Comparison run failed; see comparison.json for the failed process.")

    baseline = _median_run(runs, BASELINE_BACKEND)
    candidate = _median_run(runs, CANDIDATE_BACKEND)
    diff_path = output_dir / "source_transcript.diff"
    _write_source_diff(Path(baseline["source_transcript"]), Path(candidate["source_transcript"]), diff_path)
    composite_path = output_dir / "review_composite.mp4"
    _write_review_composite(Path(baseline["video"]), Path(candidate["video"]), composite_path)
    report["artifacts"] = {
        "manifest": "manifest.json",
        "source_transcript_diff": diff_path.name,
        "review_composite": composite_path.name,
    }
    _write_json(output_dir / "comparison.json", report)
    _write_review(output_dir, report, diff_path, composite_path)
    return output_dir


def _manifest(config: ComparisonConfig, range_path: Path) -> dict[str, Any]:
    source_stat = config.video_path.stat()
    fonts_dir, box_background = _effective_cli_style(config)
    return {
        "comparison_variable": "transcription_backend",
        "media_range": {
            "source": str(config.video_path.resolve()),
            "source_name": config.video_path.name,
            "source_bytes": source_stat.st_size,
            "source_modified_at": datetime.fromtimestamp(
                source_stat.st_mtime,
                timezone.utc,
            ).isoformat(),
            "start": config.start,
            "end": config.end,
            "prepared_media": str(range_path.resolve()),
        },
        "shared": {
            "model_checkpoint": config.model,
            "machine": {
                "system": platform.system(),
                "machine": platform.machine(),
                "release": platform.release(),
            },
            "runtime_versions": _runtime_versions(),
            "language": config.language,
            "temperature": config.temperature,
            "prompt": None,
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
                "fonts_dir": fonts_dir,
                "font_size": config.font_size,
                "outline_width": config.outline_width,
                "box_background": box_background,
                "margin_v": config.margin_v,
                "margin_h": config.margin_h,
                "alignment": config.alignment,
                "time_buffer": config.time_buffer,
            },
            "effective_cli_options": _common_cli_options(
                config,
                range_path,
                fonts_dir,
                box_background,
            ),
        },
        "baseline": {"transcription_backend": BASELINE_BACKEND},
        "candidate": {"transcription_backend": CANDIDATE_BACKEND},
    }


def _runtime_versions() -> dict[str, Optional[str]]:
    def installed_version(distribution: str) -> Optional[str]:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "openai-whisper": installed_version("openai-whisper"),
        "mlx-whisper": installed_version("mlx-whisper"),
    }


def _effective_cli_style(config: ComparisonConfig) -> tuple[str, bool]:
    defaults = _cli_defaults()
    fonts_dir = config.fonts_dir if config.fonts_dir is not None else defaults.fonts_dir
    if not fonts_dir:
        raise RuntimeError("Hermecho CLI has no effective fonts directory.")
    return fonts_dir, defaults.box_background


def _cli_defaults() -> argparse.Namespace:
    from .cli import parse_args

    return parse_args(["comparison.mp4"])


def _common_cli_options(
    config: ComparisonConfig,
    range_path: Path,
    fonts_dir: str,
    box_background: bool,
) -> dict[str, Any]:
    return {
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
        "fonts_dir": fonts_dir,
        "font_size": config.font_size,
        "outline_width": config.outline_width,
        "box_background": box_background,
        "margin_v": config.margin_v,
        "margin_h": config.margin_h,
        "alignment": config.alignment,
        "stage_cooldown": config.stage_cooldown,
        "save_source_transcript": True,
    }


def _extract_review_range(source: Path, target: Path, start: str, end: str) -> None:
    duration = _timestamp_seconds(end) - _timestamp_seconds(start)
    if duration <= 0:
        raise ValueError("Comparison end must be after start.")
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-ss",
            start,
            "-t",
            str(duration),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(target),
        ]
    )


def _timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _run_once(
    config: ComparisonConfig,
    range_path: Path,
    run_dir: Path,
    backend: str,
    *,
    warm_cache: bool,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "run_config.json"
    _write_json(
        config_path,
        {"cli_args": _pipeline_args(config, range_path, run_dir, backend, warm_cache)},
    )
    command = [
        sys.executable,
        "-m",
        "hermecho.asr_comparison",
        "_run",
        "--backend",
        backend,
        "--metrics-file",
        str(metrics_path),
        "--run-dir",
        str(run_dir),
        "--config",
        str(config_path),
    ]
    if warm_cache:
        command.append("--warm-cache")
    result = subprocess.run(command, capture_output=True, text=True)
    metrics = _read_json(metrics_path)
    record: dict[str, Any] = {
        "backend": backend,
        "fresh_process": True,
        "warm_cache": warm_cache,
        "returncode": result.returncode,
        "transcription_seconds": metrics.get("transcription_seconds"),
        "end_to_end_seconds": metrics.get("end_to_end_seconds"),
        "peak_memory_bytes": metrics.get("peak_memory_bytes"),
    }
    if not warm_cache and result.returncode == 0:
        try:
            record.update(_run_artifacts(run_dir))
        except OSError as error:
            record["artifact_error"] = str(error)
    record["completed"] = result.returncode == 0 and _has_metrics(record) and (
        warm_cache or all(key in record for key in ("source_transcript", "final_subtitles", "video"))
    )
    return record


def _pipeline_args(
    config: ComparisonConfig,
    range_path: Path,
    run_dir: Path,
    backend: str,
    warm_cache: bool,
) -> list[str]:
    fonts_dir, box_background = _effective_cli_style(config)
    args = [
        range_path.name,
        "--input_dir",
        str(range_path.parent),
        "--output_dir",
        str(run_dir),
        "--model",
        config.model,
        "--transcription-backend",
        backend,
        "--target_language",
        config.target_language,
        "--translation_model",
        config.translation_model,
        "--time_buffer",
        str(config.time_buffer),
        "--reference_file",
        str(config.reference_file),
        "--locked-terms-file",
        str(config.locked_terms_file),
        "--temperature",
        str(config.temperature),
        "--font_name",
        config.font_name,
        "--font_size",
        str(config.font_size),
        "--outline_width",
        str(config.outline_width),
        "--margin_v",
        str(config.margin_v),
        "--margin_h",
        str(config.margin_h),
        "--alignment",
        str(config.alignment),
        "--stage-cooldown",
        str(config.stage_cooldown),
        "--save-source-transcript",
    ]
    if box_background:
        args.append("--box_background")
    if config.language:
        args.extend(["--language", config.language])
    args.extend(["--fonts-dir", fonts_dir])
    if warm_cache:
        args.append("--transcribe-only")
    return args


def _run_artifacts(run_dir: Path) -> dict[str, str]:
    return {
        "source_transcript": str(_one_artifact(run_dir, "*source*.srt")),
        "final_subtitles": str(_one_artifact(run_dir, "*subtitles.srt")),
        "video": str(_one_artifact(run_dir, "*translated.mp4")),
    }


def _one_artifact(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern} artifact under {run_dir}, found {len(matches)}.")
    return matches[0]


def _has_metrics(record: dict[str, Any]) -> bool:
    return all(_valid_metric(record.get(key)) for key in (
        "transcription_seconds",
        "end_to_end_seconds",
        "peak_memory_bytes",
    ))


def _valid_metric(value: Any) -> bool:
    return _nonnegative_number(value)


def _comparison_report(
    config: ComparisonConfig,
    runs: list[dict[str, Any]],
    warmups: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "comparison_variable": "transcription_backend",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {"model_checkpoint": config.model, "path": "manifest.json"},
        "runs": runs,
        "warmups": warmups,
        "summary": _timing_summary(runs),
    }


def _timing_summary(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for backend in (BASELINE_BACKEND, CANDIDATE_BACKEND):
        matching = [run for run in runs if run.get("backend") == backend and run.get("completed")]
        if not matching:
            continue
        summary[backend] = {
            "median_transcription_seconds": statistics.median(run["transcription_seconds"] for run in matching),
            "median_end_to_end_seconds": statistics.median(run["end_to_end_seconds"] for run in matching),
            "peak_memory_bytes": max(run["peak_memory_bytes"] for run in matching),
        }
    return summary


def _median_run(runs: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    matching = [run for run in runs if run["backend"] == backend and run["completed"]]
    median = statistics.median(run["transcription_seconds"] for run in matching)
    return min(matching, key=lambda run: (abs(run["transcription_seconds"] - median), run["iteration"]))


def _write_source_diff(baseline: Path, candidate: Path, target: Path) -> None:
    diff = difflib.unified_diff(
        baseline.read_text(encoding="utf-8").splitlines(keepends=True),
        candidate.read_text(encoding="utf-8").splitlines(keepends=True),
        fromfile="baseline/source_transcript.srt",
        tofile="candidate/source_transcript.srt",
    )
    target.write_text("".join(diff), encoding="utf-8")


def _write_review_composite(baseline: Path, candidate: Path, target: Path) -> None:
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(baseline),
            "-i",
            str(candidate),
            "-filter_complex",
            "[0:v]setpts=PTS-STARTPTS[baseline];"
            "[1:v]setpts=PTS-STARTPTS[candidate];"
            "[baseline][candidate]hstack=inputs=2[video];"
            "[0:a]asetpts=PTS-STARTPTS[audio]",
            "-map",
            "[video]",
            "-map",
            "[audio]",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(target),
        ]
    )


def _run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def _write_review(output_dir: Path, report: dict[str, Any], diff_path: Path, composite_path: Path) -> None:
    summary = report["summary"]
    timing_rows = []
    for backend in (BASELINE_BACKEND, CANDIDATE_BACKEND):
        values = summary[backend]
        timing_rows.append(
            f"| {backend} | {values['median_transcription_seconds']:.3f} | "
            f"{values['median_end_to_end_seconds']:.3f} | {values['peak_memory_bytes']} |"
        )
    checklist = "\n".join(f"- Candidate-only {check}: PENDING" for check in REVIEW_CHECKS)
    (output_dir / "review.md").write_text(
        "\n".join(
            [
                "# ASR Comparison Review",
                "",
                "## Manifests",
                "- Common manifest: `manifest.json`",
                "- Baseline manifest: `baseline_manifest.json`",
                "- Candidate manifest: `candidate_manifest.json`",
                "",
                "## Timings",
                "| Backend | Median transcription seconds | Median end-to-end seconds | Peak memory bytes |",
                "| --- | ---: | ---: | ---: |",
                *timing_rows,
                "",
                "## Review artifacts",
                f"- Source transcript diff: `{diff_path.name}`",
                f"- Unscaled synchronized Review Composite (shared Baseline audio): `{composite_path.name}`",
                "",
                "## Candidate-only regression checklist",
                checklist,
                "",
                "## Timestamped problems",
                "- PENDING: record each observed problem as `HH:MM:SS.mmm: description`, or replace with `none`.",
                "",
                "## Human Approval",
                "- Reviewer: PENDING",
                "- Date: PENDING",
                "- Decision: PENDING",
                "",
                "Do not mark this approved unless every checklist item is `no`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _child_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warm-cache", action="store_true")
    args = parser.parse_args(argv)
    payload = _read_json(args.config)
    cli_args = payload.get("cli_args")
    if not isinstance(cli_args, list) or not all(isinstance(value, str) for value in cli_args):
        raise ValueError("Comparison child config has invalid cli_args.")

    from . import pipeline
    from .cli import main as pipeline_main

    original = pipeline.transcribe_audio
    transcription_seconds: Optional[float] = None

    def timed_transcribe(*values: Any, **kwargs: Any) -> Any:
        nonlocal transcription_seconds
        started = time.perf_counter()
        try:
            return original(*values, **kwargs)
        finally:
            transcription_seconds = time.perf_counter() - started

    pipeline.transcribe_audio = timed_transcribe
    started = time.perf_counter()
    error: Optional[str] = None
    try:
        pipeline_main(cli_args)
    except Exception as exception:
        error = str(exception)
        raise
    finally:
        pipeline.transcribe_audio = original
        _write_json(
            args.metrics_file,
            {
                "transcription_seconds": transcription_seconds,
                "end_to_end_seconds": time.perf_counter() - started,
                "peak_memory_bytes": _peak_memory_bytes(),
                "error": error,
            },
        )
    return 0


def _peak_memory_bytes() -> Optional[int]:
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> ComparisonConfig:
    parser = argparse.ArgumentParser(description="Run a repeatable Whisper-versus-MLX comparison.")
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--model", default="large")
    parser.add_argument("--language")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reference-file", type=Path, default=Path("references/tripleS.md"))
    parser.add_argument("--locked-terms-file", type=Path, default=Path("references/locked_terms.json"))
    parser.add_argument("--translation-model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--target-language", default="Traditional Chinese (Taiwan)")
    parser.add_argument("--time-buffer", type=float, default=0.1)
    parser.add_argument("--font-name", default="Heiti TC")
    parser.add_argument("--fonts-dir")
    parser.add_argument("--font-size", type=int, default=12)
    parser.add_argument("--outline-width", type=int, default=0)
    parser.add_argument("--margin-v", type=int, default=20)
    parser.add_argument("--margin-h", type=int, default=10)
    parser.add_argument("--alignment", type=int, default=2)
    parser.add_argument("--stage-cooldown", type=int, default=0)
    args = parser.parse_args(argv)
    return ComparisonConfig(**vars(args))


def main(argv: Optional[Sequence[str]] = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "_run":
        return _child_main(values[1:])
    output_dir = run_comparison(parse_args(values))
    print(f"Comparison evidence written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
