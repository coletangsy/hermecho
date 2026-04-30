"""
Experimental Gemini vs VibeVoice-ASR comparison runner.

This script is not part of the default CLI. It produces review artifacts under
``output/<video_name>/asr_comparison_<timestamp>/`` while reusing the existing
Hermecho transcription, translation, timing-review, and burn-in stages.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import traceback
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from models import srt_to_seconds
from subtitles import fill_transcription_gaps, generate_srt, split_long_segments
from timing_review import review_subtitle_timing
from transcription import (
    DEFAULT_MULTIMODAL_CHUNK_SECONDS,
    DEFAULT_MULTIMODAL_MODEL,
    transcribe_audio,
    transcribe_audio_multimodal,
)
from translation import translate_segments
from utils import extract_keywords_for_whisper, load_reference_material
from vibevoice_asr import (
    DEFAULT_VIBEVOICE_GRADIO_MAX_NEW_TOKENS,
    DEFAULT_VIBEVOICE_GRADIO_HTTP_TIMEOUT,
    DEFAULT_VIBEVOICE_GRADIO_TEMPERATURE,
    DEFAULT_VIBEVOICE_GRADIO_TOP_P,
    DEFAULT_VIBEVOICE_MODEL,
    transcribe_audio_vibevoice,
    transcribe_audio_vibevoice_gradio,
    validate_vibevoice_gradio_url,
)
from video_processing import burn_subtitles_into_video, extract_audio


class VibeVoiceTimeoutError(RuntimeError):
    """Raised when the local VibeVoice experiment exceeds its time budget."""


def _run_ffprobe_json(video_path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name",
        "-of",
        "json",
        video_path,
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def get_media_metadata(video_path: str) -> Dict[str, Any]:
    raw = _run_ffprobe_json(video_path)
    streams = raw.get("streams", [])
    audio_codec = ""
    video_codec = ""
    for stream in streams:
        if stream.get("codec_type") == "audio" and not audio_codec:
            audio_codec = str(stream.get("codec_name", ""))
        if stream.get("codec_type") == "video" and not video_codec:
            video_codec = str(stream.get("codec_name", ""))
    duration = float(raw.get("format", {}).get("duration", 0.0) or 0.0)
    return {
        "duration": duration,
        "audio_codec": audio_codec,
        "video_codec": video_codec,
    }


def extract_gradio_upload_audio(video_path: str, output_dir: str) -> str:
    """Extract compact mono 16 kHz FLAC audio for hosted Gradio upload."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{base_name}_vibevoice_upload.flac")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "flac",
        audio_path,
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return audio_path


def _system_memory_gb() -> Optional[float]:
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        for line in out.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("Memory:"):
                value = stripped.split(":", 1)[1].strip()
                parts = value.split()
                try:
                    amount = float(parts[0])
                except (IndexError, ValueError):
                    return None
                unit = parts[1].upper() if len(parts) > 1 else "GB"
                if unit.startswith("TB"):
                    return amount * 1024
                if unit.startswith("MB"):
                    return amount / 1024
                return amount
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / (1024 ** 3), 2)
    except (AttributeError, OSError, ValueError):
        return None


def collect_vibevoice_preflight(
    model_id: str,
    device: str,
    dtype: str,
    tokenizer_chunk_size: Optional[int],
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "tokenizer_chunk_size": tokenizer_chunk_size,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "system_memory_gb": _system_memory_gb(),
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "mps_available": False,
    }
    try:
        import torch  # type: ignore
    except Exception:
        return info

    info["torch_available"] = True
    info["torch_version"] = getattr(torch, "__version__", None)
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        info["cuda_available"] = False
    try:
        mps = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(mps and mps.is_available())
    except Exception:
        info["mps_available"] = False
    return info


def compute_segment_stats(
    segments: List[Dict[str, Any]],
    media_duration: float,
) -> Dict[str, Any]:
    durations = [
        max(0.0, float(seg["end"]) - float(seg["start"]))
        for seg in segments
        if "start" in seg and "end" in seg
    ]
    coverage = round(sum(durations), 3)
    return {
        "segment_count": len(segments),
        "coverage_seconds": coverage,
        "coverage_percent": round((coverage / media_duration) * 100, 2)
        if media_duration > 0
        else 0.0,
        "average_segment_duration": round(sum(durations) / len(durations), 3)
        if durations
        else 0.0,
    }


def _overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return max(0.0, min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"])))


def _format_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02}:{m:02}:{s:05.2f}"


def _best_overlap(
    segment: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    best = None
    best_overlap = 0.0
    for candidate in candidates:
        amount = _overlap(segment, candidate)
        if amount > best_overlap:
            best = candidate
            best_overlap = amount
    return best


def _text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _comparison_rows(
    gemini_segments: List[Dict[str, Any]],
    vibevoice_segments: List[Dict[str, Any]],
    whisper_segments: Optional[List[Dict[str, Any]]] = None,
    whisper_no_prompt_segments: Optional[List[Dict[str, Any]]] = None,
    limit: int = 16,
) -> List[str]:
    whisper_segments = whisper_segments or []
    whisper_no_prompt_segments = whisper_no_prompt_segments or []
    rows = [
        (
            "| Time | Gemini source | VibeVoice source | Whisper source "
            "| Whisper (no prompt) source | VibeVoice similarity | Whisper similarity "
            "| Whisper (no prompt) similarity |"
        )
        if whisper_no_prompt_segments
        else "| Time | Gemini source | VibeVoice source | Whisper source | VibeVoice similarity | Whisper similarity |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
        if whisper_no_prompt_segments
        else "| --- | --- | --- | --- | --- | --- |",
    ]
    if not gemini_segments:
        return rows

    step = max(1, len(gemini_segments) // limit)
    sampled = gemini_segments[0::step][:limit]
    for seg in sampled:
        vibe_match = _best_overlap(seg, vibevoice_segments) if vibevoice_segments else None
        whisper_match = _best_overlap(seg, whisper_segments) if whisper_segments else None
        whisper_no_prompt_match = (
            _best_overlap(seg, whisper_no_prompt_segments)
            if whisper_no_prompt_segments
            else None
        )
        g_text = str(seg.get("text", "")).replace("|", "\\|")
        v_text = str(vibe_match.get("text", "") if vibe_match else "").replace("|", "\\|")
        w_text = str(whisper_match.get("text", "") if whisper_match else "").replace("|", "\\|")
        wnp_text = str(
            whisper_no_prompt_match.get("text", "") if whisper_no_prompt_match else ""
        ).replace("|", "\\|")
        v_sim = _text_similarity(g_text, v_text) if vibe_match else 0.0
        w_sim = _text_similarity(g_text, w_text) if whisper_match else 0.0
        wnp_sim = (
            _text_similarity(g_text, wnp_text) if whisper_no_prompt_match else 0.0
        )
        if whisper_no_prompt_segments:
            rows.append(
                f"| {_format_time(float(seg['start']))}-{_format_time(float(seg['end']))} "
                f"| {g_text[:120]} | {v_text[:120]} | {w_text[:120]} | {wnp_text[:120]} "
                f"| {v_sim:.2f} | {w_sim:.2f} | {wnp_sim:.2f} |"
            )
        else:
            rows.append(
                f"| {_format_time(float(seg['start']))}-{_format_time(float(seg['end']))} "
                f"| {g_text[:120]} | {v_text[:120]} | {w_text[:120]} | {v_sim:.2f} | {w_sim:.2f} |"
            )
    return rows


def _hotspot_rows(
    gemini_segments: List[Dict[str, Any]],
    vibevoice_segments: List[Dict[str, Any]],
    whisper_segments: Optional[List[Dict[str, Any]]] = None,
    whisper_no_prompt_segments: Optional[List[Dict[str, Any]]] = None,
    limit: int = 12,
) -> List[str]:
    whisper_segments = whisper_segments or []
    whisper_no_prompt_segments = whisper_no_prompt_segments or []
    scored: List[Dict[str, Any]] = []
    for seg in gemini_segments:
        vibe_match = _best_overlap(seg, vibevoice_segments)
        whisper_match = _best_overlap(seg, whisper_segments)
        whisper_no_prompt_match = _best_overlap(seg, whisper_no_prompt_segments)
        scores = []
        if vibe_match:
            scores.append(_text_similarity(str(seg.get("text", "")), str(vibe_match.get("text", ""))))
        else:
            scores.append(0.0)
        if whisper_match:
            scores.append(_text_similarity(str(seg.get("text", "")), str(whisper_match.get("text", ""))))
        elif whisper_segments:
            scores.append(0.0)
        if whisper_no_prompt_match:
            scores.append(
                _text_similarity(
                    str(seg.get("text", "")),
                    str(whisper_no_prompt_match.get("text", "")),
                )
            )
        elif whisper_no_prompt_segments:
            scores.append(0.0)
        scored.append(
            {
                "segment": seg,
                "vibe_match": vibe_match,
                "whisper_match": whisper_match,
                "whisper_no_prompt_match": whisper_no_prompt_match,
                "score": min(scores) if scores else 0.0,
            }
        )
    scored.sort(key=lambda row: row["score"])

    rows = [
        "| Time | Gemini | VibeVoice | Whisper | Whisper (no prompt) | Lowest similarity |"
        if whisper_no_prompt_segments
        else "| Time | Gemini | VibeVoice | Whisper | Lowest similarity |",
        "| --- | --- | --- | --- | --- | --- |"
        if whisper_no_prompt_segments
        else "| --- | --- | --- | --- | --- |",
    ]
    for row in scored[:limit]:
        seg = row["segment"]
        vibe_match = row["vibe_match"]
        whisper_match = row["whisper_match"]
        whisper_no_prompt_match = row["whisper_no_prompt_match"]
        g_text = str(seg.get("text", "")).replace("|", "\\|")
        v_text = str(vibe_match.get("text", "") if vibe_match else "").replace("|", "\\|")
        w_text = str(whisper_match.get("text", "") if whisper_match else "").replace("|", "\\|")
        wnp_text = str(
            whisper_no_prompt_match.get("text", "") if whisper_no_prompt_match else ""
        ).replace("|", "\\|")
        if whisper_no_prompt_segments:
            rows.append(
                f"| {_format_time(float(seg['start']))}-{_format_time(float(seg['end']))} "
                f"| {g_text[:120]} | {v_text[:120]} | {w_text[:120]} | {wnp_text[:120]} "
                f"| {row['score']:.2f} |"
            )
        else:
            rows.append(
                f"| {_format_time(float(seg['start']))}-{_format_time(float(seg['end']))} "
                f"| {g_text[:120]} | {v_text[:120]} | {w_text[:120]} | {row['score']:.2f} |"
            )
    return rows


def _artifact_lines(artifacts: Dict[str, str]) -> List[str]:
    labels = {
        "gemini_baseline_dir": "Gemini baseline directory",
        "gemini_source_srt": "Gemini source SRT",
        "gemini_translated_srt": "Gemini translated SRT",
        "gemini_burned_video": "Gemini burned MP4",
        "vibevoice_raw_json": "VibeVoice raw JSON",
        "vibevoice_raw_text": "VibeVoice raw text",
        "vibevoice_upload_audio": "VibeVoice upload audio",
        "vibevoice_source_srt": "VibeVoice source SRT",
        "vibevoice_translated_srt": "VibeVoice translated SRT",
        "vibevoice_burned_video": "VibeVoice burned MP4",
        "whisper_source_srt": "Whisper source SRT",
        "whisper_translated_srt": "Whisper translated SRT",
        "whisper_burned_video": "Whisper burned MP4",
    }
    optional_labels = {
        "whisper_no_prompt_source_srt": "Whisper no-prompt source SRT",
        "whisper_no_prompt_translated_srt": "Whisper no-prompt translated SRT",
        "whisper_no_prompt_burned_video": "Whisper no-prompt burned MP4",
    }
    lines = []
    for key, label in labels.items():
        path = artifacts.get(key)
        status = path if path else "not generated"
        lines.append(f"- {label}: `{status}`")
    for key, label in optional_labels.items():
        path = artifacts.get(key)
        if path:
            lines.append(f"- {label}: `{path}`")
    return lines


def build_comparison_report(
    output_path: str,
    input_video: str,
    media_metadata: Dict[str, Any],
    gemini_segments: List[Dict[str, Any]],
    vibevoice_segments: List[Dict[str, Any]],
    artifacts: Dict[str, str],
    whisper_segments: Optional[List[Dict[str, Any]]] = None,
    whisper_no_prompt_segments: Optional[List[Dict[str, Any]]] = None,
    vibevoice_error: Optional[str] = None,
    vibevoice_preflight: Optional[Dict[str, Any]] = None,
    vibevoice_provider_info: Optional[Dict[str, Any]] = None,
) -> str:
    media_duration = float(media_metadata.get("duration", 0.0) or 0.0)
    whisper_segments = whisper_segments or []
    whisper_no_prompt_segments = whisper_no_prompt_segments or []
    gemini_stats = compute_segment_stats(gemini_segments, media_duration)
    vibevoice_stats = compute_segment_stats(vibevoice_segments, media_duration)
    whisper_stats = compute_segment_stats(whisper_segments, media_duration)
    whisper_no_prompt_stats = compute_segment_stats(
        whisper_no_prompt_segments,
        media_duration,
    )

    lines = [
        "# Gemini vs VibeVoice-ASR vs Whisper Comparison",
        "",
        "## Input Metadata",
        f"- Input: `{input_video}`",
        f"- Duration: `{media_duration:.2f}s`",
        f"- Audio codec: `{media_metadata.get('audio_codec', '')}`",
        f"- Video codec: `{media_metadata.get('video_codec', '')}`",
        "",
        "## Artifacts",
        *_artifact_lines(artifacts),
        "",
    ]

    if vibevoice_preflight:
        lines.extend(
            [
                "## VibeVoice Hardware Preflight",
                f"- Model: `{vibevoice_preflight.get('model_id')}`",
                f"- Device request: `{vibevoice_preflight.get('device')}`",
                f"- Dtype request: `{vibevoice_preflight.get('dtype')}`",
                f"- Tokenizer chunk size: `{vibevoice_preflight.get('tokenizer_chunk_size')}`",
                f"- Platform: `{vibevoice_preflight.get('platform')}`",
                f"- Machine: `{vibevoice_preflight.get('machine')}`",
                f"- Python: `{vibevoice_preflight.get('python_version')}`",
                f"- System memory: `{vibevoice_preflight.get('system_memory_gb')} GB`",
                f"- Torch available: `{vibevoice_preflight.get('torch_available')}`",
                f"- Torch version: `{vibevoice_preflight.get('torch_version')}`",
                f"- CUDA available: `{vibevoice_preflight.get('cuda_available')}`",
                f"- MPS available: `{vibevoice_preflight.get('mps_available')}`",
                "",
            ]
        )

    if vibevoice_provider_info:
        settings = vibevoice_provider_info.get("api_settings", {})
        lines.extend(
            [
                "## VibeVoice Provider",
                f"- Provider: `{vibevoice_provider_info.get('provider')}`",
                f"- Gradio URL: `{vibevoice_provider_info.get('gradio_url', '')}`",
                f"- Gradio version: `{vibevoice_provider_info.get('gradio_version', '')}`",
                f"- API name: `{vibevoice_provider_info.get('api_name', '')}`",
                f"- Upload audio: `{vibevoice_provider_info.get('upload_audio_path', '')}`",
                f"- Upload audio size: `{vibevoice_provider_info.get('upload_audio_size_mb', 0):.2f} MB`",
                f"- Max new tokens: `{settings.get('max_new_tokens')}`",
                f"- Enable sampling: `{settings.get('enable_sampling')}`",
                f"- Temperature: `{settings.get('temperature')}`",
                f"- Top-p: `{settings.get('top_p')}`",
                f"- HTTP timeout: `{settings.get('http_timeout_seconds')}`",
                "",
            ]
        )

    lines.extend(
        [
        "## Segment Stats",
        "| Engine | Segments | Coverage | Coverage % | Avg duration |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| Gemini | {gemini_stats['segment_count']} | "
            f"{gemini_stats['coverage_seconds']:.2f}s | "
            f"{gemini_stats['coverage_percent']:.2f}% | "
            f"{gemini_stats['average_segment_duration']:.2f}s |"
        ),
        (
            f"| VibeVoice-ASR | {vibevoice_stats['segment_count']} | "
            f"{vibevoice_stats['coverage_seconds']:.2f}s | "
            f"{vibevoice_stats['coverage_percent']:.2f}% | "
            f"{vibevoice_stats['average_segment_duration']:.2f}s |"
        ),
        (
            f"| Whisper | {whisper_stats['segment_count']} | "
            f"{whisper_stats['coverage_seconds']:.2f}s | "
            f"{whisper_stats['coverage_percent']:.2f}% | "
            f"{whisper_stats['average_segment_duration']:.2f}s |"
        ),
        *(
            [
                (
                    f"| Whisper (no prompt) | {whisper_no_prompt_stats['segment_count']} | "
                    f"{whisper_no_prompt_stats['coverage_seconds']:.2f}s | "
                    f"{whisper_no_prompt_stats['coverage_percent']:.2f}% | "
                    f"{whisper_no_prompt_stats['average_segment_duration']:.2f}s |"
                )
            ]
            if whisper_no_prompt_segments
            else []
        ),
        "",
        "## Transcript Comparison Excerpts",
        *_comparison_rows(
            gemini_segments,
            vibevoice_segments,
            whisper_segments,
            whisper_no_prompt_segments,
        ),
        "",
        "## Flagged Disagreement Hotspots",
        *_hotspot_rows(
            gemini_segments,
            vibevoice_segments,
            whisper_segments,
            whisper_no_prompt_segments,
        ),
        "",
        ]
    )
    if whisper_no_prompt_segments:
        lines.extend(
            [
                "## Overall Assessment",
                (
                    "Gemini remains the cleanest subtitle-ready baseline overall, "
                    "but it has lower timeline coverage than VibeVoice-ASR and both "
                    "Whisper variants."
                ),
                (
                    "VibeVoice-ASR covers the full timeline and captures music or "
                    "speaker context, but it still needs cleanup of labels such as "
                    "`[Speaker]`, `[Music]`, and `[Lyric]` before final subtitle use."
                ),
                (
                    "Prompted Whisper remains useful as a local/offline baseline, "
                    "but the reference prompt caused obvious repeated-name "
                    "hallucinations near the beginning, middle silence, and ending."
                ),
                (
                    "Whisper (no prompt) removes most of those prompt-driven "
                    "hallucinations and aligns better with Gemini in many spoken "
                    "sections, but it still has lower coverage and some end-of-video "
                    "misrecognitions."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Overall Assessment",
                (
                    "Gemini remains the cleanest subtitle-ready baseline overall, "
                    "VibeVoice-ASR provides broader timeline coverage with labels "
                    "that need cleanup, and Whisper is useful as a local/offline "
                    "cross-check."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Translation Review Notes",
            (
                "All completed paths use the same Hermecho translation and burn-in "
                "settings. Gemini may use multimodal timing review, while VibeVoice "
                "and Whisper translated subtitles preserve the source SRT timings."
            ),
            (
                "Different source segment boundaries can change translated subtitle "
                "readability even when the translation model and prompt are identical."
            ),
        ]
    )

    if vibevoice_error:
        lines.extend(
            [
                "",
                "## VibeVoice-ASR Blocker",
                "The VibeVoice-ASR attempt did not complete.",
                "",
                "```text",
                vibevoice_error.strip(),
                "```",
                "",
                "CUDA/container command to finish the VibeVoice side elsewhere:",
                "",
                "```bash",
                "docker run --gpus all --rm -it -v \"$PWD:/work\" -w /work pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime bash",
                "pip install 'transformers>=5.3.0' accelerate soundfile librosa",
                "PYTHONPATH=src python -m asr_comparison --skip-gemini --baseline-dir output/Wsp9Z6-S0LA/asr_comparison_20260430_103344 --video Wsp9Z6-S0LA.mp4",
                "```",
            ]
        )

    text = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def _prepare_source_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return fill_transcription_gaps(split_long_segments(segments))


def load_srt_segments(path: str) -> List[Dict[str, Any]]:
    """Load simple SRT files into Hermecho segment dictionaries."""
    with open(path, "r", encoding="utf-8-sig") as fh:
        text = fh.read()

    segments: List[Dict[str, Any]] = []
    for block in text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        timing_index = None
        for idx, line in enumerate(lines):
            if "-->" in line:
                timing_index = idx
                break
        if timing_index is None:
            continue

        start_text, end_text = [
            part.strip() for part in lines[timing_index].split("-->", 1)
        ]
        end_text = end_text.split()[0]
        body = " ".join(lines[timing_index + 1 :]).strip()
        if not body:
            continue

        try:
            start = srt_to_seconds(start_text)
            end = srt_to_seconds(end_text)
        except ValueError:
            continue
        if end < start:
            start, end = end, start
        segments.append({"start": start, "end": end, "text": body})

    return segments


def load_gemini_baseline(baseline_dir: str) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Load previously generated Gemini comparison artifacts."""
    baseline_dir = os.path.abspath(baseline_dir)
    expected = {
        "gemini_source_srt": "gemini_source.srt",
        "gemini_translated_srt": "gemini_translated.srt",
        "gemini_burned_video": "gemini_burned.mp4",
    }
    missing = [
        filename
        for filename in expected.values()
        if not os.path.exists(os.path.join(baseline_dir, filename))
    ]
    if missing:
        raise FileNotFoundError(
            f"Baseline directory {baseline_dir} is missing: {', '.join(missing)}"
        )

    artifacts = {"gemini_baseline_dir": baseline_dir}
    artifacts.update(
        {
            key: os.path.join(baseline_dir, filename)
            for key, filename in expected.items()
        }
    )
    return load_srt_segments(artifacts["gemini_source_srt"]), artifacts


def load_existing_engine_artifacts(
    baseline_dir: str,
    label: str,
    require_complete: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Load existing comparison artifacts for one engine when present."""
    baseline_dir = os.path.abspath(baseline_dir)
    expected = {
        f"{label}_source_srt": f"{label}_source.srt",
        f"{label}_translated_srt": f"{label}_translated.srt",
        f"{label}_burned_video": f"{label}_burned.mp4",
    }
    existing = {
        key: os.path.join(baseline_dir, filename)
        for key, filename in expected.items()
        if os.path.exists(os.path.join(baseline_dir, filename))
    }
    if require_complete:
        missing = [
            filename
            for filename in expected.values()
            if not os.path.exists(os.path.join(baseline_dir, filename))
        ]
        if missing:
            raise FileNotFoundError(
                f"Baseline directory {baseline_dir} is missing: {', '.join(missing)}"
            )
    source_path = existing.get(f"{label}_source_srt")
    segments = load_srt_segments(source_path) if source_path else []
    return segments, existing


def _write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2)


def _write_text(path: str, value: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(value)


def _run_with_timeout(func, timeout_seconds: Optional[float]) -> Any:
    if not timeout_seconds:
        return func()
    if not hasattr(signal, "SIGALRM"):
        return func()

    def _handle_timeout(_signum, _frame) -> None:
        raise VibeVoiceTimeoutError(
            f"VibeVoice-ASR exceeded timeout of {timeout_seconds:.0f} seconds."
        )

    previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return func()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _process_downstream(
    label: str,
    video_path: str,
    audio_path: str,
    source_segments: List[Dict[str, Any]],
    output_dir: str,
    reference_material: Optional[str],
    args: argparse.Namespace,
    apply_timing_review: bool = True,
) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    source_srt = os.path.join(output_dir, f"{label}_source.srt")
    generate_srt(source_segments, source_srt)
    artifacts[f"{label}_source_srt"] = source_srt

    translated = translate_segments(
        source_segments,
        target_language=args.target_language,
        translation_model=args.translation_model,
        reference_material=reference_material,
    )
    if not translated:
        return artifacts

    if apply_timing_review and args.timing_review:
        reviewed = review_subtitle_timing(
            audio_path,
            translated,
            chunk_seconds=args.timing_review_chunk_seconds,
            review_model=args.timing_review_model,
            source_language=args.language,
            target_language=args.target_language,
        )
        if reviewed is not None:
            translated = reviewed

    translated_srt = os.path.join(output_dir, f"{label}_translated.srt")
    generate_srt(translated, translated_srt)
    artifacts[f"{label}_translated_srt"] = translated_srt

    burned_video = os.path.join(output_dir, f"{label}_burned.mp4")
    burn_subtitles_into_video(
        video_path,
        os.path.abspath(translated_srt),
        os.path.abspath(burned_video),
        font_name=args.font_name,
        font_size=args.font_size,
        outline_width=args.outline_width,
        use_box_background=args.box_background,
        margin_v=args.margin_v,
        margin_h=args.margin_h,
        alignment=args.alignment,
    )
    if os.path.exists(burned_video):
        artifacts[f"{label}_burned_video"] = burned_video
    return artifacts


def _process_whisper(
    video_path: str,
    audio_path: str,
    output_dir: str,
    reference_material: Optional[str],
    prompt: Optional[str],
    args: argparse.Namespace,
    label: str = "whisper",
) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    raw_whisper = transcribe_audio(
        audio_path,
        model=args.whisper_model,
        language=args.language,
        initial_prompt=prompt,
        temperature=args.temperature,
    )
    if raw_whisper is None:
        return [], {}

    whisper_segments = _prepare_source_segments(raw_whisper)
    artifacts = _process_downstream(
        label,
        video_path,
        audio_path,
        whisper_segments,
        output_dir,
        reference_material,
        args,
        apply_timing_review=False,
    )
    return whisper_segments, artifacts


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an experimental Gemini vs VibeVoice-ASR vs Whisper full-output comparison."
    )
    parser.add_argument("--video", default="Wsp9Z6-S0LA.mp4")
    parser.add_argument("--input-dir", default="input")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--reference-file", default="references/tripleS.md")
    parser.add_argument("--language", default="ko")
    parser.add_argument("--target-language", default="Traditional Chinese (Taiwan)")
    parser.add_argument("--translation-model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--multimodal-model", default=DEFAULT_MULTIMODAL_MODEL)
    parser.add_argument("--multimodal-chunk-seconds", type=float, default=DEFAULT_MULTIMODAL_CHUNK_SECONDS)
    parser.add_argument("--vibevoice-model", default=DEFAULT_VIBEVOICE_MODEL)
    parser.add_argument("--vibevoice-provider", choices=["local", "gradio"], default="local")
    parser.add_argument("--vibevoice-gradio-url")
    parser.add_argument("--vibevoice-gradio-max-new-tokens", type=int, default=DEFAULT_VIBEVOICE_GRADIO_MAX_NEW_TOKENS)
    parser.add_argument("--vibevoice-gradio-enable-sampling", action="store_true")
    parser.add_argument("--vibevoice-gradio-temperature", type=float, default=DEFAULT_VIBEVOICE_GRADIO_TEMPERATURE)
    parser.add_argument("--vibevoice-gradio-top-p", type=float, default=DEFAULT_VIBEVOICE_GRADIO_TOP_P)
    parser.add_argument("--vibevoice-gradio-http-timeout", type=float, default=DEFAULT_VIBEVOICE_GRADIO_HTTP_TIMEOUT)
    parser.add_argument("--vibevoice-device", default="auto")
    parser.add_argument("--vibevoice-dtype", default="auto")
    parser.add_argument("--vibevoice-tokenizer-chunk-size", type=int)
    parser.add_argument(
        "--vibevoice-timeout-seconds",
        type=float,
        help="Abort the local VibeVoice attempt after this many seconds and write the blocker into the report.",
    )
    parser.add_argument(
        "--vibevoice-acoustic-tokenizer-chunk-size",
        dest="vibevoice_tokenizer_chunk_size",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--vibevoice-transcribe-only", action="store_true")
    parser.add_argument("--initial-prompt", default="This is a conversation in Korean and English.")
    parser.add_argument("--timing-review", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timing-review-model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--timing-review-chunk-seconds", type=float, default=120.0)
    parser.add_argument("--font-name", default="Arial Unicode MS")
    parser.add_argument("--font-size", type=int, default=12)
    parser.add_argument("--outline-width", type=int, default=0)
    parser.add_argument("--box-background", action="store_true", default=True)
    parser.add_argument("--margin-v", type=int, default=20)
    parser.add_argument("--margin-h", type=int, default=10)
    parser.add_argument("--alignment", type=int, default=2, choices=list(range(1, 10)))
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--skip-vibevoice", action="store_true")
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--include-whisper-no-prompt", action="store_true")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--baseline-dir",
        help=(
            "Existing asr_comparison output directory to reuse for cached engine "
            "artifacts and new Whisper outputs."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    load_dotenv()
    args = _parse_args()
    video_path = os.path.abspath(os.path.join(args.input_dir, args.video))
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        os.path.abspath(args.baseline_dir)
        if args.baseline_dir
        else os.path.join(args.output_dir, video_name, f"asr_comparison_{timestamp}")
    )
    os.makedirs(output_dir, exist_ok=True)

    metadata = get_media_metadata(video_path)
    audio_path = extract_audio(video_path)
    if not audio_path:
        raise SystemExit("Audio extraction failed.")

    artifacts: Dict[str, str] = {}
    gemini_segments: List[Dict[str, Any]] = []
    vibevoice_segments: List[Dict[str, Any]] = []
    whisper_segments: List[Dict[str, Any]] = []
    whisper_no_prompt_segments: List[Dict[str, Any]] = []
    vibevoice_error: Optional[str] = None
    vibevoice_provider_info: Optional[Dict[str, Any]] = None
    vibevoice_preflight = None
    if args.vibevoice_provider == "local":
        vibevoice_preflight = collect_vibevoice_preflight(
            model_id=args.vibevoice_model,
            device=args.vibevoice_device,
            dtype=args.vibevoice_dtype,
            tokenizer_chunk_size=args.vibevoice_tokenizer_chunk_size,
        )

    try:
        keywords = extract_keywords_for_whisper(args.reference_file)
        prompt = args.initial_prompt
        if keywords:
            prompt = f"{prompt} Context: {keywords}"
        reference_material = load_reference_material(args.reference_file)

        if args.baseline_dir:
            artifacts["gemini_baseline_dir"] = os.path.abspath(args.baseline_dir)
            for label in ("gemini", "vibevoice", "whisper", "whisper_no_prompt"):
                cached_segments, cached_artifacts = load_existing_engine_artifacts(
                    args.baseline_dir,
                    label,
                )
                if label == "gemini":
                    gemini_segments = cached_segments
                elif label == "vibevoice":
                    vibevoice_segments = cached_segments
                elif label == "whisper":
                    whisper_segments = cached_segments
                else:
                    whisper_no_prompt_segments = cached_segments
                artifacts.update(cached_artifacts)

        if not gemini_segments and not args.skip_gemini:
            raw_gemini = transcribe_audio_multimodal(
                audio_path,
                language=args.language,
                multimodal_model=args.multimodal_model,
                initial_prompt=prompt,
                chunk_seconds=args.multimodal_chunk_seconds,
            )
            if raw_gemini:
                gemini_segments = _prepare_source_segments(raw_gemini)
                artifacts.update(
                    _process_downstream(
                        "gemini",
                        video_path,
                        audio_path,
                        gemini_segments,
                        output_dir,
                        reference_material,
                        args,
                    )
                )

        if not vibevoice_segments and not args.skip_vibevoice:
            raw_path = os.path.join(output_dir, "vibevoice_raw.json")
            try:
                if args.vibevoice_provider == "gradio":
                    if not args.vibevoice_gradio_url:
                        raise ValueError("--vibevoice-gradio-url is required with --vibevoice-provider gradio")
                    gradio_endpoint = validate_vibevoice_gradio_url(args.vibevoice_gradio_url)
                    upload_audio_path = extract_gradio_upload_audio(video_path, output_dir)
                    artifacts["vibevoice_upload_audio"] = upload_audio_path
                    upload_size = os.path.getsize(upload_audio_path) / (1024 * 1024)
                    gradio_settings = {
                        "max_new_tokens": args.vibevoice_gradio_max_new_tokens,
                        "enable_sampling": args.vibevoice_gradio_enable_sampling,
                        "temperature": args.vibevoice_gradio_temperature,
                        "top_p": args.vibevoice_gradio_top_p,
                        "http_timeout_seconds": args.vibevoice_gradio_http_timeout,
                    }
                    vibevoice_provider_info = {
                        "provider": "gradio",
                        "gradio_url": args.vibevoice_gradio_url,
                        "gradio_version": gradio_endpoint.get("gradio_version"),
                        "api_name": gradio_endpoint.get("api_name"),
                        "upload_audio_path": upload_audio_path,
                        "upload_audio_size_mb": upload_size,
                        "api_settings": gradio_settings,
                    }
                    raw_vibevoice, normalized, srt_download = _run_with_timeout(
                        lambda: transcribe_audio_vibevoice_gradio(
                            upload_audio_path,
                            gradio_url=args.vibevoice_gradio_url,
                            hotwords_context=keywords,
                            max_new_tokens=gradio_settings["max_new_tokens"],
                            enable_sampling=gradio_settings["enable_sampling"],
                            temperature=gradio_settings["temperature"],
                            top_p=gradio_settings["top_p"],
                            http_timeout_seconds=gradio_settings["http_timeout_seconds"],
                        ),
                        args.vibevoice_timeout_seconds,
                    )
                    raw_text = str(raw_vibevoice.get("raw_text", ""))
                    if raw_text:
                        raw_text_path = os.path.join(output_dir, "vibevoice_raw.txt")
                        _write_text(raw_text_path, raw_text)
                        artifacts["vibevoice_raw_text"] = raw_text_path
                    vibevoice_provider_info["api_settings"] = raw_vibevoice.get("api_settings", gradio_settings)
                else:
                    raw_vibevoice, normalized = _run_with_timeout(
                        lambda: transcribe_audio_vibevoice(
                            audio_path,
                            model_id=args.vibevoice_model,
                            prompt=prompt,
                            device=args.vibevoice_device,
                            dtype=args.vibevoice_dtype,
                            tokenizer_chunk_size=args.vibevoice_tokenizer_chunk_size,
                        ),
                        args.vibevoice_timeout_seconds,
                    )
                    srt_download = None
                _write_json(raw_path, raw_vibevoice)
                artifacts["vibevoice_raw_json"] = raw_path
                if normalized:
                    vibevoice_segments = _prepare_source_segments(normalized)
                    source_srt = os.path.join(output_dir, "vibevoice_source.srt")
                    if args.vibevoice_provider == "gradio" and srt_download:
                        shutil.copyfile(srt_download, source_srt)
                    else:
                        generate_srt(vibevoice_segments, source_srt)
                    artifacts["vibevoice_source_srt"] = source_srt
                    if not args.vibevoice_transcribe_only:
                        artifacts.update(
                            _process_downstream(
                                "vibevoice",
                                video_path,
                                audio_path,
                                vibevoice_segments,
                                output_dir,
                                reference_material,
                                args,
                                apply_timing_review=False,
                            )
                        )
            except Exception:
                vibevoice_error = traceback.format_exc()
                _write_json(raw_path, {"error": vibevoice_error})
                artifacts["vibevoice_raw_json"] = raw_path

        if not args.skip_whisper:
            if not whisper_segments:
                new_segments, new_artifacts = _process_whisper(
                    video_path=video_path,
                    audio_path=audio_path,
                    output_dir=output_dir,
                    reference_material=reference_material,
                    prompt=prompt,
                    args=args,
                )
                whisper_segments = new_segments
                artifacts.update(new_artifacts)
            if args.include_whisper_no_prompt and not whisper_no_prompt_segments:
                no_prompt_segments, no_prompt_artifacts = _process_whisper(
                    video_path=video_path,
                    audio_path=audio_path,
                    output_dir=output_dir,
                    reference_material=reference_material,
                    prompt=None,
                    args=args,
                    label="whisper_no_prompt",
                )
                whisper_no_prompt_segments = no_prompt_segments
                artifacts.update(no_prompt_artifacts)

        report_path = os.path.join(output_dir, "comparison_report.md")
        build_comparison_report(
            output_path=report_path,
            input_video=os.path.relpath(video_path),
            media_metadata=metadata,
            gemini_segments=gemini_segments,
            vibevoice_segments=vibevoice_segments,
            whisper_segments=whisper_segments,
            whisper_no_prompt_segments=whisper_no_prompt_segments,
            artifacts=artifacts,
            vibevoice_error=vibevoice_error,
            vibevoice_preflight=vibevoice_preflight,
            vibevoice_provider_info=vibevoice_provider_info,
        )
        print(f"Comparison output written to {output_dir}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    main()
