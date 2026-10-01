"""Compare timed OpenRouter transcripts with local Whisper on one video."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from .transcription import transcribe_audio


TRANSCRIPTION_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
MODELS = (
    "google/gemini-3.5-transcribe",
    "microsoft/mai-transcribe-2",
)
CHUNK_SECONDS = 60
OVERLAP_SECONDS = 1
PROBE_START = 20
PROBE_SECONDS = 30
REVIEW_WINDOWS = ((25, 55), (90, 120), (210, 240), (345, 375), (430, 460))


class CostUnknownError(RuntimeError):
    """A request may have been billed, but its cost was not reported."""


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid media duration: {path}")
    return duration


def _extract_audio(source: Path, destination: Path, start: float = 0,
                   duration: float | None = None) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        command.extend(["-ss", str(start)])
    command.extend(["-i", str(source), "-map", "0:a:0", "-vn"])
    if duration is not None:
        command.extend(["-t", str(duration)])
    command.extend(["-ac", "1", "-ar", "16000", "-codec:a", "libmp3lame",
                    "-b:a", "64k", str(destination)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Audio extraction produced no data: {destination}")


def _normalise_words(response: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    raw_words = response.get("words")
    if raw_words is None and isinstance(response.get("segments"), list):
        raw_words = []
        for segment in response["segments"]:
            if not isinstance(segment, dict) or not isinstance(segment.get("words"), list):
                raise ValueError("Response contains a segment without word-level timestamps")
            raw_words.extend(segment["words"])
    if not isinstance(raw_words, list) or not raw_words:
        raise ValueError("Response has no word-level timestamps")
    words = []
    previous_end = 0.0
    for raw in raw_words:
        if not isinstance(raw, dict) or not isinstance(raw.get("word"), str) or not raw["word"].strip():
            raise ValueError("Response contains a malformed word")
        try:
            if isinstance(raw["start"], bool) or isinstance(raw["end"], bool):
                raise ValueError("Boolean timestamps are not numeric evidence")
            start, end = float(raw["start"]), float(raw["end"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("Response contains a word without numeric timestamps") from error
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0
                or start > end or end > duration + 2 or start < previous_end):
            raise ValueError("Response contains invalid or unordered word timestamps")
        words.append({"word": raw["word"], "start": start, "end": end})
        previous_end = end
    return words


def _request_transcription(audio_path: Path, model: str, api_key: str,
                           duration: float) -> dict[str, Any]:
    payload = {
        "model": model,
        "input_audio": {
            "data": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            "format": "mp3",
        },
        "language": "ko",
        "response_format": "verbose_json",
        "timestamp_granularities": ["word"],
    }
    request = Request(
        TRANSCRIPTION_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=120) as response:
            body = json.load(response)
            generation_id = response.headers.get("X-Generation-Id")
    except HTTPError as error:
        detail = error.read(1000).decode("utf-8", errors="replace")
        if error.code >= 500:
            raise CostUnknownError(f"OpenRouter HTTP {error.code}: {detail}") from error
        raise RuntimeError(f"OpenRouter HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise CostUnknownError(f"OpenRouter connection failed: {error}") from error
    except json.JSONDecodeError as error:
        raise CostUnknownError("OpenRouter returned invalid JSON; cost is unknown") from error
    if not isinstance(body, dict):
        raise CostUnknownError("OpenRouter returned a non-object response; cost is unknown")
    usage = body.get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        raise CostUnknownError("OpenRouter response has no usable usage.cost")
    result = {
        "model": model,
        "text": body.get("text"),
        "usage": usage,
        "cost_usd": float(cost),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "generation_id": generation_id,
    }
    try:
        result["words"] = _normalise_words(body, duration)
    except ValueError as error:
        result["words"] = []
        result["word_error"] = str(error)
    return result


def _local_words(segments: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    return _normalise_words({"segments": segments}, duration)


def _word_text(words: list[dict[str, Any]], start: float, end: float) -> str:
    selected = [entry["word"].strip() for entry in words
                if start <= (entry["start"] + entry["end"]) / 2 < end]
    return " ".join(selected)


def _timing_summary(words: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = []
    zero_duration = []
    long_words = []
    for word in words:
        detail = {"start": round(word["start"], 3),
                  "end": round(word["end"], 3), "word": word["word"]}
        if word["start"] == word["end"]:
            zero_duration.append(detail)
        if word["end"] - word["start"] > 3:
            long_words.append(detail)
    for left, right in zip(words, words[1:]):
        gap = right["start"] - left["end"]
        if gap >= 5:
            gaps.append({"start": round(left["end"], 3),
                         "end": round(right["start"], 3)})
    return {"timed_words": len(words), "first_word_second": words[0]["start"] if words else None,
            "last_word_second": words[-1]["end"] if words else None, "gaps_at_least_5_seconds": gaps,
            "zero_duration_words": zero_duration, "words_over_3_seconds": long_words}


def _review_document(output_dir: Path, duration: float,
                     results: dict[str, dict[str, Any]]) -> None:
    lines = ["# ASR evaluation review", "",
             "Listen to each audio clip while checking omissions, repetitions, names,",
             "hallucinations, and word timing. Previous model output is not ground truth.", ""]
    for index, (start, end) in enumerate(REVIEW_WINDOWS, 1):
        if end > duration:
            continue
        clip = output_dir / f"review_{index:02d}.mp3"
        _extract_audio(output_dir / "audio.mp3", clip, start, end - start)
        lines.extend([f"## {start:06.1f}–{end:06.1f} seconds", "",
                      f"Audio: `{clip.name}`", ""])
        for name, result in results.items():
            words = result.get("words")
            if isinstance(words, list) and words:
                lines.extend([f"**{name}**: {_word_text(words, start, end)}", ""])
        lines.extend(["Reviewer: PENDING", "Findings: PENDING", ""])
    (output_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(video_path: Path, output_dir: Path, max_cost_usd: float) -> dict[str, Any]:
    """Run one isolated evaluation; preserve partial artifacts on failure."""
    from .openrouter_transcription import TRANSCRIPTION_ASSEMBLY_RULES, _words_for_output

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    if not math.isfinite(max_cost_usd) or max_cost_usd <= 0:
        raise ValueError("--max-cost-usd must be a positive finite number")
    duration = _ffprobe_duration(video_path)
    if duration < max(end for _, end in REVIEW_WINDOWS):
        raise ValueError("Video is too short for the fixed review windows")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "audio.mp3"
    _extract_audio(video_path, audio_path)
    audio_duration = _ffprobe_duration(audio_path)
    with video_path.open("rb") as source_file:
        source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
    _write_json(output_dir / "manifest.json", {
        "source": str(video_path.resolve()), "source_sha256": source_hash,
        "duration_seconds": duration, "audio_seconds": audio_duration,
        "language": "ko", "prompt": None, "baseline": "whisper:large",
        "candidates": MODELS, "chunk_seconds": CHUNK_SECONDS,
        "assembly_rules": TRANSCRIPTION_ASSEMBLY_RULES,
        "overlap_seconds": OVERLAP_SECONDS, "probe": [PROBE_START, PROBE_SECONDS],
        "review_window_seconds": REVIEW_WINDOWS,
        "max_cost_usd": max_cost_usd,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    spent = 0.0
    results: dict[str, dict[str, Any]] = {}
    eligible = []
    probe_costs: dict[str, float] = {}
    cost_unknown = False
    probe_path = output_dir / "probe.mp3"
    _extract_audio(audio_path, probe_path, PROBE_START, PROBE_SECONDS)
    for model in MODELS:
        name = model.replace("/", "_")
        if cost_unknown or spent >= max_cost_usd:
            results[model] = {"status": "cost_unknown" if cost_unknown else "budget_exhausted"}
            continue
        print(f"Probing word timestamps: {model}", flush=True)
        try:
            probe = _request_transcription(probe_path, model, api_key, PROBE_SECONDS)
        except CostUnknownError as error:
            cost_unknown = True
            results[model] = {"status": "cost_unknown", "error": str(error)}
        except (RuntimeError, ValueError) as error:
            results[model] = {"status": "probe_failed", "error": str(error)}
        else:
            spent += probe["cost_usd"]
            probe_costs[model] = probe["cost_usd"]
            _write_json(output_dir / f"{name}_probe.json", probe)
            if "word_error" in probe:
                results[model] = {"status": "probe_failed", "error": probe["word_error"],
                                  "probe_cost_usd": probe["cost_usd"]}
            else:
                eligible.append(model)
                results[model] = {"status": "probe_passed", "probe_cost_usd": probe["cost_usd"]}
        _write_json(output_dir / "progress.json", {"spent_usd": spent,
                                                    "cost_unknown": cost_unknown, "models": results})

    if eligible:
        print("Running local Whisper large baseline...", flush=True)
        started = time.monotonic()
        segments = transcribe_audio(str(audio_path), "large", "ko", backend="whisper")
        if segments is None:
            raise RuntimeError("Local Whisper baseline failed")
        baseline = {"model": "whisper:large", "words": _local_words(segments, audio_duration),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "elapsed_scope": "local transcription only", "cost_usd": 0.0}
        baseline["timing"] = _timing_summary(baseline["words"])
        results["whisper:large"] = baseline
        _write_json(output_dir / "whisper_large.json", baseline)

    for model in eligible:
        name = model.replace("/", "_")
        collected: list[dict[str, Any]] = []
        completed = []
        chunks = []
        for index, core_start in enumerate(range(0, math.ceil(audio_duration), CHUNK_SECONDS), 1):
            core_end = min(audio_duration, core_start + CHUNK_SECONDS)
            start = max(0.0, core_start - OVERLAP_SECONDS)
            end = min(audio_duration, core_end + OVERLAP_SECONDS)
            if cost_unknown or spent >= max_cost_usd:
                results[model] = {"status": "cost_unknown" if cost_unknown else "budget_exhausted",
                                  "chunks": chunks}
                break
            path = output_dir / f"chunk_{index:02d}.mp3"
            if not path.exists():
                _extract_audio(audio_path, path, start, end - start)
            print(f"Transcribing {model} chunk {index} ({start:.0f}–{end:.0f}s)", flush=True)
            try:
                response = _request_transcription(path, model, api_key, end - start)
            except CostUnknownError as error:
                cost_unknown = True
                results[model] = {"status": "cost_unknown", "chunk": index,
                                  "error": str(error), "chunks": chunks}
                break
            except (RuntimeError, ValueError) as error:
                results[model] = {"status": "chunk_failed", "chunk": index,
                                  "error": str(error), "chunks": chunks}
                break
            spent += response["cost_usd"]
            response["audio_start_second"] = start
            response["audio_end_second"] = end
            _write_json(output_dir / f"{name}_chunk_{index:02d}.json", response)
            if "word_error" in response:
                results[model] = {"status": "chunk_failed", "chunk": index,
                                  "error": response["word_error"], "chunks": chunks}
                break
            chunks.append({"index": index, "start": start, "end": end,
                           "cost_usd": response["cost_usd"],
                           "elapsed_seconds": response["elapsed_seconds"],
                           "words_kept": None})
            completed.append((
                {"start": start, "end": end, "core_start": core_start, "core_end": core_end},
                response,
            ))
            try:
                reconciled = _words_for_output(completed, audio_duration)
            except RuntimeError as error:
                results[model] = {"status": "chunk_failed", "chunk": index,
                                  "error": str(error), "chunks": chunks}
                break
            chunks[-1]["words_kept"] = len(reconciled) - len(collected)
            collected = reconciled
            results[model] = {"status": "running", "chunks": chunks}
            _write_json(output_dir / "progress.json", {"spent_usd": spent,
                                                        "cost_unknown": cost_unknown, "models": results})
        else:
            full_cost = sum(chunk["cost_usd"] for chunk in chunks)
            results[model] = {"status": "complete" if collected else "no_timed_words", "words": collected,
                              "chunks": chunks, "timing": _timing_summary(collected),
                              "probe_cost_usd": probe_costs[model],
                              "full_cost_usd": full_cost,
                              "cost_usd": probe_costs[model] + full_cost,
                              "elapsed_seconds": sum(chunk["elapsed_seconds"] for chunk in chunks),
                              "elapsed_scope": "full-video API requests only"}
            _write_json(output_dir / f"{name}.json", results[model])

    _write_json(output_dir / "progress.json", {"spent_usd": spent,
                                                "cost_unknown": cost_unknown, "models": results})
    _review_document(output_dir, audio_duration, results)
    report = {"source": str(video_path.resolve()), "spent_usd": spent,
              "cost_unknown": cost_unknown,
              "max_cost_usd": max_cost_usd, "models": results,
              "human_review": "PENDING"}
    _write_json(output_dir / "comparison.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    args = parser.parse_args(argv)
    report = run_evaluation(args.video_path, args.output_dir, args.max_cost_usd)
    print(f"Evaluation artifacts: {args.output_dir}")
    print(f"OpenRouter cost: US${report['spent_usd']:.4f}")
    for model, result in report["models"].items():
        print(f"{model}: {result['status'] if 'status' in result else 'complete'}")
    if report["cost_unknown"] or any(
        report["models"][model].get("status") != "complete" for model in MODELS
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
