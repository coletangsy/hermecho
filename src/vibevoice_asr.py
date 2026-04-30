"""
Experimental VibeVoice-ASR adapter.

This module is intentionally isolated from the normal Hermecho CLI path. It
keeps optional VibeVoice dependencies out of ``requirements.txt`` while making
the model output consumable by the existing translation/timing/burn pipeline.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Tuple

from models import srt_to_seconds


DEFAULT_VIBEVOICE_MODEL = "microsoft/VibeVoice-ASR-HF"
DEFAULT_VIBEVOICE_GRADIO_MAX_NEW_TOKENS = 4096
DEFAULT_VIBEVOICE_GRADIO_TEMPERATURE = 0.0
DEFAULT_VIBEVOICE_GRADIO_TOP_P = 1.0
DEFAULT_VIBEVOICE_GRADIO_HTTP_TIMEOUT = 300.0


def import_transformers_vibevoice() -> Tuple[Any, Any, Any]:
    """Import optional VibeVoice Transformers symbols lazily."""
    import torch  # type: ignore
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration  # type: ignore

    return AutoProcessor, VibeVoiceAsrForConditionalGeneration, torch


def import_gradio_client() -> Tuple[Any, Any]:
    """Import optional Gradio client symbols lazily."""
    from gradio_client import Client  # type: ignore

    try:
        from gradio_client import handle_file  # type: ignore
    except ImportError:
        handle_file = None
    return Client, handle_file


def _lookup_case_insensitive(item: Dict[str, Any], *names: str) -> Any:
    lowered = {str(k).lower(): v for k, v in item.items()}
    for name in names:
        if name in item:
            return item[name]
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def _coerce_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return srt_to_seconds(text)
    except ValueError:
        return None


def _iter_candidate_rows(raw: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(raw, dict):
        return

    for key in (
        "segments",
        "transcription",
        "transcriptions",
        "results",
        "chunks",
        "utterances",
    ):
        val = _lookup_case_insensitive(raw, key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    yield item
            return

    if any(str(k).lower() in {"start", "end", "content", "text"} for k in raw):
        yield raw


def normalize_vibevoice_output(raw: Any) -> List[Dict[str, Any]]:
    """
    Convert VibeVoice structured output to Hermecho segment dicts.

    Transformers currently decodes VibeVoice-ASR parsed output as a list of
    dicts like ``{"Start": 0, "End": 15.43, "Speaker": 0, "Content": "..."}``.
    The normalizer also accepts common lowercase/snake_case variants so saved
    JSON from wrappers can be reused.
    """
    segments: List[Dict[str, Any]] = []
    for item in _iter_candidate_rows(raw):
        timestamp = _lookup_case_insensitive(item, "timestamp", "timestamps")
        start: Optional[float] = None
        end: Optional[float] = None
        if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
            start = _coerce_seconds(timestamp[0])
            end = _coerce_seconds(timestamp[1])

        if start is None:
            start = _coerce_seconds(
                _lookup_case_insensitive(item, "start", "start_time", "begin")
            )
        if end is None:
            end = _coerce_seconds(
                _lookup_case_insensitive(item, "end", "end_time", "finish")
            )

        text = _lookup_case_insensitive(
            item,
            "text",
            "content",
            "sentence",
            "transcript",
            "transcription",
        )
        text = "" if text is None else str(text).strip()

        if start is None or end is None or not text:
            continue
        if end < start:
            start, end = end, start
        segments.append({"start": start, "end": end, "text": text})

    segments.sort(key=lambda seg: (seg["start"], seg["end"]))
    return segments


def parse_vibevoice_json_text(text: str) -> Any:
    """Parse raw VibeVoice JSON text, trimming chat special tokens if present."""
    stripped = text.strip()
    if "<|im_start|>assistant" in stripped:
        stripped = stripped.split("<|im_start|>assistant", 1)[1]
    if "<|im_end|>" in stripped:
        stripped = stripped.split("<|im_end|>", 1)[0]
    if "<|endoftext|>" in stripped:
        stripped = stripped.split("<|endoftext|>", 1)[0]
    return json.loads(stripped.strip())


def parse_srt_text(text: str) -> List[Dict[str, Any]]:
    """Parse SRT text into Hermecho segment dictionaries."""
    segments: List[Dict[str, Any]] = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for block in normalized.split("\n\n"):
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

        start = _coerce_seconds(start_text)
        end = _coerce_seconds(end_text)
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        segments.append({"start": start, "end": end, "text": body})
    return segments


def normalize_vibevoice_text_output(text: str) -> List[Dict[str, Any]]:
    """Normalize raw Gradio textbox output as JSON first, then SRT."""
    if not text.strip():
        return []
    try:
        return normalize_vibevoice_output(parse_vibevoice_json_text(text))
    except (json.JSONDecodeError, TypeError, ValueError):
        return parse_srt_text(text)


def validate_vibevoice_gradio_url(gradio_url: str, timeout_seconds: float = 15.0) -> Dict[str, Any]:
    """
    Validate the hosted Gradio app exposes the expected public API.

    Returns the dependency metadata for ``transcribe_wrapper`` so callers can
    persist exact endpoint details in experiment reports.
    """
    base_url = gradio_url.rstrip("/")
    config_url = f"{base_url}/config"
    with urllib.request.urlopen(config_url, timeout=timeout_seconds) as response:
        config = json.load(response)

    for dependency in config.get("dependencies", []):
        if dependency.get("api_name") == "transcribe_wrapper":
            return {
                "url": gradio_url,
                "config_url": config_url,
                "gradio_version": config.get("version"),
                "api_name": "transcribe_wrapper",
                "input_count": len(dependency.get("inputs", [])),
                "output_count": len(dependency.get("outputs", [])),
            }
    raise RuntimeError(f"{config_url} does not expose public API transcribe_wrapper")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _read_srt_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as fh:
        return parse_srt_text(fh.read())


def _extract_file_path(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value if os.path.exists(value) else None
    if isinstance(value, dict):
        for key in ("path", "name"):
            path = value.get(key)
            if isinstance(path, str) and os.path.exists(path):
                return path
        for child in value.values():
            path = _extract_file_path(child)
            if path:
                return path
    if isinstance(value, (list, tuple)):
        for child in value:
            path = _extract_file_path(child)
            if path:
                return path
    return None


def _preview_value_to_upload(value: Any, handle_file: Any) -> Any:
    if isinstance(value, dict) and value.get("__type__") == "update":
        value = value.get("value")
    if isinstance(value, str) and os.path.exists(value):
        return handle_file(value) if handle_file else value
    return value


def transcribe_audio_vibevoice(
    audio_path: str,
    model_id: str = DEFAULT_VIBEVOICE_MODEL,
    prompt: Optional[str] = None,
    device: str = "auto",
    dtype: str = "auto",
    tokenizer_chunk_size: Optional[int] = None,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Run VibeVoice-ASR locally with optional dependencies.

    Raises ImportError/RuntimeError from the underlying stack so the experiment
    runner can capture exact blocker details in its report.
    """
    AutoProcessor, VibeVoiceAsrForConditionalGeneration, _torch = (
        import_transformers_vibevoice()
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model_kwargs: Dict[str, Any] = {}
    if device != "default":
        model_kwargs["device_map"] = device
    if dtype != "default":
        model_kwargs["torch_dtype"] = dtype
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        model_id,
        **model_kwargs,
    )
    inputs = processor.apply_transcription_request(
        audio=audio_path,
        prompt=prompt,
    ).to(model.device, model.dtype)
    generate_kwargs: Dict[str, Any] = {}
    if tokenizer_chunk_size is not None:
        generate_kwargs["acoustic_tokenizer_chunk_size"] = tokenizer_chunk_size

    output_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    parsed = processor.decode(generated_ids, return_format="parsed")[0]
    return parsed, normalize_vibevoice_output(parsed)


def transcribe_audio_vibevoice_gradio(
    audio_path: str,
    gradio_url: str,
    hotwords_context: Optional[str] = None,
    max_new_tokens: int = DEFAULT_VIBEVOICE_GRADIO_MAX_NEW_TOKENS,
    enable_sampling: bool = False,
    temperature: float = DEFAULT_VIBEVOICE_GRADIO_TEMPERATURE,
    top_p: float = DEFAULT_VIBEVOICE_GRADIO_TOP_P,
    http_timeout_seconds: float = DEFAULT_VIBEVOICE_GRADIO_HTTP_TIMEOUT,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[str]]:
    """
    Run VibeVoice-ASR through the hosted Gradio playground.

    The public API currently returns raw textbox text plus an SRT download. The
    downloaded SRT is preferred for segment timing; if it is unavailable, the
    raw textbox is normalized as JSON or SRT text.
    """
    endpoint_info = validate_vibevoice_gradio_url(gradio_url)
    Client, handle_file = import_gradio_client()
    client = Client(gradio_url, httpx_kwargs={"timeout": http_timeout_seconds})
    upload_value = handle_file(audio_path) if handle_file else audio_path
    settings = {
        "max_new_tokens": max_new_tokens,
        "enable_sampling": enable_sampling,
        "temperature": temperature,
        "top_p": top_p,
        "http_timeout_seconds": http_timeout_seconds,
        "hotwords_context": hotwords_context or "",
    }

    audio_preview = None
    video_preview = None
    try:
        preview_result = client.predict(upload_value, api_name="/update_media_preview")
    except Exception:
        preview_result = None
    if isinstance(preview_result, (list, tuple)):
        if len(preview_result) > 0:
            audio_preview = _preview_value_to_upload(preview_result[0], handle_file)
        if len(preview_result) > 1:
            video_preview = _preview_value_to_upload(preview_result[1], handle_file)

    result = client.predict(
        upload_value,
        None,
        None,
        audio_preview,
        video_preview,
        max_new_tokens,
        temperature,
        top_p,
        enable_sampling,
        hotwords_context or "",
        api_name="/transcribe_wrapper",
    )

    if not isinstance(result, (str, bytes, dict, list, tuple)) and hasattr(result, "__iter__"):
        final_result = None
        for final_result in result:
            pass
        result = final_result

    raw_text = ""
    srt_result: Any = None
    if isinstance(result, (list, tuple)):
        if result:
            raw_text = "" if result[0] is None else str(result[0])
        if len(result) > 3:
            srt_result = result[3]
    else:
        raw_text = "" if result is None else str(result)

    srt_path = _extract_file_path(srt_result)
    segments = _read_srt_file(srt_path) if srt_path else normalize_vibevoice_text_output(raw_text)
    raw = {
        "provider": "gradio",
        "endpoint": endpoint_info,
        "api_settings": settings,
        "raw_text": raw_text,
        "srt_path": srt_path,
        "preview_result": _json_safe(preview_result),
        "result": _json_safe(result),
    }
    return raw, segments, srt_path
