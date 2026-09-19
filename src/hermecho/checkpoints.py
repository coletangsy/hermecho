"""Atomic, versioned checkpoints for resumable pipeline stages."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import unicodedata
from typing import Any, Dict, List, Optional, Sequence


CHECKPOINT_VERSION = 1


def fingerprint_data(value: Any) -> str:
    """Return a stable fingerprint for JSON-compatible stage inputs."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_file(path: str) -> str:
    """Return a content fingerprint without retaining the source audio."""
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _json_object_without_duplicates(pairs: List[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint key: {key}")
        result[key] = value
    return result


def _is_complete_transcription(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and set(record) == {"status", "fingerprint", "segments"}
        and record["status"] == "complete"
        and isinstance(record["fingerprint"], str)
        and bool(record["fingerprint"])
        and isinstance(record["segments"], list)
        and bool(record["segments"])
        and all(_is_transcription_segment(segment) for segment in record["segments"])
    )


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_transcription_segment(segment: Any) -> bool:
    return (
        isinstance(segment, dict)
        and isinstance(segment.get("text"), str)
        and _is_finite_number(segment.get("start"))
        and _is_finite_number(segment.get("end"))
        and segment["start"] <= segment["end"]
    )


def _is_accepted_chunk(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and set(record) == {"status", "fingerprint", "translations"}
        and record["status"] == "accepted"
        and isinstance(record["fingerprint"], str)
        and bool(record["fingerprint"])
        and isinstance(record["translations"], dict)
        and bool(record["translations"])
        and all(
            isinstance(translation_id, str)
            and isinstance(text, str)
            and bool(text.strip())
            for translation_id, text in record["translations"].items()
        )
    )


def _source_content_signature(value: str) -> str:
    return "".join(
        character
        for character in value
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _is_source_sentence(sentence: Any) -> bool:
    if not isinstance(sentence, dict):
        return False
    words = sentence.get("source_words")
    indices = sentence.get("source_word_indices")
    if (
        not isinstance(sentence.get("text"), str)
        or not sentence["text"].strip()
        or not _is_finite_number(sentence.get("start"))
        or not _is_finite_number(sentence.get("end"))
        or sentence["start"] > sentence["end"]
        or not isinstance(words, list)
        or not words
        or not isinstance(indices, list)
        or len(indices) != len(words)
        or any(type(index) is not int or index < 0 for index in indices)
        or any(right != left + 1 for left, right in zip(indices, indices[1:]))
    ):
        return False
    previous_end = None
    for word in words:
        if (
            not isinstance(word, dict)
            or not isinstance(word.get("word"), str)
            or not word["word"].strip()
            or not _is_finite_number(word.get("start"))
            or not _is_finite_number(word.get("end"))
            or word["start"] > word["end"]
        ):
            return False
        if previous_end is not None and word["start"] < previous_end:
            return False
        previous_end = word["end"]
    if sentence["start"] != words[0]["start"] or sentence["end"] != words[-1]["end"]:
        return False
    source_text = "".join(word["word"] for word in words)
    return bool(_source_content_signature(source_text)) and _source_content_signature(
        sentence["text"]
    ) == _source_content_signature(source_text)


def _is_source_grouping(record: Any) -> bool:
    if not (
        isinstance(record, dict)
        and set(record) == {"status", "fingerprint", "sentences"}
        and record["status"] == "accepted"
        and isinstance(record["fingerprint"], str)
        and bool(record["fingerprint"])
        and isinstance(record["sentences"], list)
        and bool(record["sentences"])
        and all(_is_source_sentence(sentence) for sentence in record["sentences"])
    ):
        return False
    next_index = 0
    for sentence in record["sentences"]:
        indices = sentence["source_word_indices"]
        if indices[0] != next_index:
            return False
        next_index = indices[-1] + 1
    return True


def _is_checkpoint_state(value: Any) -> bool:
    version = value.get("version") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or isinstance(version, bool)
        or version != CHECKPOINT_VERSION
    ):
        return False
    if set(value) - {"version", "transcription", "source_grouping", "translation"}:
        return False

    transcription = value.get("transcription")
    if transcription is not None and not _is_complete_transcription(transcription):
        return False

    source_grouping = value.get("source_grouping")
    if source_grouping is not None and not _is_source_grouping(source_grouping):
        return False

    translation = value.get("translation")
    if translation is None:
        return True
    if (
        not isinstance(translation, dict)
        or set(translation) != {"fingerprint", "chunks"}
        or not isinstance(translation["fingerprint"], str)
        or not translation["fingerprint"]
        or not isinstance(translation["chunks"], dict)
        or not translation["chunks"]
    ):
        return False
    return all(
        isinstance(chunk_index, str) and _is_accepted_chunk(chunk)
        for chunk_index, chunk in translation["chunks"].items()
    )


class CheckpointStore:
    """Keep one current, validated checkpoint set for one video."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as checkpoint:
                state = json.load(checkpoint, object_pairs_hook=_json_object_without_duplicates)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return {"version": CHECKPOINT_VERSION}
        return state if _is_checkpoint_state(state) else {"version": CHECKPOINT_VERSION}

    def _write_state(self, state: Dict[str, Any]) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(self.path)}.",
            suffix=".tmp",
            dir=directory,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
                json.dump(
                    state,
                    temporary_file,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def load_transcription(self, fingerprint: str) -> Optional[List[Dict]]:
        record = self._state.get("transcription")
        if not _is_complete_transcription(record) or record["fingerprint"] != fingerprint:
            return None
        return copy.deepcopy(record["segments"])

    def save_transcription(self, fingerprint: str, segments: List[Dict]) -> None:
        record = {
            "status": "complete",
            "fingerprint": fingerprint,
            "segments": copy.deepcopy(segments),
        }
        if not _is_complete_transcription(record):
            raise ValueError("only completed transcriptions can be checkpointed")
        state = {
            "version": CHECKPOINT_VERSION,
            "transcription": record,
        }
        self._write_state(state)
        self._state = state

    def load_source_sentences(self, fingerprint: str) -> Optional[List[Dict]]:
        """Load a matching, fully validated Source Sentence grouping."""
        record = self._state.get("source_grouping")
        if not _is_source_grouping(record) or record["fingerprint"] != fingerprint:
            return None
        return copy.deepcopy(record["sentences"])

    def save_source_sentences(self, fingerprint: str, sentences: List[Dict]) -> None:
        """Persist an accepted Source Sentence grouping atomically."""
        record = {
            "status": "accepted",
            "fingerprint": fingerprint,
            "sentences": copy.deepcopy(sentences),
        }
        if not _is_source_grouping(record):
            raise ValueError("only complete Source Sentence groupings can be checkpointed")
        state = copy.deepcopy(self._state)
        previous = state.get("source_grouping")
        grouping_changed = (
            not isinstance(previous, dict)
            or previous.get("fingerprint") != fingerprint
            or previous.get("sentences") != record["sentences"]
        )
        if grouping_changed:
            state.pop("translation", None)
        state["source_grouping"] = record
        self._write_state(state)
        self._state = state

    def discard_stale_translation(self, fingerprint: str) -> None:
        """Drop translation chunks that cannot belong to this stage input."""
        translation = self._state.get("translation")
        if not isinstance(translation, dict) or translation.get("fingerprint") == fingerprint:
            return
        state = copy.deepcopy(self._state)
        state.pop("translation", None)
        self._write_state(state)
        self._state = state

    def load_accepted_translation_chunk(
        self,
        translation_fingerprint: str,
        chunk_index: int,
        chunk_fingerprint: str,
        expected_ids: Sequence[str],
    ) -> Optional[Dict[str, str]]:
        translation = self._state.get("translation")
        if (
            not isinstance(translation, dict)
            or translation.get("fingerprint") != translation_fingerprint
        ):
            return None
        chunk = translation.get("chunks", {}).get(str(chunk_index))
        if not _is_accepted_chunk(chunk) or chunk["fingerprint"] != chunk_fingerprint:
            return None
        expected_id_set = set(expected_ids)
        translations = chunk["translations"]
        if len(expected_id_set) != len(expected_ids) or set(translations) != expected_id_set:
            return None
        return copy.deepcopy(translations)

    def save_accepted_translation_chunk(
        self,
        translation_fingerprint: str,
        chunk_index: int,
        chunk_fingerprint: str,
        translations: Dict[str, str],
    ) -> None:
        record = {
            "status": "accepted",
            "fingerprint": chunk_fingerprint,
            "translations": copy.deepcopy(translations),
        }
        if not _is_accepted_chunk(record):
            raise ValueError("only Translation-Gate-accepted chunks can be checkpointed")

        existing = self._state.get("translation")
        chunks = (
            copy.deepcopy(existing["chunks"])
            if isinstance(existing, dict)
            and existing.get("fingerprint") == translation_fingerprint
            and isinstance(existing.get("chunks"), dict)
            else {}
        )
        chunks[str(chunk_index)] = record
        state = copy.deepcopy(self._state)
        state["translation"] = {
            "fingerprint": translation_fingerprint,
            "chunks": chunks,
        }
        self._write_state(state)
        self._state = state
