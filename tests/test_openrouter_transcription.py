import base64
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import MagicMock, patch

from hermecho.openrouter_transcription import (
    DEFAULT_OPENROUTER_TRANSCRIPTION_MODEL,
    OpenRouterRequestError,
    OpenRouterWordTimestampError,
    _aggregate_metadata,
    _chunk_specs,
    _request_transcription,
    _response_words,
    _words_for_output,
    transcribe_openrouter,
)


class _Response:
    def __init__(self, body, headers=None):
        self.body = body
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, *_args):
        if isinstance(self.body, bytes):
            return self.body
        return json.dumps(self.body).encode("utf-8")


def _words(*values):
    return [{"word": word, "start": start, "end": end} for word, start, end in values]


class TestOpenRouterTranscription(unittest.TestCase):
    def test_overlap_timestamp_drift_neither_duplicates_nor_drops_a_word(self) -> None:
        specs = _chunk_specs(120.0)
        for first_start, second_start in ((59.8, 60.0), (60.0, 59.8)):
            with self.subTest(first_start=first_start, second_start=second_start):
                chunks = [
                    (specs[0], {"words": _words(("hello", first_start, first_start + 0.1))}),
                    (specs[1], {"words": _words(("hello", second_start - 59, second_start - 58.9))}),
                ]
                words = _words_for_output(chunks, 120.0)
                self.assertEqual([word["word"] for word in words], ["hello"])
                self.assertIn(words[0], chunks[0][1]["words"])

    def test_overlap_preserves_repeated_words_and_original_evidence(self) -> None:
        specs = _chunk_specs(120.0)
        left = _words((" Ha!", 59.7, 59.8), ("ha", 60.1, 60.2))
        right = _words(("ha", 0.75, 0.85), ("HA.", 1.15, 1.25))
        words = _words_for_output([(specs[0], {"words": left}),
                                   (specs[1], {"words": right})], 120.0)
        self.assertEqual(words, left)

    def test_conflicting_overlap_evidence_blocks_instead_of_guessing(self) -> None:
        specs = _chunk_specs(120.0)
        left = _words(("one", 59.5, 59.6), ("two", 60.1, 60.2))
        conflicts = [
            _words(("one", 0.5, 0.6)),  # missing the second shared word
            _words(("other", 0.5, 0.6), ("two", 1.1, 1.2)),
            _words(("two", 0.5, 0.6), ("one", 1.1, 1.2)),
            _words(("one", 1.0, 1.1), ("two", 1.5, 1.6)),
        ]
        for right in conflicts:
            with self.subTest(right=right), self.assertRaisesRegex(RuntimeError, "overlapping transcripts disagree"):
                _words_for_output([(specs[0], {"words": left}),
                                   (specs[1], {"words": right})], 120.0)

    def test_overlap_includes_zero_duration_words_and_float_edges(self) -> None:
        specs = _chunk_specs(120.0)
        left = _words(("edge", 58.95, 59 - 1e-12), ("zero", 61, 61))
        right = _words(("edge", 0, 0.02), ("zero", 2, 2))
        self.assertEqual(_words_for_output([
            (specs[0], {"words": left}), (specs[1], {"words": right}),
        ], 120.0), left)

    def test_overlap_reconciles_multiple_boundaries_without_reordering(self) -> None:
        specs = _chunk_specs(180.0)
        chunks = [
            (specs[0], {"words": _words(("first", 59.8, 59.9))}),
            (specs[1], {"words": _words(("first", 1.0, 1.1), ("second", 61.0, 61.1))}),
            (specs[2], {"words": _words(("second", 0.8, 0.9), ("last", 3.0, 3.1))}),
        ]
        words = _words_for_output(chunks, 180.0)
        self.assertEqual([word["word"] for word in words], ["first", "second", "last"])
        self.assertEqual([word["start"] for word in words], [59.8, 120.0, 122.0])

    def test_missing_chunk_metadata_does_not_become_zero_cost_or_latency(self) -> None:
        for records in ([], [({}, {})], [({}, {"metadata": {
            "cost_usd": 0.1, "elapsed_seconds": 1.0,
        }}), ({}, {})]):
            with self.subTest(records=records):
                result = _aggregate_metadata(records)
                self.assertIsNone(result["cost_usd"])
                self.assertIsNone(result["elapsed_seconds"])
                self.assertEqual(len(result["provider"]), len(records))

    def test_payload_defaults_to_auto_language_and_mai_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            audio_path = Path(temporary_dir) / "chunk.mp3"
            audio_path.write_bytes(b"audio")

            def fake_urlopen(request, timeout):
                self.assertEqual(timeout, 120)
                payload = json.loads(request.data)
                self.assertEqual(payload["model"], DEFAULT_OPENROUTER_TRANSCRIPTION_MODEL)
                self.assertNotIn("language", payload)
                self.assertEqual(payload["temperature"], 0.25)
                self.assertEqual(payload["response_format"], "verbose_json")
                self.assertEqual(payload["timestamp_granularities"], ["word"])
                self.assertEqual(payload["input_audio"]["format"], "mp3")
                self.assertEqual(
                    base64.b64decode(payload["input_audio"]["data"]), b"audio"
                )
                self.assertEqual(
                    payload["provider"]["options"]["azure"]["enhancedMode"]
                    ["modelOptions"]["transcribeStyle"],
                    "verbatim",
                )
                return _Response(
                    {
                        "text": "hello",
                        "words": _words(("hello", 0, 1)),
                        "usage": {"cost": 0.004},
                        "model": "microsoft/mai-transcribe-2",
                        "provider": "azure",
                    },
                    {"X-Generation-Id": "generation-1"},
                )

            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), patch(
                "hermecho.openrouter_transcription.urlopen", side_effect=fake_urlopen
            ):
                body, metadata = _request_transcription(
                    audio_path,
                    DEFAULT_OPENROUTER_TRANSCRIPTION_MODEL,
                    None,
                    0.25,
                    "test-key",
                )
        self.assertEqual(body["text"], "hello")
        self.assertEqual(metadata["cost_usd"], 0.004)
        self.assertEqual(metadata["model"], "microsoft/mai-transcribe-2")
        self.assertEqual(metadata["provider"], "azure")
        self.assertEqual(metadata["generation_id"], "generation-1")
        self.assertIsInstance(metadata["elapsed_seconds"], float)

    def test_payload_pins_language_and_does_not_add_azure_options_to_other_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            audio_path = Path(temporary_dir) / "chunk.mp3"
            audio_path.write_bytes(b"audio")

            def fake_urlopen(request, timeout):
                payload = json.loads(request.data)
                self.assertEqual(payload["model"], "other/model")
                self.assertEqual(payload["language"], "ko")
                self.assertNotIn("provider", payload)
                return _Response({"text": "안녕", "words": _words(("안녕", 0, 1))})

            with patch(
                "hermecho.openrouter_transcription.urlopen", side_effect=fake_urlopen
            ):
                _request_transcription(audio_path, "other/model", "ko", 0.0, "test-key")

    def test_malformed_or_incomplete_evidence_is_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            audio_path = Path(temporary_dir) / "chunk.mp3"
            audio_path.write_bytes(b"audio")
            responses = [
                _Response(b"not-json"),
                _Response({"text": "hello"}),
                _Response(
                    {"text": "hello", "words": _words(("hello", 2, 1))}
                ),
            ]
            with patch(
                "hermecho.openrouter_transcription.urlopen",
                side_effect=responses,
            ):
                for index, _ in enumerate(responses):
                    with self.assertRaises(RuntimeError) as error:
                        body, _metadata = _request_transcription(
                            audio_path,
                            "other/model",
                            None,
                            0.0,
                            "test-key",
                        )
                        if index:
                            _response_words(body, 10)
                    self.assertNotIsInstance(error.exception, OpenRouterRequestError)

    def test_retries_invalid_word_timestamps_before_checkpointing_chunk(self) -> None:
        responses = [
            ({"text": "hello world", "words": _words(
                ("hello", 0.0, 0.8), ("world", 0.7, 1.1),
            )}, {
                "cost_usd": 0.01, "model": "model", "provider": "provider-a",
                "generation_id": "generation-1", "elapsed_seconds": 0.1,
            }),
            ({"text": "hello world", "words": _words(
                ("hello", 0.0, 0.8), ("world", 0.8, 1.1),
            )}, {
                "cost_usd": 0.02, "model": "model", "provider": "provider-a",
                "generation_id": "generation-2", "elapsed_seconds": 0.2,
            }),
        ]

        def fake_request(*_args):
            response = responses.pop(0)
            return response

        def fake_extract(_source, destination, _start, _duration):
            destination.write_bytes(b"chunk")

        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / "source.mp3"
            source.write_bytes(b"source")
            checkpoint = Path(temporary_dir) / "checkpoint.json"
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), patch(
                "hermecho.openrouter_transcription._ffprobe_duration", return_value=1.5
            ), patch(
                "hermecho.openrouter_transcription._extract_audio", side_effect=fake_extract
            ), patch(
                "hermecho.openrouter_transcription._request_transcription",
                side_effect=fake_request,
            ) as request, patch(
                "hermecho.openrouter_transcription.emit_progress"
            ) as progress:
                segments = transcribe_openrouter(
                    str(source), "model", checkpoint_path=str(checkpoint)
                )
                state = json.loads(checkpoint.read_text())

        assert request.call_count == 2
        assert [word["word"] for word in segments[0]["words"]] == ["hello", "world"]
        metadata = state["chunks"]["0"]["metadata"]
        assert metadata["attempt_count"] == 2
        assert metadata["cost_usd"] == 0.03
        assert metadata["elapsed_seconds"] == 0.3
        assert metadata["generation_id"] == "generation-2"
        assert len(metadata["retry_diagnostics"]) == 1
        retry = [call for call in progress.call_args_list if "Retrying" in call.args[2]]
        assert len(retry) == 1
        assert retry[0].kwargs["current"] == 1
        assert retry[0].kwargs["total"] == 1

    def test_exhausted_timestamp_retries_do_not_cache_invalid_chunk(self) -> None:
        invalid = {"text": "hello world", "words": _words(
            ("hello", 0.0, 0.8), ("world", 0.7, 1.1),
        )}

        def fake_request(*_args):
            return invalid, {
                "cost_usd": 0.01, "model": "model", "provider": "provider-a",
                "generation_id": "generation", "elapsed_seconds": 0.1,
            }

        def fake_extract(_source, destination, _start, _duration):
            destination.write_bytes(b"chunk")

        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / "source.mp3"
            source.write_bytes(b"source")
            checkpoint = Path(temporary_dir) / "checkpoint.json"
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), patch(
                "hermecho.openrouter_transcription._ffprobe_duration", return_value=1.5
            ), patch(
                "hermecho.openrouter_transcription._extract_audio", side_effect=fake_extract
            ), patch(
                "hermecho.openrouter_transcription._request_transcription",
                side_effect=fake_request,
            ) as request:
                with self.assertRaisesRegex(OpenRouterWordTimestampError, "attempt 3"):
                    transcribe_openrouter(
                        str(source), "model", checkpoint_path=str(checkpoint)
                    )

            state = json.loads(checkpoint.read_text())

        assert request.call_count == 3
        assert state["status"] == "partial"
        assert state["chunks"] == {}

    def test_exhausted_retry_history_resumes_and_aggregates_each_charge_once(self) -> None:
        invalid = {"text": "hello world", "words": _words(
            ("hello", 0.0, 0.8), ("world", 0.7, 1.1),
        )}
        responses = [
            ({"text": "one two three edge", "words": _words(
                ("one", 50.0, 50.2), ("two", 51.0, 51.2),
                ("three", 52.0, 52.2), ("edge", 59.5, 59.8),
            )}, {
                "cost_usd": 0.1, "model": "model", "provider": "provider-a",
                "generation_id": "generation-0", "elapsed_seconds": 0.1,
            }),
            *[(invalid, {
                "cost_usd": cost, "model": "model", "provider": "provider-a",
                "generation_id": f"generation-{index}", "elapsed_seconds": 0.1,
            }) for index, cost in enumerate((0.01, 0.02, 0.03), start=1)],
        ]

        def fake_request(*_args):
            return responses.pop(0)

        def fake_extract(_source, destination, _start, _duration):
            destination.write_bytes(b"chunk")

        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / "source.mp3"
            source.write_bytes(b"source")
            checkpoint = Path(temporary_dir) / "checkpoint.json"
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), patch(
                "hermecho.openrouter_transcription._ffprobe_duration", return_value=62.0
            ), patch(
                "hermecho.openrouter_transcription._extract_audio", side_effect=fake_extract
            ), patch(
                "hermecho.openrouter_transcription._request_transcription",
                side_effect=fake_request,
            ) as request:
                with self.assertRaises(OpenRouterWordTimestampError):
                    transcribe_openrouter(
                        str(source), "model", checkpoint_path=str(checkpoint)
                    )
                partial = json.loads(checkpoint.read_text())
                self.assertEqual(list(partial["chunks"]), ["0"])
                self.assertEqual(len(partial["retry_history"]), 1)
                history = partial["retry_history"][0]
                self.assertEqual(history["scope"], "chunk")
                self.assertEqual(history["identifier"], "1")
                self.assertEqual(history["attempt_count"], 3)
                self.assertEqual(
                    [attempt["generation_id"] for attempt in history["attempts"]],
                    ["generation-1", "generation-2", "generation-3"],
                )
                self.assertTrue(all("text" not in attempt and "words" not in attempt
                                    for attempt in history["attempts"]))

                responses.append((
                    {"text": "edge hello world tail", "words": _words(
                        ("edge", 0.5, 0.8), ("hello", 2.1, 2.3),
                        ("world", 2.5, 2.8), ("tail", 2.9, 3.0),
                    )},
                    {
                        "cost_usd": 0.04, "model": "model", "provider": "provider-a",
                        "generation_id": "generation-4", "elapsed_seconds": 0.1,
                    },
                ))
                result = transcribe_openrouter(
                    str(source), "model", checkpoint_path=str(checkpoint)
                )
                state = json.loads(checkpoint.read_text())

        self.assertEqual(request.call_count, 5)
        self.assertEqual(
            [word["word"] for word in result[0]["words"]],
            ["one", "two", "three", "edge", "hello", "world", "tail"],
        )
        self.assertEqual(state["metadata"]["cost_usd"], 0.2)
        self.assertEqual(state["metadata"]["elapsed_seconds"], 0.5)
        self.assertEqual(len(state["retry_history"]), 1)
        self.assertCountEqual(
            state["metadata"]["generation_id"],
            [f"generation-{index}" for index in range(5)],
        )

    def test_request_failure_after_invalid_timestamp_persists_prior_metadata(self) -> None:
        invalid = {"text": "hello world", "words": _words(
            ("hello", 0.0, 0.8), ("world", 0.7, 1.1),
        )}
        responses = [
            (invalid, {
                "cost_usd": 0.01, "model": "model", "provider": "provider-a",
                "generation_id": "generation-invalid", "elapsed_seconds": 0.1,
            }),
            OpenRouterRequestError("timeout"),
        ]

        def fake_request(*_args):
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        def fake_extract(_source, destination, _start, _duration):
            destination.write_bytes(b"chunk")

        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / "source.mp3"
            source.write_bytes(b"source")
            checkpoint = Path(temporary_dir) / "checkpoint.json"
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), patch(
                "hermecho.openrouter_transcription._ffprobe_duration", return_value=1.5
            ), patch(
                "hermecho.openrouter_transcription._extract_audio", side_effect=fake_extract
            ), patch(
                "hermecho.openrouter_transcription._request_transcription",
                side_effect=fake_request,
            ) as request:
                with self.assertRaises(OpenRouterRequestError):
                    transcribe_openrouter(
                        str(source), "model", checkpoint_path=str(checkpoint)
                    )
            state = json.loads(checkpoint.read_text())

        self.assertEqual(request.call_count, 2)
        self.assertEqual(state["chunks"], {})
        self.assertEqual(len(state["retry_history"]), 1)
        history = state["retry_history"][0]
        self.assertEqual(history["status"], "interrupted")
        self.assertEqual(history["attempt_count"], 1)
        self.assertEqual(history["attempts"][0]["generation_id"], "generation-invalid")
        self.assertEqual(history["attempts"][0]["cost_usd"], 0.01)
        self.assertTrue(all("text" not in attempt and "words" not in attempt
                            for attempt in history["attempts"]))
        self.assertEqual(len(history["diagnostics"]), 1)

    def test_retry_history_with_missing_cost_stays_unknown(self) -> None:
        result = _aggregate_metadata(
            [({}, {"metadata": {"cost_usd": 0.1, "elapsed_seconds": 1.0}})],
            retry_history=[{
                "attempts": [{"cost_usd": None, "elapsed_seconds": None}],
                "diagnostics": ["invalid timestamps"],
            }],
        )
        self.assertIsNone(result["cost_usd"])
        self.assertIsNone(result["elapsed_seconds"])

    def test_silence_does_not_hide_malformed_nested_transcript_evidence(self) -> None:
        with self.assertRaises(RuntimeError):
            _response_words(
                {"text": "", "segments": [{"text": "lost speech", "words": []}]},
                10,
            )
        with self.assertRaises(RuntimeError):
            _response_words(
                {"text": "hello", "words": _words(("other", 0, 1))},
                10,
            )

    def test_temperature_must_match_openrouter_range(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), patch(
            "hermecho.openrouter_transcription._ffprobe_duration", return_value=1.0
        ):
            with self.assertRaisesRegex(RuntimeError, "between 0 and 1"):
                transcribe_openrouter("audio.mp3", "model", temperature=1.01)

    def test_only_transient_http_failures_are_request_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            audio_path = Path(temporary_dir) / "chunk.mp3"
            audio_path.write_bytes(b"audio")
            for status, expected in (
                (400, RuntimeError), (401, RuntimeError), (403, RuntimeError),
                (404, RuntimeError), (422, RuntimeError), (408, OpenRouterRequestError),
                (429, OpenRouterRequestError), (503, OpenRouterRequestError),
            ):
                error = HTTPError(
                    "https://openrouter.ai/api/v1/audio/transcriptions",
                    status,
                    "error",
                    {},
                    None,
                )
                with patch(
                    "hermecho.openrouter_transcription.urlopen", side_effect=error
                ):
                    with self.assertRaises(expected) as raised:
                        _request_transcription(audio_path, "other/model", None, 0.0, "test-key")
                    self.assertIs(type(raised.exception), expected)

    def test_resumes_chunks_and_invalidates_when_configuration_changes(self) -> None:
        responses = [
            ({"text": "a b", "words": _words(("a", 1, 2), ("b", 60, 61))}, {"cost": 0.1}),
            OpenRouterRequestError("interrupted second chunk"),
            ({"text": "b", "words": _words(("b", 1, 2))}, {"cost": 0.2}),
        ]

        def fake_request(_path, _model, _language, _temperature, _api_key):
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            body, usage = response
            return {**body, "usage": usage}, {
                "cost_usd": usage["cost"],
                "model": "actual/model",
                "provider": "provider-a",
                "generation_id": "generation",
                "elapsed_seconds": 0.1,
            }

        def fake_extract(_source, destination, _start, _duration):
            destination.write_bytes(b"chunk")

        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / "source.mp3"
            source.write_bytes(b"source")
            checkpoint = Path(temporary_dir) / "nested" / "checkpoint.json"
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), patch(
                "hermecho.openrouter_transcription._ffprobe_duration", return_value=61.0
            ), patch(
                "hermecho.openrouter_transcription._extract_audio", side_effect=fake_extract
            ), patch(
                "hermecho.openrouter_transcription._request_transcription",
                side_effect=fake_request,
            ) as request:
                with self.assertRaises(OpenRouterRequestError):
                    transcribe_openrouter(
                        str(source), "model-a", None, checkpoint_path=str(checkpoint)
                    )
                self.assertEqual(request.call_count, 2)
                partial = json.loads(checkpoint.read_text())
                self.assertEqual(partial["status"], "partial")
                self.assertEqual(list(partial["chunks"]), ["0"])
                first = transcribe_openrouter(
                    str(source), "model-a", None, checkpoint_path=str(checkpoint)
                )
                self.assertEqual(request.call_count, 3)
                second = transcribe_openrouter(
                    str(source), "model-a", None, checkpoint_path=str(checkpoint)
                )
                self.assertEqual(request.call_count, 3)
                self.assertEqual(second, first)

                responses.extend(
                    [
                        ({"text": "c d", "words": _words(("c", 1, 2), ("d", 60, 61))}, {"cost": 0.3}),
                        ({"text": "d", "words": _words(("d", 1, 2))}, {"cost": 0.4}),
                    ]
                )
                changed = transcribe_openrouter(
                    str(source), "model-b", None, checkpoint_path=str(checkpoint)
                )
                self.assertEqual(request.call_count, 5)
                self.assertEqual([word["word"] for word in changed[0]["words"]], ["c", "d"])

    def test_reconciliation_deduplicates_overlap_and_offsets_words(self) -> None:
        calls = 0

        def fake_request(_path, _model, _language, _temperature, _api_key):
            nonlocal calls
            calls += 1
            if calls == 1:
                words = _words(("left", 59.7, 59.8), ("right", 60.2, 60.3))
            else:
                words = _words(("left", 0.7, 0.8), ("right", 1.2, 1.3))
            return {"text": "left right", "words": words}, {
                "cost_usd": None,
                "model": None,
                "provider": None,
                "generation_id": None,
                "elapsed_seconds": 0.0,
            }

        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / "source.mp3"
            source.write_bytes(b"source")

            def fake_extract(_source, destination, _start, _duration):
                destination.write_bytes(b"chunk")

            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), patch(
                "hermecho.openrouter_transcription._ffprobe_duration", return_value=61.0
            ), patch(
                "hermecho.openrouter_transcription._extract_audio", side_effect=fake_extract
            ), patch(
                "hermecho.openrouter_transcription._request_transcription",
                side_effect=fake_request,
            ):
                result = transcribe_openrouter(str(source), "model", None)

        self.assertEqual([word["word"] for word in result[0]["words"]], ["left", "right"])
        self.assertEqual([word["start"] for word in result[0]["words"]], [59.7, 60.2])
        self.assertEqual([word["end"] for word in result[0]["words"]], [59.8, 60.3])

    def test_silence_chunk_is_valid_and_force_bypasses_cache(self) -> None:
        request = MagicMock(
            return_value=(
                {"text": "", "words": []},
                {
                    "cost_usd": None,
                    "model": None,
                    "provider": None,
                    "generation_id": None,
                    "elapsed_seconds": 0.0,
                },
            )
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / "source.mp3"
            source.write_bytes(b"source")
            checkpoint = Path(temporary_dir) / "checkpoint.json"

            def fake_extract(_source, destination, _start, _duration):
                destination.write_bytes(b"chunk")

            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), patch(
                "hermecho.openrouter_transcription._ffprobe_duration", return_value=61.0
            ), patch(
                "hermecho.openrouter_transcription._extract_audio", side_effect=fake_extract
            ), patch(
                "hermecho.openrouter_transcription._request_transcription",
                request,
            ):
                self.assertEqual(
                    transcribe_openrouter(
                        str(source), "model", None, checkpoint_path=str(checkpoint)
                    ),
                    [],
                )
                self.assertEqual(request.call_count, 2)
                transcribe_openrouter(
                    str(source), "model", None, checkpoint_path=str(checkpoint), force=True
                )
                self.assertEqual(request.call_count, 4)


if __name__ == "__main__":
    unittest.main()
