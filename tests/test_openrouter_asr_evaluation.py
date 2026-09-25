import json
import unittest
from pathlib import Path
from unittest.mock import patch

from hermecho.openrouter_asr_evaluation import (
    CostUnknownError,
    _normalise_words,
    _request_transcription,
    _timing_summary,
    main,
)


class _Response:
    def __init__(self, body):
        self.body = body
        self.headers = {"X-Generation-Id": "generation-1"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, *_args):
        return json.dumps(self.body).encode("utf-8")


class TestOpenRouterAsrEvaluation(unittest.TestCase):
    def test_rejects_text_without_word_timestamps(self) -> None:
        with self.assertRaisesRegex(ValueError, "no word-level timestamps"):
            _normalise_words({"text": "hello"}, 10)

    def test_accepts_nested_words_and_rejects_invalid_timing(self) -> None:
        result = {"segments": [{"words": [
            {"word": "안녕", "start": 0.1, "end": 0.5},
            {"word": "하세요", "start": 0.5, "end": 1.0},
        ]}]}
        self.assertEqual(len(_normalise_words(result, 10)), 2)
        result["segments"][0]["words"][1]["start"] = -1
        with self.assertRaisesRegex(ValueError, "invalid or unordered"):
            _normalise_words(result, 10)

    def test_request_uses_openrouter_transcription_contract_and_reads_cost(self) -> None:
        body = {"text": "안녕", "words": [{"word": "안녕", "start": 0, "end": 0.8}],
                "usage": {"cost": 0.004}}

        def fake_urlopen(request, timeout):
            self.assertEqual(timeout, 120)
            self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/audio/transcriptions")
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "microsoft/mai-transcribe-2")
            self.assertEqual(payload["language"], "ko")
            self.assertEqual(payload["response_format"], "verbose_json")
            self.assertEqual(payload["timestamp_granularities"], ["word"])
            self.assertEqual(payload["input_audio"]["format"], "mp3")
            return _Response(body)

        with patch("pathlib.Path.read_bytes", return_value=b"audio"), patch(
            "hermecho.openrouter_asr_evaluation.urlopen", side_effect=fake_urlopen
        ):
            result = _request_transcription(Path("audio.mp3"), "microsoft/mai-transcribe-2",
                                            "test-key", 1)
        self.assertEqual(result["cost_usd"], 0.004)
        self.assertEqual(result["generation_id"], "generation-1")

    def test_rejects_missing_cost_so_budget_cannot_be_guessed(self) -> None:
        body = {"words": [{"word": "안녕", "start": 0, "end": 0.8}]}
        with patch("pathlib.Path.read_bytes", return_value=b"audio"), patch(
            "hermecho.openrouter_asr_evaluation.urlopen", return_value=_Response(body)
        ):
            with self.assertRaisesRegex(CostUnknownError, "usage.cost"):
                _request_transcription(Path("audio.mp3"), "microsoft/mai-transcribe-2",
                                       "test-key", 1)

    def test_records_cost_when_response_has_no_timed_words(self) -> None:
        body = {"text": "안녕", "usage": {"cost": 0.003}}
        with patch("pathlib.Path.read_bytes", return_value=b"audio"), patch(
            "hermecho.openrouter_asr_evaluation.urlopen", return_value=_Response(body)
        ):
            result = _request_transcription(Path("audio.mp3"), "microsoft/mai-transcribe-2",
                                            "test-key", 1)
        self.assertEqual(result["cost_usd"], 0.003)
        self.assertEqual(result["word_error"], "Response has no word-level timestamps")

    def test_timing_summary_flags_zero_and_long_word_spans(self) -> None:
        summary = _timing_summary([
            {"word": "a", "start": 1.0, "end": 1.0},
            {"word": "b", "start": 8.0, "end": 12.0},
        ])
        self.assertEqual(len(summary["zero_duration_words"]), 1)
        self.assertEqual(len(summary["words_over_3_seconds"]), 1)
        self.assertEqual(len(summary["gaps_at_least_5_seconds"]), 1)

    def test_cli_reports_partial_evaluation_as_failure(self) -> None:
        report = {"spent_usd": 0.01, "cost_unknown": False, "models": {
            "google/gemini-3.5-transcribe": {"status": "probe_failed"},
            "microsoft/mai-transcribe-2": {"status": "complete"},
        }}
        with patch("hermecho.openrouter_asr_evaluation.run_evaluation", return_value=report):
            with self.assertRaises(SystemExit) as exit_result:
                main(["input.mp4", "--output-dir", "output/eval", "--max-cost-usd", "10"])
        self.assertEqual(exit_result.exception.code, 1)
