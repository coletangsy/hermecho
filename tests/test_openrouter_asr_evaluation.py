import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermecho.openrouter_asr_evaluation import (
    CostUnknownError,
    _normalise_words,
    _request_transcription,
    _timing_summary,
    main,
    run_evaluation,
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

    def test_rejects_overlapping_words_even_with_ordered_starts(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid or unordered"):
            _normalise_words({"words": [
                {"word": "first", "start": 0, "end": 2},
                {"word": "second", "start": 1, "end": 3},
            ]}, 10)

    def test_rejects_malformed_nested_segment_without_dropping_it(self) -> None:
        with self.assertRaisesRegex(ValueError, "segment without word-level"):
            _normalise_words({"segments": [
                {"words": [{"word": "first", "start": 0, "end": 1}]},
                {"text": "lost speech", "words": "invalid"},
            ]}, 10)

    def test_rejects_boolean_and_overflowing_timestamps(self) -> None:
        for invalid_start in (True, 10 ** 1000):
            with self.subTest(start_type=type(invalid_start).__name__):
                with self.assertRaisesRegex(ValueError, "numeric timestamps"):
                    _normalise_words({"words": [
                        {"word": "first", "start": invalid_start, "end": 1},
                    ]}, 10)

    def test_evaluation_stops_requests_at_budget_or_unknown_cost(self) -> None:
        for response, expected_status in (
            ({"words": [{"word": "a", "start": 0, "end": 1}],
              "cost_usd": 1.0}, "budget_exhausted"),
            (CostUnknownError("unknown cost"), "cost_unknown"),
        ):
            with self.subTest(expected_status=expected_status), tempfile.TemporaryDirectory() as root:
                video = Path(root) / "video.mp4"
                video.write_bytes(b"fake video")
                output = Path(root) / "evaluation"

                def extract(_source, destination, *_args):
                    destination.write_bytes(b"fake audio")

                with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}), \
                        patch("hermecho.openrouter_asr_evaluation._ffprobe_duration", return_value=462), \
                        patch("hermecho.openrouter_asr_evaluation._extract_audio", side_effect=extract), \
                        patch("hermecho.openrouter_asr_evaluation._request_transcription",
                              side_effect=[response]) as request, \
                        patch("hermecho.openrouter_asr_evaluation.transcribe_audio", return_value=[{
                            "text": "a", "start": 0, "end": 1,
                            "words": [{"word": "a", "start": 0, "end": 1}],
                        }]):
                    report = run_evaluation(video, output, 1.0)
                self.assertEqual(request.call_count, 1)
                self.assertEqual(report["models"]["microsoft/mai-transcribe-2"]["status"], expected_status)
                self.assertEqual(report["cost_unknown"], expected_status == "cost_unknown")
                self.assertTrue((output / "progress.json").is_file())
                self.assertTrue((output / "comparison.json").is_file())

    def test_failed_chunk_preserves_completed_calls_and_offsets(self) -> None:
        def response(word, start, end):
            return {"words": [{"word": word, "start": start, "end": end}],
                    "cost_usd": 0.01, "elapsed_seconds": 0.1}

        with tempfile.TemporaryDirectory() as root:
            video = Path(root) / "video.mp4"
            video.write_bytes(b"video")
            output = Path(root) / "evaluation"

            def extract(_source, destination, *_args):
                destination.write_bytes(b"audio")

            calls = [response("probe", 0, 1), response("probe", 0, 1),
                     response("first", 0, 1), RuntimeError("failed chunk"),
                     *[response("candidate", 1, 2) for _ in range(8)]]
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}), \
                    patch("hermecho.openrouter_asr_evaluation._ffprobe_duration", return_value=462), \
                    patch("hermecho.openrouter_asr_evaluation._extract_audio", side_effect=extract), \
                    patch("hermecho.openrouter_asr_evaluation._request_transcription", side_effect=calls), \
                    patch("hermecho.openrouter_asr_evaluation.transcribe_audio", return_value=[{
                        "text": "baseline", "start": 0, "end": 1,
                        "words": [{"word": "baseline", "start": 0, "end": 1}],
                    }]):
                report = run_evaluation(video, output, 10.0)
            failed = report["models"]["google/gemini-3.5-transcribe"]
            self.assertEqual(failed["status"], "chunk_failed")
            self.assertEqual(len(failed["chunks"]), 1)
            self.assertTrue((output / "google_gemini-3.5-transcribe_chunk_01.json").exists())
            complete = report["models"]["microsoft/mai-transcribe-2"]
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(complete["words"][1]["start"], 60)
            self.assertEqual(len(list(output.glob("review_*.mp3"))), 5)

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

    def test_empty_timing_summary_keeps_unknown_bounds(self) -> None:
        summary = _timing_summary([])
        self.assertEqual(summary["timed_words"], 0)
        self.assertIsNone(summary["first_word_second"])
        self.assertIsNone(summary["last_word_second"])

    def test_cli_reports_partial_evaluation_as_failure(self) -> None:
        report = {"spent_usd": 0.01, "cost_unknown": False, "models": {
            "google/gemini-3.5-transcribe": {"status": "probe_failed"},
            "microsoft/mai-transcribe-2": {"status": "complete"},
        }}
        with patch("hermecho.openrouter_asr_evaluation.run_evaluation", return_value=report):
            with self.assertRaises(SystemExit) as exit_result:
                main(["input.mp4", "--output-dir", "output/eval", "--max-cost-usd", "10"])
        self.assertEqual(exit_result.exception.code, 1)
