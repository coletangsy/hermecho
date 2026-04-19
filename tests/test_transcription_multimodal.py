"""
Unit tests for multimodal OpenRouter transcription helpers.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from transcription import (
    DEFAULT_MULTIMODAL_MODEL,
    _build_multimodal_prompt,
    _infer_openrouter_audio_format,
    _normalize_multimodal_segments,
    _parse_json_object_from_model_text,
    _repair_multimodal_segment_times,
    transcribe_audio_multimodal,
)


class TestBuildMultimodalPrompt(unittest.TestCase):
    """Multimodal user prompt encodes segmentation policy."""

    def test_content_not_clock_driven(self) -> None:
        """Splits follow delivery, not a time budget."""
        text = _build_multimodal_prompt("ko", None)
        self.assertIn("not by a time quota", text)
        self.assertIn("one JSON segment per phase", text)

    def test_appends_initial_prompt(self) -> None:
        text = _build_multimodal_prompt("ko", "names: Foo")
        self.assertIn("names: Foo", text)


class TestInferOpenrouterAudioFormat(unittest.TestCase):
    """Map file extensions to OpenRouter input_audio format."""

    def test_known_extensions(self) -> None:
        self.assertEqual(_infer_openrouter_audio_format("/a/b.wav"), "wav")
        self.assertEqual(_infer_openrouter_audio_format("x.MP3"), "mp3")
        self.assertEqual(_infer_openrouter_audio_format("c.m4a"), "m4a")

    def test_unknown_defaults_to_mp3(self) -> None:
        self.assertEqual(_infer_openrouter_audio_format("noext"), "mp3")


class TestParseJsonObjectFromModelText(unittest.TestCase):
    """JSON extraction from model output."""

    def test_plain_json(self) -> None:
        raw = '{"segments":[]}'
        self.assertEqual(
            _parse_json_object_from_model_text(raw),
            {"segments": []},
        )

    def test_markdown_fence(self) -> None:
        raw = '```json\n{"segments":[]}\n```'
        self.assertEqual(
            _parse_json_object_from_model_text(raw),
            {"segments": []},
        )

    def test_invalid_returns_none(self) -> None:
        self.assertIsNone(_parse_json_object_from_model_text("not json"))


class TestNormalizeMultimodalSegments(unittest.TestCase):
    """Segment list normalization."""

    def test_sorts_and_filters(self) -> None:
        parsed = {
            "segments": [
                {"start": 2.0, "end": 3.0, "text": "b"},
                {"start": 0.0, "end": 1.0, "text": "a"},
                {"start": "x", "end": 1.0, "text": "skip"},
            ]
        }
        out = _normalize_multimodal_segments(parsed)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "a")
        self.assertEqual(out[1]["text"], "b")

    def test_swaps_inverted_times(self) -> None:
        parsed = {"segments": [{"start": 5.0, "end": 1.0, "text": "x"}]}
        out = _normalize_multimodal_segments(parsed)
        self.assertEqual(out[0]["start"], 1.0)
        self.assertEqual(out[0]["end"], 5.0)


class TestRepairMultimodalSegmentTimes(unittest.TestCase):
    """Post-process LLM segment times for a known clip length."""

    def test_clamps_to_clip_end(self) -> None:
        segs = [
            {"text": "a", "start": -1.0, "end": 5.0},
            {"text": "b", "start": 8.0, "end": 100.0},
        ]
        out = _repair_multimodal_segment_times(segs, clip_end=10.0)
        self.assertEqual(out[0]["start"], 0.0)
        self.assertEqual(out[0]["end"], 5.0)
        self.assertEqual(out[1]["end"], 10.0)

    def test_trims_overlap(self) -> None:
        segs = [
            {"text": "a", "start": 0.0, "end": 5.0},
            {"text": "b", "start": 3.0, "end": 7.0},
        ]
        out = _repair_multimodal_segment_times(segs, clip_end=20.0)
        self.assertLessEqual(out[0]["end"], out[1]["start"])


class TestTranscribeAudioMultimodal(unittest.TestCase):
    """OpenRouter multimodal transcription entry point."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("transcription.requests.post")
    def test_posts_and_parses_segments(
        self,
        mock_post: MagicMock,
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "segments": [
                                    {"start": 0.0, "end": 1.0, "text": "hello"},
                                ]
                            }
                        )
                    }
                }
            ],
        }
        mock_post.return_value = mock_resp

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"fake")

        try:
            with patch(
                "transcription._audio_duration_seconds",
                return_value=10.0,
            ):
                out = transcribe_audio_multimodal(
                    path,
                    language="ko",
                    multimodal_model=DEFAULT_MULTIMODAL_MODEL,
                    chunk_seconds=300.0,
                )
        finally:
            os.unlink(path)

        self.assertEqual(len(out or []), 1)
        self.assertEqual(out[0]["text"], "hello")
        mock_post.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"x")
        try:
            out = transcribe_audio_multimodal(path, language="ko")
        finally:
            os.unlink(path)
        self.assertIsNone(out)
