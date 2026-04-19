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
    _extract_openrouter_choice_message,
    _infer_openrouter_audio_format,
    _json_object_response_format_for_model,
    _normalize_multimodal_segments,
    _openrouter_message_content_to_text,
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

    def test_none_returns_none(self) -> None:
        self.assertIsNone(_parse_json_object_from_model_text(None))

    def test_non_string_returns_none(self) -> None:
        self.assertIsNone(_parse_json_object_from_model_text(["x"]))


class TestOpenrouterMessageContentToText(unittest.TestCase):
    """Normalize provider-specific message content shapes."""

    def test_none(self) -> None:
        self.assertIsNone(_openrouter_message_content_to_text(None))

    def test_string(self) -> None:
        self.assertEqual(
            _openrouter_message_content_to_text('  {"a":1}  '),
            '{"a":1}',
        )

    def test_list_of_text_parts(self) -> None:
        payload = json.dumps({"segments": []})
        content = [
            {"type": "text", "text": payload[:5]},
            {"type": "text", "text": payload[5:]},
        ]
        self.assertEqual(_openrouter_message_content_to_text(content), payload)


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


class TestExtractOpenrouterChoiceMessage(unittest.TestCase):
    """Normalize OpenRouter chat completion bodies before parsing content."""

    def test_top_level_error_dict(self) -> None:
        body = {"error": {"message": "bad request", "code": 400}}
        msg, err = _extract_openrouter_choice_message(body)
        self.assertIsNone(msg)
        self.assertIn("400", err or "")
        self.assertIn("bad request", err or "")

    def test_missing_choices_lists_keys(self) -> None:
        body = {"usage": {"total_tokens": 1}}
        msg, err = _extract_openrouter_choice_message(body)
        self.assertIsNone(msg)
        self.assertIn("choices", err or "")
        self.assertIn("usage", err or "")

    def test_success_returns_message(self) -> None:
        body = {
            "choices": [
                {"message": {"content": '{"segments":[]}'}}
            ]
        }
        msg, err = _extract_openrouter_choice_message(body)
        self.assertIsNone(err)
        assert msg is not None
        self.assertEqual(msg.get("content"), '{"segments":[]}')


class TestJsonObjectResponseFormatForModel(unittest.TestCase):
    """response_format json_object compatibility per model slug."""

    def test_gpt_audio_omits_format(self) -> None:
        self.assertIsNone(
            _json_object_response_format_for_model("openai/gpt-audio")
        )

    def test_gemini_includes_format(self) -> None:
        self.assertEqual(
            _json_object_response_format_for_model(
                "google/gemini-3.1-pro-preview"
            ),
            {"type": "json_object"},
        )


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
                )
        finally:
            os.unlink(path)

        self.assertEqual(len(out or []), 1)
        self.assertEqual(out[0]["text"], "hello")
        mock_post.assert_called_once()

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("transcription.requests.post")
    def test_gemini_31_pro_retries_gpt_on_http_500(
        self,
        mock_post: MagicMock,
    ) -> None:
        """Primary Gemini 3.1 Pro failure triggers one GPT fallback retry."""
        bad = MagicMock()
        bad.status_code = 500
        bad.text = '{"error":"internal"}'
        good = MagicMock()
        good.status_code = 200
        good.json.return_value = {
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 4,
            },
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "segments": [
                                    {"start": 0.0, "end": 1.0, "text": "retry"},
                                ]
                            }
                        )
                    }
                }
            ],
        }
        mock_post.side_effect = [bad, good]

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"x")

        try:
            with patch(
                "transcription._audio_duration_seconds",
                return_value=5.0,
            ):
                out = transcribe_audio_multimodal(
                    path,
                    language="ko",
                    multimodal_model="google/gemini-3.1-pro-preview",
                )
        finally:
            os.unlink(path)

        self.assertEqual(len(out or []), 1)
        self.assertEqual(out[0]["text"], "retry")
        self.assertEqual(mock_post.call_count, 2)
        first = mock_post.call_args_list[0][1]["json"]
        self.assertEqual(first["response_format"], {"type": "json_object"})
        second = mock_post.call_args_list[1][1]["json"]
        self.assertEqual(second["model"], "openai/gpt-audio")
        self.assertNotIn("response_format", second)

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("transcription.requests.post")
    def test_list_message_content_parses(
        self,
        mock_post: MagicMock,
    ) -> None:
        """Some APIs return message content as a list of typed parts."""
        payload = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": json.dumps(payload)},
                        ]
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
                )
        finally:
            os.unlink(path)

        self.assertEqual(len(out or []), 1)
        self.assertEqual(out[0]["text"], "hi")

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("transcription._transcribe_clip_openrouter")
    @patch("transcription._ffmpeg_extract_chunk", return_value=True)
    @patch(
        "transcription._finalize_multimodal_segments_for_audio",
        side_effect=lambda segs, _path: segs,
    )
    def test_long_audio_uses_multiple_chunks(
        self,
        _mock_fin: MagicMock,
        _mock_ffmpeg: MagicMock,
        mock_clip: MagicMock,
    ) -> None:
        """Media longer than chunk_seconds is transcribed per ffmpeg window."""
        state = {"i": 0}

        def fake_clip(
            chunk_path: str,
            model: str,
            lang: str,
            prompt: str,
        ) -> tuple:
            state["i"] += 1
            n = state["i"]
            return (
                [{"start": 0.0, "end": 1.0, "text": f"part{n}"}],
                {"total_tokens": 1},
            )

        mock_clip.side_effect = fake_clip

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"fake")

        try:
            with patch(
                "transcription._audio_duration_seconds",
                return_value=950.0,
            ):
                out = transcribe_audio_multimodal(
                    path,
                    language="ko",
                    multimodal_model=DEFAULT_MULTIMODAL_MODEL,
                    chunk_seconds=400.0,
                )
        finally:
            os.unlink(path)

        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(mock_clip.call_count, 3)
        self.assertEqual(out[0]["start"], 0.0)
        self.assertEqual(out[1]["start"], 400.0)
        self.assertEqual(out[2]["start"], 800.0)

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("transcription.requests.post")
    @patch("transcription.time.sleep")
    def test_transient_502_retries_then_succeeds(
        self,
        mock_sleep: MagicMock,
        mock_post: MagicMock,
    ) -> None:
        """Gateway 502 on first POST is retried before parsing the body."""
        bad = MagicMock()
        bad.status_code = 502
        bad.text = "bad gateway"
        good = MagicMock()
        good.status_code = 200
        good.json.return_value = {
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "segments": [
                                    {"start": 0.0, "end": 1.0, "text": "ok"},
                                ]
                            }
                        )
                    }
                }
            ],
        }
        mock_post.side_effect = [bad, good]

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"x")

        try:
            with patch(
                "transcription._audio_duration_seconds",
                return_value=8.0,
            ):
                out = transcribe_audio_multimodal(
                    path,
                    language="ko",
                    multimodal_model=DEFAULT_MULTIMODAL_MODEL,
                    chunk_seconds=0.0,
                )
        finally:
            os.unlink(path)

        self.assertEqual(len(out or []), 1)
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called()

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
