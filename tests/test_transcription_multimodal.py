"""
Unit tests for multimodal Gemini transcription helpers.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from transcription import (
    DEFAULT_MULTIMODAL_MODEL,
    _build_multimodal_prompt,
    _infer_gemini_inline_audio_mime_type,
    _normalize_multimodal_segments,
    _parse_json_object_from_model_text,
    _repair_multimodal_segment_times,
    transcribe_audio_multimodal,
)


class TestBuildMultimodalPrompt(unittest.TestCase):
    def test_content_not_clock_driven(self) -> None:
        text = _build_multimodal_prompt("ko", None)
        self.assertIn("not by a time quota", text)
        self.assertIn("one JSON segment per phase", text)
        self.assertIn("exactly ONE JSON value", text)

    def test_appends_initial_prompt(self) -> None:
        text = _build_multimodal_prompt("ko", "names: Foo")
        self.assertIn("names: Foo", text)


class TestInferGeminiAudioMimeType(unittest.TestCase):
    def test_known_extensions(self) -> None:
        self.assertEqual(_infer_gemini_inline_audio_mime_type("/a/b.wav"), "audio/wav")
        self.assertEqual(_infer_gemini_inline_audio_mime_type("x.MP3"), "audio/mpeg")
        self.assertEqual(_infer_gemini_inline_audio_mime_type("c.m4a"), "audio/mp4")
        self.assertEqual(_infer_gemini_inline_audio_mime_type("d.flac"), "audio/flac")

    def test_unknown_defaults_to_mpeg(self) -> None:
        self.assertEqual(_infer_gemini_inline_audio_mime_type("noext"), "audio/mpeg")


class TestParseJsonObjectFromModelText(unittest.TestCase):
    def test_plain_json(self) -> None:
        raw = '{"segments":[]}'
        self.assertEqual(_parse_json_object_from_model_text(raw), {"segments": []})

    def test_markdown_fence(self) -> None:
        raw = '```json\n{"segments":[]}\n```'
        self.assertEqual(_parse_json_object_from_model_text(raw), {"segments": []})

    def test_invalid_returns_none(self) -> None:
        self.assertIsNone(_parse_json_object_from_model_text("not json"))

    def test_none_returns_none(self) -> None:
        self.assertIsNone(_parse_json_object_from_model_text(None))

    def test_non_string_returns_none(self) -> None:
        self.assertIsNone(_parse_json_object_from_model_text(["x"]))

    def test_leading_prose_then_json_object(self) -> None:
        raw = 'Here is the transcript:\n{"segments":[{"start":0,"end":1,"text":"a"}]}'
        out = _parse_json_object_from_model_text(raw)
        self.assertEqual(out, {"segments": [{"start": 0, "end": 1, "text": "a"}]})

    def test_trailing_prose_after_json(self) -> None:
        raw = '{"segments":[]}\nthanks'
        self.assertEqual(_parse_json_object_from_model_text(raw), {"segments": []})

    def test_top_level_segment_array(self) -> None:
        raw = '[{"start":0.0,"end":1.0,"text":"only"}]'
        out = _parse_json_object_from_model_text(raw)
        self.assertEqual(out, {"segments": [{"start": 0.0, "end": 1.0, "text": "only"}]})


class TestNormalizeMultimodalSegments(unittest.TestCase):
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
    def _make_mock_response(self, segments_json: dict) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(segments_json)
        mock_resp.usage_metadata = None
        return mock_resp

    def _make_mock_client(self, response: MagicMock) -> MagicMock:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response
        mock_uploaded = MagicMock()
        mock_uploaded.uri = "https://generativelanguage.googleapis.com/v1beta/files/abc"
        mock_uploaded.name = "files/abc"
        mock_client.files.upload.return_value = mock_uploaded
        return mock_client

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    @patch("transcription._make_gemini_client")
    def test_parses_segments_from_sdk_response(
        self, mock_make_client: MagicMock
    ) -> None:
        segments_data = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}
        mock_resp = self._make_mock_response(segments_data)
        mock_client = self._make_mock_client(mock_resp)
        mock_make_client.return_value = mock_client

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"fake")

        try:
            with patch("transcription._audio_duration_seconds", return_value=10.0):
                out = transcribe_audio_multimodal(
                    path,
                    language="ko",
                    multimodal_model=DEFAULT_MULTIMODAL_MODEL,
                    chunk_seconds=0.0,
                )
        finally:
            os.unlink(path)

        self.assertIsNotNone(out)
        self.assertEqual(len(out or []), 1)
        self.assertEqual(out[0]["text"], "hello")  # type: ignore[index]
        mock_client.models.generate_content.assert_called_once()

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    @patch("transcription._make_gemini_client")
    @patch("transcription._ffmpeg_extract_chunk", return_value=True)
    @patch(
        "transcription._finalize_multimodal_segments_for_audio",
        side_effect=lambda segs, _path: segs,
    )
    def test_long_audio_calls_multiple_chunks(
        self,
        _mock_fin: MagicMock,
        _mock_ffmpeg: MagicMock,
        mock_make_client: MagicMock,
    ) -> None:
        call_count = {"n": 0}

        def fake_generate(**_kwargs):
            call_count["n"] += 1
            n = call_count["n"]
            mock_r = MagicMock()
            mock_r.text = json.dumps(
                {"segments": [{"start": 0.0, "end": 1.0, "text": f"part{n}"}]}
            )
            mock_r.usage_metadata = None
            return mock_r

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = fake_generate
        mock_uploaded = MagicMock()
        mock_uploaded.uri = "https://example.com/files/x"
        mock_uploaded.name = "files/x"
        mock_client.files.upload.return_value = mock_uploaded
        mock_make_client.return_value = mock_client

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"fake")

        try:
            with patch("transcription._audio_duration_seconds", return_value=950.0):
                with patch("transcription.time.sleep"):
                    out = transcribe_audio_multimodal(
                        path,
                        language="ko",
                        multimodal_model=DEFAULT_MULTIMODAL_MODEL,
                        chunk_seconds=400.0,
                    )
        finally:
            os.unlink(path)

        self.assertIsNotNone(out)
        self.assertEqual(call_count["n"], 3)

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_returns_none(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"x")
        try:
            out = transcribe_audio_multimodal(path, language="ko")
        finally:
            os.unlink(path)
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
