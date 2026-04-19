"""
Unit tests for multimodal subtitle timing review helpers.
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from timing_review import (
    _clamp_and_repair_subtitles_in_order,
    build_review_payload_cues,
    parse_review_response,
    review_subtitle_timing,
    review_timing_chunk,
    slice_segments_for_window,
)


class TestSliceSegmentsForWindow(unittest.TestCase):
    """Overlap selection for timeline chunks."""

    def test_overlapping_middle(self) -> None:
        segs = [
            {"start": 0.0, "end": 1.0, "text": "a"},
            {"start": 5.0, "end": 7.0, "text": "b"},
            {"start": 10.0, "end": 12.0, "text": "c"},
        ]
        # [4, 11) overlaps segment 1 [5,7) and segment 2 [10,12).
        out = slice_segments_for_window(segs, 4.0, 11.0)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0], 1)
        self.assertEqual(out[0][1]["text"], "b")
        self.assertEqual(out[1][0], 2)
        self.assertEqual(out[1][1]["text"], "c")

    def test_single_segment_window(self) -> None:
        segs = [
            {"start": 0.0, "end": 1.0, "text": "a"},
            {"start": 5.0, "end": 7.0, "text": "b"},
        ]
        out = slice_segments_for_window(segs, 5.5, 8.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1]["text"], "b")


class TestBuildReviewPayloadCues(unittest.TestCase):
    """Clip-relative cue payloads for the model."""

    def test_relative_times_and_ids(self) -> None:
        indexed = [
            (0, {"start": 10.0, "end": 12.0, "source_text": "안녕", "text": "你好"}),
            (1, {"start": 12.0, "end": 15.0, "source_text": "b", "text": "B"}),
        ]
        cues = build_review_payload_cues(indexed, window_start=10.0, window_span=5.0)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0]["id"], 0)
        self.assertEqual(cues[0]["start"], 0.0)
        self.assertEqual(cues[0]["end"], 2.0)
        self.assertEqual(cues[0]["text"], "안녕")
        self.assertEqual(cues[0]["translation"], "你好")
        self.assertEqual(cues[1]["id"], 1)
        self.assertAlmostEqual(cues[1]["start"], 2.0)
        self.assertAlmostEqual(cues[1]["end"], 5.0)


class TestClampAndRepairInOrder(unittest.TestCase):
    """Order-preserving monotonic repair (no sort-by-start)."""

    def test_preserves_cue_order_and_text_keys(self) -> None:
        segs = [
            {"start": 10.0, "end": 11.5, "text": "A", "source_text": "a"},
            {"start": 11.0, "end": 13.0, "text": "B", "source_text": "b"},
        ]
        out = _clamp_and_repair_subtitles_in_order(segs, clip_end=20.0)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "A")
        self.assertEqual(out[1]["text"], "B")
        self.assertGreaterEqual(float(out[1]["start"]), float(out[0]["end"]))

    def test_no_reorder_when_second_start_is_earlier(self) -> None:
        """Times out of order must not swap dicts (text stays with list index)."""
        segs = [
            {"start": 5.0, "end": 6.0, "text": "line_one"},
            {"start": 2.0, "end": 3.0, "text": "line_two"},
        ]
        out = _clamp_and_repair_subtitles_in_order(segs, clip_end=10.0)
        self.assertEqual(out[0]["text"], "line_one")
        self.assertEqual(out[1]["text"], "line_two")


class TestParseReviewResponse(unittest.TestCase):
    """JSON timing rows from the model."""

    def test_valid(self) -> None:
        raw = (
            '{"segments":['
            '{"id":1,"start":2.0,"end":3.0},'
            '{"id":0,"start":0.0,"end":1.0}'
            "]}"
        )
        rows = parse_review_response(raw, expected_count=2)
        self.assertIsNotNone(rows)
        assert rows is not None
        self.assertEqual(rows[0]["id"], 0)
        self.assertEqual(rows[1]["id"], 1)

    def test_wrong_count_returns_none(self) -> None:
        raw = '{"segments":[{"id":0,"start":0.0,"end":1.0}]}'
        self.assertIsNone(parse_review_response(raw, expected_count=2))


class TestReviewTimingChunk(unittest.TestCase):
    """OpenRouter request shape for one chunk."""

    @patch("timing_review.requests.post")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
    def test_posts_temperature_zero_and_input_audio(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"segments":[{"id":0,"start":0.0,"end":1.0}]}'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        mock_post.return_value = mock_resp

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"\x00\x01")

        try:
            out, _usage = review_timing_chunk(
                path,
                [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "text": "a",
                        "translation": "A",
                    }
                ],
                "m",
                "ko",
                "zh",
                10.0,
                "test-label",
            )
        finally:
            os.unlink(path)

        self.assertIsNotNone(out)
        mock_post.assert_called_once()
        kwargs = mock_post.call_args.kwargs
        body = kwargs["json"]
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        content = body["messages"][0]["content"]
        self.assertTrue(any(b.get("type") == "input_audio" for b in content))


class TestReviewSubtitleTimingOrchestrator(unittest.TestCase):
    """End-to-end merge with ffmpeg and API mocked."""

    @patch("timing_review.review_timing_chunk")
    @patch("timing_review._ffmpeg_extract_chunk", return_value=True)
    @patch("timing_review._audio_duration_seconds", return_value=5.0)
    def test_updates_absolute_times(
        self,
        _mock_dur: MagicMock,
        _mock_ffmpeg: MagicMock,
        mock_review: MagicMock,
    ) -> None:
        mock_review.return_value = (
            [
                {"id": 0, "start": 0.0, "end": 1.0},
                {"id": 1, "start": 1.0, "end": 3.0},
            ],
            {"prompt_tokens": 1},
        )

        segs = [
            {
                "start": 0.5,
                "end": 2.0,
                "text": "A",
                "source_text": "a",
            },
            {
                "start": 2.5,
                "end": 4.0,
                "text": "B",
                "source_text": "b",
            },
        ]

        with patch("timing_review.os.path.exists", return_value=True):
            with patch("timing_review.tempfile.mkdtemp", return_value="/tmp/x"):
                with patch("timing_review.shutil.rmtree"):
                    out = review_subtitle_timing(
                        "/fake/audio.mp3",
                        segs,
                        chunk_seconds=120.0,
                        review_model="m",
                        source_language="ko",
                        target_language="zh",
                    )

        self.assertIsNotNone(out)
        assert out is not None
        self.assertAlmostEqual(out[0]["start"], 0.0)
        # Global repair may shorten ends to enforce a small gap vs next start.
        self.assertLessEqual(out[0]["end"], 1.0)
        self.assertGreater(out[0]["end"], 0.5)
        self.assertEqual(out[0]["text"], "A")
        self.assertAlmostEqual(out[1]["start"], 1.0)
        self.assertAlmostEqual(out[1]["end"], 3.0)


if __name__ == "__main__":
    unittest.main()
