"""
Unit tests for subtitle burn filter construction and ffmpeg capability checks.
"""
import json
import subprocess
import types
import unittest
from unittest.mock import patch

from hermecho.video_processing import (
    _build_subtitle_style_options,
    _build_subtitles_filter,
    _ffmpeg_supports_subtitles_filter,
    burn_subtitles_into_video,
)


class TestSubtitleFilterConstruction(unittest.TestCase):

    def test_build_style_escapes_font_name(self) -> None:
        style = _build_subtitle_style_options(
            font_name="Ping:Fang's",
            font_size=12,
            outline_width=2,
            use_box_background=False,
            margin_v=25,
            margin_h=20,
            alignment=2,
        )
        self.assertIn("FontName=Ping\\:Fang\\'s", style)
        self.assertIn("BorderStyle=1", style)

    def test_build_subtitles_filter_uses_filename_option(self) -> None:
        style = "FontName=PingFang TC,FontSize=12,Outline=2"
        flt = _build_subtitles_filter(
            srt_path="/tmp/a:b's.srt",
            style_options=style,
        )
        self.assertTrue(flt.startswith("subtitles=filename='"))
        self.assertIn("/tmp/a\\:b\\'s.srt", flt)
        self.assertIn(":force_style='", flt)


class TestFfmpegCapabilityDetection(unittest.TestCase):

    @patch("hermecho.video_processing.subprocess.run")
    def test_supports_subtitles_filter_when_present(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffmpeg", "-hide_banner", "-filters"],
            returncode=0,
            stdout=" ... subtitles        V->V       Render text subtitles\n",
            stderr="",
        )
        self.assertTrue(_ffmpeg_supports_subtitles_filter())

    @patch("hermecho.video_processing.subprocess.run")
    def test_reports_false_when_subtitles_filter_missing(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffmpeg", "-hide_banner", "-filters"],
            returncode=0,
            stdout=" ... scale            V->V       Scale video\n",
            stderr="",
        )
        self.assertFalse(_ffmpeg_supports_subtitles_filter())


class TestSubtitleBurnProgress(unittest.TestCase):

    @patch("hermecho.video_processing._ffmpeg_supports_subtitles_filter", return_value=True)
    @patch("hermecho.video_processing._video_duration_seconds", return_value=10.0)
    @patch("hermecho.video_processing.subprocess.Popen")
    @patch("builtins.print")
    def test_burn_subtitles_emits_structured_progress(
        self,
        mock_print,
        mock_popen,
        _mock_duration,
        _mock_filter,
    ) -> None:
        process = types.SimpleNamespace(
            stdout=[
                "out_time_us=1000000\n",
                "out_time_us=5000000\n",
                "progress=end\n",
            ],
            stderr=[],
            returncode=0,
            wait=lambda: None,
        )
        mock_popen.return_value = process

        burn_subtitles_into_video(
            "/tmp/in.mp4",
            "/tmp/subs.srt",
            "/tmp/out.mp4",
        )

        progress_lines = [
            call.args[0]
            for call in mock_print.call_args_list
            if call.args and call.args[0].startswith("HERMECHO_PROGRESS ")
        ]
        events = [
            json.loads(line.removeprefix("HERMECHO_PROGRESS "))
            for line in progress_lines
        ]

        self.assertIn(
            {
                "stage": "burn_in",
                "status": "running",
                "message": "Burning subtitles 5/10s",
                "current": 5,
                "total": 10,
                "pct": 50,
            },
            events,
        )
        self.assertIn(
            {
                "stage": "burn_in",
                "status": "complete",
                "message": "Successfully burned subtitles into the video",
                "pct": 100,
            },
            events,
        )
