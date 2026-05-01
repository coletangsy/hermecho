"""
Unit tests for subtitle burn filter construction and ffmpeg capability checks.
"""
import subprocess
import unittest
from unittest.mock import patch

from hermecho.video_processing import (
    _build_subtitle_style_options,
    _build_subtitles_filter,
    _ffmpeg_supports_subtitles_filter,
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
