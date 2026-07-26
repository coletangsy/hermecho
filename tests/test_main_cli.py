import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from hermecho import cli
from hermecho.pipeline import PipelineConfig


class TestCliArguments(unittest.TestCase):
    def test_removed_transcription_and_timing_flags_are_rejected(self) -> None:
        removed_flags = [
            "--whisper",
            "--multimodal-model",
            "--multimodal-chunk-seconds",
            "--initial_prompt",
            "--timing-review",
            "--no-timing-review",
            "--timing-review-model",
            "--timing-review-chunk-seconds",
        ]

        for flag in removed_flags:
            argv = ["main.py", "clip.mp4", flag]
            if flag in {
                "--multimodal-model",
                "--multimodal-chunk-seconds",
                "--initial_prompt",
                "--timing-review-model",
                "--timing-review-chunk-seconds",
            }:
                argv.append("value")
            with self.subTest(flag=flag), patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    cli.parse_args()

    def test_parse_args_maps_defaults_to_pipeline_config(self) -> None:
        args = cli.parse_args(["clip.mp4"])
        config = cli.config_from_args(args)

        self.assertIsInstance(config, PipelineConfig)
        self.assertEqual(config.video_filename, "clip.mp4")
        self.assertEqual(config.model, "large")
        self.assertIsNone(config.language)
        self.assertEqual(config.target_language, "Traditional Chinese (Taiwan)")
        self.assertEqual(config.translation_model, "deepseek/deepseek-v4-pro")
        self.assertEqual(config.font_name, "Heiti TC")
        self.assertFalse(config.transcribe_only)
        self.assertFalse(config.srt_only)
        self.assertTrue(config.box_background)

    def test_compatibility_wrapper_delegates_to_package_cli(self) -> None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        wrapper_path = os.path.join(root, "src", "main.py")
        spec = importlib.util.spec_from_file_location("hermecho_compat_main", wrapper_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with patch("hermecho.cli.main") as package_main:
            module.main()

        package_main.assert_called_once_with()


class TestPipelineOrchestration(unittest.TestCase):
    def test_portrait_pipeline_limits_cues_and_uses_them_for_both_outputs(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_path = tmp.name
            tmp.write(b"fake")

        output_dir = tempfile.mkdtemp()
        config = PipelineConfig(
            video_filename="portrait.mp4",
            input_dir="input",
            output_dir=output_dir,
            language="ko",
            stage_cooldown=0,
        )
        translated = [
            {
                "start": 0.0,
                "end": 4.0,
                "text": "這是前段字幕，這是後段需要以字數分割的直式影片文字內容",
            },
            {
                "start": 4.0,
                "end": 10.0,
                "text": "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥天地玄黃宇宙洪荒",
            },
        ]
        ffprobe_result = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "width": 1920,
                            "height": 1080,
                            "side_data_list": [{"rotation": 90}],
                        }
                    ]
                }
            ),
            stderr="",
        )

        try:
            with patch("hermecho.pipeline.extract_audio", return_value=audio_path), \
                patch("hermecho.pipeline.transcribe_audio", return_value=translated), \
                patch("hermecho.pipeline.translate_segments", return_value=translated) as translate, \
                patch("hermecho.pipeline.adjust_subtitle_timing", return_value=translated), \
                patch("hermecho.pipeline.load_reference_material", return_value=""), \
                patch("hermecho.pipeline.burn_subtitles_into_video") as burn, \
                patch("hermecho.video_processing.subprocess.run", return_value=ffprobe_result):
                cli.process_video(config)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

        srt_path = next(os.path.join(root, name) for root, _, files in os.walk(output_dir) for name in files if name.endswith(".srt"))
        with open(srt_path, encoding="utf-8") as srt:
            self.assertEqual(
                srt.read(),
                """1
00:00:00,000 --> 00:00:01,037
這是前段字幕，

2
00:00:01,037 --> 00:00:04,000
這是後段需要以字數分割的
直式影片文字內容

3
00:00:04,000 --> 00:00:08,800
甲乙丙丁戊己庚辛壬癸子丑
寅卯辰巳午未申酉戌亥天地

4
00:00:08,800 --> 00:00:10,000
玄黃宇宙洪荒

""",
            )
        burn.assert_called_once()
        burn_args, burn_kwargs = burn.call_args
        self.assertEqual(
            burn_args[:2],
            (
                os.path.abspath(os.path.join("input", "portrait.mp4")),
                os.path.abspath(srt_path),
            ),
        )
        self.assertTrue(burn_args[2].endswith("_translated.mp4"))
        self.assertEqual(
            burn_kwargs,
            {
                "font_name": "Heiti TC",
                "font_size": 12,
                "outline_width": 0,
                "use_box_background": True,
                "margin_v": 20,
                "margin_h": 10,
                "alignment": 2,
            },
        )
        self.assertTrue(translate.call_args.kwargs["preserve_punctuation"])

    def test_default_pipeline_transcribes_with_whisper_and_adjusts_timing(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_path = tmp.name
            tmp.write(b"fake")

        args = argparse.Namespace(
            video_filename="clip.mp4",
            transcribe_only=False,
            srt_only=True,
            save_source_transcript=False,
            model="tiny",
            language="ko",
            target_language="Traditional Chinese (Taiwan)",
            translation_model="openrouter-test",
            time_buffer=0.25,
            input_dir="input",
            output_dir=tempfile.mkdtemp(),
            reference_file="references/tripleS.md",
            temperature=0.0,
            font_name="PingFang TC",
            font_size=12,
            outline_width=0,
            box_background=True,
            margin_v=20,
            margin_h=10,
            alignment=2,
            stage_cooldown=0,
        )
        transcribed = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        translated = [{"start": 0.0, "end": 1.0, "text": "你好"}]
        adjusted = [{"start": 0.0, "end": 1.2, "text": "你好"}]

        try:
            config = PipelineConfig(**vars(args))
            with patch("hermecho.pipeline.extract_audio", return_value=audio_path), \
                patch("hermecho.pipeline.transcribe_audio", return_value=transcribed) as transcribe, \
                patch("hermecho.pipeline.translate_segments", return_value=translated), \
                patch("hermecho.pipeline.adjust_subtitle_timing", return_value=adjusted) as adjust, \
                patch("hermecho.pipeline.generate_srt") as generate_srt, \
                patch("hermecho.pipeline.is_portrait_video", return_value=False), \
                patch("hermecho.pipeline.load_reference_material", return_value=""):
                cli.process_video(config)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

        transcribe.assert_called_once_with(
            audio_path,
            model="tiny",
            language="ko",
            temperature=0.0,
        )
        adjust.assert_called_once_with(translated, 0.25)
        generate_srt.assert_called_once_with(adjusted, generate_srt.call_args.args[1])

    @patch.dict(sys.modules, {"timing_review": None})
    def test_full_pipeline_does_not_import_or_call_timing_review(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_path = tmp.name
            tmp.write(b"fake")

        args = argparse.Namespace(
            video_filename="clip.mp4",
            transcribe_only=False,
            srt_only=True,
            save_source_transcript=False,
            model="tiny",
            language="ko",
            target_language="Traditional Chinese (Taiwan)",
            translation_model="openrouter-test",
            time_buffer=0.1,
            input_dir="input",
            output_dir=tempfile.mkdtemp(),
            reference_file="references/tripleS.md",
            temperature=0.0,
            font_name="PingFang TC",
            font_size=12,
            outline_width=0,
            box_background=True,
            margin_v=20,
            margin_h=10,
            alignment=2,
            stage_cooldown=0,
        )
        transcribed = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        translated = [{"start": 0.0, "end": 1.0, "text": "你好"}]

        try:
            config = PipelineConfig(**vars(args))
            with patch("hermecho.pipeline.extract_audio", return_value=audio_path), \
                patch("hermecho.pipeline.transcribe_audio", return_value=transcribed), \
                patch("hermecho.pipeline.translate_segments", return_value=translated), \
                patch("hermecho.pipeline.adjust_subtitle_timing", return_value=translated), \
                patch("hermecho.pipeline.generate_srt"), \
                patch("hermecho.pipeline.is_portrait_video", return_value=False), \
                patch("hermecho.pipeline.load_reference_material", return_value=""):
                cli.process_video(config)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)


if __name__ == "__main__":
    unittest.main()
