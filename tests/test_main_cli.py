import argparse
import importlib.util
import os
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
        self.assertEqual(config.language, "ko")
        self.assertEqual(config.target_language, "Traditional Chinese (Taiwan)")
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
            translation_model="gemini-test",
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
            translation_model="gemini-test",
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
                patch("hermecho.pipeline.load_reference_material", return_value=""):
                cli.process_video(config)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)


if __name__ == "__main__":
    unittest.main()
