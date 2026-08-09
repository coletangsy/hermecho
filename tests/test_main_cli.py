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
        self.assertEqual(config.transcription_backend, "auto")
        self.assertIsNone(config.language)
        self.assertEqual(config.target_language, "Traditional Chinese (Taiwan)")
        self.assertEqual(config.translation_model, "deepseek/deepseek-v4-pro")
        self.assertEqual(config.locked_terms_file, "references/locked_terms.json")
        self.assertEqual(config.font_name, "Heiti TC")
        self.assertEqual(
            config.fonts_dir,
            "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData",
        )
        self.assertFalse(config.transcribe_only)
        self.assertFalse(config.srt_only)
        self.assertTrue(config.box_background)

    def test_parse_args_accepts_explicit_transcription_backends(self) -> None:
        for backend in ("mlx", "whisper"):
            with self.subTest(backend=backend):
                config = cli.config_from_args(
                    cli.parse_args(["clip.mp4", "--transcription-backend", backend])
                )

                self.assertEqual(config.transcription_backend, backend)

    def test_parse_args_preserves_fonts_dir(self) -> None:
        config = cli.config_from_args(
            cli.parse_args(["clip.mp4", "--fonts-dir", "/tmp/pingfang"])
        )

        self.assertEqual(config.fonts_dir, "/tmp/pingfang")

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
    def test_translation_gate_blocks_srt_and_video_after_retries(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_path = tmp.name
            tmp.write(b"fake")

        config = PipelineConfig(
            video_filename="clip.mp4",
            input_dir="input",
            output_dir=tempfile.mkdtemp(),
            language="ko",
            stage_cooldown=0,
        )
        transcribed = [{"start": 0.0, "end": 1.0, "text": "hello"}]

        try:
            with patch("hermecho.pipeline.extract_audio", return_value=audio_path), \
                patch("hermecho.pipeline.transcribe_audio", return_value=transcribed), \
                patch("hermecho.pipeline.is_portrait_video", return_value=False), \
                patch("hermecho.pipeline.load_reference_material", return_value=""), \
                patch(
                    "hermecho.translation._translate_chunk",
                    return_value=({"translations": {"0": "  "}}, None),
                ) as translate_chunk, \
                patch("hermecho.translation.time.sleep"), \
                patch("hermecho.pipeline.generate_srt") as generate_srt, \
                patch("hermecho.pipeline.burn_subtitles_into_video") as burn, \
                patch("builtins.print") as mock_print:
                cli.process_video(config)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

        generate_srt.assert_not_called()
        burn.assert_not_called()
        self.assertEqual(translate_chunk.call_count, 3)
        messages = [str(call.args[0]) for call in mock_print.call_args_list if call.args]
        self.assertTrue(
            any("0" in message and "empty_translation" in message for message in messages)
        )

    def test_pipeline_excludes_no_speech_from_source_and_translation_delivery(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_path = tmp.name
            tmp.write(b"fake")

        transcribed = [
            {"start": 0.0, "end": 1.0, "text": "first"},
            {"start": 7.0, "end": 8.0, "text": "second"},
        ]
        translated = [
            {"start": 0.0, "end": 1.0, "text": "甲"},
            {"start": 7.0, "end": 8.0, "text": "乙"},
        ]

        try:
            for mode in ("transcribe_only", "save_source_transcript"):
                with self.subTest(mode=mode):
                    config = PipelineConfig(
                        video_filename="clip.mp4",
                        input_dir="input",
                        output_dir=tempfile.mkdtemp(),
                        transcribe_only=mode == "transcribe_only",
                        srt_only=True,
                        save_source_transcript=mode == "save_source_transcript",
                        language="ko",
                        stage_cooldown=0,
                    )
                    with patch("hermecho.pipeline.extract_audio", return_value=audio_path), \
                        patch("hermecho.pipeline.transcribe_audio", return_value=transcribed), \
                        patch("hermecho.pipeline.is_portrait_video", return_value=False), \
                        patch("hermecho.pipeline.load_reference_material", return_value=""), \
                        patch("hermecho.pipeline.load_locked_terms", return_value={}), \
                        patch("hermecho.pipeline.translate_segments", return_value=translated) as translate, \
                        patch("hermecho.pipeline.adjust_subtitle_timing", return_value=translated), \
                        patch("hermecho.pipeline.generate_srt") as generate_srt:
                        cli.process_video(config)

                    for call in generate_srt.call_args_list:
                        self.assertNotIn(
                            "[no speech]",
                            [segment["text"] for segment in call.args[0]],
                        )
                    if mode == "save_source_transcript":
                        self.assertNotIn(
                            "[no speech]",
                            [segment["text"] for segment in translate.call_args.args[0]],
                        )
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def test_pipeline_keeps_silence_boundary_for_subtitle_timing(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_path = tmp.name
            tmp.write(b"fake")

        transcribed = [
            {"start": 0.0, "end": 1.0, "text": "first"},
            {"start": 7.0, "end": 8.0, "text": "second"},
        ]
        translated = [
            {"start": 0.0, "end": 1.0, "text": "甲"},
            {"start": 7.0, "end": 8.0, "text": "乙"},
        ]
        config = PipelineConfig(
            video_filename="clip.mp4",
            input_dir="input",
            output_dir=tempfile.mkdtemp(),
            language="ko",
            srt_only=True,
            stage_cooldown=0,
        )

        try:
            with patch("hermecho.pipeline.extract_audio", return_value=audio_path), \
                patch("hermecho.pipeline.transcribe_audio", return_value=transcribed), \
                patch("hermecho.pipeline.is_portrait_video", return_value=False), \
                patch("hermecho.pipeline.load_reference_material", return_value=""), \
                patch("hermecho.pipeline.load_locked_terms", return_value={}), \
                patch("hermecho.pipeline.translate_segments", return_value=translated), \
                patch("hermecho.pipeline.generate_srt") as generate_srt:
                cli.process_video(config)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

        final_segments = generate_srt.call_args.args[0]
        self.assertNotIn("[no speech]", [segment["text"] for segment in final_segments])
        self.assertLessEqual(final_segments[0]["end"], 1.0)

    def test_pipeline_blocks_missing_or_malformed_locked_terms_files(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_path = tmp.name
            tmp.write(b"fake")

        try:
            with tempfile.TemporaryDirectory() as temporary_dir:
                malformed_path = os.path.join(temporary_dir, "malformed.json")
                with open(malformed_path, "w", encoding="utf-8") as locked_terms:
                    locked_terms.write("{")

                for label, locked_terms_path in (
                    ("missing", os.path.join(temporary_dir, "missing.json")),
                    ("malformed", malformed_path),
                ):
                    with self.subTest(locked_terms=label):
                        config = PipelineConfig(
                            video_filename="clip.mp4",
                            input_dir="input",
                            output_dir=tempfile.mkdtemp(),
                            language="ko",
                            locked_terms_file=locked_terms_path,
                            stage_cooldown=0,
                        )
                        with patch("hermecho.pipeline.extract_audio", return_value=audio_path), \
                            patch(
                                "hermecho.pipeline.transcribe_audio",
                                return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
                            ), \
                            patch("hermecho.pipeline.is_portrait_video", return_value=False), \
                            patch("hermecho.pipeline.load_reference_material", return_value=""), \
                            patch("hermecho.pipeline.translate_segments") as translate, \
                            patch("hermecho.pipeline.generate_srt") as generate_srt, \
                            patch("hermecho.pipeline.burn_subtitles_into_video") as burn:
                            cli.process_video(config)

                        translate.assert_not_called()
                        generate_srt.assert_not_called()
                        burn.assert_not_called()
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def test_pipeline_blocks_invalid_utf8_locked_terms_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_path = tmp.name
            tmp.write(b"fake")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            locked_terms_path = tmp.name
            tmp.write(b"\xff")

        config = PipelineConfig(
            video_filename="clip.mp4",
            input_dir="input",
            output_dir=tempfile.mkdtemp(),
            language="ko",
            locked_terms_file=locked_terms_path,
            stage_cooldown=0,
        )

        try:
            with patch("hermecho.pipeline.extract_audio", return_value=audio_path), \
                patch(
                    "hermecho.pipeline.transcribe_audio",
                    return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
                ), \
                patch("hermecho.pipeline.is_portrait_video", return_value=False), \
                patch("hermecho.pipeline.load_reference_material", return_value=""), \
                patch("hermecho.pipeline.translate_segments") as translate, \
                patch("hermecho.pipeline.generate_srt") as generate_srt, \
                patch("hermecho.pipeline.burn_subtitles_into_video") as burn, \
                patch("builtins.print") as mock_print:
                cli.process_video(config)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            if os.path.exists(locked_terms_path):
                os.unlink(locked_terms_path)

        translate.assert_not_called()
        generate_srt.assert_not_called()
        burn.assert_not_called()
        messages = [str(call.args[0]) for call in mock_print.call_args_list if call.args]
        self.assertTrue(any("--locked-terms-file" in message for message in messages))
        self.assertTrue(any("translation_gate" in message for message in messages))

    def test_mlx_preflight_blocks_extraction_when_unavailable(self) -> None:
        cases = (
            ("unsupported model", "Darwin", "arm64", "tiny", "supports only large-v3"),
            ("unsupported platform", "Linux", "x86_64", "large", "requires Apple Silicon"),
            ("missing runtime", "Darwin", "arm64", "large", 'install -e ".[mlx]"'),
        )

        for name, system, machine, model, expected_error in cases:
            with self.subTest(name=name), \
                patch("hermecho.transcription.platform.system", return_value=system), \
                patch("hermecho.transcription.platform.machine", return_value=machine), \
                patch.dict(sys.modules, {"mlx_whisper": None}), \
                patch("hermecho.pipeline.extract_audio", return_value=None) as extract, \
                patch("builtins.print") as mock_print:
                cli.process_video(
                    PipelineConfig(
                        video_filename="clip.mp4",
                        model=model,
                        transcription_backend="mlx",
                        stage_cooldown=0,
                    )
                )

            extract.assert_not_called()
            messages = "\n".join(call.args[0] for call in mock_print.call_args_list)
            self.assertIn(expected_error, messages)
            self.assertIn('"stage": "transcription", "status": "error"', messages)

    def test_portrait_pipeline_applies_delivery_profile_to_both_outputs(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_path = tmp.name
            tmp.write(b"fake")

        output_dir = tempfile.mkdtemp()
        config = PipelineConfig(
            video_filename="portrait.mp4",
            input_dir="input",
            output_dir=output_dir,
            language="ko",
            fonts_dir="/tmp/pingfang",
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
00:00:00,000 --> 00:00:04,000
這是前段字幕，這是後段需要
以字數分割的直式影片文字內容

2
00:00:04,000 --> 00:00:10,000
甲乙丙丁戊己庚辛壬癸子丑寅卯辰
巳午未申酉戌亥天地玄黃宇宙洪荒

""",
            )
        report_path = next(
            os.path.join(root, name)
            for root, _, files in os.walk(output_dir)
            for name in files
            if name.endswith("_delivery_gate.txt")
        )
        with open(report_path, encoding="utf-8") as report:
            self.assertIn("Repair Limits: 6", report.read())
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
                "fonts_dir": "/tmp/pingfang",
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
        translated = [{"start": 0.0, "end": 1.0, "text": "你好，世界。"}]
        adjusted = [{"start": 0.0, "end": 1.2, "text": "你好，世界。"}]

        try:
            config = PipelineConfig(**vars(args))
            with patch("hermecho.pipeline.extract_audio", return_value=audio_path), \
                patch("hermecho.pipeline.transcribe_audio", return_value=transcribed) as transcribe, \
                patch("hermecho.pipeline.translate_segments", return_value=translated) as translate, \
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
            backend="auto",
        )
        adjust.assert_called_once_with(
            translated,
            0.25,
            silence_boundaries=[],
        )
        generate_srt.assert_called_once_with(adjusted, generate_srt.call_args.args[1])
        self.assertTrue(translate.call_args.kwargs["preserve_punctuation"])
        self.assertEqual(generate_srt.call_args.args[0][0]["text"], "你好，世界。")

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
