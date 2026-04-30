import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from asr_comparison import (
    build_comparison_report,
    collect_vibevoice_preflight,
    compute_segment_stats,
    load_gemini_baseline,
    load_existing_engine_artifacts,
    load_srt_segments,
    _process_whisper,
    _process_downstream,
    _parse_args,
)
from vibevoice_asr import normalize_vibevoice_output, transcribe_audio_vibevoice
from vibevoice_asr import normalize_vibevoice_text_output, transcribe_audio_vibevoice_gradio


class TestNormalizeVibeVoiceOutput(unittest.TestCase):
    def test_normalizes_nested_segments_with_timestamp_pairs(self) -> None:
        raw = {
            "segments": [
                {"timestamp": [1.25, 2.5], "text": " hello "},
                {"start": "00:00:03,000", "end": "00:00:04,250", "sentence": "world"},
                {"start": 5, "end": 4, "text": "swapped"},
                {"start": 9, "end": 10, "text": "   "},
            ]
        }

        out = normalize_vibevoice_output(raw)

        self.assertEqual(
            out,
            [
                {"start": 1.25, "end": 2.5, "text": "hello"},
                {"start": 3.0, "end": 4.25, "text": "world"},
                {"start": 4.0, "end": 5.0, "text": "swapped"},
            ],
        )

    def test_normalizes_transcription_result_list(self) -> None:
        raw = {
            "transcription": [
                {"start_time": 0, "end_time": 1.2, "transcript": "first"},
                {"begin": 2.0, "finish": 3.0, "content": "second"},
            ]
        }

        out = normalize_vibevoice_output(raw)

        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "first")
        self.assertEqual(out[1]["start"], 2.0)

    def test_normalizes_gradio_text_output_from_json(self) -> None:
        text = (
            '<|im_start|>assistant\n'
            '[{"Start": 0.0, "End": 1.5, "Content": "안녕"}]'
            '<|im_end|>'
        )

        out = normalize_vibevoice_text_output(text)

        self.assertEqual(out, [{"start": 0.0, "end": 1.5, "text": "안녕"}])

    def test_normalizes_gradio_text_output_from_srt(self) -> None:
        text = "1\n00:00:01,000 --> 00:00:02,250\nhello\n\n"

        out = normalize_vibevoice_text_output(text)

        self.assertEqual(out, [{"start": 1.0, "end": 2.25, "text": "hello"}])


class TestAsrComparisonReport(unittest.TestCase):
    def test_load_srt_segments_handles_multiline_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sample.srt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    "1\n"
                    "00:00:01,250 --> 00:00:02,500\n"
                    "hello\n"
                    "world\n\n"
                    "2\n"
                    "00:00:03,000 --> 00:00:04,000\n"
                    "again\n\n"
                )

            segments = load_srt_segments(path)

        self.assertEqual(
            segments,
            [
                {"start": 1.25, "end": 2.5, "text": "hello world"},
                {"start": 3.0, "end": 4.0, "text": "again"},
            ],
        )

    def test_load_gemini_baseline_requires_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "gemini_source.srt"), "w", encoding="utf-8") as fh:
                fh.write("1\n00:00:00,000 --> 00:00:01,000\n안녕\n\n")
            with open(os.path.join(tmp, "gemini_translated.srt"), "w", encoding="utf-8") as fh:
                fh.write("1\n00:00:00,000 --> 00:00:01,000\n你好\n\n")
            with open(os.path.join(tmp, "gemini_burned.mp4"), "wb") as fh:
                fh.write(b"video")

            segments, artifacts = load_gemini_baseline(tmp)

        self.assertEqual(segments[0]["text"], "안녕")
        self.assertTrue(artifacts["gemini_source_srt"].endswith("gemini_source.srt"))
        self.assertIn("gemini_baseline_dir", artifacts)

    def test_load_gemini_baseline_reports_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_gemini_baseline(tmp)

        self.assertIn("gemini_source.srt", str(ctx.exception))
        self.assertIn("gemini_translated.srt", str(ctx.exception))
        self.assertIn("gemini_burned.mp4", str(ctx.exception))

    def test_load_existing_engine_artifacts_accepts_partial_cached_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "vibevoice_source.srt"), "w", encoding="utf-8") as fh:
                fh.write("1\n00:00:00,000 --> 00:00:01,000\n안녕\n\n")

            segments, artifacts = load_existing_engine_artifacts(tmp, "vibevoice")

        self.assertEqual(segments, [{"start": 0.0, "end": 1.0, "text": "안녕"}])
        self.assertEqual(list(artifacts.keys()), ["vibevoice_source_srt"])

    def test_compute_segment_stats(self) -> None:
        stats = compute_segment_stats(
            [
                {"start": 1.0, "end": 2.0, "text": "a"},
                {"start": 4.0, "end": 7.0, "text": "b"},
            ],
            media_duration=10.0,
        )

        self.assertEqual(stats["segment_count"], 2)
        self.assertEqual(stats["coverage_seconds"], 4.0)
        self.assertEqual(stats["coverage_percent"], 40.0)
        self.assertEqual(stats["average_segment_duration"], 2.0)

    def test_build_report_includes_artifacts_and_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "comparison_report.md")
            artifacts = {
                "gemini_source_srt": "gemini_source.srt",
                "gemini_translated_srt": "gemini_translated.srt",
                "gemini_burned_video": "gemini_burned.mp4",
            }

            text = build_comparison_report(
                output_path=report_path,
                input_video="input/Wsp9Z6-S0LA.mp4",
                media_metadata={
                    "duration": 462.309297,
                    "audio_codec": "aac",
                    "video_codec": "av1",
                },
                gemini_segments=[{"start": 0, "end": 1, "text": "안녕"}],
                vibevoice_segments=[],
                artifacts=artifacts,
                vibevoice_error="torch import failed",
                vibevoice_preflight={
                    "model_id": "microsoft/VibeVoice-ASR-HF",
                    "device": "auto",
                    "dtype": "auto",
                    "tokenizer_chunk_size": 64000,
                    "torch_available": False,
                },
            )

            self.assertTrue(os.path.exists(report_path))
            self.assertIn("input/Wsp9Z6-S0LA.mp4", text)
            self.assertIn("gemini_source.srt", text)
            self.assertIn("torch import failed", text)
            self.assertIn("docker run --gpus all", text)
            self.assertIn("## VibeVoice Hardware Preflight", text)
            self.assertIn("microsoft/VibeVoice-ASR-HF", text)

    def test_build_report_includes_gradio_provider_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "comparison_report.md")

            text = build_comparison_report(
                output_path=report_path,
                input_video="input/Wsp9Z6-S0LA.mp4",
                media_metadata={"duration": 10.0},
                gemini_segments=[],
                vibevoice_segments=[],
                artifacts={"vibevoice_upload_audio": "upload.flac"},
                vibevoice_provider_info={
                    "provider": "gradio",
                    "gradio_url": "https://example.gradio.live/",
                    "gradio_version": "6.9.0",
                    "api_name": "transcribe_wrapper",
                    "upload_audio_path": "upload.flac",
                    "upload_audio_size_mb": 16.25,
                    "api_settings": {
                        "max_new_tokens": 4096,
                        "enable_sampling": False,
                        "temperature": 0.0,
                        "top_p": 1.0,
                    },
                },
            )

        self.assertIn("## VibeVoice Provider", text)
        self.assertIn("https://example.gradio.live/", text)
        self.assertIn("16.25 MB", text)

    def test_build_report_includes_whisper_artifacts_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "comparison_report.md")

            text = build_comparison_report(
                output_path=report_path,
                input_video="input/Wsp9Z6-S0LA.mp4",
                media_metadata={"duration": 10.0},
                gemini_segments=[{"start": 0, "end": 2, "text": "gemini"}],
                vibevoice_segments=[{"start": 0, "end": 2, "text": "vibe"}],
                whisper_segments=[{"start": 0, "end": 3, "text": "whisper"}],
                artifacts={
                    "whisper_source_srt": "whisper_source.srt",
                    "whisper_translated_srt": "whisper_translated.srt",
                    "whisper_burned_video": "whisper_burned.mp4",
                },
            )

        self.assertIn("# Gemini vs VibeVoice-ASR vs Whisper Comparison", text)
        self.assertIn("Whisper source SRT", text)
        self.assertIn("| Whisper | 1 | 3.00s | 30.00% | 3.00s |", text)
        self.assertIn("| Time | Gemini source | VibeVoice source | Whisper source |", text)

    def test_build_report_includes_optional_whisper_no_prompt_artifacts_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "comparison_report.md")

            text = build_comparison_report(
                output_path=report_path,
                input_video="input/Wsp9Z6-S0LA.mp4",
                media_metadata={"duration": 10.0},
                gemini_segments=[{"start": 0, "end": 2, "text": "gemini"}],
                vibevoice_segments=[{"start": 0, "end": 2, "text": "vibe"}],
                whisper_segments=[{"start": 0, "end": 3, "text": "whisper"}],
                whisper_no_prompt_segments=[
                    {"start": 0, "end": 4, "text": "whisper no prompt"}
                ],
                artifacts={
                    "whisper_no_prompt_source_srt": "whisper_no_prompt_source.srt",
                    "whisper_no_prompt_translated_srt": "whisper_no_prompt_translated.srt",
                    "whisper_no_prompt_burned_video": "whisper_no_prompt_burned.mp4",
                },
            )

        self.assertIn("Whisper no-prompt source SRT", text)
        self.assertIn(
            "| Whisper (no prompt) | 1 | 4.00s | 40.00% | 4.00s |",
            text,
        )
        self.assertIn("Whisper (no prompt) source", text)
        self.assertIn("Whisper (no prompt) similarity", text)

    def test_build_report_omits_whisper_no_prompt_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "comparison_report.md")

            text = build_comparison_report(
                output_path=report_path,
                input_video="input/Wsp9Z6-S0LA.mp4",
                media_metadata={"duration": 10.0},
                gemini_segments=[],
                vibevoice_segments=[],
                artifacts={},
            )

        self.assertNotIn("Whisper (no prompt)", text)
        self.assertNotIn("whisper_no_prompt", text)


class TestProcessDownstream(unittest.TestCase):
    def _args(self) -> MagicMock:
        args = MagicMock()
        args.target_language = "Traditional Chinese (Taiwan)"
        args.translation_model = "gemini-3.1-flash-lite-preview"
        args.timing_review = True
        args.timing_review_chunk_seconds = 120.0
        args.timing_review_model = "gemini-3.1-flash-lite-preview"
        args.language = "ko"
        args.font_name = "Arial Unicode MS"
        args.font_size = 12
        args.outline_width = 0
        args.box_background = True
        args.margin_v = 20
        args.margin_h = 10
        args.alignment = 2
        return args

    @patch("asr_comparison.burn_subtitles_into_video")
    @patch("asr_comparison.review_subtitle_timing")
    @patch("asr_comparison.translate_segments")
    def test_vibevoice_downstream_skips_timing_review_and_preserves_source_timings(
        self,
        mock_translate: MagicMock,
        mock_review: MagicMock,
        mock_burn: MagicMock,
    ) -> None:
        source_segments = [
            {"start": 1.25, "end": 2.5, "text": "안녕"},
            {"start": 3.0, "end": 4.75, "text": "다시"},
        ]
        mock_translate.return_value = [
            {**segment, "text": translated}
            for segment, translated in zip(source_segments, ["你好", "再次"])
        ]

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _process_downstream(
                "vibevoice",
                "input.mp4",
                "audio.wav",
                source_segments,
                tmp,
                "reference",
                self._args(),
                apply_timing_review=False,
            )

            translated_segments = load_srt_segments(artifacts["vibevoice_translated_srt"])

        mock_review.assert_not_called()
        mock_burn.assert_called_once()
        self.assertEqual(
            [(seg["start"], seg["end"]) for seg in translated_segments],
            [(1.25, 2.5), (3.0, 4.75)],
        )
        self.assertEqual([seg["text"] for seg in translated_segments], ["你好", "再次"])

    @patch("asr_comparison.burn_subtitles_into_video")
    @patch("asr_comparison.review_subtitle_timing")
    @patch("asr_comparison.translate_segments")
    def test_gemini_downstream_uses_timing_review_when_enabled(
        self,
        mock_translate: MagicMock,
        mock_review: MagicMock,
        _mock_burn: MagicMock,
    ) -> None:
        source_segments = [{"start": 1.0, "end": 2.0, "text": "안녕"}]
        translated = [{"start": 1.0, "end": 2.0, "text": "你好"}]
        reviewed = [{"start": 1.5, "end": 2.5, "text": "你好"}]
        mock_translate.return_value = translated
        mock_review.return_value = reviewed

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _process_downstream(
                "gemini",
                "input.mp4",
                "audio.wav",
                source_segments,
                tmp,
                "reference",
                self._args(),
            )

            translated_segments = load_srt_segments(artifacts["gemini_translated_srt"])

        mock_review.assert_called_once()
        self.assertEqual(
            [(seg["start"], seg["end"]) for seg in translated_segments],
            [(1.5, 2.5)],
        )

    @patch("asr_comparison.burn_subtitles_into_video")
    @patch("asr_comparison.review_subtitle_timing")
    @patch("asr_comparison.translate_segments")
    @patch("asr_comparison.transcribe_audio")
    def test_whisper_path_generates_labeled_artifacts_without_timing_review(
        self,
        mock_transcribe: MagicMock,
        mock_translate: MagicMock,
        mock_review: MagicMock,
        mock_burn: MagicMock,
    ) -> None:
        mock_transcribe.return_value = [
            {"start": 1.0, "end": 2.0, "text": "안녕"},
            {"start": 3.0, "end": 4.5, "text": "다시"},
        ]
        mock_translate.return_value = [
            {"start": 1.0, "end": 2.0, "text": "你好"},
            {"start": 3.0, "end": 4.5, "text": "再次"},
        ]
        mock_burn.side_effect = lambda _video, _srt, out, **_kwargs: open(out, "wb").close()
        args = self._args()
        args.whisper_model = "large-v3"
        args.temperature = 0.0

        with tempfile.TemporaryDirectory() as tmp:
            segments, artifacts = _process_whisper(
                video_path="input.mp4",
                audio_path="audio.wav",
                output_dir=tmp,
                reference_material="reference",
                prompt="conversation Context: tripleS",
                args=args,
            )

            source_segments = load_srt_segments(artifacts["whisper_source_srt"])
            translated_segments = load_srt_segments(artifacts["whisper_translated_srt"])

        mock_transcribe.assert_called_once_with(
            "audio.wav",
            model="large-v3",
            language="ko",
            initial_prompt="conversation Context: tripleS",
            temperature=0.0,
        )
        mock_review.assert_not_called()
        mock_burn.assert_called_once()
        self.assertIn("whisper_burned_video", artifacts)
        self.assertEqual(segments, source_segments)
        self.assertEqual(
            [(seg["start"], seg["end"]) for seg in translated_segments],
            [(1.0, 2.0), (3.0, 4.5)],
        )

    @patch("asr_comparison.burn_subtitles_into_video")
    @patch("asr_comparison.review_subtitle_timing")
    @patch("asr_comparison.translate_segments")
    @patch("asr_comparison.transcribe_audio")
    def test_whisper_no_prompt_path_uses_none_prompt_and_labeled_artifacts(
        self,
        mock_transcribe: MagicMock,
        mock_translate: MagicMock,
        mock_review: MagicMock,
        mock_burn: MagicMock,
    ) -> None:
        mock_transcribe.return_value = [{"start": 1.0, "end": 2.0, "text": "안녕"}]
        mock_translate.return_value = [{"start": 1.0, "end": 2.0, "text": "你好"}]
        mock_burn.side_effect = lambda _video, _srt, out, **_kwargs: open(out, "wb").close()
        args = self._args()
        args.whisper_model = "large-v3"
        args.temperature = 0.0

        with tempfile.TemporaryDirectory() as tmp:
            segments, artifacts = _process_whisper(
                video_path="input.mp4",
                audio_path="audio.wav",
                output_dir=tmp,
                reference_material=None,
                prompt=None,
                args=args,
                label="whisper_no_prompt",
            )

        mock_transcribe.assert_called_once_with(
            "audio.wav",
            model="large-v3",
            language="ko",
            initial_prompt=None,
            temperature=0.0,
        )
        mock_review.assert_not_called()
        self.assertEqual(segments[0]["text"], "안녕")
        self.assertIn("whisper_no_prompt_source_srt", artifacts)
        self.assertIn("whisper_no_prompt_translated_srt", artifacts)
        self.assertIn("whisper_no_prompt_burned_video", artifacts)


class TestVibeVoicePreflight(unittest.TestCase):
    def test_collect_preflight_has_requested_options_without_torch(self) -> None:
        with patch.dict("sys.modules", {"torch": None}):
            info = collect_vibevoice_preflight(
                model_id="microsoft/VibeVoice-ASR-HF",
                device="auto",
                dtype="auto",
                tokenizer_chunk_size=64000,
            )

        self.assertEqual(info["model_id"], "microsoft/VibeVoice-ASR-HF")
        self.assertEqual(info["device"], "auto")
        self.assertEqual(info["dtype"], "auto")
        self.assertEqual(info["tokenizer_chunk_size"], 64000)
        self.assertFalse(info["torch_available"])

    def test_parse_args_exposes_vibevoice_options(self) -> None:
        args = _parse_args(
            [
                "--skip-gemini",
                "--vibevoice-device",
                "mps",
                "--vibevoice-dtype",
                "bfloat16",
                "--vibevoice-tokenizer-chunk-size",
                "64000",
                "--vibevoice-transcribe-only",
                "--baseline-dir",
                "output/run",
                "--vibevoice-timeout-seconds",
                "120",
                "--vibevoice-provider",
                "gradio",
                "--vibevoice-gradio-url",
                "https://example.gradio.live/",
                "--vibevoice-gradio-http-timeout",
                "600",
                "--skip-whisper",
                "--whisper-model",
                "large-v3",
            ]
        )

        self.assertTrue(args.skip_gemini)
        self.assertEqual(args.vibevoice_device, "mps")
        self.assertEqual(args.vibevoice_dtype, "bfloat16")
        self.assertEqual(args.vibevoice_tokenizer_chunk_size, 64000)
        self.assertTrue(args.vibevoice_transcribe_only)
        self.assertEqual(args.baseline_dir, "output/run")
        self.assertEqual(args.vibevoice_timeout_seconds, 120)
        self.assertEqual(args.vibevoice_provider, "gradio")
        self.assertEqual(args.vibevoice_gradio_url, "https://example.gradio.live/")
        self.assertEqual(args.vibevoice_gradio_http_timeout, 600)
        self.assertTrue(args.skip_whisper)
        self.assertEqual(args.whisper_model, "large-v3")

    def test_parse_args_exposes_whisper_no_prompt_option(self) -> None:
        args = _parse_args(["--include-whisper-no-prompt"])

        self.assertTrue(args.include_whisper_no_prompt)


class TestTranscribeAudioVibeVoice(unittest.TestCase):
    @patch("vibevoice_asr.import_transformers_vibevoice")
    def test_passes_device_dtype_and_tokenizer_chunk_size(
        self,
        mock_import: MagicMock,
    ) -> None:
        mock_processor = MagicMock()
        mock_inputs = {"input_ids": MagicMock()}
        mock_inputs["input_ids"].shape = [1, 3]
        mock_prepared = MagicMock()
        mock_prepared.__getitem__.side_effect = mock_inputs.__getitem__
        mock_prepared.to.return_value = mock_prepared
        mock_processor.apply_transcription_request.return_value = mock_prepared
        mock_processor.decode.return_value = [
            [{"Start": 0.0, "End": 1.0, "Content": "hello"}]
        ]

        mock_model = MagicMock()
        mock_model.device = "mps"
        mock_model.dtype = "bfloat16"
        mock_model.generate.return_value = MagicMock()

        mock_auto_processor = MagicMock()
        mock_auto_model = MagicMock()
        mock_auto_processor.from_pretrained.return_value = mock_processor
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_import.return_value = (mock_auto_processor, mock_auto_model, None)

        raw, segments = transcribe_audio_vibevoice(
            "audio.mp3",
            model_id="microsoft/VibeVoice-ASR-HF",
            prompt="names",
            device="mps",
            dtype="bfloat16",
            tokenizer_chunk_size=64000,
        )

        mock_auto_model.from_pretrained.assert_called_once_with(
            "microsoft/VibeVoice-ASR-HF",
            device_map="mps",
            torch_dtype="bfloat16",
        )
        mock_model.generate.assert_called_once_with(
            **mock_prepared,
            acoustic_tokenizer_chunk_size=64000,
        )
        self.assertEqual(raw[0]["Content"], "hello")
        self.assertEqual(segments[0]["text"], "hello")

    @patch("vibevoice_asr.validate_vibevoice_gradio_url")
    @patch("vibevoice_asr.import_gradio_client")
    def test_gradio_prefers_downloaded_srt_and_passes_api_settings(
        self,
        mock_import_gradio: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        mock_validate.return_value = {
            "gradio_version": "6.9.0",
            "api_name": "transcribe_wrapper",
        }
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = os.path.join(tmp, "audio.flac")
            preview_path = os.path.join(tmp, "preview.mp3")
            srt_path = os.path.join(tmp, "result.srt")
            with open(audio_path, "wb") as fh:
                fh.write(b"audio")
            with open(preview_path, "wb") as fh:
                fh.write(b"preview")
            with open(srt_path, "w", encoding="utf-8") as fh:
                fh.write("1\n00:00:01,000 --> 00:00:02,000\nfrom srt\n\n")

            mock_client = MagicMock()
            mock_client.predict.side_effect = [
                ({"visible": True, "value": preview_path, "__type__": "update"}, None),
                (
                    '[{"Start": 9, "End": 10, "Content": "from raw"}]',
                    "<div></div>",
                    "<div></div>",
                    {"path": srt_path},
                ),
            ]
            mock_client_class = MagicMock(return_value=mock_client)
            mock_handle_file = MagicMock(side_effect=lambda path: {"path": path})
            mock_import_gradio.return_value = (mock_client_class, mock_handle_file)

            raw, segments, downloaded_srt = transcribe_audio_vibevoice_gradio(
                audio_path,
                gradio_url="https://example.gradio.live/",
                hotwords_context="tripleS",
                max_new_tokens=2048,
                enable_sampling=False,
                temperature=0.0,
                top_p=1.0,
                http_timeout_seconds=600,
            )

        mock_client_class.assert_called_once_with(
            "https://example.gradio.live/",
            httpx_kwargs={"timeout": 600},
        )
        self.assertEqual(mock_client.predict.call_count, 2)
        mock_client.predict.assert_any_call(
            {"path": audio_path},
            api_name="/update_media_preview",
        )
        mock_client.predict.assert_any_call(
            {"path": audio_path},
            None,
            None,
            {"path": preview_path},
            None,
            2048,
            0.0,
            1.0,
            False,
            "tripleS",
            api_name="/transcribe_wrapper",
        )
        self.assertEqual(downloaded_srt, srt_path)
        self.assertEqual(segments, [{"start": 1.0, "end": 2.0, "text": "from srt"}])
        self.assertEqual(raw["provider"], "gradio")
        self.assertEqual(raw["api_settings"]["max_new_tokens"], 2048)


if __name__ == "__main__":
    unittest.main()
