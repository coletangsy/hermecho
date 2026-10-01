"""Remote transcription selection, cache isolation, and whole-audio fallback."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermecho.cli import config_from_args, parse_args
from hermecho.checkpoints import CheckpointStore, fingerprint_data, fingerprint_file
from hermecho.openrouter_transcription import OpenRouterRequestError
from hermecho.pipeline import PipelineConfig, process_video


class TestOpenRouterPipeline(unittest.TestCase):
    @staticmethod
    def _segments(text="hello."):
        return [{"start": 0.0, "end": 1.0, "text": text,
                 "words": [{"word": text, "start": 0.0, "end": 1.0}]}]

    def test_cli_exposes_separate_remote_model(self):
        config = config_from_args(parse_args([
            "clip.mp4", "--transcription-backend", "openrouter",
            "--transcription-model", "other/timed-model", "--model", "tiny",
        ]))
        self.assertEqual(config.transcription_backend, "openrouter")
        self.assertEqual(config.transcription_model, "other/timed-model")
        self.assertEqual(config.model, "tiny")
        defaults = config_from_args(parse_args(["clip.mp4"]))
        self.assertEqual(defaults.transcription_backend, "auto")
        self.assertEqual(defaults.transcription_model, "microsoft/mai-transcribe-2")

    def test_remote_preflight_blocks_missing_key_or_invalid_config(self):
        cases = [({}, {}), ({"OPENROUTER_API_KEY": "test"}, {"transcription_model": " "}),
                 ({"OPENROUTER_API_KEY": "test"}, {"temperature": float("nan")})]
        for environment, options in cases:
            with self.subTest(options=options), patch.dict(os.environ, environment, clear=True), \
                    patch("hermecho.pipeline.extract_audio") as extract:
                process_video(PipelineConfig(
                    "clip.mp4", transcription_backend="openrouter", **options,
                ))
                extract.assert_not_called()

    def test_remote_cache_reuses_only_matching_model_and_language(self):
        with tempfile.TemporaryDirectory() as root:
            audio = Path(root) / "audio.mp3"
            config = PipelineConfig("clip.mp4", output_dir=root, transcribe_only=True,
                                    transcription_backend="openrouter", stage_cooldown=0)

            def extract(_path):
                audio.write_bytes(b"same audio")
                return str(audio)

            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}), \
                    patch("hermecho.pipeline.extract_audio", side_effect=extract), \
                    patch("hermecho.openrouter_transcription.transcribe_openrouter",
                          return_value=self._segments()) as remote, \
                    patch("hermecho.pipeline.generate_srt") as srt:
                process_video(config)
                process_video(config)
                self.assertEqual(remote.call_count, 1)
                config.transcription_model = "other/timed-model"
                process_video(config)
                config.language = "ko"
                process_video(config)
                self.assertEqual(remote.call_count, 3)
                self.assertEqual(srt.call_count, 4)
                self.assertEqual(remote.call_args.args[1:3], ("other/timed-model", "ko"))

    def test_midpoint_assembled_checkpoint_is_recomputed(self):
        with tempfile.TemporaryDirectory() as root:
            audio = Path(root) / "audio.mp3"
            audio.write_bytes(b"same audio")
            config = PipelineConfig("clip.mp4", output_dir=root, transcribe_only=True,
                                    transcription_backend="openrouter", stage_cooldown=0)
            old_fingerprint = fingerprint_data({
                "audio": fingerprint_file(str(audio)), "backend": "openrouter",
                "language": None, "model": config.transcription_model,
                "temperature": 0.0, "rules": "openrouter-word-chunks-v1",
            })
            checkpoint = CheckpointStore(str(Path(root) / "clip" / ".hermecho-checkpoint.json"))
            checkpoint.save_transcription(old_fingerprint, [{
                "start": 59.8, "end": 60.1, "text": "hello hello",
                "words": [{"word": "hello", "start": 59.8, "end": 59.9},
                          {"word": "hello", "start": 60.0, "end": 60.1}],
            }], require_words=True)
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}), \
                    patch("hermecho.pipeline.extract_audio", return_value=str(audio)), \
                    patch("hermecho.openrouter_transcription.transcribe_openrouter",
                          return_value=self._segments()) as remote, \
                    patch("hermecho.pipeline.generate_srt") as srt:
                process_video(config)
            remote.assert_called_once()
            self.assertEqual(srt.call_args.args[0], self._segments())

    def test_conflicting_raw_overlap_cache_blocks_without_fallback_or_reupload(self):
        with tempfile.TemporaryDirectory() as root:
            audio = Path(root) / "audio.mp3"

            def extract(_path):
                audio.write_bytes(b"same audio")
                return str(audio)

            def extract_chunk(_source, destination, *_args):
                destination.write_bytes(b"chunk")

            bodies = [
                ({"text": "hello", "words": [{"word": "hello", "start": 59.8, "end": 59.9}]}, {}),
                ({"text": "different", "words": [{"word": "different", "start": 1, "end": 1.1}]}, {}),
            ]
            from hermecho.transcription import transcribe_audio

            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}), \
                    patch("hermecho.pipeline.extract_audio", side_effect=extract), \
                    patch("hermecho.pipeline.transcribe_audio", wraps=transcribe_audio) as transcribe, \
                    patch("hermecho.openrouter_transcription._ffprobe_duration", return_value=61), \
                    patch("hermecho.openrouter_transcription._extract_audio", side_effect=extract_chunk), \
                    patch("hermecho.openrouter_transcription._request_transcription", side_effect=bodies) as request, \
                    patch("hermecho.pipeline.generate_srt") as srt:
                config = PipelineConfig("clip.mp4", output_dir=root, transcribe_only=True,
                                        transcription_backend="openrouter", stage_cooldown=0)
                process_video(config)
                process_video(config)
            self.assertEqual(request.call_count, 2)
            self.assertEqual(transcribe.call_count, 2)
            self.assertTrue(all(call.kwargs["backend"] == "openrouter" for call in transcribe.call_args_list))
            srt.assert_not_called()
            self.assertFalse((Path(root) / "clip" / ".hermecho-checkpoint.json").exists())
            self.assertTrue((Path(root) / "clip" / ".openrouter-transcription.json").exists())

    def test_request_failure_uses_only_whisper_and_local_fingerprint(self):
        with tempfile.TemporaryDirectory() as root:
            audio = Path(root) / "audio.mp3"
            audio.write_bytes(b"audio")
            expected_fingerprint = fingerprint_data({
                "audio": fingerprint_file(str(audio)), "backend": "whisper",
                "model": "tiny", "language": "ko", "temperature": 0.0,
            })
            local = self._segments("local words.")
            config = PipelineConfig("clip.mp4", output_dir=root, transcribe_only=True,
                                    transcription_backend="openrouter", model="tiny",
                                    language="ko", stage_cooldown=0)
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}), \
                    patch("hermecho.pipeline.extract_audio", return_value=str(audio)), \
                    patch("hermecho.pipeline.transcribe_audio",
                          side_effect=[OpenRouterRequestError("timeout"), local]) as transcribe, \
                    patch("hermecho.pipeline.generate_srt") as srt:
                process_video(config)
            self.assertEqual(transcribe.call_args.kwargs["backend"], "whisper")
            self.assertEqual(srt.call_args.args[0], local)
            checkpoint = CheckpointStore(str(Path(root) / "clip" / ".hermecho-checkpoint.json"))
            self.assertEqual(checkpoint.load_transcription(expected_fingerprint), local)

    def test_invalid_response_blocks_without_local_fallback_or_delivery(self):
        with tempfile.TemporaryDirectory() as root:
            audio = Path(root) / "audio.mp3"
            audio.write_bytes(b"audio")
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}), \
                    patch("hermecho.pipeline.extract_audio", return_value=str(audio)), \
                    patch("hermecho.pipeline.transcribe_audio",
                          side_effect=RuntimeError("no word timestamps")) as transcribe, \
                    patch("hermecho.pipeline.generate_srt") as srt, \
                    patch("hermecho.pipeline.translate_segments") as translate:
                process_video(PipelineConfig("clip.mp4", output_dir=root,
                    transcription_backend="openrouter", stage_cooldown=0))
            self.assertEqual(transcribe.call_count, 1)
            srt.assert_not_called()
            translate.assert_not_called()
            self.assertFalse((Path(root) / "clip" / ".hermecho-checkpoint.json").exists())

    def test_auto_with_key_stays_local(self):
        with tempfile.TemporaryDirectory() as root:
            audio = Path(root) / "audio.mp3"
            audio.write_bytes(b"audio")
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}), \
                    patch("hermecho.pipeline.extract_audio", return_value=str(audio)), \
                    patch("hermecho.pipeline.resolve_transcription_backend", return_value="whisper"), \
                    patch("hermecho.pipeline.transcribe_audio", return_value=self._segments()) as transcribe, \
                    patch("hermecho.pipeline.generate_srt"):
                process_video(PipelineConfig("clip.mp4", output_dir=root,
                    transcribe_only=True, stage_cooldown=0))
            self.assertEqual(transcribe.call_args.kwargs["backend"], "whisper")
            self.assertNotIn("transcription_model", transcribe.call_args.kwargs)

    def test_remote_words_feed_translated_delivery(self):
        with tempfile.TemporaryDirectory() as root:
            audio = Path(root) / "audio.mp3"
            audio.write_bytes(b"audio")

            def translate(sentences, **_kwargs):
                return [{**sentence, "text": "你好。"} for sentence in sentences]

            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}), \
                    patch("hermecho.pipeline.extract_audio", return_value=str(audio)), \
                    patch("hermecho.openrouter_transcription.transcribe_openrouter",
                          return_value=self._segments()), \
                    patch("hermecho.pipeline.load_reference_material", return_value=""), \
                    patch("hermecho.pipeline.load_locked_terms", return_value={}), \
                    patch("hermecho.pipeline.is_portrait_video", return_value=False), \
                    patch("hermecho.pipeline.translate_segments", side_effect=translate) as translation, \
                    patch("hermecho.pipeline.generate_srt") as srt:
                process_video(PipelineConfig("clip.mp4", output_dir=root,
                    transcription_backend="openrouter", srt_only=True, stage_cooldown=0))
            self.assertEqual(translation.call_args.args[0][0]["source_words"],
                             self._segments()[0]["words"])
            self.assertEqual(srt.call_args.args[0][0]["text"], "你好。")


if __name__ == "__main__":
    unittest.main()
