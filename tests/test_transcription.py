import os
import sys
import types
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from hermecho.transcription import transcribe_audio


class TestTranscribeAudio(unittest.TestCase):
    @patch("hermecho.transcription.os.path.exists", return_value=False)
    def test_missing_audio_path_returns_none(self, _mock_exists: MagicMock) -> None:
        out = transcribe_audio("/missing/audio.mp3", model="tiny", language="ko")

        self.assertIsNone(out)

    def test_auto_and_explicit_whisper_use_whisper_with_no_prompt_options(self) -> None:
        mock_whisper_model = MagicMock()
        segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        mock_whisper_model.transcribe.return_value = {
            "segments": segments,
            "language": "ko",
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"fake")

        try:
            fake_whisper = types.SimpleNamespace(load_model=MagicMock(return_value=mock_whisper_model))
            with patch.dict(sys.modules, {"whisper": fake_whisper}):
                for backend in ("auto", "whisper"):
                    with self.subTest(backend=backend):
                        fake_whisper.load_model.reset_mock()
                        mock_whisper_model.transcribe.reset_mock()
                        out = transcribe_audio(
                            path,
                            model="base",
                            language=None,
                            temperature=0.2,
                            backend=backend,
                        )

                        self.assertEqual(out, segments)
                        fake_whisper.load_model.assert_called_once_with("base")
                        mock_whisper_model.transcribe.assert_called_once()
                        kwargs = mock_whisper_model.transcribe.call_args.kwargs
                        self.assertNotIn("initial_prompt", kwargs)
                        self.assertNotIn("carry_initial_prompt", kwargs)
                        self.assertIsNone(kwargs["language"])
                        self.assertEqual(kwargs["temperature"], 0.2)
        finally:
            os.unlink(path)

    def test_mlx_large_models_use_large_v3_and_normalise_segments_and_words(self) -> None:
        mlx_result = {
            "language": "ko",
            "segments": [
                {
                    "id": 0,
                    "start": 0,
                    "end": 1,
                    "text": " 안녕",
                    "words": [
                        {
                            "word": " 안녕",
                            "start": 0,
                            "end": 1,
                            "probability": 0.99,
                        }
                    ],
                }
            ],
        }
        fake_mlx = types.SimpleNamespace(transcribe=MagicMock(return_value=mlx_result))

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"fake")

        try:
            with patch("platform.system", return_value="Darwin"), \
                patch("platform.machine", return_value="arm64"), \
                patch.dict(sys.modules, {"mlx_whisper": fake_mlx}), \
                patch("builtins.print") as mock_print:
                for model in ("large", "large-v3"):
                    with self.subTest(model=model):
                        fake_mlx.transcribe.reset_mock()
                        out = transcribe_audio(
                            path,
                            model=model,
                            language="ko",
                            temperature=0.2,
                            backend="mlx",
                        )

                        self.assertEqual(out, mlx_result["segments"])
                        fake_mlx.transcribe.assert_called_once_with(
                            path,
                            path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                            language="ko",
                            word_timestamps=True,
                            verbose=True,
                            temperature=0.2,
                            condition_on_previous_text=False,
                            no_speech_threshold=0.85,
                            compression_ratio_threshold=1.7,
                        )
        finally:
            os.unlink(path)

        self.assertIsInstance(out[0]["start"], float)
        self.assertIsInstance(out[0]["words"][0]["start"], float)
        mock_print.assert_any_call("MLX Whisper detected language: ko")

    def test_mlx_requires_apple_silicon(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"fake")

        try:
            with patch("platform.system", return_value="Darwin"), \
                patch("platform.machine", return_value="x86_64"), \
                patch("builtins.print") as mock_print:
                out = transcribe_audio(path, model="large", language="ko", backend="mlx")
        finally:
            os.unlink(path)

        self.assertIsNone(out)
        self.assertIn("requires Apple Silicon", str(mock_print.call_args_list))

    def test_mlx_missing_runtime_has_install_message(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"fake")

        try:
            with patch("platform.system", return_value="Darwin"), \
                patch("platform.machine", return_value="arm64"), \
                patch.dict(sys.modules, {"mlx_whisper": None}), \
                patch("builtins.print") as mock_print:
                out = transcribe_audio(path, model="large", language="ko", backend="mlx")
        finally:
            os.unlink(path)

        self.assertIsNone(out)
        self.assertIn('install -e ".[mlx]"', str(mock_print.call_args_list))

    def test_mlx_runtime_failure_does_not_fallback_to_whisper(self) -> None:
        fake_mlx = types.SimpleNamespace(transcribe=MagicMock(side_effect=RuntimeError("failed")))
        fake_whisper = types.SimpleNamespace(load_model=MagicMock())

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"fake")

        try:
            with patch("platform.system", return_value="Darwin"), \
                patch("platform.machine", return_value="arm64"), \
                patch.dict(sys.modules, {"mlx_whisper": fake_mlx, "whisper": fake_whisper}):
                out = transcribe_audio(path, model="large", language="ko", backend="mlx")
        finally:
            os.unlink(path)

        self.assertIsNone(out)
        fake_whisper.load_model.assert_not_called()

    def test_empty_whisper_segments_returns_empty_list(self) -> None:
        mock_whisper_model = MagicMock()
        mock_whisper_model.transcribe.return_value = {
            "segments": [],
            "language": "ko",
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
            tmp.write(b"fake")

        try:
            fake_whisper = types.SimpleNamespace(load_model=MagicMock(return_value=mock_whisper_model))
            with patch.dict(sys.modules, {"whisper": fake_whisper}):
                out = transcribe_audio(path, model="tiny", language="ko")
        finally:
            os.unlink(path)

        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
