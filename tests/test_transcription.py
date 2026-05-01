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

    def test_whisper_transcribe_receives_no_prompt_options(self) -> None:
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
                out = transcribe_audio(
                    path,
                    model="base",
                    language="ko",
                    temperature=0.2,
                )
        finally:
            os.unlink(path)

        self.assertEqual(out, segments)
        fake_whisper.load_model.assert_called_once_with("base")
        mock_whisper_model.transcribe.assert_called_once()
        kwargs = mock_whisper_model.transcribe.call_args.kwargs
        self.assertNotIn("initial_prompt", kwargs)
        self.assertNotIn("carry_initial_prompt", kwargs)
        self.assertEqual(kwargs["language"], "ko")
        self.assertEqual(kwargs["temperature"], 0.2)

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
