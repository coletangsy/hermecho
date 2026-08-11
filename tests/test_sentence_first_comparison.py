import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from hermecho.sentence_first_comparison import ComparisonConfig, main, run_comparison
from hermecho.sentence_first import evidence_allows_sentence_first


class TestSentenceFirstComparison(unittest.TestCase):
    def test_main_loads_dotenv_before_running_the_comparison(self) -> None:
        with patch(
            "hermecho.sentence_first_comparison.load_dotenv",
        ) as load_dotenv, patch(
            "hermecho.sentence_first_comparison.parse_args",
            return_value=object(),
        ), patch(
            "hermecho.sentence_first_comparison.run_comparison",
            return_value=Path("output/sentence-first-comparison"),
        ):
            main([])

        load_dotenv.assert_called_once_with()

    def test_run_reports_a_failed_delivery_path_before_artifact_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_video = root / "20251231_w-yGSP1c3bg.mp4"
            source_video.write_bytes(b"video")
            audio_path = root / "audio.mp3"
            audio_path.write_bytes(b"audio")

            def extract_range(_source: Path, target: Path, _start: str, _end: str) -> None:
                target.write_bytes(b"range")

            def failed_process(config) -> None:
                run_dir = Path(config.output_dir) / config.video_filename.removesuffix(".mp4")
                run_dir.mkdir(parents=True)
                (run_dir / "transcript_source.srt").write_text("source", encoding="utf-8")

            with patch(
                "hermecho.sentence_first_comparison._extract_review_range",
                side_effect=extract_range,
            ), patch(
                "hermecho.video_processing.extract_audio",
                return_value=str(audio_path),
            ), patch(
                "hermecho.transcription.resolve_transcription_backend",
                return_value="whisper",
            ), patch(
                "hermecho.transcription.transcribe_audio",
                return_value=[
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "안녕.",
                        "words": [{"word": "안녕.", "start": 0.0, "end": 1.0}],
                    }
                ],
            ), patch("hermecho.pipeline.process_video", side_effect=failed_process):
                with self.assertRaisesRegex(RuntimeError, "legacy delivery did not finish"):
                    run_comparison(
                        ComparisonConfig(
                            video_path=source_video,
                            output_dir=root / "comparison",
                        )
                    )

    def test_run_freezes_one_transcript_for_both_delivery_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_video = root / "20251231_w-yGSP1c3bg.mp4"
            source_video.write_bytes(b"video")
            output_dir = root / "comparison"
            audio_path = root / "audio.mp3"
            audio_path.write_bytes(b"audio")
            transcription = [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "안녕.",
                    "words": [{"word": "안녕.", "start": 0.0, "end": 1.0}],
                }
            ]

            def extract_range(_source: Path, target: Path, _start: str, _end: str) -> None:
                target.write_bytes(b"range")

            def fake_process(config) -> None:
                run_dir = Path(config.output_dir) / config.video_filename.removesuffix(".mp4")
                run_dir.mkdir(parents=True)
                (run_dir / f"{config.subtitle_delivery}_transcript_source.srt").write_text(
                    "source", encoding="utf-8"
                )
                (run_dir / f"{config.subtitle_delivery}_subtitles.srt").write_text(
                    "subtitle", encoding="utf-8"
                )
                (run_dir / f"{config.subtitle_delivery}_translated.mp4").write_bytes(b"video")
                (run_dir / f"{config.subtitle_delivery}_delivery_gate.txt").write_text(
                    "Structural Defects: 0", encoding="utf-8"
                )

            def fake_composite(_baseline: Path, _candidate: Path, target: Path) -> None:
                target.write_bytes(b"composite")

            with patch(
                "hermecho.sentence_first_comparison._extract_review_range",
                side_effect=extract_range,
            ), patch(
                "hermecho.video_processing.extract_audio",
                return_value=str(audio_path),
            ), patch(
                "hermecho.transcription.resolve_transcription_backend",
                return_value="whisper",
            ), patch(
                "hermecho.transcription.transcribe_audio",
                return_value=transcription,
            ), patch("hermecho.pipeline.process_video", side_effect=fake_process), patch(
                "hermecho.sentence_first_comparison._write_review_composite",
                side_effect=fake_composite,
            ):
                result = run_comparison(
                    ComparisonConfig(
                        video_path=source_video,
                        output_dir=output_dir,
                        transcription_backend="whisper",
                    )
                )

            self.assertEqual(result, output_dir)
            report = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(report["baseline"], "legacy")
            self.assertEqual(report["candidate"], "sentence-first")
            self.assertEqual(
                report["delivery_gates"],
                {"baseline": "passed", "candidate": "passed"},
            )
            self.assertTrue((output_dir / "review_composite.mp4").is_file())
            self.assertTrue((output_dir / "review.md").is_file())
            self.assertFalse(evidence_allows_sentence_first(output_dir))
            self.assertFalse(audio_path.exists())


if __name__ == "__main__":
    unittest.main()
