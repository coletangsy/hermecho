import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermecho import cli
from hermecho.asr_comparison import (
    ComparisonConfig,
    evidence_allows_mlx,
    run_comparison,
)


class TestAsrComparisonEvidence(unittest.TestCase):
    def _write_evidence(
        self,
        directory: Path,
        *,
        mlx_times: list[float] = [2.0, 3.0, 4.0],
        approved: bool = True,
    ) -> None:
        runs = []
        for iteration, (whisper_time, mlx_time) in enumerate(
            zip([5.0, 6.0, 7.0], mlx_times), start=1
        ):
            runs.extend(
                [
                    {
                        "backend": "whisper",
                        "iteration": iteration,
                        "fresh_process": True,
                        "warm_cache": False,
                        "completed": True,
                        "returncode": 0,
                        "transcription_seconds": whisper_time,
                        "end_to_end_seconds": whisper_time + 1,
                        "peak_memory_bytes": 100,
                    },
                    {
                        "backend": "mlx",
                        "iteration": iteration,
                        "fresh_process": True,
                        "warm_cache": False,
                        "completed": True,
                        "returncode": 0,
                        "transcription_seconds": mlx_time,
                        "end_to_end_seconds": mlx_time + 1,
                        "peak_memory_bytes": 100,
                    },
                ]
            )
        (directory / "comparison.json").write_text(
            json.dumps(
                {
                    "comparison_variable": "transcription_backend",
                    "manifest": {"model_checkpoint": "large", "path": "manifest.json"},
                    "runs": runs,
                    "warmups": [
                        {
                            "backend": backend,
                            "fresh_process": True,
                            "warm_cache": True,
                            "completed": True,
                            "returncode": 0,
                            "transcription_seconds": 1.0,
                            "end_to_end_seconds": 2.0,
                            "peak_memory_bytes": 100,
                        }
                        for backend in ("whisper", "mlx")
                    ],
                    "artifacts": {
                        "manifest": "manifest.json",
                        "source_transcript_diff": "source_transcript.diff",
                        "review_composite": "review_composite.mp4",
                    },
                }
            ),
            encoding="utf-8",
        )
        source = directory / "20251231_w-yGSP1c3bg.mp4"
        prepared_media = directory / "media_range.mp4"
        prepared_media.write_bytes(b"video")
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "comparison_variable": "transcription_backend",
                    "media_range": {
                        "source": str(source),
                        "source_name": source.name,
                        "start": "00:29:30.000",
                        "end": "00:39:30.000",
                        "prepared_media": str(prepared_media),
                    },
                    "shared": {
                        "model_checkpoint": "large",
                        "machine": {},
                        "runtime_versions": {},
                        "language": "ko",
                        "temperature": 0.0,
                        "prompt": None,
                        "references": {
                            "reference_file": "references/tripleS.md",
                            "locked_terms_file": "references/locked_terms.json",
                        },
                        "translation": {
                            "provider": "OpenRouter",
                            "model": "deepseek/deepseek-v4-pro",
                            "target_language": "Traditional Chinese (Taiwan)",
                        },
                        "subtitle_style": {
                            "font_name": "Heiti TC",
                            "fonts_dir": "/System/fonts",
                            "font_size": 12,
                            "outline_width": 0,
                            "box_background": True,
                            "margin_v": 20,
                            "margin_h": 10,
                            "alignment": 2,
                            "time_buffer": 0.1,
                        },
                        "effective_cli_options": {
                            "video_filename": "media_range.mp4",
                            "input_dir": str(directory),
                            "model": "large",
                            "language": "ko",
                            "temperature": 0.0,
                            "target_language": "Traditional Chinese (Taiwan)",
                            "translation_model": "deepseek/deepseek-v4-pro",
                            "reference_file": "references/tripleS.md",
                            "locked_terms_file": "references/locked_terms.json",
                            "time_buffer": 0.1,
                            "font_name": "Heiti TC",
                            "fonts_dir": "/System/fonts",
                            "font_size": 12,
                            "outline_width": 0,
                            "box_background": True,
                            "margin_v": 20,
                            "margin_h": 10,
                            "alignment": 2,
                            "stage_cooldown": 0,
                            "save_source_transcript": True,
                        },
                    },
                    "baseline": {"transcription_backend": "whisper"},
                    "candidate": {"transcription_backend": "mlx"},
                }
            ),
            encoding="utf-8",
        )
        (directory / "source_transcript.diff").write_text("diff", encoding="utf-8")
        (directory / "review_composite.mp4").write_bytes(b"video")
        decision = "approved" if approved else "PENDING"
        reviewer = "Ada" if approved else "PENDING"
        date = "2026-08-10" if approved else "PENDING"
        (directory / "review.md").write_text(
            "\n".join(
                [
                    "# Review",
                    "",
                    "## Human Approval",
                    f"- Reviewer: {reviewer}",
                    f"- Date: {date}",
                    f"- Decision: {decision}",
                    "- Candidate-only missing speech: no",
                    "- Candidate-only repetition: no",
                    "- Candidate-only hallucination: no",
                    "- Candidate-only name regression: no",
                    "- Candidate-only timing regression: no",
                    "- Candidate-only unreadable subtitle: no",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def test_evidence_rejects_incomplete_manifest_and_invalid_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence_dir = Path(tmp)
            self._write_evidence(evidence_dir)
            (evidence_dir / "manifest.json").write_text("{}", encoding="utf-8")
            self.assertFalse(evidence_allows_mlx(evidence_dir, model="large"))

            self._write_evidence(evidence_dir)
            manifest_path = evidence_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["media_range"]["start"] = "00:00:00.000"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertFalse(evidence_allows_mlx(evidence_dir, model="large"))

    def test_evidence_rejects_empty_or_missing_effective_manifest_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence_dir = Path(tmp)
            self._write_evidence(evidence_dir)
            manifest_path = evidence_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["shared"]["references"]["reference_file"] = ""
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertFalse(evidence_allows_mlx(evidence_dir, model="large"))

            self._write_evidence(evidence_dir)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["shared"]["subtitle_style"].pop("margin_v")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertFalse(evidence_allows_mlx(evidence_dir, model="large"))

    def test_evidence_rejects_missing_artifact_warmup_or_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence_dir = Path(tmp)
            self._write_evidence(evidence_dir)
            evidence_path = evidence_dir / "comparison.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["warmups"][1]["completed"] = False
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            self.assertFalse(evidence_allows_mlx(evidence_dir, model="large"))

            self._write_evidence(evidence_dir)
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["runs"][0].pop("peak_memory_bytes")
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            self.assertFalse(evidence_allows_mlx(evidence_dir, model="large"))

            self._write_evidence(evidence_dir)
            (evidence_dir / "review_composite.mp4").unlink()
            self.assertFalse(evidence_allows_mlx(evidence_dir, model="large"))

    def test_evidence_requires_faster_mlx_and_explicit_human_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence_dir = Path(tmp)
            self._write_evidence(evidence_dir)
            self.assertTrue(evidence_allows_mlx(evidence_dir, model="large"))

            self._write_evidence(evidence_dir, mlx_times=[7.0, 8.0, 9.0])
            self.assertFalse(evidence_allows_mlx(evidence_dir, model="large"))

            self._write_evidence(evidence_dir, approved=False)
            self.assertFalse(evidence_allows_mlx(evidence_dir, model="large"))

            self._write_evidence(evidence_dir)
            review_path = evidence_dir / "review.md"
            review_path.write_text(
                review_path.read_text(encoding="utf-8").replace(
                    "- Candidate-only hallucination: no",
                    "- Candidate-only hallucination: yes",
                ),
                encoding="utf-8",
            )
            self.assertFalse(evidence_allows_mlx(evidence_dir, model="large"))
            self.assertFalse(evidence_allows_mlx(evidence_dir, model="large-v3"))


class TestComparisonRun(unittest.TestCase):
    def test_run_writes_manifest_review_diff_and_unscaled_shared_audio_composite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "20251231_w-yGSP1c3bg.mp4"
            source.write_bytes(b"video")
            output_dir = root / "comparison"
            commands: list[list[str]] = []
            child_cli_args: list[list[str]] = []

            def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                if command[0] == "ffmpeg":
                    Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
                    Path(command[-1]).write_bytes(b"video")
                    return subprocess.CompletedProcess(command, 0, "", "")

                backend = command[command.index("--backend") + 1]
                metrics_path = Path(command[command.index("--metrics-file") + 1])
                run_dir = Path(command[command.index("--run-dir") + 1])
                config_path = Path(command[command.index("--config") + 1])
                child_cli_args.append(
                    json.loads(config_path.read_text(encoding="utf-8"))["cli_args"]
                )
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(
                    json.dumps(
                        {
                            "transcription_seconds": 2.0 if backend == "mlx" else 5.0,
                            "end_to_end_seconds": 9.0,
                            "peak_memory_bytes": 123,
                        }
                    ),
                    encoding="utf-8",
                )
                if "--warm-cache" not in command:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "source.srt").write_text(f"{backend} source", encoding="utf-8")
                    (run_dir / "subtitles.srt").write_text("Traditional Chinese", encoding="utf-8")
                    (run_dir / "translated.mp4").write_bytes(b"video")
                return subprocess.CompletedProcess(command, 0, "", "")

            config = ComparisonConfig(
                video_path=source,
                output_dir=output_dir,
                model="large",
                language="ko",
                temperature=0.2,
                reference_file=Path("references/tripleS.md"),
                locked_terms_file=Path("references/locked_terms.json"),
                translation_model="deepseek/deepseek-v4-pro",
            )
            with patch("hermecho.asr_comparison.subprocess.run", side_effect=fake_run):
                run_comparison(config)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["media_range"]["start"], "00:29:30.000")
            self.assertEqual(manifest["shared"]["model_checkpoint"], "large")
            self.assertIsNone(manifest["shared"]["prompt"])
            self.assertEqual(
                manifest["shared"]["references"]["locked_terms_file"],
                "references/locked_terms.json",
            )
            self.assertEqual(manifest["shared"]["translation"]["provider"], "OpenRouter")
            self.assertEqual(manifest["shared"]["subtitle_style"]["font_name"], "Heiti TC")
            self.assertEqual(
                manifest["shared"]["subtitle_style"]["fonts_dir"],
                cli.parse_args(["clip.mp4"]).fonts_dir,
            )
            self.assertEqual(
                manifest["shared"]["subtitle_style"]["box_background"],
                cli.parse_args(["clip.mp4"]).box_background,
            )
            self.assertEqual(
                manifest["shared"]["effective_cli_options"]["box_background"],
                cli.parse_args(["clip.mp4"]).box_background,
            )
            self.assertTrue(manifest["shared"]["effective_cli_options"]["save_source_transcript"])
            self.assertIn("machine", manifest["shared"])
            self.assertIn("python", manifest["shared"]["runtime_versions"])
            self.assertEqual(manifest["baseline"], {"transcription_backend": "whisper"})
            self.assertEqual(manifest["candidate"], {"transcription_backend": "mlx"})
            self.assertTrue((output_dir / "source_transcript.diff").exists())
            report = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["whisper"]["median_transcription_seconds"], 5.0)
            self.assertEqual(report["summary"]["mlx"]["median_transcription_seconds"], 2.0)
            self.assertTrue((output_dir / report["artifacts"]["review_composite"]).exists())
            review = (output_dir / "review.md").read_text(encoding="utf-8")
            self.assertIn("- Decision: PENDING", review)
            self.assertIn("## Timestamped problems", review)
            self.assertFalse(evidence_allows_mlx(output_dir, model="large"))

            approved_review = review.replace("- Reviewer: PENDING", "- Reviewer: Ada")
            approved_review = approved_review.replace("- Date: PENDING", "- Date: 2026-08-10")
            approved_review = approved_review.replace("- Decision: PENDING", "- Decision: approved")
            for check in (
                "missing speech",
                "repetition",
                "hallucination",
                "name regression",
                "timing regression",
                "unreadable subtitle",
            ):
                approved_review = approved_review.replace(
                    f"- Candidate-only {check}: PENDING",
                    f"- Candidate-only {check}: no",
                )
            (output_dir / "review.md").write_text(approved_review, encoding="utf-8")
            self.assertTrue(evidence_allows_mlx(output_dir, model="large"))

            measured_backends = [
                command[command.index("--backend") + 1]
                for command in commands
                if command[0] != "ffmpeg" and "--warm-cache" not in command
            ]
            self.assertEqual(measured_backends, ["whisper", "mlx"] * 3)
            default_fonts_dir = cli.parse_args(["clip.mp4"]).fonts_dir
            for args in child_cli_args:
                self.assertEqual(args[args.index("--fonts-dir") + 1], default_fonts_dir)
                self.assertIn("--box_background", args)
            range_command = next(
                command
                for command in commands
                if command[0] == "ffmpeg" and command[-1] == str(output_dir / "media_range.mp4")
            )
            codec_index = range_command.index("-c:v")
            self.assertEqual(range_command[codec_index + 1], "libx264")
            self.assertEqual(range_command.count("libx264"), 1)
            self.assertEqual(range_command.count(str(output_dir / "media_range.mp4")), 1)
            composite = next(command for command in commands if command[0] == "ffmpeg" and "hstack" in " ".join(command))
            self.assertIn("[0:a]", " ".join(composite))
            self.assertNotIn("scale=", " ".join(composite))


if __name__ == "__main__":
    unittest.main()
