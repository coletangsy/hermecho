import os
import tempfile
import unittest
import unicodedata
from unittest.mock import patch

from hermecho.pipeline import PipelineConfig, process_video
from hermecho.subtitles import (
    adjust_subtitle_timing,
    apply_delivery_profile,
    delivery_gate_report,
    delivery_profile_for_orientation,
    split_long_segments,
    visual_cell_count,
)


class TestVisualCells(unittest.TestCase):
    def test_counts_full_width_and_half_width_characters(self) -> None:
        self.assertEqual(visual_cell_count("臺Ａa 1，"), 4.5)

    def test_combining_marks_do_not_consume_visual_cells_or_change_layout(self) -> None:
        composed = "甲" * 6 + "café" + "乙" * 7
        decomposed = "甲" * 6 + "cafe\u0301" + "乙" * 7
        portrait = delivery_profile_for_orientation(is_portrait=True)

        self.assertEqual(visual_cell_count(composed), visual_cell_count(decomposed))
        composed_result = apply_delivery_profile(
            [{"start": 0.0, "end": 3.0, "text": composed}], portrait
        )
        decomposed_result = apply_delivery_profile(
            [{"start": 0.0, "end": 3.0, "text": decomposed}], portrait
        )

        self.assertEqual(
            unicodedata.normalize("NFC", composed_result.cues[0]["text"]),
            unicodedata.normalize("NFC", decomposed_result.cues[0]["text"]),
        )


class TestDeliveryProfiles(unittest.TestCase):
    def test_split_uses_segment_timing_when_source_word_timing_is_invalid(self) -> None:
        segments = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "valid",
                "words": [{"word": "valid", "start": 0.0, "end": 1.0}],
            },
            {
                "start": 1.0,
                "end": 2.0,
                "text": "fallback",
                "words": [{"word": "fallback", "start": 1.5, "end": 1.5}],
            },
            {"start": 2.0, "end": 2.0, "text": "zero"},
            {"start": 2.0, "end": 3.0, "text": "  "},
        ]

        cleaned = split_long_segments(segments)

        self.assertEqual([segment["text"] for segment in cleaned], ["valid", "fallback"])
        self.assertNotIn("words", cleaned[1])
        self.assertFalse(
            apply_delivery_profile(
                cleaned,
                delivery_profile_for_orientation(is_portrait=False),
            ).blocked
        )

    def test_adjust_timing_keeps_original_duration_when_buffer_cannot_fit(self) -> None:
        adjusted = adjust_subtitle_timing(
            [
                {"start": 0.0, "end": 0.1, "text": "first"},
                {"start": 0.1, "end": 1.0, "text": "second"},
            ],
            time_buffer=0.1,
        )

        self.assertEqual(adjusted[0]["end"], 0.1)

    def test_portrait_and_landscape_use_independent_line_limits(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        landscape = delivery_profile_for_orientation(is_portrait=False)

        self.assertEqual((portrait.warning_line_cells, portrait.repair_line_cells), (10, 12))
        self.assertEqual((landscape.warning_line_cells, landscape.repair_line_cells), (16, 20))

        result = apply_delivery_profile(
            [{"start": 0.0, "end": 3.0, "text": "甲" * 17}],
            landscape,
        )

        self.assertFalse(result.blocked)
        self.assertEqual(result.cues[0]["text"], "甲" * 17)
        self.assertEqual(
            [(diagnostic.severity, diagnostic.code) for diagnostic in result.diagnostics],
            [("Warning", "line_cells")],
        )

    def test_wrap_keeps_half_width_words_intact(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        result = apply_delivery_profile(
            [{"start": 0.0, "end": 3.0, "text": "甲" * 10 + " tripleS"}],
            portrait,
        )

        self.assertEqual(result.cues[0]["text"], "甲" * 7 + "\n" + "甲" * 3 + " tripleS")
        self.assertNotIn("triple\nS", result.cues[0]["text"])

    def test_wrap_preserves_translation_whitespace(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        short_result = apply_delivery_profile(
            [{"start": 0.0, "end": 3.0, "text": "A  B"}], portrait
        )
        text = "甲" * 10 + " A  B" + "乙" * 10
        wrapped_result = apply_delivery_profile(
            [{"start": 0.0, "end": 3.0, "text": text}], portrait
        )

        self.assertEqual(short_result.cues[0]["text"], "A  B")
        self.assertIn("\n", wrapped_result.cues[0]["text"])
        self.assertEqual(wrapped_result.cues[0]["text"].replace("\n", ""), text)

    def test_wrap_normalizes_existing_layout_breaks(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        result = apply_delivery_profile(
            [{"start": 0.0, "end": 3.0, "text": "hello\r\nworld\n一"}], portrait
        )

        self.assertLessEqual(len(result.cues[0]["text"].splitlines()), 2)
        self.assertEqual(result.cues[0]["text"], "hello world 一")

    def test_wrap_keeps_common_half_width_tokens_intact(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)

        for token in ("O'Neil", "v1.2", "https://x/y"):
            with self.subTest(token=token):
                result = apply_delivery_profile(
                    [{"start": 0.0, "end": 3.0, "text": "甲" * 9 + token + "乙" * 9}],
                    portrait,
                )

                self.assertTrue(
                    any(token in line for line in result.cues[0]["text"].splitlines())
                )

    def test_wrap_keeps_unicode_half_width_tokens_intact(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)

        for token in ("café", "cafe\u0301", "O’Neil"):
            with self.subTest(token=token):
                result = apply_delivery_profile(
                    [{"start": 0.0, "end": 3.0, "text": "甲" * 9 + token + "乙" * 9}],
                    portrait,
                )

                self.assertTrue(
                    any(token in line for line in result.cues[0]["text"].splitlines())
                )

        for character in ("甲", "Ａ"):
            with self.subTest(character=character):
                full_width_result = apply_delivery_profile(
                    [{"start": 0.0, "end": 3.0, "text": character * 13}],
                    portrait,
                )
                self.assertEqual(
                    full_width_result.cues[0]["text"],
                    character * 6 + "\n" + character * 7,
                )

    def test_wrap_prefers_a_punctuation_boundary(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        result = apply_delivery_profile(
            [{"start": 0.0, "end": 3.0, "text": "甲" * 6 + "，" + "乙" * 8}],
            portrait,
        )

        self.assertEqual(result.cues[0]["text"], "甲" * 6 + "，\n" + "乙" * 8)

    def test_wrap_prefers_balance_before_a_punctuation_tiebreak(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        result = apply_delivery_profile(
            [{"start": 0.0, "end": 3.0, "text": "甲，" + "乙" * 12}],
            portrait,
        )

        self.assertEqual(result.cues[0]["text"], "甲，" + "乙" * 5 + "\n" + "乙" * 7)

    def test_cps_and_duration_thresholds_are_strict(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        result = apply_delivery_profile(
            [
                {"start": 0.0, "end": 1.0, "text": "甲" * 8},
                {"start": 1.0, "end": 2.0, "text": "甲" * 9},
                {"start": 2.0, "end": 3.0, "text": "甲" * 12},
                {"start": 3.0, "end": 4.0, "text": "甲" * 13},
                {"start": 4.0, "end": 4.5, "text": "甲"},
                {"start": 5.0, "end": 5.49, "text": "甲"},
                {"start": 6.0, "end": 13.0, "text": "甲"},
                {"start": 14.0, "end": 24.0, "text": "甲"},
                {"start": 25.0, "end": 35.01, "text": "甲"},
            ],
            portrait,
        )

        findings = {
            (diagnostic.cue_index, diagnostic.code): diagnostic.severity
            for diagnostic in result.diagnostics
        }
        self.assertNotIn((1, "cps"), findings)
        self.assertEqual(findings[(2, "cps")], "Warning")
        self.assertEqual(findings[(3, "cps")], "Warning")
        self.assertEqual(findings[(4, "cps")], "Repair Limit")
        self.assertEqual(findings[(5, "duration")], "Warning")
        self.assertEqual(findings[(6, "duration")], "Repair Limit")
        self.assertNotIn((7, "duration"), findings)
        self.assertEqual(findings[(8, "duration")], "Warning")
        self.assertEqual(findings[(9, "duration")], "Repair Limit")

    def test_cue_cell_thresholds_are_profile_specific(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        portrait_result = apply_delivery_profile(
            [
                {"start": 0.0, "end": 3.0, "text": "甲" * 20},
                {"start": 4.0, "end": 7.0, "text": "甲" * 21},
                {"start": 8.0, "end": 11.0, "text": "甲" * 24},
                {"start": 12.0, "end": 15.0, "text": "甲" * 25},
            ],
            portrait,
        )
        portrait_findings = {
            (diagnostic.cue_index, diagnostic.code): diagnostic.severity
            for diagnostic in portrait_result.diagnostics
        }
        self.assertNotIn((1, "cue_cells"), portrait_findings)
        self.assertEqual(portrait_findings[(2, "cue_cells")], "Warning")
        self.assertEqual(portrait_findings[(3, "cue_cells")], "Warning")
        self.assertEqual(portrait_findings[(4, "cue_cells")], "Repair Limit")

        landscape = delivery_profile_for_orientation(is_portrait=False)
        landscape_result = apply_delivery_profile(
            [
                {"start": 0.0, "end": 4.0, "text": "甲" * 32},
                {"start": 5.0, "end": 9.0, "text": "甲" * 33},
                {"start": 10.0, "end": 15.0, "text": "甲" * 40},
                {"start": 16.0, "end": 21.0, "text": "甲" * 41},
            ],
            landscape,
        )
        landscape_findings = {
            (diagnostic.cue_index, diagnostic.code): diagnostic.severity
            for diagnostic in landscape_result.diagnostics
        }
        self.assertNotIn((1, "cue_cells"), landscape_findings)
        self.assertEqual(landscape_findings[(2, "cue_cells")], "Warning")
        self.assertEqual(landscape_findings[(3, "cue_cells")], "Warning")
        self.assertEqual(landscape_findings[(4, "cue_cells")], "Repair Limit")

    def test_line_cell_thresholds_preserve_an_unbreakable_half_width_word(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        portrait_result = apply_delivery_profile(
            [
                {"start": 0.0, "end": 3.0, "text": "a" * 20},
                {"start": 4.0, "end": 7.0, "text": "a" * 22},
                {"start": 8.0, "end": 11.0, "text": "a" * 24},
                {"start": 12.0, "end": 15.0, "text": "a" * 25},
            ],
            portrait,
        )
        portrait_findings = {
            (diagnostic.cue_index, diagnostic.code): diagnostic.severity
            for diagnostic in portrait_result.diagnostics
        }
        self.assertNotIn((1, "line_cells"), portrait_findings)
        self.assertEqual(portrait_findings[(2, "line_cells")], "Warning")
        self.assertEqual(portrait_findings[(3, "line_cells")], "Warning")
        self.assertEqual(portrait_findings[(4, "line_cells")], "Repair Limit")

        landscape = delivery_profile_for_orientation(is_portrait=False)
        landscape_result = apply_delivery_profile(
            [
                {"start": 0.0, "end": 3.0, "text": "a" * 32},
                {"start": 4.0, "end": 7.0, "text": "a" * 34},
                {"start": 8.0, "end": 11.0, "text": "a" * 40},
                {"start": 12.0, "end": 15.0, "text": "a" * 41},
            ],
            landscape,
        )
        landscape_findings = {
            (diagnostic.cue_index, diagnostic.code): diagnostic.severity
            for diagnostic in landscape_result.diagnostics
        }
        self.assertNotIn((1, "line_cells"), landscape_findings)
        self.assertEqual(landscape_findings[(2, "line_cells")], "Warning")
        self.assertEqual(landscape_findings[(3, "line_cells")], "Warning")
        self.assertEqual(landscape_findings[(4, "line_cells")], "Repair Limit")

    def test_repair_limits_are_best_effort_with_a_gate_report(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        result = apply_delivery_profile(
            [{"start": 0.0, "end": 1.0, "text": "甲" * 25}],
            portrait,
        )

        self.assertFalse(result.blocked)
        self.assertIn(
            ("Repair Limit", "cue_cells"),
            [(diagnostic.severity, diagnostic.code) for diagnostic in result.diagnostics],
        )
        report = delivery_gate_report(result, portrait)
        self.assertIn("Delivery Gate (portrait)", report)
        self.assertIn("Repair Limits: ", report)
        self.assertIn("Repair Limit cue 1: cue_cells", report)

    def test_tolerates_float_noise_at_a_cue_boundary(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        result = apply_delivery_profile(
            [
                {"start": 0.0, "end": 0.30000000000000004, "text": "甲"},
                {"start": 0.3, "end": 1.0, "text": "乙"},
            ],
            portrait,
        )

        self.assertFalse(result.blocked)
        self.assertFalse(any(diagnostic.code == "overlap_timing" for diagnostic in result.diagnostics))

    def test_structural_defects_block_delivery(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        result = apply_delivery_profile(
            [
                {"start": 0.0, "end": 1.0, "text": ""},
                {"start": 2.0, "end": 2.0, "text": "甲"},
                {"start": 4.0, "end": 3.0, "text": "乙"},
                {"start": 5.0, "end": 7.0, "text": "丙"},
                {"start": 6.5, "end": 8.0, "text": "丁"},
                {
                    "start": 9.0,
                    "end": 10.0,
                    "text": "戊",
                    "words": [{"word": "word", "start": 9.0}],
                },
                {
                    "start": 11.0,
                    "end": 12.0,
                    "text": "己",
                    "words": [
                        {"word": "one", "start": 11.0, "end": 11.5},
                        {"word": "two", "start": 11.5, "end": 12.0},
                    ],
                    "source_word_indices": [0, 2],
                },
                {"start": float("nan"), "end": 14.0, "text": "庚"},
            ],
            portrait,
        )

        self.assertTrue(result.blocked)
        self.assertTrue(
            {
                "empty_piece",
                "non_positive_duration",
                "reversed_timing",
                "overlap_timing",
                "missing_source_word_timing",
                "invalid_source_coverage",
                "invalid_timing",
            }.issubset({diagnostic.code for diagnostic in result.diagnostics})
        )

    def test_negative_cue_and_source_word_timing_are_structural(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        result = apply_delivery_profile(
            [
                {"start": -1.0, "end": 1.0, "text": "甲"},
                {"start": 1.0, "end": -1.0, "text": "乙"},
                {
                    "start": 2.0,
                    "end": 3.0,
                    "text": "丙",
                    "words": [{"word": "word", "start": -0.5, "end": 0.5}],
                },
                {
                    "start": 4.0,
                    "end": 5.0,
                    "text": "丁",
                    "words": [{"word": "word", "start": 0.0, "end": -0.5}],
                },
            ],
            portrait,
        )

        self.assertTrue(result.blocked)
        self.assertEqual(
            [d.code for d in result.diagnostics].count("invalid_timing"),
            2,
        )
        self.assertEqual(
            [d.code for d in result.diagnostics].count("invalid_source_word_timing"),
            2,
        )

    def test_none_source_words_falls_back_to_legacy_words(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        result = apply_delivery_profile(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "甲",
                    "source_words": None,
                    "words": [{"word": "word", "start": -0.5, "end": 0.5}],
                }
            ],
            portrait,
        )

        self.assertTrue(result.blocked)
        self.assertIn(
            "invalid_source_word_timing",
            [diagnostic.code for diagnostic in result.diagnostics],
        )

    def test_source_word_indices_allow_a_nonzero_continuous_range(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        words = [
            {"word": "one", "start": 0.0, "end": 0.5},
            {"word": "two", "start": 0.5, "end": 1.0},
        ]
        valid = apply_delivery_profile(
            [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "甲",
                    "words": words,
                    "source_word_indices": [4, 5],
                }
            ],
            portrait,
        )

        self.assertFalse(valid.blocked)
        for indices in ([4, 6], [4, 4], [4, "5"]):
            with self.subTest(indices=indices):
                invalid = apply_delivery_profile(
                    [
                        {
                            "start": 0.0,
                            "end": 2.0,
                            "text": "甲",
                            "words": words,
                            "source_word_indices": indices,
                        }
                    ],
                    portrait,
                )

                self.assertTrue(invalid.blocked)
                self.assertIn(
                    "invalid_source_coverage",
                    [diagnostic.code for diagnostic in invalid.diagnostics],
                )

    def test_no_speech_is_not_an_empty_delivery_piece(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        result = apply_delivery_profile(
            [{"start": 0.0, "end": 1.0, "text": "", "source_text": "[no speech]"}],
            portrait,
        )

        self.assertFalse(result.blocked)
        self.assertEqual(result.cues, [])
        self.assertEqual(result.diagnostics, [])

    def test_diagnostics_number_only_delivery_cues_after_no_speech(self) -> None:
        portrait = delivery_profile_for_orientation(is_portrait=True)
        warning = apply_delivery_profile(
            [
                {"start": 0.0, "end": 1.0, "text": "", "source_text": "[no speech]"},
                {"start": 1.0, "end": 1.75, "text": "甲"},
            ],
            portrait,
        )
        structural = apply_delivery_profile(
            [
                {"start": 0.0, "end": 1.0, "text": "", "source_text": "[no speech]"},
                {"start": 2.0, "end": 2.0, "text": "乙"},
            ],
            portrait,
        )

        self.assertIn("Warning cue 1: duration", delivery_gate_report(warning, portrait))
        self.assertIn(
            "Structural Defect cue 1: non_positive_duration",
            delivery_gate_report(structural, portrait),
        )


class TestDeliveryPipeline(unittest.TestCase):
    def test_best_effort_delivery_writes_srt_and_report(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
            audio_path = audio_file.name
        output_dir = tempfile.mkdtemp()
        config = PipelineConfig(
            video_filename="portrait.mp4",
            input_dir="input",
            output_dir=output_dir,
            srt_only=True,
            stage_cooldown=0,
        )
        translated = [{"start": 0.0, "end": 1.0, "text": "甲" * 25}]

        try:
            with patch("hermecho.pipeline.extract_audio", return_value=audio_path), \
                patch("hermecho.pipeline.transcribe_audio", return_value=translated), \
                patch("hermecho.pipeline.translate_segments", return_value=translated), \
                patch("hermecho.pipeline.load_reference_material", return_value=""), \
                patch("hermecho.pipeline.is_portrait_video", return_value=True):
                process_video(config)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

        output_files = [
            os.path.join(root, filename)
            for root, _, filenames in os.walk(output_dir)
            for filename in filenames
        ]
        srt_paths = [path for path in output_files if path.endswith(".srt")]
        report_paths = [path for path in output_files if path.endswith("_delivery_gate.txt")]
        self.assertEqual(len(srt_paths), 1)
        self.assertEqual(len(report_paths), 1)
        with open(report_paths[0], encoding="utf-8") as report_file:
            self.assertIn("Repair Limit", report_file.read())

    def test_structural_defect_writes_report_but_stops_final_srt(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
            audio_path = audio_file.name
        output_dir = tempfile.mkdtemp()
        config = PipelineConfig(
            video_filename="portrait.mp4",
            input_dir="input",
            output_dir=output_dir,
            srt_only=True,
            stage_cooldown=0,
        )
        translated = [{"start": 0.0, "end": 0.0, "text": "甲"}]

        try:
            with patch("hermecho.pipeline.extract_audio", return_value=audio_path), \
                patch("hermecho.pipeline.transcribe_audio", return_value=translated), \
                patch("hermecho.pipeline.translate_segments", return_value=translated), \
                patch("hermecho.pipeline.load_reference_material", return_value=""), \
                patch("hermecho.pipeline.is_portrait_video", return_value=True):
                process_video(config)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

        output_files = [
            os.path.join(root, filename)
            for root, _, filenames in os.walk(output_dir)
            for filename in filenames
        ]
        self.assertFalse(any(path.endswith(".srt") for path in output_files))
        report_paths = [path for path in output_files if path.endswith("_delivery_gate.txt")]
        self.assertEqual(len(report_paths), 1)
        with open(report_paths[0], encoding="utf-8") as report_file:
            self.assertIn("Structural Defect", report_file.read())


if __name__ == "__main__":
    unittest.main()
