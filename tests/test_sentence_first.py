from unittest.mock import Mock
import json
import tempfile
from pathlib import Path
import unittest

from hermecho.subtitles import PORTRAIT_DELIVERY_PROFILE
from hermecho.sentence_first import (
    SentenceFirstError,
    build_delivery_cues,
    build_source_sentences,
    resolve_subtitle_delivery,
)


class TestSourceSentences(unittest.TestCase):
    def test_merges_segments_at_sentence_boundaries_and_omits_no_speech(self) -> None:
        first_words = [
            {"word": "첫 ", "start": 0.0, "end": 0.4},
            {"word": "문장", "start": 0.45, "end": 0.8},
        ]
        ending_words = [{"word": "입니다.", "start": 0.85, "end": 1.2}]
        second_words = [{"word": "다음", "start": 2.1, "end": 2.5}]
        sentences = build_source_sentences(
            [
                {"start": 0.0, "end": 0.8, "text": "첫 문장", "words": first_words},
                {"start": 0.85, "end": 1.2, "text": "입니다.", "words": ending_words},
                {"start": 1.2, "end": 2.1, "text": "[no speech]"},
                {"start": 2.1, "end": 2.5, "text": "다음", "words": second_words},
            ]
        )

        self.assertEqual(
            [sentence["text"] for sentence in sentences],
            ["첫 문장입니다.", "다음"],
        )
        self.assertEqual(sentences[0]["source_words"], first_words + ending_words)
        self.assertEqual(sentences[0]["source_word_indices"], [0, 1, 2])
        self.assertEqual(sentences[0]["start"], 0.0)
        self.assertEqual(sentences[0]["end"], 1.2)
        self.assertNotIn("[no speech]", [sentence["text"] for sentence in sentences])

    def test_requires_word_timestamps(self) -> None:
        with self.assertRaisesRegex(SentenceFirstError, "Source Word timestamps"):
            build_source_sentences([{"start": 0.0, "end": 1.0, "text": "hello"}])

    def test_ignores_empty_segments_without_word_evidence(self) -> None:
        sentences = build_source_sentences(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "안녕.",
                    "words": [{"word": "안녕.", "start": 0.0, "end": 1.0}],
                },
                {"start": 1.0, "end": 1.0, "text": "", "words": []},
            ]
        )

        self.assertEqual([sentence["text"] for sentence in sentences], ["안녕."])

    def test_preserves_point_timed_source_words(self) -> None:
        words = [
            {"word": "첫", "start": 0.0, "end": 0.5},
            {"word": "번째", "start": 0.5, "end": 0.5},
            {"word": "문장.", "start": 0.5, "end": 1.0},
        ]

        sentences = build_source_sentences(
            [{"start": 0.0, "end": 1.0, "text": "첫 번째 문장.", "words": words}]
        )

        self.assertEqual(sentences[0]["source_words"], words)

    def test_safety_boundary_uses_the_nearest_word_pause(self) -> None:
        segments = [
            {
                "start": start,
                "end": start + 4.0,
                "text": f"詞{index}",
                "words": [{"word": f"詞{index} ", "start": start, "end": start + 4.0}],
            }
            for index, start in enumerate((0.0, 4.1, 8.2, 12.3))
        ]

        sentences = build_source_sentences(segments, safety_duration=10.0)

        self.assertEqual(sentences[0]["end"], 8.1)
        self.assertEqual(len(sentences), 2)


class TestSentenceFirstDelivery(unittest.TestCase):
    def test_fitting_sentence_is_one_cue_without_repair_or_alignment(self) -> None:
        fit_repair = Mock()
        align = Mock()
        result = build_delivery_cues(
            [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "你好。",
                    "source_words": [
                        {"word": "안녕", "start": 0.0, "end": 1.0},
                        {"word": "하세요", "start": 1.1, "end": 2.0},
                    ],
                    "source_word_indices": [0, 1],
                }
            ],
            profile=None,
            fit_repair=fit_repair,
            align=align,
        )

        self.assertFalse(result.blocked)
        self.assertEqual([cue["text"] for cue in result.cues], ["你好。"])
        fit_repair.assert_not_called()
        align.assert_not_called()

    def test_point_timed_word_is_allowed_inside_a_positive_cue(self) -> None:
        result = build_delivery_cues(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "你好。",
                    "source_words": [
                        {"word": "안", "start": 0.0, "end": 0.5},
                        {"word": "녕", "start": 0.5, "end": 0.5},
                        {"word": "요", "start": 0.5, "end": 1.0},
                    ],
                    "source_word_indices": [0, 1, 2],
                }
            ],
            profile=None,
        )

        self.assertFalse(result.blocked)

    def test_tolerates_float_noise_at_a_contiguous_source_word_boundary(self) -> None:
        sentences = build_source_sentences(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "안녕.",
                    "words": [
                        {"word": "안", "start": 0.0, "end": 0.30000000000000004},
                        {"word": "녕.", "start": 0.3, "end": 1.0},
                    ],
                }
            ]
        )

        result = build_delivery_cues([{**sentences[0], "text": "你好。"}], profile=None)

        self.assertFalse(result.blocked)

    def test_isolated_point_timed_word_remains_blocked(self) -> None:
        result = build_delivery_cues(
            [
                {
                    "start": 1.0,
                    "end": 1.0,
                    "text": "你好。",
                    "source_words": [{"word": "안녕", "start": 1.0, "end": 1.0}],
                    "source_word_indices": [0],
                }
            ],
            profile=None,
        )

        self.assertTrue(result.blocked)
        self.assertTrue(
            any(diagnostic.code == "non_positive_duration" for diagnostic in result.diagnostics)
        )

    def test_cps_repair_runs_before_alignment_and_rechecks_the_result(self) -> None:
        fit_repair = Mock(return_value="短句。")
        align = Mock()
        result = build_delivery_cues(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥。",
                    "source_words": [{"word": "source", "start": 0.0, "end": 1.0}],
                    "source_word_indices": [0],
                }
            ],
            PORTRAIT_DELIVERY_PROFILE,
            fit_repair=fit_repair,
            align=align,
        )

        self.assertFalse(result.blocked)
        self.assertEqual([cue["text"] for cue in result.cues], ["短句。"])
        fit_repair.assert_called_once()
        align.assert_not_called()

    def test_alignment_preserves_text_covers_words_and_extends_into_gap(self) -> None:
        from hermecho.subtitles import DeliveryProfile

        profile = DeliveryProfile(
            name="test",
            warning_line_cells=2,
            repair_line_cells=3,
            warning_cue_cells=3,
            repair_cue_cells=3,
            warning_cps=100,
            repair_cps=100,
            warning_min_duration=0,
            warning_max_duration=100,
            repair_min_duration=0,
            repair_max_duration=100,
        )
        align = Mock(
            return_value=[
                {"text": "甲乙", "end_source_word_index": 1},
                {"text": "丙丁", "end_source_word_index": 3},
            ]
        )
        result = build_delivery_cues(
            [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "甲乙丙丁",
                    "source_words": [
                        {"word": "a", "start": 0.0, "end": 0.2},
                        {"word": "b", "start": 0.3, "end": 0.5},
                        {"word": "c", "start": 1.0, "end": 1.2},
                        {"word": "d", "start": 1.3, "end": 1.5},
                    ],
                    "source_word_indices": [5, 6, 7, 8],
                }
            ],
            profile,
            align=align,
            time_buffer=0.2,
        )

        self.assertFalse(result.blocked)
        self.assertEqual([cue["text"] for cue in result.cues], ["甲乙", "丙丁"])
        self.assertEqual(result.cues[0]["source_word_indices"], [5, 6])
        self.assertEqual(result.cues[1]["source_word_indices"], [7, 8])
        self.assertEqual(result.cues[0]["end"], 0.7)
        self.assertEqual(result.cues[1]["start"], 1.0)
        align.assert_called_once()

    def test_alignment_merges_point_timed_piece_into_previous_cue(self) -> None:
        from hermecho.subtitles import DeliveryProfile

        profile = DeliveryProfile(
            name="test",
            warning_line_cells=100,
            repair_line_cells=100,
            warning_cue_cells=100,
            repair_cue_cells=100,
            warning_cps=8,
            repair_cps=12,
            warning_min_duration=0,
            warning_max_duration=100,
            repair_min_duration=0,
            repair_max_duration=100,
        )
        align = Mock(
            return_value=[
                {"text": "甲乙丙丁", "end_source_word_index": 0},
                {"text": "戊己庚辛", "end_source_word_index": 1},
                {"text": "壬癸子丑寅", "end_source_word_index": 2},
            ]
        )

        result = build_delivery_cues(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "甲乙丙丁戊己庚辛壬癸子丑寅",
                    "source_words": [
                        {"word": "a", "start": 0.0, "end": 0.3},
                        {"word": "b", "start": 0.3, "end": 0.3},
                        {"word": "c", "start": 0.3, "end": 1.0},
                    ],
                    "source_word_indices": [0, 1, 2],
                }
            ],
            profile,
            align=align,
        )

        self.assertFalse(result.blocked)
        self.assertEqual([cue["text"] for cue in result.cues], ["甲乙丙丁戊己庚辛", "壬癸子丑寅"])
        self.assertEqual(result.cues[0]["source_word_indices"], [0, 1])

    def test_invalid_alignment_falls_back_to_the_unsplit_cue(self) -> None:
        from hermecho.subtitles import DeliveryProfile

        profile = DeliveryProfile(
            name="test",
            warning_line_cells=2,
            repair_line_cells=3,
            warning_cue_cells=3,
            repair_cue_cells=3,
            warning_cps=100,
            repair_cps=100,
            warning_min_duration=0,
            warning_max_duration=100,
            repair_min_duration=0,
            repair_max_duration=100,
        )
        align = Mock(return_value=[{"text": "甲", "end_source_word_index": 0}])
        result = build_delivery_cues(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "甲乙丙丁",
                    "source_words": [
                        {"word": "a", "start": 0.0, "end": 0.2},
                        {"word": "b", "start": 0.3, "end": 0.5},
                        {"word": "c", "start": 0.6, "end": 0.8},
                        {"word": "d", "start": 0.9, "end": 1.0},
                    ],
                    "source_word_indices": [0, 1, 2, 3],
                }
            ],
            profile,
            align=align,
        )

        self.assertFalse(result.blocked)
        self.assertEqual(align.call_count, 2)
        self.assertTrue(
            any(diagnostic.severity == "Repair Limit" for diagnostic in result.diagnostics)
        )

    def test_unresolved_presentation_limit_is_best_effort_without_structural_failure(self) -> None:
        fit_repair = Mock(return_value=None)
        result = build_delivery_cues(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥。",
                    "source_words": [{"word": "source", "start": 0.0, "end": 1.0}],
                    "source_word_indices": [0],
                }
            ],
            fit_repair=fit_repair,
        )

        self.assertFalse(result.blocked)
        self.assertEqual(fit_repair.call_count, 2)
        self.assertTrue(
            any(
                diagnostic.severity == "Repair Limit"
                for diagnostic in result.diagnostics
            )
        )


class TestSentenceFirstPromotion(unittest.TestCase):
    def test_auto_uses_legacy_until_human_approval_then_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            evidence_dir = Path(temporary_dir)
            self.assertEqual(resolve_subtitle_delivery("auto", evidence_dir), "legacy")
            self.assertEqual(resolve_subtitle_delivery("legacy", evidence_dir), "legacy")
            self.assertEqual(
                resolve_subtitle_delivery("sentence-first", evidence_dir),
                "sentence-first",
            )
            for name in (
                "manifest.json",
                "review_composite.mp4",
                "transcript.json",
                "media_range.mp4",
            ):
                (evidence_dir / name).write_text("evidence", encoding="utf-8")
            (evidence_dir / "comparison.json").write_text(
                json.dumps(
                    {
                        "comparison_variable": "subtitle_delivery",
                        "baseline": "legacy",
                        "candidate": "sentence-first",
                        "media_range": {
                            "source_name": "20251231_w-yGSP1c3bg.mp4",
                            "start": "00:29:30.000",
                            "end": "00:39:30.000",
                            "prepared_media": "media_range.mp4",
                            "shared_audio": "review_composite.mp4",
                        },
                        "frozen_transcription": {"fingerprint": "frozen"},
                        "delivery_gates": {"baseline": "passed", "candidate": "passed"},
                        "artifacts": {
                            "manifest": "manifest.json",
                            "review_composite": "review_composite.mp4",
                            "transcript": "transcript.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (evidence_dir / "review.md").write_text(
                "\n".join(
                    [
                        "## Review checklist",
                        *[f"- {check.title()}: pass" for check in (
                            "translation completeness",
                            "meaning boundaries",
                            "timing",
                            "readability",
                            "locked terms",
                            "punctuation",
                            "presentation warnings",
                        )],
                        "## Timestamped Candidate-only regressions",
                        "- none.",
                        "## Human Approval",
                        "- Reviewer: reviewer",
                        "- Date: 2026-08-11",
                        "- Decision: approved",
                        *[
                            f"- Candidate-only {check}: no"
                            for check in (
                                "translation completeness",
                                "meaning boundaries",
                                "timing",
                                "readability",
                                "locked terms",
                                "punctuation",
                                "presentation warnings",
                            )
                        ],
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(resolve_subtitle_delivery("auto", evidence_dir), "sentence-first")


if __name__ == "__main__":
    unittest.main()
