import json
from pathlib import Path
from unittest.mock import patch

from hermecho.pipeline import PipelineConfig, process_video


def test_source_srt_accepts_studio_identifiers_settings_and_decimal_separator(tmp_path):
    from hermecho.subtitle_bundle import read_source_srt
    source = tmp_path / "source.srt"
    source.write_text("cue-id\n-00:00:00.500 --> 00:00:01.001 align:start\nHello\n\n00:00:02,000 --> 00:00:03,000\n[no speech]\n")
    assert read_source_srt(str(source)) == [
        {"start": -0.5, "end": 1.001, "text": "Hello"},
        {"start": 2.0, "end": 3.0, "text": "[no speech]"},
    ]


def test_translation_preserves_source_no_speech_markers_without_model_requests():
    from hermecho.translation import translate_segments
    source = [{"start": 2.0, "end": 3.0, "text": "[no speech]"}]
    with patch("hermecho.translation._translate_chunk") as translate:
        result = translate_segments(source, target_language="zh-TW", translation_model="test", reference_material=None, preserve_markers=True)
    assert result == [{**source[0], "source_text": "[no speech]"}]
    translate.assert_not_called()


def test_translation_preserves_markers_between_translated_source_cues():
    from hermecho.translation import translate_segments
    source = [{"start": i, "end": i + 1, "text": text} for i, text in enumerate(["Hello", "[no speech]", "Bye"])]
    with patch("hermecho.translation._translate_chunk", return_value=({"translations": {"0": "你好", "2": "再見"}}, None)):
        result = translate_segments(source, target_language="zh-TW", translation_model="test", reference_material=None, preserve_markers=True)
    assert result is not None
    assert [cue["text"] for cue in result] == ["你好", "[no speech]", "再見"]
    assert [(cue["start"], cue["end"]) for cue in result] == [(0, 1), (1, 2), (2, 3)]


def test_source_srt_pipeline_preserves_marker_timing_without_model_request(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    source = tmp_path / "source.srt"
    source.write_text(
        "first\n00:00:00,500 --> 00:00:01.500\nHello\n\n"
        "00:00:02,000 --> 00:00:03,000\n[no speech]\n\n"
        "last\n00:00:04,000 --> 00:00:05,000\nBye\n"
    )
    with patch("hermecho.pipeline._video_duration_seconds", return_value=6), \
         patch("hermecho.pipeline.is_portrait_video", return_value=False), \
         patch("hermecho.pipeline.load_reference_material", return_value=""), \
         patch("hermecho.pipeline.load_locked_terms", return_value={}), \
         patch(
             "hermecho.translation._translate_chunk",
             return_value=({"translations": {"0": "你好", "2": "再見"}}, None),
         ) as translate_chunk:
        process_video(
            PipelineConfig(
                "clip.mp4",
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                source_srt=str(source),
                srt_only=True,
                stage_cooldown=0,
            )
        )

    bundle = json.loads(next((tmp_path / "out").rglob("*_subtitle_bundle.json")).read_text())
    assert [(cue["start_ms"], cue["end_ms"]) for cue in bundle["source_cues"]] == [
        (500, 1500),
        (2000, 3000),
        (4000, 5000),
    ]
    assert [cue["text"] for cue in bundle["translation_cues"]] == ["你好", "[no speech]", "再見"]
    assert [cue["_translation_id"] for cue in translate_chunk.call_args.args[0]] == ["0", "2"]


def test_pipeline_preserves_all_source_pairs_without_nontranslation_requests(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"source")
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"audio")
    source = [
        {"start": 0.001, "end": 1.234, "text": "hello.", "words": [{"word": "hello.", "start": 0.001, "end": 1.234}]},
        {"start": 2.0, "end": 2.0, "text": "zero.", "words": [{"word": "zero.", "start": 2.0, "end": 2.0}]},
    ]
    def translation(sentences, **kwargs):
        return [{**sentence, "text": "很長的完整翻譯" * 20} for sentence in sentences]
    with patch("hermecho.pipeline.extract_audio", return_value=str(audio)), \
         patch("hermecho.pipeline.transcribe_audio", return_value=source), \
         patch("hermecho.pipeline.translate_segments", side_effect=translation), \
         patch("hermecho.pipeline.is_portrait_video", return_value=False), \
         patch("hermecho.pipeline.load_reference_material", return_value=""), \
         patch("hermecho.pipeline.load_locked_terms", return_value={}), \
         patch("hermecho.translation.review_source_sentence_boundaries") as boundary, \
         patch("hermecho.translation.align_translation_sentence") as align, \
         patch("hermecho.translation.fit_repair_translation_sentence") as repair:
        process_video(PipelineConfig("clip.mp4", input_dir=str(tmp_path), output_dir=str(tmp_path / "out"), srt_only=True, save_source_transcript=True, language="ko", stage_cooldown=0))
    bundles = list((tmp_path / "out").rglob("*_subtitle_bundle.json"))
    assert len(bundles) == 1
    bundle = json.loads(bundles[0].read_text())
    assert [(c["start_ms"], c["end_ms"]) for c in bundle["source_cues"]] == [(1, 1234), (2000, 2000)]
    assert [(c["start_ms"], c["end_ms"]) for c in bundle["translation_cues"]] == [(1, 1234), (2000, 2000)]
    assert bundle["translation_cues"][0]["text"] == "很長的完整翻譯" * 20
    assert bundle["omitted"] == [{"cue_id": "translation-1", "start_ms": 2000, "end_ms": 2000, "reason": "non_positive_duration"}]
    for request in (boundary, align, repair):
        request.assert_not_called()


def test_imported_source_translation_skips_asr_and_keeps_invalid_times(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    source = tmp_path / "source.srt"
    source.write_text("1\n-00:00:00,500 --> 00:00:01,500\nHello\n\n2\n00:00:02,000 --> 00:00:02,000\nZero\n\n")
    with patch("hermecho.pipeline.transcribe_audio") as asr, \
         patch("hermecho.pipeline.extract_audio") as audio, \
         patch("hermecho.pipeline._video_duration_seconds", return_value=3), \
         patch("hermecho.pipeline.is_portrait_video", return_value=False), \
         patch("hermecho.pipeline.load_reference_material", return_value=""), \
         patch("hermecho.pipeline.load_locked_terms", return_value={}), \
         patch("hermecho.pipeline.translate_segments", side_effect=lambda cues, **kwargs: [{**c, "text": "翻譯"} for c in cues]):
        process_video(PipelineConfig("clip.mp4", input_dir=str(tmp_path), output_dir=str(tmp_path / "out"), source_srt=str(source), srt_only=True))
    bundle = json.loads(next((tmp_path / "out").rglob("*_subtitle_bundle.json")).read_text())
    assert [(c["start_ms"], c["end_ms"]) for c in bundle["translation_cues"]] == [(-500, 1500), (2000, 2000)]
    assert all(c["source_words"] == [] for c in bundle["source_cues"])
    assert "-00:00:00,500" in next((tmp_path / "out").rglob("*_subtitles.srt")).read_text()
    asr.assert_not_called()
    audio.assert_not_called()


def test_standalone_tool_failure_is_not_reported_as_completion():
    import pytest
    from hermecho.video_processing import burn_subtitles_into_video
    with patch("hermecho.video_processing._ffmpeg_supports_subtitles_filter", return_value=False):
        with pytest.raises(RuntimeError, match="subtitles filter"):
            burn_subtitles_into_video("video.mp4", "captions.srt", "out.mp4")
