import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from hermecho.sentence_first import build_source_sentences
from hermecho.transcription import repair_mlx_word_timing
from hermecho.checkpoints import CheckpointStore, fingerprint_data, fingerprint_file
from hermecho.pipeline import PipelineConfig, process_video
import pytest


def segments(words):
    return [{"start": w["start"], "end": w["end"], "text": w["word"], "words": [w]} for w in words]


def word(text, start, end):
    return {"word": text, "start": start, "end": end}


def test_cached_mlx_overlap_is_retranscribed_with_anchors_and_raw_audit():
    original = segments([
        word("one", 410, 410.2), word("two", 411, 411.2), word("three", 412, 412.2),
        word("yes", 418.4, 418.92), word("맞아요", 418.85, 419.46),
        word("four", 423, 423.2), word("five", 424, 424.2), word("six", 425, 425.2),
    ])
    unchanged = copy.deepcopy(original)
    fresh = copy.deepcopy(original)
    fresh[4]["start"] = fresh[4]["words"][0]["start"] = 419.0
    # Native clipping returns absolute timestamps from fresh inference.
    with tempfile.TemporaryDirectory() as directory:
        audit = Path(directory) / "audit.json"
        with patch("hermecho.openrouter_transcription._ffprobe_duration", return_value=440), \
             patch("hermecho.checkpoints.fingerprint_file", return_value="audio-hash"), \
             patch("hermecho.openrouter_transcription._extract_audio"), \
             patch("hermecho.transcription._transcribe_with_mlx", return_value=fresh) as transcribe:
            repaired = repair_mlx_word_timing("audio.mp3", original, "large", "ko", 0, audit_path=str(audit))
        build_source_sentences(repaired)
        assert transcribe.call_count == 1
        assert transcribe.call_args.kwargs["clip_timestamps"] == "408,432"
        assert original == unchanged
        record = json.loads(audit.read_text())
        assert record["original_segments"] == unchanged
        assert record["status"] == "complete"
        words = [w for s in repaired for w in s["words"]]
        assert [w["word"] for w in words] == [s["text"] for s in original]
        assert words[4]["start"] == 419.0
        assert words[:3] == [s["words"][0] for s in original[:3]]
        assert words[-3:] == [s["words"][0] for s in original[-3:]]


def test_valid_mlx_transcript_never_loads_runtime_or_rewrites_segments():
    original = segments([word("hello", 0, 1), word("world", 1, 2)])
    with patch("hermecho.transcription._transcribe_with_mlx") as transcribe:
        assert repair_mlx_word_timing("audio.mp3", original, "large", "ko", 0) is original
    transcribe.assert_not_called()


def test_separate_mlx_overlaps_are_repaired_without_rejecting_pending_windows():
    first = segments([
        word("a", 10, 10.2), word("b", 11, 11.2), word("c", 12, 12.2),
        word("yes", 18.4, 18.92), word("right", 18.85, 19.46),
        word("d", 23, 23.2), word("e", 24, 24.2), word("f", 25, 25.2),
    ])
    second = copy.deepcopy(first)
    for segment in second:
        segment["start"] += 100
        segment["end"] += 100
        for w in segment["words"]:
            w["start"] += 100
            w["end"] += 100
    fresh = [copy.deepcopy(first), copy.deepcopy(second)]
    for group in fresh:
        group[4]["start"] = group[4]["words"][0]["start"] = group[3]["end"]
    with patch("hermecho.openrouter_transcription._ffprobe_duration", return_value=150), \
         patch("hermecho.checkpoints.fingerprint_file", return_value="audio-hash"), \
         patch("hermecho.transcription._transcribe_with_mlx", side_effect=fresh) as transcribe:
        repaired = repair_mlx_word_timing("audio.mp3", first + second, "large", "ko", 0)
    build_source_sentences(repaired)
    assert transcribe.call_count == 2
    assert len([w for s in repaired for w in s["words"]]) == 16


def test_unmatched_fresh_words_block_without_changing_original_evidence():
    original = segments([word("first", 10, 11), word("second", 10.9, 12)])
    unchanged = copy.deepcopy(original)
    with patch("hermecho.openrouter_transcription._ffprobe_duration", return_value=30), \
         patch("hermecho.checkpoints.fingerprint_file", return_value="audio-hash"), \
         patch("hermecho.openrouter_transcription._extract_audio"), \
         patch("hermecho.transcription._transcribe_with_mlx", return_value=segments([word("other", 2, 3)])):
        with pytest.raises(RuntimeError, match="no matching word/time anchors"):
            repair_mlx_word_timing("audio.mp3", original, "large", "ko", 0)
    assert original == unchanged


def test_pipeline_repairs_cached_mlx_words_and_reuses_the_repaired_checkpoint():
    original = segments([word("hello", 0, 0.8), word("world.", 0.73, 1.2)])
    repaired = segments([word("hello", 0, 0.8), word("world.", 0.8, 1.2)])
    repairs = []

    def recover(_audio, current, *_args, **_kwargs):
        if current == original:
            repairs.append(current)
            return repaired
        return current

    def translate(source, *_args, **_kwargs):
        return [{**s, "source_text": s["text"], "text": "你好。"} for s in source]

    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "audio.mp3"
        audio.write_bytes(b"audio")
        (Path(directory) / "clip.mp4").write_bytes(b"video")
        checkpoint = CheckpointStore(str(Path(directory) / "clip" / ".hermecho-checkpoint.json"))
        fingerprint = fingerprint_data({"audio": fingerprint_file(str(audio)), "backend": "mlx",
                                        "language": "ko", "model": "large", "temperature": 0.0})
        checkpoint.save_transcription(fingerprint, original)
        config = PipelineConfig("clip.mp4", input_dir=directory, output_dir=directory, language="ko", srt_only=True,
                                transcription_backend="mlx", stage_cooldown=0)
        def extract(_video):
            audio.write_bytes(b"audio")
            return str(audio)
        with patch("hermecho.pipeline.validate_mlx_backend", return_value=None), \
             patch("hermecho.pipeline.extract_audio", side_effect=extract), \
             patch("hermecho.pipeline.transcribe_audio") as transcribe, \
             patch("hermecho.pipeline.repair_mlx_word_timing", side_effect=recover), \
             patch("hermecho.pipeline.translate_segments", side_effect=translate), \
             patch("hermecho.pipeline.load_reference_material", return_value=""), \
             patch("hermecho.pipeline.load_locked_terms", return_value={}), \
             patch("hermecho.pipeline.is_portrait_video", return_value=False), \
             patch("hermecho.pipeline._video_duration_seconds", return_value=3):
            process_video(config)
            process_video(config)
        transcribe.assert_not_called()
        assert len(repairs) == 1
        loaded = CheckpointStore(checkpoint.path).load_transcription(fingerprint, require_words=True)
        assert loaded == repaired
        assert len(list((Path(directory) / "clip").glob("*_subtitles.srt"))) == 2
