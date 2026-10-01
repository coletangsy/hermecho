import json
import subprocess
from unittest.mock import patch

import pytest

from hermecho.openrouter_transcription import OpenRouterRequestError, transcribe_openrouter


def words(*items):
    return [{"word": text, "start": start, "end": start + 0.2} for text, start in items]


@pytest.mark.parametrize("repair_failure", [False, True, "extraction", "response"])
@pytest.mark.parametrize("missing_edge_word", [False, True])
def test_conflicting_chunk_edges_use_cached_authoritative_boundary_audio(tmp_path, repair_failure, missing_edge_word):
    audio = tmp_path / "source.mp3"
    audio.write_bytes(b"source")
    checkpoint = tmp_path / "checkpoint.json"
    left = words(("a", 50), ("b", 51), ("c", 52), ("wrong", 59.5))
    right = words(("different", 0.5), ("r", 4), ("s", 5), ("t", 6), ("tail", 31))
    boundary = words(("a", 0), ("b", 1), ("c", 2), ("fixed", 9.5),
                     ("r", 13), ("s", 14), ("t", 15))
    if missing_edge_word:
        # The reported Korean failure: the earlier audio cut has no overlap
        # word, while the next response recognises 이번에 at 60.72 seconds.
        left.pop()
        right[0] = {"word": "이번에", "start": 1.72, "end": 2.0}
        boundary[3] = {"word": "이번에", "start": 10.72, "end": 11.0}
    responses = [({"text": " ".join(w["word"] for w in value), "words": value},
                  {"cost_usd": 0.01, "elapsed_seconds": 1.0})
                 for value in [left, right, boundary]]
    if repair_failure is True:
        responses[-1] = OpenRouterRequestError("timeout")
    elif repair_failure == "response":
        responses[-1] = ({"text": "uncovered", "words": []}, {})

    def extract(_source, destination, _start, _duration):
        if repair_failure == "extraction" and _start == 50:
            raise subprocess.CalledProcessError(1, ["ffmpeg"])
        destination.write_bytes(b"chunk")

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), patch(
        "hermecho.openrouter_transcription._ffprobe_duration", return_value=120.0,
    ), patch("hermecho.openrouter_transcription._extract_audio", side_effect=extract), patch(
        "hermecho.openrouter_transcription._request_transcription", side_effect=responses,
    ) as request:
        if repair_failure:
            with pytest.raises(RuntimeError, match="boundary repair near 60s failed") as error:
                transcribe_openrouter(str(audio), "model", "ko", checkpoint_path=str(checkpoint))
            assert not isinstance(error.value, OpenRouterRequestError)
            assert len(json.loads(checkpoint.read_text())["chunks"]) == 2
            assert error.value.__cause__ is not None
            assert request.call_count == (2 if repair_failure == "extraction" else 3)
            return
        first = transcribe_openrouter(str(audio), "model", "ko", checkpoint_path=str(checkpoint))
        second = transcribe_openrouter(str(audio), "model", "ko", checkpoint_path=str(checkpoint))

    assert first == second
    assert request.call_count == 3
    assert [w["word"] for w in first[0]["words"]] == [
        "a", "b", "c", "이번에" if missing_edge_word else "fixed", "r", "s", "t", "tail",
    ]
    assert [w["start"] for w in first[0]["words"]] == [50, 51, 52, 60.72 if missing_edge_word else 59.5, 63, 64, 65, 90]
    state = json.loads(checkpoint.read_text())
    assert len(state["chunks"]) == 2
    assert len(state["boundary_repairs"]) == 1
    assert state["metadata"]["cost_usd"] == pytest.approx(0.03)


def test_boundary_repair_without_common_word_and_time_anchors_still_blocks(tmp_path):
    from hermecho.openrouter_transcription import _splice_boundary_words

    with pytest.raises(RuntimeError, match="anchors"):
        _splice_boundary_words(
            words(("left", 55)), words(("right", 65)), words(("unknown", 60)), 60,
        )


def test_boundary_repair_does_not_guess_between_repeated_word_anchors():
    from hermecho.openrouter_transcription import _splice_boundary_words

    left = words(("ha", 58.1), ("ha", 58.2), ("ha", 58.3))
    right = words(("r", 63), ("s", 64), ("t", 65))
    bridge = words(("ha", 58.0), ("ha", 58.1), ("ha", 58.2), ("ha", 58.3)) + right
    with pytest.raises(RuntimeError, match="ambiguous anchors"):
        _splice_boundary_words(left, right, bridge, 60)


def test_assembly_migration_reuses_raw_evidence_and_records_new_policy(tmp_path):
    import hermecho.openrouter_transcription as transcription

    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")
    checkpoint = tmp_path / "checkpoint.json"
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), patch.object(
        transcription, "_ffprobe_duration", return_value=3.0,
    ), patch.object(transcription, "_extract_audio"), patch.object(
        transcription, "_request_transcription", return_value=({"text": "hi", "words": words(("hi", 1))}, {}),
    ) as request:
        with patch.object(transcription, "TRANSCRIPTION_ASSEMBLY_RULES", "old-rules"):
            first = transcribe_openrouter(str(audio), checkpoint_path=str(checkpoint))
        old = json.loads(checkpoint.read_text())
        second = transcribe_openrouter(str(audio), checkpoint_path=str(checkpoint))
        current = json.loads(checkpoint.read_text())
        assert first == second
        assert old["chunks"] == current["chunks"]
        assert request.call_count == 1
        assert current["assembly_policy"]["rules"] == transcription.TRANSCRIPTION_ASSEMBLY_RULES
        assert old["assembly_fingerprint"] != current["assembly_fingerprint"]
        with patch.object(transcription, "_words_for_output", side_effect=RuntimeError("assembly blocked")):
            with pytest.raises(RuntimeError, match="assembly blocked"):
                transcribe_openrouter(str(audio), checkpoint_path=str(checkpoint))
        assert json.loads(checkpoint.read_text())["status"] == "partial"
