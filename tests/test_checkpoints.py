import os
import tempfile
import unittest
from unittest.mock import patch

from hermecho.checkpoints import CheckpointStore


class TestCheckpointStore(unittest.TestCase):
    def test_source_sentence_grouping_is_persisted_and_invalidates_translation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = os.path.join(temporary_dir, "checkpoint.json")
            first_sentence = {
                "start": 0.0,
                "end": 1.0,
                "text": "안녕하세요.",
                "source_words": [
                    {"word": "안녕", "start": 0.0, "end": 0.5},
                    {"word": "하세요.", "start": 0.5, "end": 1.0},
                ],
                "source_word_indices": [0, 1],
            }
            store = CheckpointStore(checkpoint_path)
            store.save_accepted_translation_chunk("translation", 0, "chunk", {"0": "你好。"})
            store.save_source_sentences("grouping", [first_sentence])

            self.assertEqual(store.load_source_sentences("grouping"), [first_sentence])
            self.assertIsNone(store.load_accepted_translation_chunk("translation", 0, "chunk", ["0"]))

            store.save_accepted_translation_chunk("translation", 0, "chunk", {"0": "你好。"})
            split_sentences = [
                {
                    "start": 0.0,
                    "end": 0.5,
                    "text": "안녕",
                    "source_words": [first_sentence["source_words"][0]],
                    "source_word_indices": [0],
                },
                {
                    "start": 0.5,
                    "end": 1.0,
                    "text": "하세요.",
                    "source_words": [first_sentence["source_words"][1]],
                    "source_word_indices": [1],
                },
            ]
            store.save_source_sentences("grouping", split_sentences)
            self.assertIsNone(store.load_accepted_translation_chunk("translation", 0, "chunk", ["0"]))

            resumed = CheckpointStore(checkpoint_path)
            self.assertEqual(resumed.load_source_sentences("grouping"), split_sentences)

    def test_invalid_source_sentence_grouping_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = os.path.join(temporary_dir, "checkpoint.json")
            with open(checkpoint_path, "w", encoding="utf-8") as checkpoint:
                checkpoint.write(
                    '{"version": 1, "source_grouping": {"status": "accepted", '
                    '"fingerprint": "grouping", "sentences": [{"start": 0.0, '
                    '"end": 1.0, "text": "bad", "source_words": [{'
                    '"word": "good", "start": 0.0, "end": 1.0}], '
                    '"source_word_indices": [0]}]}}'
                )

            store = CheckpointStore(checkpoint_path)

            self.assertIsNone(store.load_source_sentences("grouping"))

    def test_interrupted_atomic_replace_keeps_previous_completed_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = os.path.join(temporary_dir, "checkpoint.json")
            segments = [{"start": 0.0, "end": 1.0, "text": "first"}]
            store = CheckpointStore(checkpoint_path)
            store.save_transcription("first", segments)

            with patch(
                "hermecho.checkpoints.os.replace",
                side_effect=OSError("interrupted write"),
            ):
                with self.assertRaises(OSError):
                    store.save_transcription("second", [{"start": 1.0, "end": 2.0, "text": "second"}])

            resumed = CheckpointStore(checkpoint_path)
            self.assertEqual(resumed.load_transcription("first"), segments)
            self.assertIsNone(resumed.load_transcription("second"))

    def test_partial_or_version_incompatible_checkpoint_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = os.path.join(temporary_dir, "checkpoint.json")
            with open(checkpoint_path, "w", encoding="utf-8") as checkpoint:
                checkpoint.write('{"version":')

            store = CheckpointStore(checkpoint_path)
            self.assertIsNone(store.load_transcription("fingerprint"))

            with open(checkpoint_path, "w", encoding="utf-8") as checkpoint:
                checkpoint.write('{"version": 999}')

            store = CheckpointStore(checkpoint_path)
            self.assertIsNone(store.load_transcription("fingerprint"))

    def test_checkpoint_diagnostics_are_not_accepted_translation_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = os.path.join(temporary_dir, "checkpoint.json")
            with open(checkpoint_path, "w", encoding="utf-8") as checkpoint:
                checkpoint.write(
                    '{"version": 1, "transcription": {'
                    '"status": "complete", "fingerprint": "asr", '
                    '"segments": [{"start": 0.0, "end": 1.0, "text": "source"}]}, '
                    '"translation": {"fingerprint": "translation", "chunks": {'
                    '"0": {"status": "accepted", "fingerprint": "chunk", '
                    '"translations": {"0": "accepted"}, "diagnostics": ["rejected"]}}}}'
                )

            store = CheckpointStore(checkpoint_path)
            self.assertIsNone(
                store.load_accepted_translation_chunk(
                    "translation",
                    0,
                    "chunk",
                    ["0"],
                )
            )

    def test_translation_checkpoint_without_local_transcription_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = os.path.join(temporary_dir, "checkpoint.json")
            with open(checkpoint_path, "w", encoding="utf-8") as checkpoint:
                checkpoint.write(
                    '{"version": 1, "translation": {"fingerprint": "translation", '
                    '"chunks": {"0": {"status": "accepted", "fingerprint": "chunk", '
                    '"translations": {"0": "accepted"}}}}}'
                )

            store = CheckpointStore(checkpoint_path)

            self.assertEqual(
                store.load_accepted_translation_chunk(
                    "translation",
                    0,
                    "chunk",
                    ["0"],
                ),
                {"0": "accepted"},
            )


if __name__ == "__main__":
    unittest.main()
