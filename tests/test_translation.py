import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from hermecho.translation import _translate_chunk, translate_segments
from hermecho.utils import load_locked_terms


class TestOpenRouterTranslation(unittest.TestCase):
    def test_locked_terms_mapping_is_machine_readable(self) -> None:
        locked_terms = load_locked_terms("references/locked_terms.json")

        self.assertEqual(locked_terms["트리플에스"], "tripleS")

    def test_locked_terms_normalize_whitespace_before_enforcement(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as tmp:
            locked_terms_path = tmp.name
            tmp.write('{" 트리플에스 ": " tripleS "}')

        try:
            locked_terms = load_locked_terms(locked_terms_path)
        finally:
            os.unlink(locked_terms_path)

        self.assertEqual(locked_terms, {"트리플에스": "tripleS"})
        with patch(
            "hermecho.translation._translate_chunk",
            return_value=({"translations": {"0": "錯誤名稱"}}, None),
        ) as translate_chunk, patch("hermecho.translation.time.sleep"):
            translated = translate_segments(
                [{"start": 0.0, "end": 1.0, "text": "트리플에스"}],
                target_language="Traditional Chinese (Taiwan)",
                translation_model="test-model",
                reference_material=None,
                locked_terms=locked_terms,
            )

        self.assertIsNone(translated)
        self.assertEqual(translate_chunk.call_count, 3)

    def test_locked_terms_reject_duplicate_trimmed_and_invalid_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            mappings = {
                "duplicate": '{"term": "first", "term": "second"}',
                "trim_collision": '{" term ": "first", "term": "second"}',
                "invalid_schema": '{"term": 1}',
            }
            for label, content in mappings.items():
                with self.subTest(mapping=label):
                    locked_terms_path = os.path.join(temporary_dir, f"{label}.json")
                    with open(locked_terms_path, "w", encoding="utf-8") as locked_terms_file:
                        locked_terms_file.write(content)
                    with patch("builtins.print") as mock_print:
                        locked_terms = load_locked_terms(locked_terms_path)

                    self.assertIsNone(locked_terms)
                    messages = [
                        str(call.args[0])
                        for call in mock_print.call_args_list
                        if call.args
                    ]
                    self.assertTrue(
                        any(
                            "--locked-terms-file" in message
                            and locked_terms_path in message
                            for message in messages
                        )
                    )

    def test_missing_locked_terms_path_names_the_option(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            locked_terms_path = os.path.join(temporary_dir, "missing.json")
            with patch("builtins.print") as mock_print:
                locked_terms = load_locked_terms(locked_terms_path)

        self.assertIsNone(locked_terms)
        mock_print.assert_any_call(
            f"Error: --locked-terms-file not found at {locked_terms_path}"
        )

    def test_missing_openrouter_api_key_fails_cleanly(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch("builtins.print") as mock_print:
            translated, usage = _translate_chunk(
                [{"start": 0.0, "end": 1.0, "text": "hello"}],
                target_language="Traditional Chinese (Taiwan)",
                translation_model="deepseek/deepseek-v4-pro",
                reference_material=None,
                context={},
            )

        self.assertIsNone(translated)
        self.assertIsNone(usage)
        mock_print.assert_any_call("Error: OPENROUTER_API_KEY is not set.")

    def test_translate_chunk_calls_openrouter_chat_completions(self) -> None:
        response = MagicMock()
        response.choices = [
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=json.dumps({"translations": {"0": "你好"}})
                )
            )
        ]
        response.usage = types.SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
        )

        client = MagicMock()
        client.chat.completions.create.return_value = response
        openai_module = types.SimpleNamespace(OpenAI=MagicMock(return_value=client))

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True), \
            patch.dict(sys.modules, {"openai": openai_module}):
            translated, usage = _translate_chunk(
                [{"start": 0.0, "end": 1.0, "text": "hello"}],
                target_language="Traditional Chinese (Taiwan)",
                translation_model="deepseek/deepseek-v4-pro",
                reference_material=None,
                context={},
            )

        self.assertEqual(translated, {"translations": {"0": "你好"}})
        self.assertEqual(
            usage,
            {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        )
        openai_module.OpenAI.assert_called_once_with(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
        )
        client.chat.completions.create.assert_called_once()
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek/deepseek-v4-pro")
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["messages"][0]["role"], "user")
        self.assertIn("hello", kwargs["messages"][0]["content"])
        self.assertEqual(
            kwargs["extra_body"]["provider"],
            {
                "order": ["alibaba", "atlas-cloud/fp8"],
                "allow_fallbacks": True,
                "require_parameters": True,
            },
        )

    def test_translate_segments_emits_structured_chunk_progress(self) -> None:
        segments = [
            {"start": float(i), "end": float(i + 1), "text": f"line {i}"}
            for i in range(201)
        ]

        def fake_translate_chunk(chunk, *_args, **_kwargs):
            return {
                "translations": {
                    segment["_translation_id"]: f"translated {i}"
                    for i, segment in enumerate(chunk)
                }
            }, None

        with patch("hermecho.translation.TOKEN_THRESHOLD", 1), \
            patch("hermecho.translation._translate_chunk", side_effect=fake_translate_chunk), \
            patch("builtins.print") as mock_print:
            translated = translate_segments(
                segments,
                target_language="Traditional Chinese (Taiwan)",
                translation_model="test-model",
                reference_material=None,
            )

        self.assertEqual(len(translated or []), 201)
        progress_lines = [
            call.args[0]
            for call in mock_print.call_args_list
            if call.args and call.args[0].startswith("HERMECHO_PROGRESS ")
        ]
        events = [
            json.loads(line.removeprefix("HERMECHO_PROGRESS "))
            for line in progress_lines
        ]

        self.assertIn(
            {
                "stage": "translation_strategy",
                "status": "running",
                "message": "Using sliding window translation",
                "total": 2,
            },
            events,
        )
        self.assertIn(
            {
                "stage": "translation",
                "status": "running",
                "message": "Translating chunk 2/2",
                "current": 2,
                "total": 2,
            },
            events,
        )

    def test_translate_segments_retry_keeps_original_chunk_source_context(self) -> None:
        segments = [
            {"start": float(i), "end": float(i + 1), "text": f"line {i}"}
            for i in range(3)
        ]
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=json.dumps({"translations": {"0": "甲", "2": "丙"}})
                        )
                    )
                ],
                usage=None,
            ),
            types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=json.dumps({"translations": {"1": "乙"}})
                        )
                    )
                ],
                usage=None,
            ),
        ]
        openai_module = types.SimpleNamespace(OpenAI=MagicMock(return_value=client))

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True), \
            patch.dict(sys.modules, {"openai": openai_module}), \
            patch("hermecho.translation.time.sleep"):
            translated = translate_segments(
                segments,
                target_language="Traditional Chinese (Taiwan)",
                translation_model="test-model",
                reference_material=None,
            )

        self.assertEqual([segment["text"] for segment in translated], ["甲", "乙", "丙"])
        self.assertEqual(client.chat.completions.create.call_count, 2)
        retry_prompt = client.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ][0]["content"]
        chunk_context = (
            "Original Chunk Context (for context, do not translate or include in output):\n"
            "---\n0: line 0\n1: line 1\n2: line 2\n---"
        )
        self.assertIn(chunk_context, retry_prompt)
        self.assertLess(
            retry_prompt.index(chunk_context),
            retry_prompt.index("Main Text to Translate (JSON Object):"),
        )
        self.assertIn('{"segments": {"1": "line 1"}}', retry_prompt)

    def test_translate_segments_retry_context_keeps_non_contiguous_accepted_source(self) -> None:
        segments = [
            {"start": float(i), "end": float(i + 1), "text": f"line {i}"}
            for i in range(3)
        ]
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=json.dumps({"translations": {"1": "乙"}})
                        )
                    )
                ],
                usage=None,
            ),
            types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=json.dumps(
                                {"translations": {"0": "甲", "2": "丙"}}
                            )
                        )
                    )
                ],
                usage=None,
            ),
        ]
        openai_module = types.SimpleNamespace(OpenAI=MagicMock(return_value=client))

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True), \
            patch.dict(sys.modules, {"openai": openai_module}), \
            patch("hermecho.translation.time.sleep"):
            translated = translate_segments(
                segments,
                target_language="Traditional Chinese (Taiwan)",
                translation_model="test-model",
                reference_material=None,
            )

        self.assertEqual([segment["text"] for segment in translated], ["甲", "乙", "丙"])
        retry_prompt = client.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ][0]["content"]
        context_start = retry_prompt.index(
            "Original Chunk Context (for context, do not translate or include in output):"
        )
        first_source = retry_prompt.index("0: line 0", context_start)
        accepted_source = retry_prompt.index("1: line 1", context_start)
        last_source = retry_prompt.index("2: line 2", context_start)
        self.assertLess(first_source, accepted_source)
        self.assertLess(accepted_source, last_source)
        self.assertLess(
            context_start,
            retry_prompt.index("Main Text to Translate (JSON Object):"),
        )
        self.assertIn(
            '{"segments": {"0": "line 0", "2": "line 2"}}',
            retry_prompt,
        )

    def test_translate_segments_rejects_duplicate_translation_ids(self) -> None:
        response = MagicMock()
        response.choices = [
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content='{"translations": {"0": "甲", "0": "乙"}}'
                )
            )
        ]
        response.usage = None
        client = MagicMock()
        client.chat.completions.create.return_value = response
        openai_module = types.SimpleNamespace(OpenAI=MagicMock(return_value=client))

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True), \
            patch.dict(sys.modules, {"openai": openai_module}), \
            patch("hermecho.translation.time.sleep"), \
            patch("builtins.print") as mock_print:
            translated = translate_segments(
                [{"start": 0.0, "end": 1.0, "text": "hello"}],
                target_language="Traditional Chinese (Taiwan)",
                translation_model="test-model",
                reference_material=None,
            )

        self.assertIsNone(translated)
        self.assertEqual(client.chat.completions.create.call_count, 3)
        messages = [str(call.args[0]) for call in mock_print.call_args_list if call.args]
        self.assertTrue(any("duplicate_id(0)" in message for message in messages))

    def test_translate_segments_retries_extra_ids(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "first"},
            {"start": 1.0, "end": 2.0, "text": "second"},
        ]

        with patch(
            "hermecho.translation._translate_chunk",
            side_effect=[
                ({"translations": {"0": "甲", "1": "乙", "9": "extra"}}, None),
                ({"translations": {"0": "甲", "1": "乙"}}, None),
            ],
        ) as translate_chunk, patch("hermecho.translation.time.sleep"):
            translated = translate_segments(
                segments,
                target_language="Traditional Chinese (Taiwan)",
                translation_model="test-model",
                reference_material=None,
            )

        self.assertEqual([segment["text"] for segment in translated], ["甲", "乙"])
        self.assertEqual(translate_chunk.call_count, 2)
        self.assertEqual(
            [segment["_translation_id"] for segment in translate_chunk.call_args_list[1].args[0]],
            ["0", "1"],
        )

    def test_translate_segments_retries_malformed_and_empty_ids(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "first"},
            {"start": 1.0, "end": 2.0, "text": "second"},
        ]

        with patch(
            "hermecho.translation._translate_chunk",
            side_effect=[
                ({"translations": {"0": 3, "1": "  "}}, None),
                ({"translations": {"0": "甲", "1": "乙"}}, None),
            ],
        ) as translate_chunk, patch("hermecho.translation.time.sleep"):
            translated = translate_segments(
                segments,
                target_language="Traditional Chinese (Taiwan)",
                translation_model="test-model",
                reference_material=None,
            )

        self.assertEqual([segment["text"] for segment in translated], ["甲", "乙"])
        self.assertEqual(
            [segment["_translation_id"] for segment in translate_chunk.call_args_list[1].args[0]],
            ["0", "1"],
        )

    def test_translate_segments_retries_locked_term_failures_only(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "트리플에스"},
            {"start": 1.0, "end": 2.0, "text": "ordinary"},
        ]

        with patch(
            "hermecho.translation._translate_chunk",
            side_effect=[
                ({"translations": {"0": "錯誤名稱", "1": "正確"}}, None),
                ({"translations": {"0": "tripleS"}}, None),
            ],
        ) as translate_chunk, patch("hermecho.translation.time.sleep"):
            translated = translate_segments(
                segments,
                target_language="Traditional Chinese (Taiwan)",
                translation_model="test-model",
                reference_material=None,
                locked_terms={"트리플에스": "tripleS"},
            )

        self.assertEqual([segment["text"] for segment in translated], ["tripleS", "正確"])
        self.assertEqual(
            [segment["_translation_id"] for segment in translate_chunk.call_args_list[1].args[0]],
            ["0"],
        )

    def test_translate_segments_omits_non_speech_before_translation(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "first"},
            {"start": 1.0, "end": 2.0, "text": "[no speech]"},
            {"start": 2.0, "end": 3.0, "text": "  "},
            {"start": 3.0, "end": 4.0, "text": "second"},
        ]

        with patch(
            "hermecho.translation._translate_chunk",
            return_value=({"translations": {"0": "甲", "3": "乙"}}, None),
        ) as translate_chunk:
            translated = translate_segments(
                segments,
                target_language="Traditional Chinese (Taiwan)",
                translation_model="test-model",
                reference_material=None,
            )

        self.assertEqual([segment["text"] for segment in translated], ["甲", "乙"])
        self.assertEqual(
            [segment["_translation_id"] for segment in translate_chunk.call_args.args[0]],
            ["0", "3"],
        )

    def test_translate_segments_preserves_punctuation_for_all_delivery_profiles(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]

        with patch(
            "hermecho.translation._translate_chunk",
            return_value=({"translations": {"0": "你好，世界。"}}, None),
        ):
            portrait = translate_segments(
                segments,
                target_language="Traditional Chinese (Taiwan)",
                translation_model="test-model",
                reference_material=None,
                preserve_punctuation=True,
            )
            landscape = translate_segments(
                segments,
                target_language="Traditional Chinese (Taiwan)",
                translation_model="test-model",
                reference_material=None,
            )

        self.assertEqual(portrait[0]["text"], "你好，世界。")
        self.assertEqual(landscape[0]["text"], "你好，世界。")

    def test_translate_segments_saves_only_gate_accepted_chunks(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "first"}]
        save_chunk = MagicMock()

        with patch(
            "hermecho.translation._translate_chunk",
            return_value=({"translations": {"0": "甲"}}, None),
        ):
            translated = translate_segments(
                segments,
                target_language="Traditional Chinese (Taiwan)",
                translation_model="test-model",
                reference_material=None,
                accepted_chunk_saver=save_chunk,
            )

        self.assertEqual([segment["text"] for segment in translated or []], ["甲"])
        save_chunk.assert_called_once()

        with patch(
            "hermecho.translation._translate_chunk",
            return_value=({"translations": {"0": "  "}}, None),
        ), patch("hermecho.translation.time.sleep"):
            translated = translate_segments(
                segments,
                target_language="Traditional Chinese (Taiwan)",
                translation_model="test-model",
                reference_material=None,
                accepted_chunk_saver=save_chunk,
            )

        self.assertIsNone(translated)
        self.assertEqual(save_chunk.call_count, 1)


if __name__ == "__main__":
    unittest.main()
