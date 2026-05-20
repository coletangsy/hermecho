import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from hermecho.translation import _translate_chunk, translate_segments


class TestOpenRouterTranslation(unittest.TestCase):
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

        self.assertEqual(translated, ["你好"])
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
            return [f"translated {i}" for i, _ in enumerate(chunk)], None

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


if __name__ == "__main__":
    unittest.main()
