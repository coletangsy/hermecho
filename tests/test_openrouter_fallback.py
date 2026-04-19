"""
Unit tests for OpenRouter Gemini 3.1 -> OpenAI GPT fallback routing.
"""
import unittest

from openrouter_fallback import (
    openrouter_fallback_model,
    openrouter_models_with_fallback,
)


class TestOpenrouterFallback(unittest.TestCase):
    """Map primary slugs to optional backup models."""

    def test_pro_preview_text_fallback_is_gpt_54(self) -> None:
        fb = openrouter_fallback_model("google/gemini-3.1-pro-preview")
        self.assertEqual(fb, "openai/gpt-5.4")

    def test_pro_preview_transcribe_fallback_is_gpt_audio(self) -> None:
        fb = openrouter_fallback_model(
            "google/gemini-3.1-pro-preview",
            multimodal_transcription=True,
        )
        self.assertEqual(fb, "openai/gpt-audio")

    def test_flash_lite_gets_gpt_mini(self) -> None:
        fb = openrouter_fallback_model(
            "google/gemini-3.1-flash-lite-preview",
        )
        self.assertEqual(fb, "openai/gpt-5.4-mini")

    def test_non_gemini_no_fallback(self) -> None:
        self.assertIsNone(
            openrouter_fallback_model("google/gemini-2.5-pro"),
        )
        self.assertIsNone(openrouter_fallback_model("anthropic/claude-3-opus"))

    def test_models_with_fallback_order(self) -> None:
        primary = "google/gemini-3.1-pro-preview"
        chain = openrouter_models_with_fallback(primary)
        self.assertEqual(
            chain,
            [primary, "openai/gpt-5.4"],
        )
        chain_audio = openrouter_models_with_fallback(
            primary,
            multimodal_transcription=True,
        )
        self.assertEqual(
            chain_audio,
            [primary, "openai/gpt-audio"],
        )

    def test_models_dedupes_identical_fallback(self) -> None:
        self.assertEqual(
            openrouter_models_with_fallback("openai/gpt-5.4-mini"),
            ["openai/gpt-5.4-mini"],
        )


if __name__ == "__main__":
    unittest.main()
