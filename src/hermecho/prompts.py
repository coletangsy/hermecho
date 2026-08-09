"""Prompt construction helpers for OpenRouter translation."""
from __future__ import annotations

import json
from typing import Dict, List, Optional


def build_translation_prompt(
    chunk_segments: List[Dict],
    target_language: str,
    reference_material: Optional[str],
    context: Dict[str, str],
    locked_terms: Optional[Dict[str, str]] = None,
) -> str:
    """Build the strict JSON translation prompt for a segment chunk."""
    main_text_dict = {
        str(segment.get("_translation_id", index)): segment["text"]
        for index, segment in enumerate(chunk_segments)
    }
    main_text_json = json.dumps({"segments": main_text_dict}, ensure_ascii=False)
    output_shape = json.dumps(
        {
            "translations": {
                translation_id: f"<{target_language} string>"
                for translation_id in main_text_dict
            }
        },
        ensure_ascii=False,
    )

    prev_context = context.get("prev", "")
    next_context = context.get("next", "")
    full_chunk_context = context.get("full_chunk", "")

    prompt_text = (
        f"You are an expert translator and editor specializing in Korean to {target_language}.\n"
        "Your task is two-fold: \n"
        "1. Translate a JSON object of Korean strings. \n"
        "2. Intelligently correct transcription errors based on the provided reference material.\n\n"
        "Contextual Information:\n"
        "To ensure accuracy, use the 'previous' and 'next' text segments for context, but **do not translate them or include them in your output**. Only translate the main JSON object.\n\n"
        "Critical Rules for Your Output:\n"
        f"1. **Translate to {target_language}**: Your primary goal is to translate the Korean text accurately.\n"
        "2. **Strict JSON Output**: Your response MUST be a single, valid JSON object with a single key named \"translations\". The value of this key must be a JSON object (dict) where each key exactly matches an input segment ID and each value is the translated string. Include exactly one entry per input segment. Do not include any conversational text, explanations, or markdown.\n"
        "3. **Preserve English Words**: If a segment contains English words, acronyms, or established brand names (e.g., 'OK', 'iPhone', 'fighting'), they MUST be kept in English. Do not translate them.\n"
        "4. **Translate Names to English**: If a word is a person's name from the reference material, you MUST translate it to their official English Stage Name (e.g., '윤서연' -> 'Yoon SeoYeon'). Do not keep it in Korean.\n"
        "5. **Intelligent Correction**: The transcription from the speech-to-text model may have errors. If you find a word that is likely a misspelled or partial version of a name from the **Reference Material** (e.g., the text says '유연' when the context suggests the member '김유연'), you MUST correct it to the full, proper name from the reference list.\n"
        "6. **Preserve Original Tone**: Maintain the original meaning and natural tone of the dialogue.\n"
        "7. **Korean short-form and 반말 (directness)**: When Korean uses "
        "informal **반말**, dropped verb endings, heavy ellipsis, slang, "
        "or telegraphic phrasing, translate with a **similarly direct** "
        f"equivalent in {target_language}—contractions, informal register "
        "where natural, and **minimal padding** of implied subjects or "
        "politeness. Do not expand short lines into long formal sentences "
        "unless the source clearly signals that level of formality.\n"
        "8. **Korean terms & direct rendering**: For fixed names, stage names, "
        "group or show titles, slogans, and any glossary-style entries in the "
        "**Reference Material**, use that material's **canonical wording** "
        "in the target language — prefer a **direct, conventional** "
        "rendering over creative paraphrase. For Korean **loanwords** "
        "already written in Latin letters in the transcript, match official "
        "or reference spellings; do not invent alternate forms. When the "
        "same Korean term has one standard fan or industry translation, "
        "use it consistently across segments.\n"
    )

    if reference_material:
        prompt_text += (
            "\nReference Material for specific terms:\n"
            f"---\n{reference_material}\n---\n"
        )

    if locked_terms:
        locked_terms_json = json.dumps(locked_terms, ensure_ascii=False)
        prompt_text += (
            "\nLocked Terms (when the source appears, use the exact target):\n"
            f"---\n{locked_terms_json}\n---\n"
        )

    prompt_text += (
        "\nPrevious Context (for context, do not translate):\n"
        f"---\n{prev_context}\n---\n"
    )
    if full_chunk_context:
        prompt_text += (
            "\nOriginal Chunk Context (for context, do not translate or include in output):\n"
            f"---\n{full_chunk_context}\n---\n"
        )
    prompt_text += (
        "\nMain Text to Translate (JSON Object):\n"
        f"---\n{main_text_json}\n---\n"
        "\nNext Context (for context, do not translate):\n"
        f"---\n{next_context}\n---\n"
        "\nYour output MUST be exactly this JSON object shape and nothing else:\n"
        f"{output_shape}\n"
        f"The dict must have exactly {len(chunk_segments)} key(s): {', '.join(main_text_dict)}."
    )
    return prompt_text
