"""
This module contains functions for translating text.
"""
import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from transcription import get_openrouter_http_referer


# Constants for the sliding window approach
# Let's set a threshold of 128k characters, which is a safe limit for models like Gemini 1.5 Pro (approx 32k tokens)
TOKEN_THRESHOLD = 128000  # Max characters to send in a single prompt
CHUNK_SIZE = 200          # Number of segments per chunk, increased for better performance
OVERLAP_SIZE = 3         # Number of segments to overlap


def _merge_api_usage_tokens(
    totals: Dict[str, int],
    usage: Optional[Dict[str, Any]],
) -> None:
    """Accumulate OpenAI/OpenRouter-style token_usage into totals."""
    if not usage or not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = usage.get(key)
        if val is not None:
            totals[key] = totals.get(key, 0) + int(val)


def _log_translation_api_tokens(
    label: str,
    usage: Optional[Dict[str, Any]],
) -> None:
    """Print token usage from a chat completion (OpenRouter / OpenAI shape)."""
    if not usage:
        print(f"{label}: (no usage metadata)")
        return
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    tt = usage.get("total_tokens")
    parts = []
    if pt is not None:
        parts.append(f"prompt_tokens={pt}")
    if ct is not None:
        parts.append(f"completion_tokens={ct}")
    if tt is not None:
        parts.append(f"total_tokens={tt}")
    if parts:
        print(f"{label}: " + ", ".join(parts))
    else:
        print(f"{label}: usage={usage!r}")


def _usage_from_langchain_message(message: Any) -> Optional[Dict[str, Any]]:
    """
    Extract token usage from an AIMessage (LangChain OpenAI-compatible).

    OpenRouter typically surfaces usage under response_metadata['token_usage'].
    Some stacks use usage_metadata with input_tokens / output_tokens.
    """
    if message is None:
        return None
    meta = getattr(message, "response_metadata", None) or {}
    u = meta.get("token_usage")
    if isinstance(u, dict) and u:
        return dict(u)
    um = getattr(message, "usage_metadata", None)
    if isinstance(um, dict) and um:
        out: Dict[str, Any] = {}
        if "input_tokens" in um:
            out["prompt_tokens"] = um["input_tokens"]
        if "output_tokens" in um:
            out["completion_tokens"] = um["output_tokens"]
        if "total_tokens" in um:
            out["total_tokens"] = um["total_tokens"]
        return out if out else dict(um)
    return None


def _message_content_to_text(message: Any) -> str:
    """Best-effort string content from an AIMessage or similar."""
    if isinstance(message, AIMessage):
        raw = message.content
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            parts: List[str] = []
            for block in raw:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "".join(parts)
        return str(raw)
    return str(message)


def _translate_chunk(
    chunk_segments: List[Dict],
    target_language: str,
    translation_model: str,
    reference_material: Optional[str],
    context: Dict[str, str],
) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
    """
    Translates a single chunk of subtitle segments using a language model.

    This function formats the text segments as a JSON array and instructs the model
    to return a translated JSON array, which is more robust for segment counting.

    Args:
        chunk_segments: A list of segment dictionaries to be translated.
        target_language: The target language for translation.
        translation_model: The name of the translation model to use.
        reference_material: Optional reference text to guide the translation.
        context: A dictionary containing 'prev' and 'next' text for contextual awareness.

    Returns:
        (translated strings, token_usage) or (None, usage) on failure.
    """
    # Serialize the main text segments into a JSON array string
    main_text_list = [seg["text"] for seg in chunk_segments]
    main_text_json = json.dumps(main_text_list, ensure_ascii=False)

    prev_context = context.get('prev', '')
    next_context = context.get('next', '')

    prompt_template = (
        f"You are an expert translator and editor specializing in Korean to {target_language}.\n"
        "Your task is two-fold: \n" 
        "1. Translate a JSON array of Korean strings. \n"
        "2. Intelligently correct transcription errors based on the provided reference material.\n\n"

        "Contextual Information:\n"
        "To ensure accuracy, use the 'previous' and 'next' text segments for context, but **do not translate them or include them in your output**. Only translate the main JSON array.\n\n"

        "Critical Rules for Your Output:\n"
        f"1. **Translate to {target_language}**: Your primary goal is to translate the Korean text accurately.\n"
        "2. **Strict JSON Output**: Your response MUST be a single, valid JSON object with a single key named \"translations\". The value of this key must be a JSON array with the exact same number of items as the input array. Do not include any conversational text, explanations, or markdown.\n"
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
        prompt_template += (
            "\nReference Material for specific terms:\n"
            f"---\n{reference_material}\n---\n"
        )

    prompt_template += (
        "\nPrevious Context (for context, do not translate):\n"
        f"---\n{prev_context}\n---\n"
        "\nMain Text to Translate (JSON Array):\n"
        f"---\n{main_text_json}\n---\n"
        "\nNext Context (for context, do not translate):\n"
        f"---\n{next_context}\n---\n"
        f"\nYour translated output must be a valid JSON array of {target_language} strings."
    )

    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatOpenAI(
        model=translation_model,
        temperature=0,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": get_openrouter_http_referer(),
            "X-Title": "Hermecho Translation",
        },
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    runnable = prompt | model

    usage: Optional[Dict[str, Any]] = None
    try:
        message = runnable.invoke({})
        usage = _usage_from_langchain_message(message)
        _log_translation_api_tokens("Translation API tokens", usage)
        response_text = _message_content_to_text(message)

        response_json = json.loads(response_text)
        translated_segments = response_json['translations']

        if len(translated_segments) != len(chunk_segments):
            print(
                f"Warning: Mismatch in segment count for a chunk. Expected {len(chunk_segments)}, got {len(translated_segments)}.")
            return None, usage

        return translated_segments, usage

    except json.JSONDecodeError:
        print("Warning: Failed to decode JSON from the model's response.")
        return None, usage
    except Exception as e:
        print(f"An unexpected error occurred during chunk translation: {e}")
        return None, usage




def translate_segments(
    segments: List[Dict],
    target_language: str,
    translation_model: str,
    reference_material: Optional[str],
) -> Optional[List[Dict]]:
    """
    Translates transcribed text segments using an optimized, two-layer strategy.
    
    For shorter texts that fit within the token threshold, it translates the entire content
    in a single batch for maximum speed. For longer texts, it uses a sliding window approach with
    robust JSON-based chunking and a concurrent fallback mechanism.

    Args:
        segments: A list of transcribed segments.
        target_language: The target language for the translation.
        translation_model: The translation model to be used.
        reference_material: Optional reference text for context-aware translation.

    Returns:
        A list of translated segments, or None if a critical error occurs.
    """
    print("Translating text using an optimized strategy...")
    translated_segments_text = []
    num_segments = len(segments)

    # Calculate the total length to decide on the translation strategy
    full_text = "\n".join([seg["text"] for seg in segments])
    # A rough estimation of the overhead from the prompt template and reference material
    prompt_overhead = len(reference_material or "") + 1000 
    total_length = len(full_text) + prompt_overhead

    try:
        # Determine strategy
        use_sliding_window = False
        usage_totals: Dict[str, int] = {}

        if total_length < TOKEN_THRESHOLD:
            print(
                "Text is short enough. Attempting to translate in a "
                "single batch."
            )
            translated_segments_text, chunk_usage = _translate_chunk(
                segments,
                target_language,
                translation_model,
                reference_material,
                context={},
            )
            _merge_api_usage_tokens(usage_totals, chunk_usage)

            if translated_segments_text is None:
                print(
                    "Single batch translation failed (likely segment mismatch). "
                    "Falling back to sliding window strategy."
                )
                use_sliding_window = True
        else:
            print("Text is too long, using sliding window translation.")
            use_sliding_window = True

        # Strategy 2: Sliding Window (Chunks)
        # This runs if the text was too long initially, OR if the single batch
        # attempt failed.
        if use_sliding_window:
            translated_segments_text = []
            num_chunks = (num_segments + CHUNK_SIZE - 1) // CHUNK_SIZE

            for i in tqdm(
                range(num_chunks),
                desc="Translating in chunks",
            ):
                start_index = i * CHUNK_SIZE
                end_index = min(start_index + CHUNK_SIZE, num_segments)
                chunk = segments[start_index:end_index]

                # Define context
                prev_start = max(0, start_index - OVERLAP_SIZE)
                prev_context_segments = segments[prev_start:start_index]
                next_start = end_index
                next_end = min(next_start + OVERLAP_SIZE, num_segments)
                next_context_segments = segments[next_start:next_end]

                context = {
                    'prev': "\n".join([seg["text"] for seg in prev_context_segments]),
                    'next': "\n".join([seg["text"] for seg in next_context_segments])
                }

                translated_chunk, u = _translate_chunk(
                    chunk,
                    target_language,
                    translation_model,
                    reference_material,
                    context,
                )
                _merge_api_usage_tokens(usage_totals, u)

                # Fallback strategy for chunk (only if chunk fails)
                if translated_chunk is None:
                    print(
                        f"Warning: Chunk {i} failed. Attempting to split "
                        "into smaller sub-chunks (size 50)."
                    )
                    translated_chunk = []
                    sub_chunk_size = 50

                    for j in range(0, len(chunk), sub_chunk_size):
                        sub_chunk = chunk[j: j + sub_chunk_size]
                        sub_result, su = _translate_chunk(
                            sub_chunk,
                            target_language,
                            translation_model,
                            reference_material,
                            context,
                        )
                        _merge_api_usage_tokens(usage_totals, su)

                        if sub_result is None:
                            print(
                                f"  Warning: Sub-chunk starting at {j} "
                                "failed. Splitting into mini-chunks "
                                "(size 10)."
                            )
                            mini_chunk_size = 10

                            for k in range(0, len(sub_chunk), mini_chunk_size):
                                mini_chunk = sub_chunk[k: k + mini_chunk_size]
                                mini_result, mu = _translate_chunk(
                                    mini_chunk,
                                    target_language,
                                    translation_model,
                                    reference_material,
                                    context,
                                )
                                _merge_api_usage_tokens(usage_totals, mu)

                                if mini_result is None:
                                    print(
                                        f"    Error: Mini-chunk starting "
                                        f"at {k} failed. Skipping translation "
                                        "for this small section."
                                    )
                                    translated_chunk.extend([""] * len(mini_chunk))
                                else:
                                    translated_chunk.extend(mini_result)
                        else:
                            translated_chunk.extend(sub_result)

                if translated_chunk:
                    translated_segments_text.extend(translated_chunk)

        _log_translation_api_tokens(
            "Translation API tokens — cumulative (reported chunks)",
            usage_totals,
        )

        # Final step: Combine translated text back into the segment structure
        final_segments = []
        for i, segment in enumerate(segments):
            translated_segment = segment.copy()
            original_text = segment.get("text", "").strip()
            translated_segment["source_text"] = original_text

            # If the original text was a placeholder for silence, ensure the
            # translation is empty
            if original_text == "[no speech]":
                translated_segment["text"] = ""
            elif i < len(translated_segments_text):
                translated_text = translated_segments_text[i]
                # Clean up punctuation
                translated_text = translated_text.replace("，", " ").replace(
                    "。", " "
                ).strip()
                translated_segment["text"] = translated_text
            else:
                translated_segment["text"] = ""
            final_segments.append(translated_segment)

        print("Text translated successfully.")
        return final_segments

    except Exception as e:
        print(f"An error occurred during text translation: {e}")
        return None