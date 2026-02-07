"""
This module contains functions for generating, adjusting, and cleaning subtitles.
"""
import logging


def split_long_segments(segments: list[dict], max_chars: int = 40, max_duration: float = 7.0) -> list[dict]:
    """
    Splits segments that are too long in character count or duration.
    Requires segments to have 'words' with timestamps (from Whisper word_timestamps=True).
    
    Args:
        segments: List of transcription segments.
        max_chars: Maximum characters allowed per segment.
        max_duration: Maximum duration (seconds) allowed per segment.
        
    Returns:
        A new list of segments with long ones split.
    """
    split_segments = []
    
    for seg in segments:
        text = seg.get("text", "").strip()
        start = seg["start"]
        end = seg["end"]
        duration = end - start
        words = seg.get("words", [])
        
        # If segment is short enough, keep it as is
        if len(text) <= max_chars and duration <= max_duration:
            split_segments.append(seg)
            continue
            
        # If no word timestamps, we can't split accurately, so keep it (or implement naive time split)
        if not words:
            split_segments.append(seg)
            continue
            
        # Split logic: Try to split into chunks that fit constraints
        current_chunk_words = []
        current_chunk_start = words[0]["start"]
        
        for i, word_info in enumerate(words):
            current_chunk_words.append(word_info)
            
            # Check if adding this word exceeds limits relative to chunk start
            chunk_text = "".join([w["word"] for w in current_chunk_words]).strip()
            chunk_duration = word_info["end"] - current_chunk_start
            
            # Look ahead to see if next word would break the limit
            next_word_breaks = False
            if i + 1 < len(words):
                next_word = words[i+1]
                next_text_len = len(chunk_text) + len(next_word["word"])
                next_duration = next_word["end"] - current_chunk_start
                if next_text_len > max_chars or next_duration > max_duration:
                    next_word_breaks = True
            
            # If we need to split here (either current is long enough, or next breaks it)
            # But ensure we have at least something in the chunk
            if next_word_breaks or i == len(words) - 1:
                split_segments.append({
                    "text": chunk_text,
                    "start": current_chunk_start,
                    "end": word_info["end"],
                    "words": current_chunk_words
                })
                # Reset for next chunk
                if i + 1 < len(words):
                    current_chunk_start = words[i+1]["start"]
                    current_chunk_words = []
                    
    return split_segments


def fill_transcription_gaps(transcribed_segments: list[dict], gap_threshold: float = 5.0, placeholder: str = "[no speech]") -> list[dict]:
    """
    Identifies and fills significant time gaps in a transcription with placeholder text.

    This function iterates through the transcribed segments and checks the time difference
    between the end of one segment and the start of the next. If the gap exceeds the
    specified threshold, a new placeholder segment is inserted.

    Args:
        transcribed_segments: The list of transcription segments from Whisper.
        gap_threshold: The minimum duration (in seconds) of a gap to be filled.
        placeholder: The text to insert for the gap.

    Returns:
        A new list of segments with gaps filled.
    """
    if not transcribed_segments:
        return []

    filled_segments = [transcribed_segments[0]]
    for i in range(len(transcribed_segments) - 1):
        current_seg = transcribed_segments[i]
        next_seg = transcribed_segments[i + 1]

        gap = next_seg["start"] - current_seg["end"]

        if gap > gap_threshold:
            logging.warning(f"Gap of {gap:.2f}s detected. Inserting placeholder.")
            filled_segments.append({
                "text": placeholder,
                "start": current_seg["end"],
                "end": next_seg["start"]
            })
        
        filled_segments.append(next_seg)
    
    return filled_segments


def adjust_subtitle_timing(segments: list[dict], time_buffer: float) -> list[dict]:
    """
    Adjusts subtitle timings to fill gaps and ensures a consistent reading pace.

    This function extends the duration of each subtitle to meet the start of the next one,
    minus a small buffer. It helps prevent subtitles from flashing on the screen too quickly.

    Args:
        segments: A list of subtitle segments (can be transcribed or translated).
        time_buffer: The buffer time (in seconds) to maintain between subtitles.

    Returns:
        The adjusted list of segments.
    """
    time_buffer = max(0, time_buffer)

    if not segments:
        return []

    adjusted_segments = [seg.copy() for seg in segments]

    for i in range(len(adjusted_segments) - 1):
        current_segment = adjusted_segments[i]
        next_segment = adjusted_segments[i + 1]

        # The ideal end time for the current segment is the start of the next one minus the buffer.
        new_end_time = next_segment['start'] - time_buffer

        # Update the end time. This extends shorter segments and shortens longer ones.
        current_segment['end'] = new_end_time

        # Ensure that the new end time does not precede the start time.
        if current_segment['end'] < current_segment['start']:
            # This can happen if the gap between segments is smaller than the time_buffer.
            # To avoid a negative or zero duration, we set the end time to be same as the start time,
            # which will make the subtitle appear as a flash. This is a safe fallback.
            current_segment['end'] = current_segment['start']

    # The last segment's end time is not modified as there's no next segment to overlap with.

    return adjusted_segments


def generate_srt(segments: list[dict], output_path: str):
    """
    Generates an SRT subtitle file from translated segments.

    Args:
        segments: A list of segments with text, start, and end times.
        output_path: The path to save the .srt file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            start_time = seg["start"]
            end_time = seg["end"]
            text = seg["text"]

            # SRT time format: HH:MM:SS,ms
            start_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}"
            end_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}"

            f.write(f"{i + 1}\n")
            f.write(f"{start_srt} --> {end_srt}\n")
            f.write(f"{text}\n\n")
    print(f"SRT file generated at {output_path}")
