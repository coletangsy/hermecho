# Hermecho

Hermecho turns source-language speech into time-aligned Traditional Chinese subtitles for delivery with video.

## Language

**Source Word**:
A recognized source-language word with its time span. It is immutable transcript evidence and is not overwritten by translation or delivery processing.
_Avoid_: ASR token, timed segment

**Source Sentence**:
A continuous, ordered span of Source Words that expresses one complete source-language meaning. It owns semantic grouping, not subtitle layout.
_Avoid_: Source segment, source cue

**Translation Sentence**:
A complete translation of one Source Sentence before being shaped for display. It retains linguistic punctuation and does not own delivery timing or line breaks.
_Avoid_: Translated segment, subtitle sentence

**Delivery Cue**:
A subtitle unit shown during a bounded time interval under layout and reading constraints. It is derived from one Translation Sentence and Source Word timing evidence.
_Avoid_: Segment, translation segment

**Delivery Profile**:
The layout and reading constraints used to derive Delivery Cues for a viewing format such as portrait or landscape. It does not alter the accepted Translation Sentence.
_Avoid_: Subtitle text, translation style

**Visual Cell**:
A relative subtitle-width unit used by a Delivery Profile. Traditional Chinese characters and full-width punctuation occupy one cell; half-width Latin characters, digits, and spaces occupy half a cell.
_Avoid_: Character count, byte length

**Translation Gate**:
The deterministic acceptance boundary between model output and accepted translation sentences. It rejects incomplete or malformed output and violations of locked terms; it does not judge naturalness.
_Avoid_: Translation quality score, model review

**Locked Term**:
A curated source expression whose required target rendering is authoritative whenever that source expression occurs. General reference context and translation suggestions are not locked terms.
_Avoid_: Prompt hint, glossary suggestion

**Fit Repair**:
A targeted revision of a Translation Sentence that cannot satisfy its Delivery Profile. It may make the sentence more concise while preserving meaning and Locked Terms, and must pass the Translation Gate again.
_Avoid_: Alignment, truncation

**Alignment**:
The mapping of unchanged Translation Sentence pieces onto continuous Source Word spans. It may split accepted text but cannot rewrite it or invent timing.
_Avoid_: Retranslation, proportional timing

**Delivery Gate**:
The deterministic acceptance boundary for Delivery Cues under their Delivery Profile and timing rules. It does not decide whether a translation is complete or natural.
_Avoid_: Translation Gate, human review

**Warning**:
A Delivery Gate finding that is recorded for review but neither triggers repair nor blocks delivery.
_Avoid_: Repair Limit, Structural Defect

**Repair Limit**:
A Delivery Profile limit whose violation triggers Fit Repair or Alignment. An unresolved violation may still be exported through Best-effort Delivery.
_Avoid_: Warning, Structural Defect

**Structural Defect**:
An invalid timing or mapping state from which a trustworthy Delivery Cue cannot be produced. It blocks final delivery.
_Avoid_: Warning, Repair Limit

**Best-effort Delivery**:
A policy that permits unresolved presentation defects to be exported with explicit diagnostics. It never bypasses the Translation Gate or structural timing validity.
_Avoid_: Partial translation, silent failure

## Review

**Comparison Run**:
Paired Baseline and Candidate executions of the same source and configuration with exactly one declared changed variable.
_Avoid_: Uncontrolled benchmark, casual comparison

**Review Composite**:
A synchronized video presenting Baseline and Candidate delivery results for direct human comparison. It is review evidence, not an approval by itself.
_Avoid_: Demo video, benchmark report

**Human Approval**:
An explicit decision that a Candidate introduces no blocking regression under the agreed review checklist.
_Avoid_: Looks okay, smoke test
