import re
from typing import List

from pydantic import BaseModel


_SRT_TS_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})$")
_SRT_TS_SHORT_RE = re.compile(r"^(\d{1,2}):(\d{2})[,:](\d{3})$")


def seconds_to_srt(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt_to_seconds(ts: str) -> float:
    ts = ts.strip()
    m = _SRT_TS_RE.match(ts)
    if m:
        h, mi, s, ms = m.groups()
        return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000
    short = _SRT_TS_SHORT_RE.match(ts)
    if short:
        mi, s, ms = short.groups()
        return int(mi) * 60 + int(s) + int(ms) / 1000
    raise ValueError(f"Invalid SRT timestamp: {ts!r}")


class TranscriptSegment(BaseModel):
    start: str
    end: str
    text: str


class TranscriptResponse(BaseModel):
    segments: List[TranscriptSegment]


class TimingReviewSegment(BaseModel):
    id: int
    start: str
    end: str


class TimingReviewResponse(BaseModel):
    segments: List[TimingReviewSegment]
