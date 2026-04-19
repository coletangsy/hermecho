"""
Centralized retry logic with exponential backoff and jitter.

All API call sites in Hermecho import from here so retry behaviour is
consistent and easy to tune in one place.
"""
from __future__ import annotations

import random
import time
from typing import Callable, Optional, Tuple, Type, TypeVar

T = TypeVar("T")

TRANSIENT_HTTP_CODES = frozenset({429, 502, 503, 504})



def compute_backoff(
    attempt: int,
    base_delay: float = 2.5,
    max_delay: float = 120.0,
    jitter: bool = True,
) -> float:
    """
    Return a delay in seconds for the given retry attempt (0-indexed).

    Uses exponential backoff: ``base_delay * 2^attempt``, capped at
    ``max_delay``.  When ``jitter`` is True, a uniform random fraction
    (±25 %) is added to spread thundering-herd retries.

    Args:
        attempt: Zero-based attempt index (first retry → attempt=1).
        base_delay: Starting delay in seconds.
        max_delay: Hard ceiling for any single delay.
        jitter: Whether to add randomised spread.

    Returns:
        Delay in seconds (≥ 0).
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        delay *= 1.0 + random.uniform(-0.25, 0.25)
    return max(0.0, delay)


def retry_on_transient(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 2.5,
    max_delay: float = 120.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    label: str = "operation",
) -> Optional[T]:
    """
    Call ``fn()`` up to ``max_attempts`` times, sleeping between failures.

    Retries on any exception in ``exceptions`` **or** when the callable
    returns ``None`` (treated as a soft failure signal).  Returns the
    first non-None result, or ``None`` if all attempts are exhausted.

    Example::

        result = retry_on_transient(
            lambda: fetch_something(),
            max_attempts=4,
            label="API call",
        )

    Args:
        fn: Zero-argument callable.  Returning ``None`` signals failure.
        max_attempts: Total attempts (including first).
        base_delay: Base backoff seconds.
        max_delay: Cap on per-sleep seconds.
        jitter: Randomise delay spread.
        exceptions: Exception types that trigger a retry.
        label: Human-readable name used in log lines.

    Returns:
        First successful (non-None) return value, or ``None``.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            result = fn()
            if result is not None:
                return result
            if attempt + 1 < max_attempts:
                delay = compute_backoff(attempt, base_delay, max_delay, jitter)
                print(
                    f"  {label}: attempt {attempt + 1}/{max_attempts} returned "
                    f"None; retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
        except exceptions as exc:
            last_exc = exc
            if attempt + 1 < max_attempts:
                delay = compute_backoff(attempt, base_delay, max_delay, jitter)
                print(
                    f"  {label}: attempt {attempt + 1}/{max_attempts} raised "
                    f"{type(exc).__name__}: {exc}; retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                print(
                    f"  {label}: all {max_attempts} attempt(s) failed. "
                    f"Last error: {exc}"
                )
    if last_exc is not None:
        print(f"  {label}: exhausted retries. Last exception: {last_exc}")
    return None


def sleep_for_status(
    status_code: int,
    attempt: int,
    *,
    base_delay: float = 2.5,
    max_delay: float = 120.0,
    jitter: bool = True,
    label: str = "HTTP request",
    max_attempts: int = 3,
) -> None:
    """
    Log and sleep when an HTTP call returns a transient error code.

    Args:
        status_code: HTTP status received.
        attempt: Zero-based attempt index (used to compute backoff).
        base_delay: Backoff base seconds.
        max_delay: Backoff ceiling.
        jitter: Spread randomisation.
        label: Logged prefix.
        max_attempts: Total attempts for log context.
    """
    delay = compute_backoff(attempt, base_delay, max_delay, jitter)
    print(
        f"  {label}: HTTP {status_code} (transient); "
        f"retrying in {delay:.1f}s "
        f"({attempt + 2}/{max_attempts})..."
    )
    time.sleep(delay)
