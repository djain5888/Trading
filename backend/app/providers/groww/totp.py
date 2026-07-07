"""Time-based one-time password (TOTP) generation.

Implements RFC 6238 with the standard library so no OTP dependency is needed.
The seed is a base32-encoded shared secret supplied via configuration (never
hardcoded). Time is taken from the injected :class:`Clock`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import struct
from datetime import datetime

from app.providers.exceptions import AuthenticationError

_DEFAULT_DIGITS = 6
_DEFAULT_PERIOD = 30


def generate_totp(
    seed: str,
    *,
    now: datetime,
    digits: int = _DEFAULT_DIGITS,
    period: int = _DEFAULT_PERIOD,
) -> str:
    """Return the current TOTP code for a base32 seed.

    Args:
        seed: Base32-encoded shared secret.
        now: The reference time (timezone-aware).
        digits: Number of code digits.
        period: Time step in seconds.

    Returns:
        The zero-padded TOTP code.

    Raises:
        AuthenticationError: If the seed is not valid base32.
    """
    key = _decode_seed(seed)
    counter = int(now.timestamp()) // period
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    binary = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(binary % (10**digits)).zfill(digits)


def _decode_seed(seed: str) -> bytes:
    """Decode a base32 TOTP seed into raw bytes."""
    normalized = seed.strip().replace(" ", "").upper()
    padding = "=" * (-len(normalized) % 8)
    try:
        return base64.b32decode(normalized + padding)
    except (ValueError, TypeError) as exc:
        raise AuthenticationError("Invalid Groww TOTP seed (not base32).") from exc
