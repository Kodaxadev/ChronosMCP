# chronos/uuid7.py
# Responsibility: RFC 9562 compliant UUIDv7 generation.
# Single function, zero dependencies beyond stdlib.

import time
import secrets


def uuid7() -> str:
    """
    RFC 9562 compliant UUIDv7.
    Uses time_ns() to avoid float precision loss vs time.time() (§2.3).
    Random fields filled with CSPRNG (secrets module).
    """
    ts_ms         = time.time_ns() // 1_000_000        # integer milliseconds
    rand_a        = secrets.randbits(12)
    rand_b        = secrets.randbits(62)
    time_hi       = (ts_ms >> 16) & 0xFFFFFFFF
    time_mid      = ts_ms & 0xFFFF
    ver_rand_a    = 0x7000 | (rand_a & 0x0FFF)
    var_rand_b_hi = 0x8000 | ((rand_b >> 48) & 0x3FFF)
    rand_b_lo     = rand_b & 0xFFFFFFFFFFFF
    return (
        f"{time_hi:08x}-{time_mid:04x}-"
        f"{ver_rand_a:04x}-{var_rand_b_hi:04x}-{rand_b_lo:012x}"
    )
