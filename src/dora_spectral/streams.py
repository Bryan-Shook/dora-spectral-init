"""Stable, purpose-specific random seeds shared by paired experiment conditions."""

import hashlib

STREAM_VERSION = "dora-streams-v1"


def initialization_stream(seed_block: int, target_matrix: str) -> int:
    """Seed spectrum-matched random directions for one named projection."""
    return _stream_seed(seed_block, "init", target_matrix)


def reference_factor_seed(seed_block: int, target_matrix: str) -> int:
    """Seed the ordinary LoRA/DoRA reference factors for one named projection."""
    return _stream_seed(seed_block, "ref-factor", target_matrix)


def data_order_stream(seed_block: int) -> int:
    """Seed a local shuffle generator, advanced continuously across epochs."""
    return _stream_seed(seed_block, "data-order")


def _stream_seed(seed_block: int, stream: str, *parts: str) -> int:
    joined = "\n".join((STREAM_VERSION, str(seed_block), stream, *parts))
    hashed = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(hashed[:8], byteorder="little")
