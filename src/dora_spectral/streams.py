import hashlib

_VERSION = "dora-streams-v1"

def initialization_stream(seed_block: int, target_matrix: str) -> int:
    return _stream_seed(seed_block, "init", target_matrix)

def reference_factor_seed(seed_block: int, target_matrix: str) -> int:
    return _stream_seed(seed_block, "ref-factor", target_matrix)

def data_order_stream(seed_block: int) -> int:
    return _stream_seed(seed_block, "data-order")

def _stream_seed(seed_block: int, stream: str, *parts: str) -> int:
    joined_args = _VERSION + "\n" + str(seed_block) + "\n" + stream
    for p in parts:
        joined_args = joined_args + "\n" + p
    hashed = hashlib.sha256(joined_args.encode("utf-8")).digest()
    return int.from_bytes(hashed[:8], byteorder="little")
