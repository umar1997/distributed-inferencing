"""Pack tensors into one binary message and unpack them again (docs/IMPLEMENTATION.md, 4.2).

Message layout:

    [4 bytes: header length N][N bytes: JSON header][tensor 1 bytes][tensor 2 bytes]...

The JSON header holds any small fields (request ID, layer, ...) plus a "tensors" list with each
tensor's name, dtype, shape and byte size, in the order the bytes follow. The numbers are sent as
raw bytes ("little-endian", the byte order every machine we use has), so nothing is rounded:
unpacking gives back exactly the same bits. The header is plain JSON, so it can be logged as-is.
"""

import json
import struct
import sys

import torch

CONTENT_TYPE = "application/x-moe-tensors"

DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
NAMES = {dtype: name for name, dtype in DTYPES.items()}

_LENGTH = struct.Struct("<I")  # 4-byte unsigned length, little-endian
MAX_HEADER_BYTES = 1 << 20

if sys.byteorder != "little":  # pragma: no cover - every CPU we target is little-endian
    raise RuntimeError("the binary message format assumes a little-endian machine")


class CodecError(ValueError):
    """A message that cannot be unpacked: wrong lengths, damaged header, unknown dtype."""


def encode(header: dict, tensors: dict[str, torch.Tensor]) -> bytes:
    """Small fields + named tensors -> one bytes message."""
    if "tensors" in header:
        raise CodecError("'tensors' is reserved in the header")
    entries, blobs = [], []
    for name, t in tensors.items():
        if t.dtype not in NAMES:
            raise CodecError(f"tensor {name!r}: dtype {t.dtype} is not supported")
        raw = t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        entries.append({"name": name, "dtype": NAMES[t.dtype], "shape": list(t.shape), "nbytes": len(raw)})
        blobs.append(raw)
    head = json.dumps({**header, "tensors": entries}, separators=(",", ":")).encode()
    return b"".join([_LENGTH.pack(len(head)), head, *blobs])


def decode(message: bytes) -> tuple[dict, dict[str, torch.Tensor]]:
    """One bytes message -> (header without 'tensors', {name: tensor})."""
    view = memoryview(message)
    if len(view) < _LENGTH.size:
        raise CodecError(f"message too short ({len(view)} bytes) to hold the header length")
    (head_len,) = _LENGTH.unpack_from(view)
    if head_len > MAX_HEADER_BYTES or _LENGTH.size + head_len > len(view):
        raise CodecError(f"header length {head_len} does not fit in a {len(view)}-byte message")
    try:
        header = json.loads(bytes(view[_LENGTH.size : _LENGTH.size + head_len]))
        entries = header.pop("tensors")
    except (ValueError, KeyError, AttributeError, TypeError) as exc:
        raise CodecError(f"damaged header: {exc}") from None

    tensors, pos = {}, _LENGTH.size + head_len
    for entry in entries:
        try:
            name, dtype_name, shape, nbytes = entry["name"], entry["dtype"], entry["shape"], entry["nbytes"]
        except (KeyError, TypeError):
            raise CodecError(f"damaged tensor entry in header: {entry!r}") from None
        if dtype_name not in DTYPES:
            raise CodecError(f"tensor {name!r}: unknown dtype {dtype_name!r}")
        dtype = DTYPES[dtype_name]
        count = 1
        for size in shape:
            count *= size
        if count * dtype.itemsize != nbytes:
            raise CodecError(f"tensor {name!r}: shape {shape} of {dtype_name} needs {count * dtype.itemsize} bytes, header says {nbytes}")
        if pos + nbytes > len(view):
            raise CodecError(f"tensor {name!r}: message ends early ({len(view) - pos} of {nbytes} bytes left)")
        raw = bytearray(view[pos : pos + nbytes])  # own copy, writable
        tensors[name] = torch.frombuffer(raw, dtype=torch.uint8).view(dtype).reshape(shape) if nbytes else torch.empty(shape, dtype=dtype)
        pos += nbytes
    if pos != len(view):
        raise CodecError(f"{len(view) - pos} unexpected bytes after the last tensor")
    return header, tensors
