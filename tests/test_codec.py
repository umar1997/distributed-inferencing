"""Phase 1.3: the binary message format packs and unpacks tensors exactly."""

import json
import struct

import pytest
import torch

from moe_dist.transport.codec import DTYPES, CodecError, decode, encode


def sample(dtype, shape):
    g = torch.Generator().manual_seed(0)
    if dtype == torch.bool:
        return torch.rand(shape, generator=g) > 0.5
    if dtype.is_floating_point:
        return torch.randn(shape, generator=g).to(dtype)
    info = torch.iinfo(dtype)
    return torch.randint(max(info.min, -1000), min(info.max, 1000), shape, generator=g, dtype=dtype)


@pytest.mark.parametrize("dtype_name", DTYPES)
@pytest.mark.parametrize("shape", [(), (0,), (3,), (1, 1024), (5, 0, 2), (2, 3, 4)])
def test_every_dtype_and_shape_survives(dtype_name, shape):
    t = sample(DTYPES[dtype_name], shape)
    header, out = decode(encode({"layer": 6}, {"x": t}))
    assert header == {"layer": 6}
    assert out["x"].dtype == t.dtype and out["x"].shape == t.shape
    assert torch.equal(out["x"], t)


def test_special_float_values_keep_their_bits():
    t = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1e-45, 3.4e38])
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        x = t.to(dtype)
        _, out = decode(encode({}, {"x": x}))
        assert torch.equal(out["x"].view(torch.uint8), x.view(torch.uint8))  # bit for bit, NaN included


def test_several_tensors_keep_names_and_order():
    tensors = {
        "hidden": torch.randn(3, 8).to(torch.bfloat16),
        "row": torch.tensor([0, 2, 1], dtype=torch.int32),
        "expert": torch.tensor([3, 3, 7], dtype=torch.int32),
        "weight": torch.rand(3).to(torch.bfloat16),
    }
    header, out = decode(encode({"request_id": "a1b2", "token_index": 14}, tensors))
    assert header == {"request_id": "a1b2", "token_index": 14}
    assert list(out) == list(tensors)
    for name in tensors:
        assert torch.equal(out[name], tensors[name])


def test_non_contiguous_input_is_packed_correctly():
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4).T  # a transposed view
    _, out = decode(encode({}, {"t": t}))
    assert torch.equal(out["t"], t)


def test_unpacked_tensors_are_writable_and_independent():
    message = encode({}, {"x": torch.zeros(4)})
    _, out = decode(message)
    out["x"] += 1  # must not fail (read-only buffer) or change the message
    _, again = decode(message)
    assert torch.equal(again["x"], torch.zeros(4))


def test_header_is_readable_json():
    message = encode({"layer": 3}, {"hidden": torch.zeros(1, 4, dtype=torch.bfloat16)})
    (n,) = struct.unpack_from("<I", message)
    head = json.loads(message[4 : 4 + n])
    assert head == {"layer": 3, "tensors": [{"name": "hidden", "dtype": "bf16", "shape": [1, 4], "nbytes": 8}]}


def valid():
    return encode({"layer": 1}, {"a": torch.arange(4, dtype=torch.int32), "b": torch.ones(2)})


def with_header(head: dict, body: bytes = b"") -> bytes:
    raw = json.dumps(head).encode()
    return struct.pack("<I", len(raw)) + raw + body


@pytest.mark.parametrize(
    "message, error",
    [
        (b"", "too short"),
        (b"\x01\x00", "too short"),
        (struct.pack("<I", 999) + b"{}", "does not fit"),
        (struct.pack("<I", 5) + b"{not}", "damaged header"),
        (with_header({"layer": 1}), "damaged header"),  # no "tensors" list
        (with_header([1, 2]), "damaged header"),
        (with_header({"tensors": [{"name": "a"}]}), "damaged tensor entry"),
        (with_header({"tensors": [{"name": "a", "dtype": "fp64", "shape": [1], "nbytes": 8}]}, b"\0" * 8), "unknown dtype"),
        (with_header({"tensors": [{"name": "a", "dtype": "fp32", "shape": [2], "nbytes": 4}]}, b"\0" * 4), "needs 8 bytes"),
        (valid()[:-3], "ends early"),
        (valid() + b"\0", "unexpected bytes"),
    ],
)
def test_broken_messages_give_clear_errors(message, error):
    with pytest.raises(CodecError, match=error):
        decode(message)


def test_unsupported_dtype_and_reserved_name_are_refused():
    with pytest.raises(CodecError, match="not supported"):
        encode({}, {"x": torch.zeros(2, dtype=torch.float64)})
    with pytest.raises(CodecError, match="reserved"):
        encode({"tensors": 1}, {})
