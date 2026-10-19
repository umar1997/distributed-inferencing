"""Phase 0: the golden reference outputs are complete and plain Transformers reproduces them."""

import json
from pathlib import Path

import pytest

from moe_dist.models.catalog import DEFAULT_MODEL, pinned_revision, short_name
from moe_dist.reference import greedy_generate, load_reference

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = json.loads((ROOT / "scripts" / "prompts.json").read_text())
GOLDEN_DIR = ROOT / "tests" / "golden" / short_name(DEFAULT_MODEL)


def load_golden(dtype: str) -> dict:
    path = GOLDEN_DIR / f"{dtype}.json"
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} missing; run scripts/make_golden.py --dtype {dtype}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_golden_file_is_complete(dtype):
    golden = load_golden(dtype)
    assert golden["model_id"] == DEFAULT_MODEL
    assert golden["revision"] == pinned_revision(DEFAULT_MODEL)
    assert [p["id"] for p in golden["prompts"]] == [p["id"] for p in PROMPTS]
    for p in golden["prompts"]:
        assert p["output_tokens"], f"{p['id']}: no tokens generated"
        assert len(p["steps"]) == len(p["output_tokens"])
        # Greedy decoding: every generated token is the step's top-1 choice.
        assert [s["top1"] for s in p["steps"]] == p["output_tokens"]


@pytest.mark.model
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_reference_reproduces_golden(dtype):
    golden = load_golden(dtype)
    model, tokenizer = load_reference(golden["model_id"], golden["revision"], dtype)
    for saved in golden["prompts"][:3]:
        fresh = greedy_generate(model, tokenizer, saved["messages"], golden["max_new_tokens"])
        assert fresh["prompt_tokens"] == saved["prompt_tokens"], f"{saved['id']}: chat template changed"
        assert fresh["output_tokens"] == saved["output_tokens"], f"{saved['id']}: tokens differ"
