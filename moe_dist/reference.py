"""Plain, unchanged Transformers: the reference every later step is compared against.

`greedy_generate` is also what the distributed engine's output gets compared with, so the
prompt formatting (chat template) and decoding settings live here, in one place.
"""

import platform

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

# How many of the highest scores to keep from the first generated step (for later comparisons).
TOP_LOGITS = 20


def load_reference(model_id: str, revision: str, dtype: str):
    """Load the full model with Transformers' simple "eager" expert loop (no special kernels)."""
    # Granite's config asks for this clean-up, but Transformers ignores it for this tokenizer
    # type anyway (and warns). Turning it off changes nothing in the output.
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, clean_up_tokenization_spaces=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=DTYPES[dtype],
        experts_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def build_inputs(tokenizer, messages: list[dict]) -> dict:
    """Chat messages -> model input, with the model's own chat template."""
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


@torch.inference_mode()
def greedy_generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> dict:
    """Greedy decoding (always the most likely token). Returns tokens, text and per-step scores."""
    inputs = build_inputs(tokenizer, messages)
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        return_dict_in_generate=True,
        output_logits=True,
    )
    new_tokens = out.sequences[0, prompt_len:].tolist()

    # For every generated token: the winner, the runner-up, and how close they were.
    # A small gap means a near-tie, where tiny rounding differences can flip the choice.
    # In bf16 exact ties (gap 0) do happen; greedy decoding then takes the lowest token ID,
    # which is what argmax does (topk may pick either), so argmax is used here.
    steps = []
    for step_logits in out.logits:
        scores = step_logits[0].float()
        top1 = scores.argmax().item()
        best = scores[top1].item()
        scores[top1] = float("-inf")
        top2 = scores.argmax().item()
        steps.append({"top1": top1, "top2": top2, "gap": best - scores[top2].item()})

    first = out.logits[0][0].float().topk(TOP_LOGITS)
    return {
        "prompt_tokens": inputs["input_ids"][0].tolist(),
        "output_tokens": new_tokens,
        "output_text": tokenizer.decode(new_tokens, skip_special_tokens=True),
        "steps": steps,
        "first_step_top": {"ids": first.indices.tolist(), "logits": first.values.tolist()},
    }


# Matching rule (docs/IMPLEMENTATION.md, Phase 1.3): when tiny number differences are allowed,
# the first different token must be at a near-tie, where the reference's two best scores are
# this close. On one machine in exact mode no difference is allowed at all.
NEAR_TIE = {"fp32": 0.01, "bf16": 0.1}


def match_reference(saved: dict, tokens: list[int], max_gap: float) -> tuple[str, int | None]:
    """Compare generated tokens with a golden prompt entry.

    Returns ("identical", None), ("near-tie", i) or ("different", i), where i is the first
    position that differs. "near-tie" means the reference's top-2 gap at i was <= max_gap, so a
    tiny rounding difference may honestly have flipped the choice.
    """
    expected = saved["output_tokens"]
    for i, (want, got) in enumerate(zip(expected, tokens)):
        if want != got:
            return ("near-tie" if saved["steps"][i]["gap"] <= max_gap else "different"), i
    if len(expected) != len(tokens):  # one stopped early (end-of-text token) where the other did not
        return "different", min(len(expected), len(tokens))
    return "identical", None


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "machine": platform.machine(),
        "system": platform.system(),
        "torch_threads": torch.get_num_threads(),
    }
