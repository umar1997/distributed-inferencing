"""The models this project supports, pinned to exact Hugging Face commits.

Every node must use the same model ID *and* revision. Details: docs/AVAILABLE_MODELS.md
"""

MODELS = {
    "ibm-granite/granite-3.0-1b-a400m-instruct": "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445",
    "Qwen/Qwen1.5-MoE-A2.7B-Chat": "ec052fda178e241c7c443468d2fa1db6618996be",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
}

DEFAULT_MODEL = "ibm-granite/granite-3.0-1b-a400m-instruct"


def pinned_revision(model_id: str) -> str:
    if model_id not in MODELS:
        raise ValueError(f"unknown model {model_id!r}; supported: {', '.join(MODELS)}")
    return MODELS[model_id]


def short_name(model_id: str) -> str:
    """'ibm-granite/granite-3.0-1b-a400m-instruct' -> 'granite-3.0-1b-a400m-instruct' (used for folder names)."""
    return model_id.split("/")[-1]
