"""Speed check: fixed prompts -> tokens per second, plain Transformers vs our dispatcher.

    .venv/bin/python scripts/benchmark.py --dtype bf16
    .venv/bin/python scripts/benchmark.py --dtype fp32 --modes local --combine fast

"plain" is unchanged Transformers. "local" is the model with dispatchers and virtual nodes in one
program (Phase 1.3): every expert call is packed into a binary message and unpacked again, but
nothing goes over a network yet. The difference shows what our own code costs.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from moe_dist.experts.dispatcher import request_scope  # noqa: E402
from moe_dist.master.model_loader import build_local_cluster  # noqa: E402
from moe_dist.models.catalog import DEFAULT_MODEL, pinned_revision  # noqa: E402
from moe_dist.reference import environment_info, greedy_generate, load_reference  # noqa: E402

PLACEMENT = {"master": "0-9", "worker-1": "10-20", "worker-2": "21-31"}


def run(model, tokenizer, prompts, max_new_tokens):
    greedy_generate(model, tokenizer, prompts[0]["messages"], 4)  # warm-up (first calls are slow)
    tokens, start = 0, time.perf_counter()
    for p in prompts:
        with request_scope(p["id"]):
            tokens += len(greedy_generate(model, tokenizer, p["messages"], max_new_tokens)["output_tokens"])
    return tokens, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--modes", nargs="+", default=["plain", "local"], choices=["plain", "local"])
    parser.add_argument("--combine", default="exact", choices=["exact", "fast"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    prompts = json.loads((ROOT / "scripts" / "prompts.json").read_text())
    revision = pinned_revision(DEFAULT_MODEL)
    print(json.dumps(environment_info()))
    for mode in args.modes:
        if mode == "plain":
            model, tokenizer = load_reference(DEFAULT_MODEL, revision, args.dtype)
            dispatchers = {}
        else:
            model, tokenizer, _, dispatchers = build_local_cluster(DEFAULT_MODEL, revision, args.dtype, PLACEMENT, args.combine)
        tokens, seconds = run(model, tokenizer, prompts, args.max_new_tokens)
        line = f"{mode:5s} {args.dtype} {'' if mode == 'plain' else args.combine:5s}: {tokens} tokens in {seconds:.1f} s = {tokens / seconds:.1f} tok/s"
        if dispatchers:
            sent = sum(d.stats["bytes_sent"] for d in dispatchers.values())
            received = sum(d.stats["bytes_received"] for d in dispatchers.values())
            calls = sum(d.stats["calls"] for d in dispatchers.values())
            line += f" | {calls} messages, {sent / 2**20:.1f} MB sent, {received / 2**20:.1f} MB received (incl. warm-up)"
        print(line, flush=True)
        del model, dispatchers
        gc.collect()


if __name__ == "__main__":
    main()
