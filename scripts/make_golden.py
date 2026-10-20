"""Create (or check) the golden reference outputs from plain, unchanged Transformers.

    python scripts/make_golden.py --dtype fp32            # write tests/golden/<model>/fp32.json
    python scripts/make_golden.py --dtype fp32 --check    # run again, compare with the saved file
    python scripts/make_golden.py --dtype bf16 --platform linux-aarch64   # tests/golden/<model>/linux-aarch64/

Different maths libraries (macOS vs Linux, ARM vs x86) round slightly differently, and in bf16 that
changes some tokens even for plain Transformers. So each platform where we compare gets its own
files; the top-level files are the macOS (Apple Silicon) reference.

Every later phase must produce the same tokens as these files (docs/IMPLEMENTATION.md, Phase 0).
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from transformers.utils import logging as hf_logging

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
hf_logging.disable_progress_bar()

from moe_dist.models.catalog import DEFAULT_MODEL, pinned_revision, short_name  # noqa: E402
from moe_dist.reference import environment_info, greedy_generate, load_reference  # noqa: E402

PROMPTS_FILE = ROOT / "scripts" / "prompts.json"
GOLDEN_DIR = ROOT / "tests" / "golden"


def golden_path(model_id: str, dtype: str, platform: str = "") -> Path:
    folder = GOLDEN_DIR / short_name(model_id)
    return (folder / platform if platform else folder) / f"{dtype}.json"


def compare(saved: dict, fresh: dict) -> list[str]:
    """Return a list of differences between two golden files (empty = identical)."""
    problems = []
    for key in ("model_id", "revision", "dtype", "max_new_tokens"):
        if saved[key] != fresh[key]:
            problems.append(f"{key}: saved {saved[key]!r}, now {fresh[key]!r}")
    saved_prompts = {p["id"]: p for p in saved["prompts"]}
    for new in fresh["prompts"]:
        old = saved_prompts.get(new["id"])
        if old is None:
            problems.append(f"{new['id']}: not in saved file")
            continue
        if old["prompt_tokens"] != new["prompt_tokens"]:
            problems.append(f"{new['id']}: prompt tokens differ (chat template changed?)")
            continue
        if old["output_tokens"] != new["output_tokens"]:
            pos = next(
                (i for i, (a, b) in enumerate(zip(old["output_tokens"], new["output_tokens"])) if a != b),
                min(len(old["output_tokens"]), len(new["output_tokens"])),
            )
            gap = old["steps"][pos]["gap"] if pos < len(old["steps"]) else float("nan")
            problems.append(f"{new['id']}: tokens differ from position {pos} (reference top-2 gap there: {gap:.4f})")
        diff = max(abs(a - b) for a, b in zip(old["first_step_top"]["logits"], new["first_step_top"]["logits"]))
        if old["first_step_top"]["ids"] != new["first_step_top"]["ids"] or diff > 0:
            problems.append(f"{new['id']}: first-step scores differ (max diff {diff:.2e})")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--check", action="store_true", help="compare with the saved file instead of writing it")
    parser.add_argument("--platform", default="", help="sub-folder for another platform, e.g. linux-aarch64")
    args = parser.parse_args()

    revision = pinned_revision(args.model)
    prompts = json.loads(PROMPTS_FILE.read_text())

    print(f"loading {args.model} @ {revision[:8]} in {args.dtype} ...")
    t0 = time.perf_counter()
    model, tokenizer = load_reference(args.model, revision, args.dtype)
    print(f"loaded in {time.perf_counter() - t0:.1f} s")

    results = []
    total_tokens, total_time = 0, 0.0
    for p in prompts:
        t0 = time.perf_counter()
        r = greedy_generate(model, tokenizer, p["messages"], args.max_new_tokens)
        dt = time.perf_counter() - t0
        total_tokens += len(r["output_tokens"])
        total_time += dt
        n = len(r["output_tokens"])
        print(f"  {p['id']:<12} prompt={len(r['prompt_tokens']):>3} new={n:>3} {n / dt:5.1f} tok/s  {r['output_text'][:60]!r}")
        results.append({"id": p["id"], "messages": p["messages"], **r})

    golden = {
        "model_id": args.model,
        "revision": revision,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment_info(),
        "speed_tok_per_s": round(total_tokens / total_time, 2),
        "prompts": results,
    }
    print(f"average speed: {golden['speed_tok_per_s']} tok/s (including prompt reading)")

    path = golden_path(args.model, args.dtype, args.platform)
    if args.check:
        problems = compare(json.loads(path.read_text()), golden)
        if problems:
            print(f"MISMATCH vs {path.relative_to(ROOT)}:")
            for line in problems:
                print("  -", line)
            sys.exit(1)
        print(f"OK: identical to {path.relative_to(ROOT)}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(golden, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
