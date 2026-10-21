"""Phase 2.4: does the cluster run inside a small laptop's memory? Run against the containers:

    docker compose -f docker/docker-compose.yml -f docker/docker-compose.8gb.yml up -d
    .venv/bin/python scripts/memory_check.py

What it does:
1. all golden prompts through the OpenAI-style API (temperature 0) -> the text must be identical,
2. one long conversation, several turns, until the context is over 1,000 tokens (the KV cache, the
   attention memory of earlier tokens, grows with the context),
3. reads from each container's Linux "cgroup" (the kernel's bookkeeping per container): its limit,
   its highest memory use since it started (this includes loading), and how often it ran out.

Ends with exit code 1 if any text differs or any container ran out of memory ("OOMKilled").
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

MASTER_URL = "http://localhost:8000"
CONTAINERS = ["moe-master-1", "moe-worker-1-1", "moe-worker-2-1"]
GOLDEN_ROOT = Path(__file__).resolve().parent.parent / "tests" / "golden"


def cgroup(container: str) -> dict:
    """Memory facts of one container, in MB, from /sys/fs/cgroup inside it."""
    def read(name):
        return subprocess.run(["docker", "exec", container, "cat", f"/sys/fs/cgroup/{name}"],
                              capture_output=True, text=True, check=True).stdout.strip()

    stat = dict(line.split() for line in read("memory.stat").splitlines())
    events = dict(line.split() for line in read("memory.events").splitlines())
    limit = read("memory.max")
    inspect = json.loads(subprocess.run(["docker", "inspect", container], capture_output=True, text=True,
                                        check=True).stdout)[0]
    mb = 2**20
    return {
        "limit": None if limit == "max" else int(limit) / mb,
        "now": int(read("memory.current")) / mb,
        "peak": int(read("memory.peak")) / mb,
        "program": int(stat["anon"]) / mb,  # the processes' own memory (weights, buffers, Python)
        "file": int(stat["file"]) / mb,  # parts of files kept in memory; Linux can drop these when short
        "oom_kills": int(events.get("oom_kill", 0)),
        "hit_limit": int(events.get("max", 0)),  # times memory use reached the limit (Linux then frees file pages)
        "oom_killed": inspect["State"]["OOMKilled"],
        "restarts": inspect["RestartCount"],
    }


def chat(http, messages, max_tokens):
    reply = http.post("/v1/chat/completions", json={"messages": messages, "temperature": 0, "max_tokens": max_tokens})
    reply.raise_for_status()
    return reply.json()


def long_notes() -> str:
    """About 700 tokens of made-up notes to talk about (fixed, so every run is the same)."""
    towns = ["Arden", "Bexley", "Corran", "Dunmore", "Elstow", "Farley", "Garth", "Hollin", "Ivers", "Jessop"]
    lines = []
    for i, town in enumerate(towns):
        lines.append(
            f"{town} has {1200 + 317 * i} inhabitants, a market every {['Monday', 'Tuesday', 'Friday'][i % 3]}, "
            f"{2 + i % 4} schools and a lighthouse built in {1820 + 13 * i}. The keeper's logbook mentions "
            f"{3 + i} storms last winter, the worst in {['January', 'February', 'December'][i % 3]}, when the "
            f"harbour wall lost {4 + 2 * i} metres. The town council plans to repair it with {15 + 5 * i} thousand "
            f"euros from the regional fund, and the ferry to the next island runs {2 + i % 3} times a day.")
    return "\n".join(lines)


def main() -> int:
    http = httpx.Client(base_url=MASTER_URL, timeout=1800)
    health = http.get("/health").json()
    if not health["ready"]:
        print(f"the master is not ready: {health.get('missing') or health.get('error')}")
        return 1
    golden = json.loads((GOLDEN_ROOT / "granite-3.0-1b-a400m-instruct" / health["platform"] /
                         f"{health['dtype']}.json").read_text())
    print(f"{health['dtype']} on {health['platform']}, nodes {health['nodes']}")
    failures = []

    start = time.perf_counter()
    for saved in golden["prompts"]:
        text = chat(http, saved["messages"], golden["max_new_tokens"])["choices"][0]["message"]["content"]
        same = text == saved["output_text"]
        failures += [] if same else [f"golden prompt {saved['id']} differs"]
        print(f"  golden {saved['id']:24s} {'identical' if same else 'DIFFERENT'}")
    print(f"golden prompts done in {time.perf_counter() - start:.0f} s")

    start = time.perf_counter()
    messages = [{"role": "user", "content": "Here are my notes:\n" + long_notes() +
                 "\n\nWhich town has the oldest lighthouse, and when was it built?"}]
    follow_ups = ["Which town lost the most metres of harbour wall?", "How much money does that town plan to spend?",
                  "Summarise the notes in five sentences.", "Which towns have a market on Friday?"]
    context = 0
    for turn in range(1 + len(follow_ups)):
        answer = chat(http, messages, 150)
        context = answer["usage"]["total_tokens"]
        reply = answer["choices"][0]["message"]["content"]
        print(f"  turn {turn + 1}: context {context} tokens, answer {answer['usage']['completion_tokens']} tokens: "
              f"{reply[:70]!r}")
        if context > 1000 and turn >= 2:
            break
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": follow_ups[turn]}]
    print(f"long conversation done in {time.perf_counter() - start:.0f} s (context reached {context} tokens)")
    if context <= 1000:
        failures.append(f"the conversation only reached {context} tokens")

    print(f"\n{'container':16s} {'limit':>7s} {'peak':>7s} {'now':>7s} {'program':>8s} {'file':>6s}  out of memory?")
    for name in CONTAINERS:
        c = cgroup(name)
        limit = f"{c['limit']:.0f}" if c["limit"] else "none"
        ran_out = c["oom_kills"] or c["oom_killed"] or c["restarts"]
        print(f"{name:16s} {limit:>7s} {c['peak']:7.0f} {c['now']:7.0f} {c['program']:8.0f} {c['file']:6.0f}  "
              f"{'YES' if ran_out else 'no'} (reached the limit {c['hit_limit']} times)")
        if ran_out:
            failures.append(f"{name} ran out of memory")
    print("(MB; peak = highest use since the container started, including loading; program = the processes' own "
          "memory; file = file pages, which Linux can free when memory is short)")

    if failures:
        print("\nFAILED: " + "; ".join(failures))
        return 1
    print("\nOK: identical text, long conversation done, no container ran out of memory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
