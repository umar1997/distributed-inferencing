"""Phase 2.1: the Docker cluster (1 master + 2 workers) gives the same tokens as plain Transformers.

Runs against containers that are already up; skipped when no master answers:

    docker compose -f docker/docker-compose.yml up -d
    .venv/bin/python -m pytest -m cluster -s
    DTYPE=fp32 docker compose -f docker/docker-compose.yml up -d   # then run again for fp32

The containers run Linux, whose maths libraries round differently from macOS. In bf16 that alone
changes tokens (Phase 2.1 finding), so the reference is plain Transformers run on the same
platform: tests/golden/<model>/<platform>/<dtype>.json (scripts/make_golden.py --platform ...).
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from moe_dist.models.catalog import short_name
from moe_dist.reference import match_reference

MASTER_URL = os.environ.get("MASTER_URL", "http://localhost:8000")
GOLDEN_ROOT = Path(__file__).resolve().parent / "golden"

pytestmark = pytest.mark.cluster


@pytest.fixture(scope="module")
def master():
    try:
        health = httpx.get(f"{MASTER_URL}/health", timeout=3).json()
    except httpx.HTTPError:
        pytest.skip(f"no master at {MASTER_URL} (start: docker compose -f docker/docker-compose.yml up -d)")
    deadline = time.monotonic() + 600
    while not health["ready"]:
        assert health["status"] != "error", health["error"]
        assert time.monotonic() < deadline, "master not ready after 10 minutes"
        time.sleep(2)
        health = httpx.get(f"{MASTER_URL}/health", timeout=3).json()
    return health


def golden_for(health):
    path = GOLDEN_ROOT / short_name(health["model_id"]) / health["platform"] / f"{health['dtype']}.json"
    if not path.exists():
        pytest.skip(f"no golden file for this platform: {path} (make it with scripts/make_golden.py --platform)")
    golden = json.loads(path.read_text())
    assert golden["revision"] == health["model_revision"]
    return golden


def test_cluster_gives_identical_tokens(master):
    golden = golden_for(master)
    assert master["nodes"] == ["master", "worker-1", "worker-2"]
    speeds = []
    with httpx.Client(base_url=MASTER_URL, timeout=600) as http:
        for saved in golden["prompts"]:
            result = http.post("/internal/generate", json={"messages": saved["messages"],
                                                           "max_new_tokens": golden["max_new_tokens"]}).json()
            assert result["prompt_tokens"] == saved["prompt_tokens"]
            verdict, at = match_reference(saved, result["output_tokens"], max_gap=0)
            assert verdict == "identical", f"{saved['id']}: first difference at token {at}"
            speeds.append(result["tokens_per_s"])
        after = http.get("/health").json()
    print(f"\n{master['dtype']} on {master['platform']}: all {len(speeds)} prompts identical; "
          f"tok/s per prompt {speeds}; calls per node {after['calls_per_node']}")
    assert all(after["calls_per_node"][n] > 0 for n in ("worker-1", "worker-2"))


def container_memory_mb() -> dict[str, float]:
    if not shutil.which("docker"):
        pytest.skip("docker command not available")
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}}"],
                         capture_output=True, text=True, check=True).stdout
    units = {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}
    memory = {}
    for line in out.splitlines():
        name, used = line.split()[0], line.split()[1]
        for unit, factor in units.items():
            if used.endswith(unit):
                memory[name] = float(used[: -len(unit)]) * factor
    return memory


def memory_breakdown_mb(container: str) -> dict[str, float]:
    """Program memory ("anon") vs. pages of files kept in memory ("file", e.g. the weight file)."""
    out = subprocess.run(["docker", "exec", container, "cat", "/sys/fs/cgroup/memory.stat"],
                         capture_output=True, text=True, check=True).stdout
    stat = dict(line.split() for line in out.splitlines())
    return {"program": int(stat["anon"]) / 2**20, "file": int(stat["file"]) / 2**20}


def test_memory_per_container(master):
    """Each worker holds about its share of the experts, the master about shared parts + own experts."""
    memory = container_memory_mb()
    per_expert_mb = {"bf16": 72, "fp32": 144}[master["dtype"]]  # 24 layers x one expert (Granite)
    shared_mb = {"bf16": 242, "fp32": 483}[master["dtype"]]
    full_mb = shared_mb + 32 * per_expert_mb
    print(f"\nmemory ({master['dtype']}, full model would be {full_mb} MB of weights):")
    for name in ("moe-master-1", "moe-worker-1-1", "moe-worker-2-1"):
        parts = memory_breakdown_mb(name)
        print(f"  {name}: {memory[name]:.0f} MB = program {parts['program']:.0f} MB + file pages {parts['file']:.0f} MB")
    # Runtime overhead (Python, PyTorch, web server) measured in Phase 2.1: ~350 MB per worker;
    # the master also has the tokenizer, Transformers' model objects and generation buffers (~800 MB).
    for name, experts in (("moe-worker-1-1", 11), ("moe-worker-2-1", 11)):
        assert memory[name] < experts * per_expert_mb + 600, name
    assert memory["moe-master-1"] < shared_mb + 10 * per_expert_mb + 900
    assert memory["moe-master-1"] < full_mb  # the master never holds the full model


def test_no_container_ran_out_of_memory(master):
    """Phase 2.4: with limits (docker-compose.8gb.yml), every container stays below its limit.

    Without limits (the normal compose file) this only checks that nothing was killed.
    """
    for name in ("moe-master-1", "moe-worker-1-1", "moe-worker-2-1"):
        read = lambda f: subprocess.run(["docker", "exec", name, "cat", f"/sys/fs/cgroup/{f}"],  # noqa: E731
                                        capture_output=True, text=True, check=True).stdout.strip()
        events = dict(line.split() for line in read("memory.events").splitlines())
        state = json.loads(subprocess.run(["docker", "inspect", name], capture_output=True, text=True,
                                          check=True).stdout)[0]["State"]
        assert events["oom_kill"] == "0" and not state["OOMKilled"], f"{name} ran out of memory"
        if read("memory.max") != "max":
            peak, limit = int(read("memory.peak")), int(read("memory.max"))
            print(f"\n{name}: peak {peak / 2**20:.0f} MB of {limit / 2**20:.0f} MB")
            assert peak < limit, name
