"""Phase 2.2: failure tests against the running Docker containers (1 master + 2 workers).

    docker compose -f docker/docker-compose.yml up -d
    .venv/bin/python -m pytest -m cluster tests/test_cluster_failures.py -s

Shorter waits: start the containers with shorter health timings first, e.g.
    DEAD_AFTER_S=6 HEARTBEAT_INTERVAL_S=1 PING_INTERVAL_S=1 docker compose -f docker/docker-compose.yml up -d

The tests stop, pause and restart containers and slow down a worker's network with `tc netem`
("network emulator": a Linux tool that adds delay to a network card). Each test puts everything
back the way it was, and the next test waits until the cluster is healthy again.
"""

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest

from moe_dist.models.catalog import short_name
from moe_dist.reference import match_reference

MASTER_URL = os.environ.get("MASTER_URL", "http://localhost:8000")
MASTER, WORKER_1, WORKER_2 = "moe-master-1", "moe-worker-1-1", "moe-worker-2-1"
GOLDEN_ROOT = Path(__file__).resolve().parent / "golden"

pytestmark = pytest.mark.cluster


def docker(*args, check=True) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check).stdout


def health() -> dict:
    return httpx.get(f"{MASTER_URL}/health", timeout=3).json()


def statuses() -> dict:
    return {n["node_id"]: n["status"] for n in httpx.get(f"{MASTER_URL}/api/nodes", timeout=3).json()["nodes"]}


def wait_for(condition, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if condition():
                return
        except httpx.HTTPError:
            pass  # the master may be restarting
        time.sleep(0.5)
    raise TimeoutError(f"{what} not reached after {timeout:.0f} s")


def logs_since(container: str, since: str) -> str:
    return subprocess.run(["docker", "logs", "--since", since, container], capture_output=True, text=True).stdout


@pytest.fixture(scope="module")
def rules():
    if not shutil.which("docker"):
        pytest.skip("docker command not available")
    try:
        answer = health()
    except httpx.HTTPError:
        pytest.skip(f"no master at {MASTER_URL} (start: docker compose -f docker/docker-compose.yml up -d)")
    return answer["health_rules"]


@pytest.fixture(autouse=True)
def cluster_healthy(rules):
    """Before and after every test: all containers running, no network delay, every worker healthy."""
    def restore():
        for name in (MASTER, WORKER_1, WORKER_2):
            docker("unpause", name, check=False)
            docker("start", name, check=False)
        for name in (WORKER_1, WORKER_2):
            docker("exec", name, "tc", "qdisc", "del", "dev", "eth0", "root", check=False)
        wait_for(lambda: health()["ready"] and set(statuses().values()) == {"healthy"}, 300, "cluster healthy")

    restore()
    yield
    restore()


def golden_check(max_new_tokens=None):
    """Run the first golden prompt and check the tokens are identical."""
    h = health()
    golden = json.loads((GOLDEN_ROOT / short_name(h["model_id"]) / h["platform"] / f"{h['dtype']}.json").read_text())
    saved = golden["prompts"][0]
    reply = httpx.post(f"{MASTER_URL}/internal/generate", timeout=600,
                       json={"messages": saved["messages"], "max_new_tokens": max_new_tokens or golden["max_new_tokens"]})
    assert reply.status_code == 200, reply.text
    verdict, at = match_reference(saved, reply.json()["output_tokens"], max_gap=0)
    assert verdict == "identical", f"first difference at token {at}"


def test_docker_stop_makes_a_worker_dead_and_docker_start_brings_it_back(rules):
    since = time.strftime("%Y-%m-%dT%H:%M:%S")
    stopped = time.monotonic()
    docker("stop", WORKER_2)
    wait_for(lambda: statuses()["worker-2"] == "dead", rules["dead_after_s"] + 10, "worker-2 dead")
    took = time.monotonic() - stopped
    print(f"\nworker-2 counted dead {took:.1f} s after docker stop (limit {rules['dead_after_s']} s)")
    assert took < rules["dead_after_s"] + 5
    assert "worker-2 status: slow → dead (no contact for" in logs_since(MASTER, since)
    reply = httpx.post(f"{MASTER_URL}/internal/generate", json={"messages": [{"role": "user", "content": "Hi"}]})
    assert reply.status_code == 503
    assert reply.json()["missing"] == {"all_layers": list(range(21, 32))}
    assert reply.json()["unavailable_nodes"] == {"worker-2": "dead"}

    docker("start", WORKER_2)
    wait_for(lambda: statuses()["worker-2"] == "healthy", 300, "worker-2 healthy again")
    assert "worker-2 status: dead → joining (registered again)" in logs_since(MASTER, since)
    golden_check()


def test_a_frozen_worker_during_a_response_ends_it_with_a_clear_error(rules):
    since = time.strftime("%Y-%m-%dT%H:%M:%S")
    result = {}
    thread = threading.Thread(target=lambda: result.update(reply=httpx.post(
        f"{MASTER_URL}/internal/generate", timeout=600,
        json={"messages": [{"role": "user", "content": "Write a long story about a lighthouse."}],
              "max_new_tokens": 400})))
    thread.start()
    time.sleep(5)  # well into the answer
    paused = time.monotonic()
    docker("pause", WORKER_2)  # frozen: the process is still there but does nothing
    thread.join(timeout=120)
    took = time.monotonic() - paused
    reply = result["reply"]
    print(f"\nanswer ended {took:.1f} s after docker pause: {reply.status_code} {reply.json()['error'][:120]}")
    assert reply.status_code == 502
    assert reply.json()["error"].startswith("worker-2 stopped responding; experts [")
    # one try + retries, each up to the call timeout
    assert took < rules["expert_call_timeout_s"] * (1 + rules["expert_call_retries"]) + 5
    assert statuses()["worker-2"] == "dead"
    assert httpx.get(f"{MASTER_URL}/health", timeout=3).json()["status"] == "ok"  # the master stays usable

    docker("unpause", WORKER_2)
    wait_for(lambda: statuses()["worker-2"] == "healthy", 120, "worker-2 healthy again")
    assert "master asked to register again" in logs_since(WORKER_2, since)
    golden_check()


def test_workers_register_again_when_the_master_restarts(rules):
    since = time.strftime("%Y-%m-%dT%H:%M:%S")
    old_epoch = health()["master_epoch"]
    docker("restart", MASTER)
    wait_for(lambda: health()["ready"], 300, "restarted master ready")
    assert health()["master_epoch"] != old_epoch
    for worker in (WORKER_1, WORKER_2):
        assert "master asked to register again" in logs_since(worker, since)
    golden_check()


def test_network_delay_makes_a_worker_slow_not_dead(rules):
    since = time.strftime("%Y-%m-%dT%H:%M:%S")
    delay_ms = int(rules["slow_rtt_ms"] + 100)
    docker("exec", WORKER_2, "tc", "qdisc", "add", "dev", "eth0", "root", "netem", "delay", f"{delay_ms}ms")
    wait_for(lambda: statuses()["worker-2"] == "slow", 60, "worker-2 slow")
    time.sleep(rules["dead_after_s"] + 2)  # longer than the dead limit: it must stay slow, not dead
    assert statuses()["worker-2"] == "slow"
    reply = httpx.post(f"{MASTER_URL}/internal/generate", timeout=600,
                       json={"messages": [{"role": "user", "content": "Hi"}], "max_new_tokens": 2})
    assert reply.status_code == 200  # a slow node is still used
    docker("exec", WORKER_2, "tc", "qdisc", "del", "dev", "eth0", "root")
    wait_for(lambda: statuses()["worker-2"] == "healthy", 60, "worker-2 healthy again")
    log = logs_since(MASTER, since)
    assert "worker-2 status: healthy → slow (typical round trip" in log and "worker-2 status: slow → healthy" in log


def run_extra_worker(name: str, **env) -> str:
    """Start one more worker container on the cluster's network, with some settings changed."""
    settings = {"NODE_ROLE": "worker", "NODE_ID": name, "LISTEN_PORT": "8100", "MASTER_ADDR": "http://master:8000",
                "ADVERTISE_ADDR": f"http://{name}:8100", "EXPERT_IDS": "0-3", "DTYPE": health()["dtype"],
                "CLUSTER_TOKEN": os.environ.get("CLUSTER_TOKEN", "change-me-dev-token"), **env}
    args = [arg for k, v in settings.items() for arg in ("-e", f"{k}={v}")]
    docker("rm", "-f", name, check=False)
    docker("run", "-d", "--name", name, "--network", "moe_default", "-v", "moe_models:/models", *args, "moe-node:dev")
    return name


@pytest.mark.parametrize("kind", ["other-dtype", "wrong-token"])
def test_a_mismatched_worker_is_refused_with_the_reason_in_both_logs(rules, kind):
    if kind == "other-dtype":
        other = "fp32" if health()["dtype"] == "bf16" else "bf16"
        env, expected = {"DTYPE": other}, f"number format (DTYPE) differs: you have '{other}'"
    else:
        env, expected = {"CLUSTER_TOKEN": "wrong-token"}, "master answered 401"
    since = time.strftime("%Y-%m-%dT%H:%M:%S")
    name = run_extra_worker("worker-odd", **env)
    try:
        wait_for(lambda: expected in logs_since(name, since), 120, "worker-odd refused")
        master_log = logs_since(MASTER, since)
        print(f"\n{kind}: " + next(line for line in logs_since(name, since).splitlines() if expected in line))
        if kind == "wrong-token":
            assert "refused /internal/register from" in master_log and "wrong or missing cluster token" in master_log
        else:
            assert f"refused registration of worker-odd: {expected}" in master_log
        assert "worker-odd" not in statuses()
    finally:
        docker("rm", "-f", name, check=False)


def test_moe_node_status_and_leave(rules):
    table = docker("exec", MASTER, "moe-node", "status", "--master", "localhost:8000")
    print("\n" + table)
    assert "coverage: OK" in table
    assert [line.split()[1] for line in table.splitlines()[1:4]] == ["healthy"] * 3

    out = docker("exec", WORKER_2, "moe-node", "leave")
    assert "left the cluster (master confirmed: True)" in out
    wait_for(lambda: statuses()["worker-2"] == "left", 10, "worker-2 left")
    wait_for(lambda: docker("inspect", "-f", "{{.State.Running}}", WORKER_2).strip() == "false", 30, "worker-2 stopped")
    missing = subprocess.run(["docker", "exec", MASTER, "moe-node", "status", "--master", "localhost:8000"],
                             capture_output=True, text=True)
    assert missing.returncode == 2 and "coverage: MISSING experts" in missing.stdout
