"""Phase 2.2: workers join and leave by themselves, and the master always knows who is alive.

The real master and worker services run in this test process (tests/servers.py) with short
health timings (dead after 2 s instead of 15 s). "Stopping a worker" shuts its web server down
without a goodbye, which to the master looks the same as a crashed or unplugged worker.
The same scenarios against Docker containers (docker stop / pause / restart, network delay) are
in tests/test_cluster_failures.py.
"""

import json
import threading
from pathlib import Path

import httpx
import pytest

from moe_dist.models.catalog import DEFAULT_MODEL, short_name
from moe_dist.reference import match_reference
from tests.servers import TOKEN, start_master, start_worker, status_of, wait_for

pytestmark = pytest.mark.model

GOLDEN = json.loads((Path(__file__).resolve().parent / "golden" / short_name(DEFAULT_MODEL) / "bf16.json").read_text())
PROMPT = GOLDEN["prompts"][0]


def generate(master, max_new_tokens=None):
    return httpx.post(f"{master.url}/internal/generate", timeout=300,
                      json={"messages": PROMPT["messages"], "max_new_tokens": max_new_tokens or GOLDEN["max_new_tokens"]})


def assert_golden(reply):
    assert reply.status_code == 200, reply.text
    verdict, at = match_reference(PROMPT, reply.json()["output_tokens"], max_gap=0)
    assert verdict == "identical", f"first difference at token {at}"


def healthy(master, *node_ids):
    """Wait until these workers are healthy and the master has finished loading its own parts."""
    return wait_for(lambda: master.app.state.engine.loaded and all(status_of(master, n) == "healthy" for n in node_ids),
                    120, f"{node_ids} healthy")


def test_master_alone_is_not_ready_until_the_workers_join():
    with start_master("0-9") as m:
        wait_for(lambda: m.app.state.engine.loaded, what="master loaded")
        health = httpx.get(f"{m.url}/health").json()
        assert not health["ready"] and health["missing_experts"] == 22  # in every layer
        reply = generate(m, 4)
        assert reply.status_code == 503
        assert reply.json() == {"error": "missing experts", "missing": {"all_layers": list(range(10, 32))},
                                "unavailable_nodes": {}}
        with start_worker("worker-1", "10-20", master_url=m.url), start_worker("worker-2", "21-31", master_url=m.url):
            healthy(m, "worker-1", "worker-2")
            assert m.logs.has("worker-1 status: joining (registered from http://127.0.0.1:")
            assert m.logs.has("worker-1 status: joining → healthy")
            assert httpx.get(f"{m.url}/health").json()["ready"]
            assert_golden(generate(m))


def test_a_stopped_worker_is_dead_within_the_limit_and_comes_back_by_itself():
    with start_master("0-9") as m, start_worker("worker-1", "10-20", master_url=m.url):
        w2 = start_worker("worker-2", "21-31", master_url=m.url).__enter__()
        healthy(m, "worker-1", "worker-2")
        w2.stop()  # no goodbye
        wait_for(lambda: status_of(m, "worker-2") == "dead", 10, "worker-2 dead")
        # healthy → slow (heartbeats late) → dead (silent for longer than DEAD_AFTER_S)
        assert m.logs.has("worker-2 status: healthy → slow (heartbeat late")
        assert m.logs.has("worker-2 status: slow → dead (no contact for 2.")
        reply = generate(m, 4)
        assert reply.status_code == 503
        assert reply.json()["missing"] == {"all_layers": list(range(21, 32))}
        assert reply.json()["unavailable_nodes"] == {"worker-2": "dead"}

        with start_worker("worker-2", "21-31", master_url=m.url, port=w2.port):  # same address again
            healthy(m, "worker-2")
            assert m.logs.has("worker-2 status: dead → joining (registered again)")
            assert_golden(generate(m))


def test_a_worker_that_dies_during_a_response_gives_a_clear_error_and_the_master_stays_usable():
    with start_master("0-9") as m, start_worker("worker-1", "10-20", master_url=m.url):
        w2 = start_worker("worker-2", "21-31", master_url=m.url).__enter__()
        healthy(m, "worker-1", "worker-2")
        result = {}
        thread = threading.Thread(target=lambda: result.update(reply=generate(m, 200)))
        thread.start()
        wait_for(lambda: w2.app.state.worker.calls > 50, 60, "worker-2 busy with the response")
        w2.stop()
        thread.join(timeout=120)
        reply = result["reply"]
        assert reply.status_code == 502
        error = reply.json()["error"]
        assert error.startswith("worker-2 stopped responding; experts [") and "] in layer " in error
        assert "are unavailable" in error
        assert status_of(m, "worker-2") == "dead"  # at once, not after the time limit
        assert m.logs.has("worker-2 status: healthy → dead (expert call failed:")

        with start_worker("worker-2", "21-31", master_url=m.url, port=w2.port):
            healthy(m, "worker-2")
            assert_golden(generate(m))


def test_workers_register_again_by_themselves_when_the_master_restarts():
    m = start_master("0-9").__enter__()
    port = m.port
    with start_worker("worker-1", "10-20", master_url=m.url) as w1, \
            start_worker("worker-2", "21-31", master_url=m.url) as w2:
        healthy(m, "worker-1", "worker-2")
        old_epoch = m.app.state.engine.controller.epoch
        m.stop()
        with start_master("0-9", port=port) as m2:
            healthy(m2, "worker-1", "worker-2")
            assert m2.app.state.engine.controller.epoch != old_epoch
            for w in (w1, w2):
                # Either a heartbeat failed while the master was away, or the new master was already
                # back by the next heartbeat; both end with the new master asking to register again.
                assert w.logs.has("master asked to register again")
                assert sum(line.startswith("registered with master") for line in w.logs) == 2
            wait_for(lambda: m2.app.state.engine.ready, what="new master ready")
            assert_golden(generate(m2))


def test_leave_says_goodbye_and_stops_the_worker():
    with start_master("0-9") as m, start_worker("worker-1", "10-31", master_url=m.url) as w:
        healthy(m, "worker-1")
        answer = httpx.post(f"{w.url}/internal/leave", headers={"Authorization": f"Bearer {TOKEN}"}).json()
        assert answer == {"ok": True, "master_confirmed": True, "stopping": True}
        assert status_of(m, "worker-1") == "left"
        assert m.logs.has("worker-1 status: healthy → left (said goodbye)")
        wait_for(lambda: not w.thread.is_alive(), 15, "worker stopped")
        assert generate(m, 4).json()["unavailable_nodes"] == {"worker-1": "left"}


def test_api_nodes_and_placement():
    with start_master("0-9") as m, start_worker("worker-1", "10-31", master_url=m.url):
        healthy(m, "worker-1")
        wait_for(lambda: m.app.state.engine.ready, what="ready")
        nodes = httpx.get(f"{m.url}/api/nodes").json()
        assert nodes["coverage"] == "ok"
        master, worker = nodes["nodes"]
        assert master["role"] == "master" and master["experts"] == {"all_layers": list(range(10))}
        assert worker["status"] == "healthy" and worker["rtt_now_ms"] is not None and worker["cpu_cores"] > 0
        placement = httpx.get(f"{m.url}/api/placement").json()
        assert placement["coverage"] == "ok"
        assert placement["nodes_by_layer"]["0"][3] == "master" and placement["nodes_by_layer"]["23"][31] == "worker-1"


def test_moe_node_status_prints_the_table(capsys):
    from typer.testing import CliRunner

    from moe_dist.cli.main import app

    with start_master("0-9") as m, start_worker("worker-1", "10-20", master_url=m.url):
        healthy(m, "worker-1")
        result = CliRunner().invoke(app, ["status", "--master", m.url])
        assert result.exit_code == 2, result.output  # experts 21-31 are missing
        lines = result.output.splitlines()
        assert lines[0].split()[:3] == ["NODE", "STATUS", "ADDRESS"]
        assert lines[1].split()[:2] == ["master", "healthy"] and lines[2].split()[:2] == ["worker-1", "healthy"]
        assert "coverage: MISSING experts {'all_layers': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]}" in result.output
    result = CliRunner().invoke(app, ["status", "--master", m.url])
    assert result.exit_code == 1 and "cannot reach the master" in result.output

