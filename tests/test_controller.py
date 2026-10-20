"""Phase 2.2: the master's health rules (docs/IMPLEMENTATION.md, section 5), with a fake clock.

No network and no model: the tests call register / heartbeat / record_ping directly and move the
clock forward by hand, so every rule is checked exactly and quickly.
"""

import pytest

from moe_dist.config import Settings
from moe_dist.master.controller import DEAD, HEALTHY, JOINING, LEFT, SLOW, Controller

SETTINGS = Settings(node_role="master", node_id="master", dtype="bf16", heartbeat_interval_s=2, dead_after_s=15,
                    slow_rtt_ms=250, recover_after_ok=3)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def setup():
    clock, logs = Clock(), []
    return Controller(SETTINGS, num_experts=32, log=logs.append, clock=clock), clock, logs


def payload(node_id="worker-1", ids=range(10, 21), ready=True, **changes):
    p = {"node_id": node_id, "advertise_addr": f"http://{node_id}:8100", "model_id": SETTINGS.model_id,
         "model_revision": SETTINGS.revision(), "dtype": "bf16", "experts": {"all_layers": list(ids)},
         "ready": ready, "info": {"cpu_cores": 4}}
    p.update(changes)
    return p


def beat(controller, node_id="worker-1", ready=True):
    return controller.heartbeat({"node_id": node_id, "master_epoch": controller.epoch, "ready": ready})


def test_a_worker_joins_then_becomes_healthy_after_good_contacts(setup):
    c, clock, logs = setup
    answer = c.register(payload(ready=False))
    assert answer == {"accepted": True, "master_epoch": c.epoch, "heartbeat_interval_s": 2}
    assert c.nodes["worker-1"].status == JOINING and c.usable() == []
    assert "worker-1 status: joining (registered from http://worker-1:8100, experts 10-20)" in logs
    for _ in range(5):  # still loading: stays joining however many heartbeats arrive
        clock.now += 2
        assert beat(c, ready=False) == {"ok": True}
    assert c.nodes["worker-1"].status == JOINING
    clock.now += 2
    beat(c, ready=True)  # ready, and well over 3 good contacts in a row
    assert c.nodes["worker-1"].status == HEALTHY
    assert [r.node_id for r in c.usable()] == ["worker-1"]
    assert logs[-1] == "worker-1 status: joining → healthy (7 good contacts in a row)"


def test_a_ready_worker_needs_three_good_contacts(setup):
    c, clock, _ = setup
    c.register(payload())  # contact 1
    beat(c)  # 2
    assert c.nodes["worker-1"].status == JOINING
    c.record_ping(c.nodes["worker-1"], 3.0)  # 3
    assert c.nodes["worker-1"].status == HEALTHY


@pytest.mark.parametrize("changes, reason", [
    ({"model_revision": "abc123"}, "model revision differs: you have 'abc123', the cluster uses 'ffec3c35"),
    ({"dtype": "fp32"}, "number format (DTYPE) differs: you have 'fp32', the cluster uses 'bf16'"),
    ({"model_id": "other/model"}, "model differs"),
    ({"node_id": "master"}, "is the master's own id"),
    ({"experts": {"all_layers": []}}, "holds no experts"),
    ({"experts": {"all_layers": [3, 40]}}, "bad expert list"),
    ({"advertise_addr": "worker-1:8100"}, "bad advertise_addr"),
])
def test_registration_is_refused_with_a_reason(setup, changes, reason):
    c, _, logs = setup
    answer = c.register(payload(**changes))
    assert answer["accepted"] is False and reason in answer["reason"]
    assert any(line.startswith("refused registration of") and reason in line for line in logs)
    assert c.nodes == {}


def test_a_live_node_id_cannot_be_taken_from_another_address(setup):
    c, _, _ = setup
    c.register(payload())
    beat(c), beat(c)
    other = c.register(payload(advertise_addr="http://10.0.0.9:8100"))
    assert not other["accepted"] and "already used by a live node at http://worker-1:8100" in other["reason"]
    assert c.register(payload())["accepted"]  # the same worker restarting (same address) is fine


def test_silence_makes_a_worker_dead_and_it_must_register_again(setup):
    c, clock, logs = setup
    c.register(payload()), beat(c), beat(c)
    assert c.nodes["worker-1"].status == HEALTHY
    version = c.version
    clock.now += 15.2
    c.tick()
    assert c.nodes["worker-1"].status == DEAD and c.usable() == []
    assert logs[-1] == "worker-1 status: healthy → dead (no contact for 15.2 s)"
    assert c.version > version  # the engine rebuilds its dispatchers
    # A late heartbeat does not bring it back by itself: the master asks it to register again.
    assert beat(c) == {"ok": False, "reregister": True}
    c.register(payload())
    assert "worker-1 status: dead → joining (registered again)" in logs
    beat(c), beat(c)
    assert c.nodes["worker-1"].status == HEALTHY


def test_a_heartbeat_from_an_unknown_node_or_old_master_asks_to_register_again(setup):
    c, _, _ = setup
    assert beat(c, "worker-9") == {"ok": False, "reregister": True}
    c.register(payload())
    assert c.heartbeat({"node_id": "worker-1", "master_epoch": "old-epoch"}) == {"ok": False, "reregister": True}


def test_slow_round_trips_make_a_worker_slow_but_still_usable(setup):
    c, clock, logs = setup
    c.register(payload()), beat(c), beat(c)
    node = c.nodes["worker-1"]
    for _ in range(3):
        clock.now += 1
        c.record_ping(node, 320.0)
    assert node.status == SLOW and c.usable() == [node]
    assert logs[-1] == "worker-1 status: healthy → slow (typical round trip 320 ms > 250 ms)"
    for _ in range(2):  # median of the last 5 is still 320 ms
        c.record_ping(node, 2.0)
    assert node.status == SLOW
    c.record_ping(node, 2.0)  # now the typical round trip is fast, and 3 good contacts in a row
    assert node.status == HEALTHY


def test_overdue_heartbeats_make_a_worker_slow_before_dead(setup):
    c, clock, logs = setup
    c.register(payload()), beat(c), beat(c)
    node = c.nodes["worker-1"]
    clock.now += 6.5  # more than 2 intervals late, but pings still answer
    c.record_ping(node, 3.0)
    assert node.status == SLOW and "heartbeat late: last one 6.5 s ago" in logs[-1]
    clock.now += 0.1
    beat(c)
    beat(c), beat(c)
    assert node.status == HEALTHY


def test_failed_pings_do_not_kill_before_the_time_limit(setup):
    c, clock, _ = setup
    c.register(payload()), beat(c), beat(c)
    node = c.nodes["worker-1"]
    for _ in range(4):  # 12 s of failed pings and no heartbeats
        clock.now += 3
        c.record_ping(node, None, "ConnectTimeout")
    assert node.status == SLOW  # heartbeats overdue
    clock.now += 3.5
    c.tick()
    assert node.status == DEAD


def test_leave_and_failed_calls(setup):
    c, _, logs = setup
    for name in ("worker-1", "worker-2"):
        c.register(payload(name)), beat(c, name), beat(c, name)
    assert c.leave("worker-1") == {"ok": True}
    assert c.nodes["worker-1"].status == LEFT and logs[-1] == "worker-1 status: healthy → left (said goodbye)"
    c.call_failed("worker-2", "worker-2 stopped responding")
    assert c.nodes["worker-2"].status == DEAD and "expert call failed" in logs[-1]
    assert c.usable() == []
    assert c.leave("worker-7") == {"ok": False, "reason": "unknown node worker-7"}


def test_describe_lists_what_the_ui_and_status_command_need(setup):
    c, _, _ = setup
    c.register(payload()), beat(c), beat(c)
    c.record_ping(c.nodes["worker-1"], 4.0)
    [row] = c.describe()
    assert row["status"] == HEALTHY and row["address"] == "http://worker-1:8100"
    assert row["experts"] == {"all_layers": list(range(10, 21))} and row["rtt_p50_ms"] == 4.0
    assert row["cpu_cores"] == 4
