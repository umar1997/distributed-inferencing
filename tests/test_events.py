"""Phase 3.2: the live event stream for the UI (GET /api/events, Server-Sent Events)."""

import json
import threading

import httpx
import pytest

from moe_dist.master import events as events_module
from moe_dist.master.events import EventHub
from tests.servers import start_master, start_worker, status_of, wait_for


def test_hub_sends_every_event_to_every_listener():
    hub = EventHub()
    assert not hub.active
    a, b = hub.subscribe(), hub.subscribe()
    hub.publish("request_start", req="a1b2", prompt_tokens=7)
    for q in (a, b):
        event = q.get_nowait()
        assert event["type"] == "request_start" and event["req"] == "a1b2" and "time" in event
    hub.unsubscribe(a), hub.unsubscribe(b)
    assert not hub.active


def test_a_slow_listener_loses_events_instead_of_slowing_the_model():
    hub = EventHub(max_queued=2)
    q = hub.subscribe()
    for i in range(5):
        hub.publish("token", token_index=i)  # never blocks
    assert hub.dropped == 3 and [q.get_nowait()["token_index"] for _ in range(2)] == [0, 1]


def test_stream_starts_with_the_first_events_and_keeps_the_connection_alive(monkeypatch):
    monkeypatch.setattr(events_module, "KEEPALIVE_S", 0.05)
    hub = EventHub()
    stream = hub.stream([{"type": "hello"}])
    assert next(stream) == 'data: {"type": "hello"}\n\n'
    assert next(stream) == ": keep-alive\n\n"
    hub.publish("node_update", nodes=[])
    assert json.loads(next(stream)[len("data: "):])["type"] == "node_update"
    stream.close()
    assert not hub.active  # closing the stream (the browser left) removes the listener


@pytest.mark.model
def test_events_show_every_token_with_every_layers_experts_and_nodes():
    with start_master("0-15") as m, start_worker("worker-1", "16-31", master_url=m.url):
        engine = m.app.state.engine
        wait_for(lambda: engine.ready and status_of(m, "worker-1") == "healthy", 120, "cluster ready")
        received, connected = [], threading.Event()

        def listen():
            with httpx.stream("GET", f"{m.url}/api/events", timeout=60) as reply:
                assert reply.headers["content-type"].startswith("text/event-stream")
                for line in reply.iter_lines():
                    if line.startswith("data: "):
                        received.append(json.loads(line[len("data: "):]))
                        connected.set()
                        kinds = {e["type"] for e in received}
                        if {"request_end", "node_update"} <= kinds:  # node updates come at least every 5 s
                            return

        listener = threading.Thread(target=listen, daemon=True)
        listener.start()
        connected.wait(10)
        wait_for(lambda: engine.events.active, 10, "listener connected")
        answer = httpx.post(f"{m.url}/v1/chat/completions", timeout=120, json={
            "messages": [{"role": "user", "content": "What is the capital of France?"}], "temperature": 0,
            "max_tokens": 8}).json()
        listener.join(timeout=60)

    kinds = [e["type"] for e in received]
    hello = received[0]
    assert kinds[0] == "hello" and hello["num_experts"] == 32 and len(hello["layers"]) == 24
    assert hello["placement"]["0"][:16] == ["master"] * 16 and hello["placement"]["23"][16:] == ["worker-1"] * 16
    assert [n["node_id"] for n in hello["nodes"]] == ["master", "worker-1"] and hello["coverage"] == "ok"

    start = next(e for e in received if e["type"] == "request_start")
    tokens = [e for e in received if e["type"] == "token" and e["req"] == start["req"]]
    end = next(e for e in received if e["type"] == "request_end")
    assert kinds.index("request_start") < kinds.index("token") < kinds.index("request_end")
    assert len(tokens) == answer["usage"]["completion_tokens"] == end["tokens"]
    assert [t["token_index"] for t in tokens] == list(range(len(tokens)))  # 0 = made by the pass over the prompt
    assert "".join(t["text"] for t in tokens) == answer["choices"][0]["message"]["content"]
    assert all(t["text"] == "" for t in tokens if t["special"])

    for t in tokens:
        assert [layer["layer"] for layer in t["layers"]] == list(range(24))  # all layers in one event
        for layer in t["layers"]:
            for expert in layer["experts"]:
                assert expert["node"] == ("master" if expert["id"] < 16 else "worker-1")
            assert layer["wait_ms"] > 0
    prompt_layer = tokens[0]["layers"][6]  # the prompt: experts with how many tokens picked them
    assert all("count" in e for e in prompt_layer["experts"])
    step_layer = tokens[1]["layers"][6]  # one new token: its 8 experts with the router's weights
    assert len(step_layer["experts"]) == 8
    assert abs(sum(e["weight"] for e in step_layer["experts"]) - 1) < 0.02
    assert end["node_time_ms"]["worker-1"]["calls"] > 0 and end["summary"].startswith("done:")
    update = next(e for e in received if e["type"] == "node_update")
    assert {n["node_id"]: n["status"] for n in update["nodes"]} == {"master": "healthy", "worker-1": "healthy"}
