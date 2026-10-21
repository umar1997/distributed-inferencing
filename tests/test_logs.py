"""Phase 3.1: log lines that let you follow one token across nodes.

The JSON format is checked line by line (every line has the required fields); with the model, a
small cluster runs at LOG_LEVEL=debug and the test follows one token: the master's "sending" line
for a layer has a matching "ran experts" line from the worker with the same req / token / layer.
"""

import io
import json
import logging

import pytest

from moe_dist.experts.dispatcher import request_scope
from moe_dist.logs import get_logger, setup_logging
from tests.servers import start_master, start_worker, status_of, wait_for


@pytest.fixture
def captured():
    """Log lines of this test in JSON (and the logger reset afterwards)."""
    stream = io.StringIO()
    setup_logging("master", "debug", "json", stream=stream)
    yield lambda: [json.loads(line) for line in stream.getvalue().splitlines()]
    logger = get_logger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def test_pretty_lines_carry_node_level_and_request_context():
    stream = io.StringIO()
    log = setup_logging("worker-1", "info", "pretty", stream=stream)
    try:
        log.info("no request here")
        with request_scope("a1b2") as info:
            info.token_index = 14
            log.info("inside a request", extra={"layer": 6})
            log.debug("hidden at info level")
        first, second = stream.getvalue().splitlines()
        assert first.split()[1:] == ["worker-1", "INFO", "no", "request", "here"]
        assert second.split()[1:] == ["worker-1", "INFO", "req=a1b2", "token=14", "layer=6", "inside", "a", "request"]
        assert len(first.split()[0]) == len("12:04:01.233")
    finally:
        log.handlers.clear()
        log.propagate = True
        log.setLevel(logging.NOTSET)


def test_json_lines_have_the_required_fields(captured):
    log = get_logger()
    with request_scope("a1b2") as info:
        info.token_index = 3
        log.debug("sending [3,7] → worker-1", extra={"layer": 6, "fields": {"total_ms": 9.8}})
    log.info("worker-2 status: healthy → slow (heartbeat late)")
    first, second = captured()
    assert {"time", "node", "level", "msg"} <= set(first) and {"time", "node", "level", "msg"} <= set(second)
    assert first["req"] == "a1b2" and first["token"] == 3 and first["layer"] == 6 and first["total_ms"] == 9.8
    assert "req" not in second and second["level"] == "info"


def test_bad_settings_are_refused():
    with pytest.raises(ValueError, match="LOG_FORMAT"):
        setup_logging("x", "info", "xml")
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        setup_logging("x", "loud", "pretty")


@pytest.mark.model
def test_a_token_can_be_followed_from_master_to_worker(captured):
    with start_master("0-9") as m, start_worker("worker-1", "10-31", master_url=m.url):
        engine = m.app.state.engine
        wait_for(lambda: engine.ready and status_of(m, "worker-1") == "healthy", 120, "cluster ready")
        result = engine.generate([{"role": "user", "content": "What is the capital of France?"}], 12)
        lines = captured()

    req = result["request_id"]
    mine = [line for line in lines if line.get("req") == req]
    received = [line for line in mine if line["msg"].startswith("request received")]
    assert len(received) == 1 and "prompt_tokens=" in received[0]["msg"]

    # Token 5, layer 6: what the master sent to worker-1 is what worker-1 says it ran.
    at = [line for line in mine if line.get("token") == 5 and line.get("layer") == 6]
    sending = next(line["msg"] for line in at if line["msg"].startswith("sending"))
    sent_to_worker = next(part.split(" → ")[0] for part in sending[len("sending "):].split(", ") if "worker-1" in part)
    ran = [line["msg"] for line in at if line["msg"].startswith(f"ran experts {sent_to_worker} ")]
    assert ran, f"no worker line for {sent_to_worker}: {[line['msg'] for line in at]}"
    assert any(line["msg"].startswith("router picked experts [") for line in at)
    assert any(line["msg"].startswith("all results back in") for line in at)
    produced = [line for line in mine if line["msg"].startswith("produced") and line.get("token") == 5]
    assert len(produced) == 1

    # The prompt (token 0) is one call per node per layer, however long it is.
    prompt_layer6 = [line for line in mine if line.get("token") == 0 and line.get("layer") == 6]
    assert sum(line["msg"].startswith("sending") for line in prompt_layer6) == 1

    done = [line for line in mine if line["msg"].startswith("done:")]
    assert len(done) == 1 and "worker-1 net" in done[0]["msg"] and "local compute" in done[0]["msg"]
    assert done[0]["tokens"] == len(result["output_tokens"]) and done[0]["node_time_ms"]["worker-1"]["calls"] > 0


@pytest.mark.model
def test_info_level_stays_quiet(captured):
    get_logger().setLevel(logging.INFO)
    with start_master("0-9") as m, start_worker("worker-1", "10-31", master_url=m.url):
        engine = m.app.state.engine
        wait_for(lambda: engine.ready and status_of(m, "worker-1") == "healthy", 120, "cluster ready")
        result = engine.generate([{"role": "user", "content": "Write a short poem about the sea."}], 64)
        lines = [line for line in captured() if line.get("req") == result["request_id"]]
    assert len(result["output_tokens"]) > 30
    assert [line["msg"].split(":")[0] for line in lines] == ["request received", "done"]
