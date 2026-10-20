"""Phase 2.1/2.2: real HTTP between master and workers, but without Docker (all on this machine).

The master and each worker are the real services, started in background threads on free ports
(tests/servers.py); workers register themselves with the master. This
separates "does the network path work" from "does Docker / Linux change anything"
(tests/test_cluster.py checks the containers).
"""

import json
from pathlib import Path

import httpx
import pytest
import torch

from moe_dist.config import Settings
from moe_dist.experts.dispatcher import ExpertDispatcher
from moe_dist.master.engine import NotReady
from moe_dist.models.catalog import DEFAULT_MODEL, short_name
from moe_dist.reference import match_reference
from moe_dist.transport import codec
from moe_dist.transport.client import NodeError, RemoteNode
from tests.servers import TOKEN, start_master, start_worker, status_of, wait_for

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / short_name(DEFAULT_MODEL)


def wait_ready(url, timeout=120):
    return wait_for(lambda: (lambda a: a if a["ready"] or a["error"] else None)(httpx.get(f"{url}/health").json()),
                    timeout, f"{url} ready")


@pytest.mark.model
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_master_and_workers_over_http_give_identical_tokens(dtype):
    golden = json.loads((GOLDEN_DIR / f"{dtype}.json").read_text())
    with start_master("0-9", dtype) as m, start_worker("worker-1", "10-20", dtype, m.url) as w1, \
            start_worker("worker-2", "21-31", dtype, m.url):
        engine = m.app.state.engine
        wait_for(lambda: engine.ready, what="master ready")
        assert httpx.get(f"{m.url}/health").json()["nodes"] == ["master", "worker-1", "worker-2"]
        for saved in golden["prompts"]:
            result = engine.generate(saved["messages"], golden["max_new_tokens"])
            verdict, at = match_reference(saved, result["output_tokens"], max_gap=0)
            assert verdict == "identical", f"{saved['id']}: first difference at token {at}"
        assert isinstance(engine.model.model.layers[0].block_sparse_moe.experts, ExpertDispatcher)
        calls = engine.stats()["calls_per_node"]
        assert all(n > 0 for n in calls.values()), calls
        assert httpx.get(f"{w1.url}/health").json()["calls_served"] == calls["worker-1"]


@pytest.mark.model
def test_worker_endpoints_and_errors():
    with start_worker("worker-1", "3,7", "bf16") as w:
        health = wait_ready(w.url)
        assert health["ready"] and health["experts"] == {"all_layers": [3, 7]}
        assert health["model_revision"] and health["dtype"] == "bf16"

        message = codec.encode(
            {"layer": 0, "combine_mode": "exact"},
            {"hidden": torch.randn(2, 1024).to(torch.bfloat16), "row": torch.tensor([0, 1], dtype=torch.int32),
             "expert": torch.tensor([3, 7], dtype=torch.int32), "weight": torch.ones(2, dtype=torch.bfloat16)},
        )
        good = RemoteNode("worker-1", w.url, TOKEN)
        header, out = codec.decode(good.forward(message))
        assert header["node_id"] == "worker-1" and header["experts_run"] == [3, 7]
        assert out["out"].shape == (2, 1024)

        with pytest.raises(NodeError, match="401"):
            RemoteNode("worker-1", w.url, "wrong-token").forward(message)
        with pytest.raises(NodeError, match="400.*not on this node"):
            bad = codec.encode({"layer": 0, "combine_mode": "exact"},
                               {"hidden": torch.randn(1, 1024).to(torch.bfloat16), "row": torch.tensor([0], dtype=torch.int32),
                                "expert": torch.tensor([5], dtype=torch.int32), "weight": torch.ones(1, dtype=torch.bfloat16)})
            good.forward(bad)
        with pytest.raises(NodeError, match="400.*too short") as bad_request:
            good.forward(b"\x01")
        assert not bad_request.value.unreachable  # a bad request says nothing about the worker's health
        assert 0 < good.ping(timeout_s=1) < 1000
        with pytest.raises(httpx.HTTPStatusError, match="401"):
            RemoteNode("worker-1", w.url, "wrong-token").ping(timeout_s=1)
    retrying = RemoteNode("worker-1", w.url, TOKEN, retries=1)
    with pytest.raises(NodeError, match="did not answer") as gone:
        retrying.forward(message)  # the worker has stopped
    assert gone.value.unreachable and retrying.calls == 2  # tried once more before giving up


@pytest.mark.model
def test_master_refuses_a_worker_with_another_number_format():
    with start_master("none", "bf16") as m, start_worker("worker-1", "0-31", "fp32", m.url) as w:
        wait_for(lambda: w.logs.has("registration refused"), what="refusal")
        reason = "number format (DTYPE) differs: you have 'fp32', the cluster uses 'bf16'"
        assert w.logs.has(reason) and m.logs.has(f"refused registration of worker-1: {reason}")  # both logs
        assert status_of(m, "worker-1") is None
        wait_for(lambda: m.app.state.engine.loaded, what="master loaded")
        health = httpx.get(f"{m.url}/health").json()
        assert not health["ready"] and health["missing"]["missing"] == {"all_layers": list(range(32))}


@pytest.mark.model
def test_master_refuses_requests_when_experts_are_missing():
    with start_master("0-8", "bf16") as m, start_worker("worker-1", "10-31", "bf16", m.url):
        engine = m.app.state.engine
        wait_for(lambda: engine.loaded and status_of(m, "worker-1") == "healthy", what="worker healthy")
        assert not engine.ready
        with pytest.raises(NotReady, match=r"missing experts \{'all_layers': \[9\]\}"):
            engine.generate([{"role": "user", "content": "Hi"}], 4)
        reply = httpx.post(f"{m.url}/internal/generate", json={"messages": [{"role": "user", "content": "Hi"}]})
        assert reply.status_code == 503
        assert reply.json() == {"error": "missing experts", "missing": {"all_layers": [9]}, "unavailable_nodes": {}}


@pytest.mark.model
def test_wrong_cluster_token_is_refused_and_logged():
    with start_master("0-9", "bf16") as m, start_worker("worker-1", "10-31", "bf16", m.url, token="wrong") as w:
        wait_for(lambda: w.logs.has("registration failed: master answered 401"), what="401 in the worker log")
        assert m.logs.has("refused /internal/register from 127.0.0.1: wrong or missing cluster token")
        assert status_of(m, "worker-1") is None


def test_addresses_are_read_with_or_without_http():
    s = Settings(node_id="worker-1", listen_port=8100, master_addr="192.168.1.10:8000")
    assert s.master_url() == "http://192.168.1.10:8000"
    assert s.advertise_url() == "http://worker-1:8100"
    assert Settings(advertise_addr="http://10.0.0.5:8100/").advertise_url() == "http://10.0.0.5:8100"
    assert Settings().master_url() == ""
