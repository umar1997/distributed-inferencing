"""Phase 2.1: real HTTP between master and workers, but without Docker (all on this machine).

Each worker is the real worker service, started in a background thread on a free port. This
separates "does the network path work" from "does Docker / Linux change anything"
(tests/test_cluster.py checks the containers).
"""

import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from moe_dist.config import Settings
from moe_dist.experts.dispatcher import ExpertDispatcher
from moe_dist.master.engine import Engine
from moe_dist.models.catalog import DEFAULT_MODEL, short_name
from moe_dist.reference import match_reference
from moe_dist.transport import codec
from moe_dist.transport.client import NodeError, RemoteNode
from moe_dist.worker.app import create_app

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / short_name(DEFAULT_MODEL)
TOKEN = "test-token"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """Run a FastAPI app with uvicorn in a background thread."""

    def __init__(self, app, port):
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.url = f"http://127.0.0.1:{port}"

    def __enter__(self):
        self.thread.start()
        while not self.server.started:
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)


def start_worker(node_id, experts, dtype, token=TOKEN):
    port = free_port()
    settings = Settings(node_role="worker", node_id=node_id, listen_port=port, expert_ids=experts, dtype=dtype,
                        cluster_token=token)
    return Server(create_app(settings), port)


def wait_ready(url, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        answer = httpx.get(f"{url}/health").json()
        if answer["ready"] or answer["error"]:
            return answer
        time.sleep(0.2)
    raise TimeoutError(url)


@pytest.mark.model
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_master_and_workers_over_http_give_identical_tokens(dtype):
    golden = json.loads((GOLDEN_DIR / f"{dtype}.json").read_text())
    with start_worker("worker-1", "10-20", dtype) as w1, start_worker("worker-2", "21-31", dtype) as w2:
        settings = Settings(node_role="master", node_id="master", expert_ids="0-9", dtype=dtype, cluster_token=TOKEN,
                            workers=f"worker-1={w1.url},worker-2={w2.url}", workers_ready_timeout_s=120)
        engine = Engine(settings, log=lambda text: None)
        engine.start()
        assert engine.ready, engine.error
        assert isinstance(engine.model.model.layers[0].block_sparse_moe.experts, ExpertDispatcher)
        for saved in golden["prompts"]:
            result = engine.generate(saved["messages"], golden["max_new_tokens"])
            verdict, at = match_reference(saved, result["output_tokens"], max_gap=0)
            assert verdict == "identical", f"{saved['id']}: first difference at token {at}"
        calls = engine.stats()["calls_per_node"]
        assert all(n > 0 for n in calls.values()), calls
        health = httpx.get(f"{w1.url}/health").json()
        assert health["calls_served"] == calls["worker-1"]


@pytest.mark.model
def test_worker_endpoints_and_errors():
    with start_worker("worker-1", "3,7", "bf16") as w:
        health = wait_ready(w.url)
        assert health["ready"] and health["experts"] == {"all_layers": [3, 7]}
        assert health["model_revision"] and health["dtype"] == "bf16"

        import torch

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
        with pytest.raises(NodeError, match="400.*too short"):
            good.forward(b"\x01")
    with pytest.raises(NodeError, match="did not answer"):
        good.forward(message)  # the worker has stopped


@pytest.mark.model
def test_master_refuses_a_worker_with_another_number_format():
    with start_worker("worker-1", "0-31", "fp32") as w:
        settings = Settings(node_role="master", node_id="master", dtype="bf16", cluster_token=TOKEN,
                            workers=f"worker-1={w.url}", workers_ready_timeout_s=120)
        engine = Engine(settings, log=lambda text: None)
        engine.start()
        assert not engine.ready
        assert "worker-1 runs" in engine.error and "fp32" in engine.error


@pytest.mark.model
def test_master_refuses_when_experts_are_missing():
    with start_worker("worker-1", "10-31", "bf16") as w:
        settings = Settings(node_role="master", node_id="master", expert_ids="0-8", dtype="bf16", cluster_token=TOKEN,
                            workers=f"worker-1={w.url}", workers_ready_timeout_s=120)
        engine = Engine(settings, log=lambda text: None)
        engine.start()
        assert not engine.ready
        assert "missing experts: [9] in all layers" in engine.error


def test_workers_setting_is_checked():
    assert Settings(workers="a=http://x:1, b=http://y:2/").worker_urls() == {"a": "http://x:1", "b": "http://y:2"}
    with pytest.raises(ValueError, match="should look like"):
        Settings(workers="worker-1:8100").worker_urls()
