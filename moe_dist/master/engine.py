"""The master's engine: build the model with dispatchers, then run one request at a time.

Start-up order:
1. load the shared parts of the model (no experts) and the master's own experts (if any),
2. wait until every worker in WORKERS says it is ready, and check it runs the same model,
   revision and number format,
3. build the placement from what each node really holds, check every expert is covered,
4. swap in the dispatchers. The master's own experts are a LocalNode (no network).
"""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoConfig

from moe_dist.config import Settings
from moe_dist.experts.dispatcher import request_scope
from moe_dist.experts.loader import model_dir
from moe_dist.experts.placement import Placement, parse_expert_ids
from moe_dist.master.model_loader import install_dispatchers, load_shared_model, local_node
from moe_dist.models import adapter_for
from moe_dist.reference import greedy_generate
from moe_dist.transport.client import RemoteNode, wait_until_ready


class Engine:
    def __init__(self, settings: Settings, log=print):
        self.settings = settings
        self.log = log
        self.model = self.tokenizer = None
        self.nodes: dict = {}
        self.dispatchers: dict = {}
        self.missing: dict = {}
        self.error: str | None = None
        self._lock = threading.Lock()  # one request at a time (Phase 2.3 adds a queue)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def start(self):
        try:
            self._build()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.log(f"master start-up failed: {self.error}")

    def _build(self):
        s = self.settings
        revision = s.revision()
        adapter = adapter_for(AutoConfig.from_pretrained(model_dir(s.model_id, revision)))
        start = time.perf_counter()
        model, tokenizer = load_shared_model(s.model_id, revision, s.dtype, adapter)
        self.log(f"shared parts loaded in {time.perf_counter() - start:.1f} s "
                 f"({sum(p.nbytes for p in model.parameters()) / 2**20:.0f} MB)")

        held = {}  # node_id -> expert numbers (same in every layer for now)
        own = parse_expert_ids(s.expert_ids, adapter.num_experts)
        if own:
            self.nodes[s.node_id] = local_node(s.node_id, s.model_id, revision, s.dtype, adapter, own)
            held[s.node_id] = own
            self.log(f"own experts loaded: {s.expert_ids}")

        remotes = [RemoteNode(nid, url, s.cluster_token, s.expert_call_timeout_s) for nid, url in s.worker_urls().items()]
        for node_id, answer in wait_until_ready(remotes, s.workers_ready_timeout_s, self.log).items():
            mine = (s.model_id, revision, s.dtype)
            theirs = (answer["model_id"], answer["model_revision"], answer["dtype"])
            if theirs != mine:
                raise RuntimeError(f"{node_id} runs {theirs}, the master runs {mine}")
            held[node_id] = answer["experts"]["all_layers"]
        self.nodes.update({r.node_id: r for r in remotes})

        placement = Placement.same_in_every_layer(held, adapter.moe_layers(), adapter.num_experts)
        self.missing = placement.missing()
        placement.check_coverage()
        executor = ThreadPoolExecutor(max_workers=max(1, len(remotes)), thread_name_prefix="expert-call")
        self.dispatchers = install_dispatchers(model, adapter, placement, self.nodes, s.combine_mode, executor)
        self.model, self.tokenizer = model, tokenizer
        self.log(f"ready: nodes {list(held)}, combine mode {s.combine_mode}")

    def generate(self, messages: list[dict], max_new_tokens: int) -> dict:
        if not self.ready:
            raise RuntimeError(self.error or "the master is still starting")
        request_id = uuid.uuid4().hex[:8]
        with self._lock, request_scope(request_id):
            start = time.perf_counter()
            result = greedy_generate(self.model, self.tokenizer, messages, max_new_tokens)
            seconds = time.perf_counter() - start
        result.update(request_id=request_id, seconds=round(seconds, 3),
                      tokens_per_s=round(len(result["output_tokens"]) / seconds, 2))
        return result

    def stats(self) -> dict:
        calls = {node_id: getattr(node, "calls", 0) for node_id, node in self.nodes.items()}
        sent = sum(d.stats["bytes_sent"] for d in self.dispatchers.values())
        received = sum(d.stats["bytes_received"] for d in self.dispatchers.values())
        return {"calls_per_node": calls, "bytes_sent": sent, "bytes_received": received}
