"""The master's engine: build the model with dispatchers, then run one request at a time.

Start-up:
1. load the shared parts of the model (no experts) and the master's own experts (if any),
2. start the controller: workers register themselves (master/controller.py) and get health states.

Requests run one at a time; the others wait in line (master/generation.py). Before every
request the engine looks at which workers are usable (healthy or slow) and builds the
placement from them and the master's own experts. If any expert is held by nobody usable, the
request is refused with the list of missing experts. The dispatchers are rebuilt only when the set
of usable workers changed.
"""

import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoConfig

from moe_dist.config import Settings
from moe_dist.experts.dispatcher import request_scope
from moe_dist.experts.loader import model_dir
from moe_dist.experts.placement import Placement, parse_expert_ids
from moe_dist.master.controller import USABLE, Controller
from moe_dist.master.generation import GenerationRequest, GenerationSession, RequestQueue
from moe_dist.master.model_loader import install_dispatchers, load_shared_model, local_node
from moe_dist.models import adapter_for
from moe_dist.transport.client import NodeError


class NotReady(RuntimeError):
    """The master cannot take a request yet: still loading, or some experts are missing."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {"error": message}


class Engine:
    def __init__(self, settings: Settings, log=print):
        self.settings = settings
        self.log = log
        self.adapter = self.model = self.tokenizer = None
        self.controller: Controller | None = None
        self.own_experts: list[int] = []
        self.local_nodes: dict = {}  # the master's own experts (no network)
        self.dispatchers: dict = {}
        self.error: str | None = None
        self._key = None  # which nodes the current dispatchers were built for
        self._past_stats = Counter()  # from dispatchers that were replaced
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="expert-call")
        self.queue = RequestQueue(settings.max_queue)  # one request at a time, the others wait in line

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def ready(self) -> bool:
        return self.loaded and not self.missing_report()

    def start(self):
        try:
            self._build()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.log(f"master start-up failed: {self.error}")

    def _build(self):
        s = self.settings
        revision = s.revision()
        self.adapter = adapter = adapter_for(AutoConfig.from_pretrained(model_dir(s.model_id, revision)))
        # Workers may register while the model is still loading; they stay "joining" meanwhile.
        self.controller = Controller(s, adapter.num_experts, self.log)
        self.controller.start_monitor()
        self.log(f"accepting registrations (master epoch {self.controller.epoch})")

        start = time.perf_counter()
        model, tokenizer = load_shared_model(s.model_id, revision, s.dtype, adapter)
        self.log(f"shared parts loaded in {time.perf_counter() - start:.1f} s "
                 f"({sum(p.nbytes for p in model.parameters()) / 2**20:.0f} MB)")
        self.own_experts = parse_expert_ids(s.expert_ids, adapter.num_experts)
        if self.own_experts:
            self.local_nodes[s.node_id] = local_node(s.node_id, s.model_id, revision, s.dtype, adapter, self.own_experts)
            self.log(f"own experts loaded: {s.expert_ids}")
        self.model, self.tokenizer = model, tokenizer

    # --- placement from the nodes that are usable right now -----------------------------------

    def _current(self):
        """(placement, nodes, key) for the master's own experts plus every usable worker."""
        held = {self.settings.node_id: self.own_experts} if self.own_experts else {}
        nodes = dict(self.local_nodes)
        workers = self.controller.usable() if self.controller else []
        for record in workers:
            held[record.node_id] = record.experts
            nodes[record.node_id] = record.remote
        placement = Placement.same_in_every_layer(held, self.adapter.moe_layers(), self.adapter.num_experts)
        key = tuple((node_id, id(nodes[node_id]), tuple(ids)) for node_id, ids in held.items())
        return placement, nodes, key

    def missing_report(self) -> dict | None:
        """None when every expert is covered; else what is missing and which unusable nodes hold it."""
        if not self.loaded:
            return None
        placement, _, _ = self._current()
        gaps = placement.missing()
        if not gaps:
            return None
        if len(gaps) == len(placement.layers) and len({tuple(ids) for ids in gaps.values()}) == 1:
            missing = {"all_layers": next(iter(gaps.values()))}
        else:
            missing = {str(layer): ids for layer, ids in gaps.items()}
        wanted = {e for ids in gaps.values() for e in ids}
        unavailable = {r.node_id: r.status for r in self.controller.records()
                       if r.status not in USABLE and wanted & set(r.experts)}
        return {"error": "missing experts", "missing": missing, "unavailable_nodes": unavailable}

    def _prepare(self):
        """Make the dispatchers match the usable nodes (called when it is a request's turn)."""
        if not self.loaded:
            raise NotReady(self.error or "the master is still starting")
        problem = self.missing_report()
        if problem:
            raise NotReady(f"missing experts {problem['missing']} (unavailable nodes: {problem['unavailable_nodes']})",
                           problem)
        placement, nodes, key = self._current()
        if key != self._key:
            for d in self.dispatchers.values():
                self._past_stats.update(d.stats)
            s = self.settings
            self.dispatchers = install_dispatchers(self.model, self.adapter, placement, nodes, s.combine_mode,
                                                   self._executor)
            self._key = key
            self.log(f"placement updated: nodes {[node_id for node_id, _, _ in key]}")

    def node_ids(self) -> list[str]:
        """The nodes taking part right now: the master (if it holds experts) first."""
        own = [self.settings.node_id] if self.own_experts else []
        return own + [r.node_id for r in (self.controller.usable() if self.controller else [])]

    def open_session(self, request: GenerationRequest) -> GenerationSession:
        """Wait for our turn (first come, first served), then get ready to generate.

        Raises Busy (queue full), NotReady (still loading / experts missing) or ValueError (bad request).
        The caller must use `session.events()` or call `session.close()`, which lets the next one in.
        """
        if not self.loaded:
            raise NotReady(self.error or "the master is still starting")
        self.queue.enter()
        try:
            self._prepare()
            request_id = uuid.uuid4().hex[:8]
            session = GenerationSession(self.model, self.tokenizer, request, request_id, self._session_closed,
                                        lambda work: self._run_request(request_id, work))
            room = self.max_context - len(session.prompt_tokens)
            if room <= 0:
                raise ValueError(f"the prompt is {len(session.prompt_tokens)} tokens; this model reads at most "
                                 f"{self.max_context}")
            request.max_new_tokens = min(request.max_new_tokens, room)
        except BaseException:
            self.queue.leave()
            raise
        session.started = time.perf_counter()
        return session

    def _run_request(self, request_id: str, work):
        """Run generate() with the request id set, and count a worker that stopped answering as dead."""
        with request_scope(request_id):
            try:
                return work()
            except NodeError as exc:
                if exc.unreachable:
                    self.controller.call_failed(exc.node_id, str(exc))
                self.log(f"request {request_id} stopped: {exc}")
                raise

    def _session_closed(self, session: GenerationSession):
        seconds = time.perf_counter() - session.started
        self.log(f"request {session.request_id} done: {len(session.output_tokens)} tokens in {seconds:.1f} s "
                 f"({session.finish_reason or 'stopped early'})")
        self.queue.leave()

    @property
    def max_context(self) -> int:
        return self.model.config.max_position_embeddings

    def generate(self, messages: list[dict], max_new_tokens: int) -> dict:
        """The whole answer at once, greedy (used by /internal/generate and the tests)."""
        session = self.open_session(GenerationRequest(messages, max_new_tokens))
        for _ in session.events():
            pass
        seconds = time.perf_counter() - session.started
        return {
            "request_id": session.request_id,
            "prompt_tokens": session.prompt_tokens,
            "output_tokens": session.output_tokens,
            "output_text": session.text,
            "finish_reason": session.finish_reason,
            "seconds": round(seconds, 3),
            "tokens_per_s": round(len(session.output_tokens) / seconds, 2),
        }

    def stats(self) -> dict:
        calls = {node_id: node.calls for node_id, node in self.local_nodes.items()}
        if self.controller:
            calls.update({r.node_id: r.remote.calls for r in self.controller.records()})
        total = Counter(self._past_stats)
        for d in self.dispatchers.values():
            total.update(d.stats)
        return {"calls_per_node": calls, "bytes_sent": total["bytes_sent"], "bytes_received": total["bytes_received"]}

    def stop(self):
        if self.controller:
            self.controller.stop()
        self._executor.shutdown(wait=False)
