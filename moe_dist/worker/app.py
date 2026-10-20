"""The worker service: holds some experts and runs them for the master.

Workers are stateless: they keep only expert weights and remember nothing between calls.

    POST /internal/expert/forward   binary message in, binary message out (docs/IMPLEMENTATION.md 4.2)
    GET  /health                    who I am, which experts I hold, ready or still loading
"""

import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from transformers import AutoConfig

from moe_dist.config import Settings
from moe_dist.experts.dispatcher import serve_forward
from moe_dist.experts.loader import load_experts, model_dir
from moe_dist.experts.placement import parse_expert_ids
from moe_dist.models import adapter_for
from moe_dist.transport.codec import CONTENT_TYPE, CodecError


class WorkerState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.started = time.time()
        self.parts = None  # {layer: ExpertSlice} once loaded
        self.act_fn = None
        self.expert_ids: list[int] = []
        self.error: str | None = None
        self.calls = 0

    def load(self):
        """Read only this worker's experts from the weight file (downloads the model if needed)."""
        try:
            s = self.settings
            adapter = adapter_for(AutoConfig.from_pretrained(model_dir(s.model_id, s.revision())))
            self.expert_ids = parse_expert_ids(s.expert_ids, adapter.num_experts)
            if not self.expert_ids:
                raise ValueError("a worker needs EXPERT_IDS (e.g. '10-20')")
            self.act_fn = adapter.act_fn()
            start = time.perf_counter()
            self.parts = load_experts(s.model_id, s.revision(), self.expert_ids, s.dtype)
            mb = sum(p.nbytes() for p in self.parts.values()) / 2**20
            print(f"{s.node_id}: loaded experts {s.expert_ids} for {len(self.parts)} layers "
                  f"({mb:.0f} MB, {s.dtype}) in {time.perf_counter() - start:.1f} s", flush=True)
        except Exception as exc:  # shown in /health instead of a silent crash
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"{self.settings.node_id}: loading failed: {self.error}", flush=True)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    state = WorkerState(settings)

    @asynccontextmanager
    async def lifespan(app):
        threading.Thread(target=state.load, daemon=True).start()
        yield

    app = FastAPI(title=f"moe worker {settings.node_id}", lifespan=lifespan)
    app.state.worker = state

    def check_token(request: Request):
        if settings.cluster_token and request.headers.get("authorization") != f"Bearer {settings.cluster_token}":
            raise HTTPException(401, "wrong or missing cluster token")

    @app.get("/health")
    def health():
        return {
            "status": "error" if state.error else "ok",
            "role": "worker",
            "node_id": settings.node_id,
            "model_id": settings.model_id,
            "model_revision": settings.revision(),
            "dtype": settings.dtype,
            "experts": {"all_layers": state.expert_ids},
            "ready": state.parts is not None,
            "error": state.error,
            "calls_served": state.calls,
            "uptime_s": round(time.time() - state.started, 1),
        }

    @app.post("/internal/expert/forward")
    async def forward(request: Request):
        check_token(request)
        if state.parts is None:
            raise HTTPException(503, f"{settings.node_id} is not ready: {state.error or 'still loading experts'}")
        body = await request.body()
        try:
            # The maths runs in a worker thread, so /health stays responsive meanwhile.
            reply = await run_in_threadpool(serve_forward, settings.node_id, state.parts, state.act_fn, body)
        except (CodecError, ValueError, KeyError) as exc:
            raise HTTPException(400, f"{settings.node_id}: bad request: {type(exc).__name__}: {exc}") from exc
        state.calls += 1
        return Response(reply, media_type=CONTENT_TYPE)

    return app
