"""The worker service: holds some experts and runs them for the master.

Workers are stateless: they keep only expert weights and remember nothing between calls.

    POST /internal/expert/forward   binary message in, binary message out (docs/IMPLEMENTATION.md 4.2)
    POST /internal/ping             {"sent_at"} -> {"sent_at", "worker_time"}: the master's health pings
    POST /internal/leave            say goodbye to the master and stop (used by `moe-node leave`)
    GET  /health                    who I am, which experts I hold, ready or still loading, registered

With MASTER_ADDR set, the worker registers itself and sends heartbeats (worker/membership.py).
"""

import os
import signal
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from transformers import AutoConfig

from moe_dist import __version__
from moe_dist.config import Settings
from moe_dist.logs import info_logger
from moe_dist.experts.dispatcher import serve_forward
from moe_dist.experts.loader import load_experts, model_dir
from moe_dist.experts.placement import parse_expert_ids
from moe_dist.models import adapter_for
from moe_dist.transport.codec import CONTENT_TYPE, CodecError
from moe_dist.worker.membership import Membership


class WorkerState:
    def __init__(self, settings: Settings, log=None):
        self.settings = settings
        self.log = log if log is not None else info_logger()
        self.started = time.time()
        self.parts = None  # {layer: ExpertSlice} once loaded
        self.act_fn = None
        self.expert_ids: list[int] = []
        self.experts_known = threading.Event()  # set once expert_ids is read (before the slow loading)
        self.error: str | None = None
        self.calls = 0
        self.busy = 0  # calls being computed right now

    def load(self):
        """Read only this worker's experts from the weight file (downloads the model if needed)."""
        try:
            s = self.settings
            adapter = adapter_for(AutoConfig.from_pretrained(model_dir(s.model_id, s.revision())))
            self.expert_ids = parse_expert_ids(s.expert_ids, adapter.num_experts)
            if not self.expert_ids:
                raise ValueError("a worker needs EXPERT_IDS (e.g. '10-20')")
            self.experts_known.set()
            self.act_fn = adapter.act_fn()
            start = time.perf_counter()
            self.parts = load_experts(s.model_id, s.revision(), self.expert_ids, s.dtype)
            mb = sum(p.nbytes() for p in self.parts.values()) / 2**20
            self.log(f"loaded experts {s.expert_ids} for {len(self.parts)} layers "
                     f"({mb:.0f} MB, {s.dtype}) in {time.perf_counter() - start:.1f} s")
        except Exception as exc:  # shown in /health and sent to the master instead of a silent crash
            self.error = f"{type(exc).__name__}: {exc}"
            self.log(f"loading failed: {self.error}")


def _stop_process():
    """After `moe-node leave`: end this process shortly (after the answer has been sent)."""
    threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM)).start()


def create_app(settings: Settings | None = None, log=None) -> FastAPI:
    settings = settings or Settings()
    if log is None:
        log = info_logger()
    state = WorkerState(settings, log)
    membership = Membership(settings, state, log) if settings.master_url() else None

    @asynccontextmanager
    async def lifespan(app):
        threading.Thread(target=state.load, daemon=True).start()
        if membership:
            membership.start()
        else:
            log("MASTER_ADDR is not set: not registering with a master")
        yield
        if membership:
            membership.stop()  # no goodbye: a stopped container should look dead to the master

    app = FastAPI(title=f"moe worker {settings.node_id}", lifespan=lifespan)
    app.state.worker = state
    app.state.membership = membership
    app.state.on_leave = _stop_process  # tests replace this so the test process keeps running

    def check_token(request: Request):
        if settings.cluster_token and request.headers.get("authorization") != f"Bearer {settings.cluster_token}":
            host = request.client.host if request.client else "?"
            log(f"refused {request.url.path} from {host}: wrong or missing cluster token")
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
            "software_version": __version__,
            "experts": {"all_layers": state.expert_ids},
            "ready": state.parts is not None,
            "error": state.error,
            "master": settings.master_url() or None,
            "registered": bool(membership and membership.registered),
            "membership_problem": membership.problem if membership else None,
            "calls_served": state.calls,
            "uptime_s": round(time.time() - state.started, 1),
        }

    @app.post("/internal/ping")
    async def ping(request: Request):
        check_token(request)
        body = await request.json()
        return {"sent_at": body.get("sent_at"), "worker_time": time.time()}

    @app.post("/internal/leave")
    def leave(request: Request):
        check_token(request)
        if membership is None:
            raise HTTPException(400, "this worker is not registered with a master (MASTER_ADDR is not set)")
        confirmed = membership.leave()
        app.state.on_leave()
        return {"ok": True, "master_confirmed": confirmed, "stopping": True}

    @app.post("/internal/expert/forward")
    async def forward(request: Request):
        check_token(request)
        if state.parts is None:
            raise HTTPException(503, f"{settings.node_id} is not ready: {state.error or 'still loading experts'}")
        body = await request.body()
        state.busy += 1
        try:
            # The maths runs in a worker thread, so /health and pings stay responsive meanwhile.
            reply = await run_in_threadpool(serve_forward, settings.node_id, state.parts, state.act_fn, body)
        except (CodecError, ValueError, KeyError) as exc:
            raise HTTPException(400, f"{settings.node_id}: bad request: {type(exc).__name__}: {exc}") from exc
        finally:
            state.busy -= 1
        state.calls += 1
        return Response(reply, media_type=CONTENT_TYPE)

    return app
