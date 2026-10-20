"""The master service.

    GET  /v1/models, POST /v1/chat/completions   the OpenAI-style API (master/openai_api.py)

    GET  /health                 loaded / ready, missing experts, nodes taking part
    GET  /api/nodes              every node with its health state, round trips, RAM, experts
    GET  /api/placement          layer -> expert -> node, and the coverage result
    POST /internal/register      a worker joins           (cluster token needed)
    POST /internal/heartbeat     a worker says "I'm alive" (cluster token needed)
    POST /internal/leave         a worker says goodbye     (cluster token needed)
    POST /internal/generate      {"messages": [...], "max_new_tokens": 64} -> tokens, text, speed
"""

import platform
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from moe_dist import __version__
from moe_dist.config import Settings
from moe_dist.master.engine import Engine, NotReady
from moe_dist.master.generation import Busy
from moe_dist.master.openai_api import error, make_router
from moe_dist.transport.client import NodeError


class GenerateRequest(BaseModel):
    messages: list[dict]
    max_new_tokens: int = 64


def create_app(settings: Settings | None = None, log=None) -> FastAPI:
    settings = settings or Settings()
    if log is None:
        log = lambda text: print(f"{settings.node_id}: {text}", flush=True)  # noqa: E731
    engine = Engine(settings, log=log)

    @asynccontextmanager
    async def lifespan(app):
        threading.Thread(target=engine.start, daemon=True).start()
        yield
        engine.stop()

    app = FastAPI(title="moe master", lifespan=lifespan)
    app.state.engine = engine
    app.include_router(make_router(engine, settings))

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError):
        # OpenAI clients expect {"error": {...}}, not FastAPI's {"detail": [...]}.
        problems = "; ".join(f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
        return error(400, problems or "bad request")

    def controller(request: Request):
        """The controller, after checking the cluster token (wrong token -> 401, logged)."""
        if settings.cluster_token and request.headers.get("authorization") != f"Bearer {settings.cluster_token}":
            host = request.client.host if request.client else "?"
            log(f"refused {request.url.path} from {host}: wrong or missing cluster token")
            raise HTTPException(401, "wrong or missing cluster token")
        if engine.controller is None:
            raise HTTPException(503, engine.error or "the master is starting")
        return engine.controller

    @app.get("/health")
    def health():
        problem = engine.missing_report()
        return {
            "status": "error" if engine.error else "ok",
            "role": "master",
            "node_id": settings.node_id,
            "model_id": settings.model_id,
            "model_revision": settings.revision(),
            "dtype": settings.dtype,
            "platform": f"{platform.system()}-{platform.machine()}".lower(),  # e.g. linux-aarch64
            "software_version": __version__,
            "combine_mode": settings.combine_mode,
            "health_rules": {k: getattr(settings, k) for k in (
                "heartbeat_interval_s", "ping_interval_s", "slow_rtt_ms", "dead_after_s", "recover_after_ok",
                "expert_call_timeout_s", "expert_call_retries")},
            "master_epoch": engine.controller.epoch if engine.controller else None,
            "loaded": engine.loaded,
            "ready": engine.ready,
            "error": engine.error,
            "missing_experts": sum(len(ids) for ids in problem["missing"].values()) if problem else 0,
            "missing": problem,
            "nodes": engine.node_ids(),
            "queue": {"running": engine.queue.running, "waiting": engine.queue.waiting},
            **engine.stats(),
        }

    @app.get("/api/nodes")
    def nodes():
        own = {
            "node_id": settings.node_id, "role": "master", "address": "", "status": "healthy" if engine.loaded else "joining",
            "reason": engine.error or ("loaded" if engine.loaded else "loading"), "ready": engine.loaded,
            "experts": {"all_layers": engine.own_experts},
        }
        workers = engine.controller.describe() if engine.controller else []
        problem = engine.missing_report()
        return {"nodes": [own, *workers], "coverage": "ok" if engine.loaded and not problem else (problem or "loading")}

    @app.get("/api/placement")
    def placement():
        if not engine.loaded:
            raise HTTPException(503, engine.error or "the master is still starting")
        current, _, _ = engine._current()
        return {
            "layers": current.layers,
            "num_experts": current.num_experts,
            "owners": {str(layer): {str(e): nodes for e, nodes in experts.items()} for layer, experts in current.owners.items()},
            "coverage": engine.missing_report() or "ok",
        }

    @app.post("/internal/register")
    async def register(request: Request):
        return controller(request).register(await request.json())

    @app.post("/internal/heartbeat")
    async def heartbeat(request: Request):
        return controller(request).heartbeat(await request.json())

    @app.post("/internal/leave")
    async def leave(request: Request):
        return controller(request).leave((await request.json()).get("node_id", ""))

    @app.post("/internal/generate")
    async def generate(request: GenerateRequest):
        try:
            return await run_in_threadpool(engine.generate, request.messages, request.max_new_tokens)
        except NotReady as exc:
            return JSONResponse(exc.details, status_code=503)
        except Busy as exc:
            return JSONResponse({"error": str(exc)}, status_code=429)
        except NodeError as exc:  # a worker stopped answering in the middle of the answer
            return JSONResponse({"error": str(exc), "node_id": exc.node_id}, status_code=502)

    return app
