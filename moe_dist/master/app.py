"""The master service (Phase 2.1: health + a simple generate call; the OpenAI API comes in 2.3).

    GET  /health               ready or still starting, missing experts, nodes
    POST /internal/generate    {"messages": [...], "max_new_tokens": 64} -> tokens, text, speed
"""

import platform
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from moe_dist.config import Settings
from moe_dist.master.engine import Engine


class GenerateRequest(BaseModel):
    messages: list[dict]
    max_new_tokens: int = 64


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    engine = Engine(settings, log=lambda text: print(f"{settings.node_id}: {text}", flush=True))

    @asynccontextmanager
    async def lifespan(app):
        threading.Thread(target=engine.start, daemon=True).start()
        yield

    app = FastAPI(title="moe master", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/health")
    def health():
        return {
            "status": "error" if engine.error else "ok",
            "role": "master",
            "node_id": settings.node_id,
            "model_id": settings.model_id,
            "model_revision": settings.revision(),
            "dtype": settings.dtype,
            "platform": f"{platform.system()}-{platform.machine()}".lower(),  # e.g. linux-aarch64
            "combine_mode": settings.combine_mode,
            "ready": engine.ready,
            "error": engine.error,
            "missing_experts": sum(len(ids) for ids in engine.missing.values()),
            "nodes": list(engine.nodes),
            **engine.stats(),
        }

    @app.post("/internal/generate")
    async def generate(request: GenerateRequest):
        if not engine.ready:
            raise HTTPException(503, engine.error or "the master is still starting")
        try:
            return await run_in_threadpool(engine.generate, request.messages, request.max_new_tokens)
        except RuntimeError as exc:  # e.g. a worker stopped answering
            raise HTTPException(502, str(exc)) from exc

    return app
