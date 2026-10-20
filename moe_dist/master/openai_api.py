"""The OpenAI-style API, so any tool that talks to Ollama, vLLM or OpenAI can chat with the cluster.

    GET  /v1/models              the one loaded model
    POST /v1/chat/completions    an answer, whole ("stream": false) or piece by piece ("stream": true)

Streaming uses Server-Sent Events (SSE): the answer arrives as lines `data: {...}` while it is being
generated, and ends with `data: [DONE]`. If a worker stops answering in the middle, the text sent so
far is kept and the stream ends with an error message.

Errors use OpenAI's format: {"error": {"message": ..., "type": ..., "code": ...}}.
If API_KEY is set, requests need the header `Authorization: Bearer <API_KEY>`.
"""

import json
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from moe_dist.master.engine import NotReady
from moe_dist.master.generation import Busy, GenerationRequest
from moe_dist.models.catalog import short_name
from moe_dist.transport.client import NodeError

MAX_STOP_TEXTS = 4  # as in OpenAI's API


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # other OpenAI fields (e.g. presence_penalty) are accepted and ignored

    model: str = ""
    messages: list[dict]
    stream: bool = False
    stream_options: dict | None = None
    temperature: float = Field(1.0, ge=0, le=2)  # OpenAI's default is 1
    top_p: float = Field(1.0, gt=0, le=1)
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)  # newer name for max_tokens
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int = 1


def error(status: int, message: str, kind: str = "invalid_request_error", code: str | None = None, **extra):
    return JSONResponse({"error": {"message": message, "type": kind, "code": code, **extra}}, status_code=status)


def message_text(content) -> str:
    """OpenAI allows the content as a string or as a list of parts ([{"type": "text", "text": ...}])."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return "" if content is None else str(content)


def make_router(engine, settings) -> APIRouter:
    router = APIRouter(prefix="/v1")
    model_names = {settings.model_id, short_name(settings.model_id)}

    def check_key(request: Request):
        if settings.api_key and request.headers.get("authorization") != f"Bearer {settings.api_key}":
            return error(401, "wrong or missing API key", "authentication_error", "invalid_api_key")
        return None

    @router.get("/models")
    def models(request: Request):
        if (refused := check_key(request)) is not None:
            return refused
        return {"object": "list", "data": [{"id": settings.model_id, "object": "model", "created": 0,
                                            "owned_by": "moe-dist"}]}

    @router.post("/chat/completions")
    async def chat(body: ChatRequest, request: Request):
        if (refused := check_key(request)) is not None:
            return refused
        if body.model and body.model not in model_names:
            return error(404, f"model {body.model!r} is not served here; this cluster serves {settings.model_id!r}",
                         code="model_not_found")
        if body.n != 1:
            return error(400, "only n=1 is supported", code="unsupported_parameter")
        stop = [body.stop] if isinstance(body.stop, str) else list(body.stop or [])
        if len(stop) > MAX_STOP_TEXTS or any(not s for s in stop):
            return error(400, f"stop: at most {MAX_STOP_TEXTS} non-empty texts")
        messages = [{"role": m.get("role", "user"), "content": message_text(m.get("content"))} for m in body.messages]
        if not messages:
            return error(400, "messages is empty")
        gen = GenerationRequest(messages, body.max_completion_tokens or body.max_tokens or settings.default_max_tokens,
                                body.temperature, body.top_p, stop, body.seed)

        try:  # waits in line until it is this request's turn
            session = await run_in_threadpool(engine.open_session, gen)
        except Busy as exc:
            return error(429, str(exc), "rate_limit_error", "busy")
        except NotReady as exc:
            return error(503, str(exc), "service_unavailable", "not_ready", details=exc.details)
        except ValueError as exc:
            return error(400, str(exc), code="context_length_exceeded")

        completion_id = f"chatcmpl-{session.request_id}"
        created = int(time.time())

        def usage():
            return {"prompt_tokens": len(session.prompt_tokens), "completion_tokens": len(session.output_tokens),
                    "total_tokens": len(session.prompt_tokens) + len(session.output_tokens)}

        if not body.stream:
            try:
                await run_in_threadpool(lambda: [None for _ in session.events()])
            except NodeError as exc:
                return error(502, str(exc), "server_error", "worker_failed", text_so_far=session.text)
            return {
                "id": completion_id, "object": "chat.completion", "created": created, "model": settings.model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": session.text},
                             "finish_reason": session.finish_reason}],
                "usage": usage(),
            }

        def chunk(delta: dict, finish_reason=None) -> str:
            data = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": settings.model_id, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
            return f"data: {json.dumps(data)}\n\n"

        def stream():
            # A plain (not async) generator: Starlette runs it in a thread, so generating does not
            # block the web server. If the client goes away, Starlette closes it, which ends generation.
            yield chunk({"role": "assistant", "content": ""})
            try:
                for kind, value in session.events():
                    if kind == "text":
                        yield chunk({"content": value})
                    else:
                        yield chunk({}, finish_reason=value)
            except NodeError as exc:
                failure = {"message": str(exc), "type": "server_error", "code": "worker_failed"}
                yield f"data: {json.dumps({'error': failure})}\n\n"
            if (body.stream_options or {}).get("include_usage"):
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': settings.model_id, 'choices': [], 'usage': usage()})}\n\n"
            yield "data: [DONE]\n\n"

        # close() again after the response, in case the client left before streaming started.
        return StreamingResponse(stream(), media_type="text/event-stream", background=BackgroundTask(session.close),
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return router
