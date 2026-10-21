"""Live events for the UI: GET /api/events, as Server-Sent Events (SSE).

SSE is a one-way stream over plain HTTP: the server keeps the response open and writes a line
`data: {...}` whenever something happens; browsers read it with `new EventSource(url)` and
reconnect by themselves. (The OpenAI streaming API uses the same technique.)

Events (every one has "type" and "time"):
- hello          sent once on connecting: the model, the nodes and the expert placement
- node_update    all nodes with their health; on every change and every few seconds
- request_start  {req, prompt_tokens, max_tokens}
- token          one per generated token: {req, token_index, text, special, layers: [{layer, experts:
                 [{id, node, weight}], wait_ms, slowest}]}, i.e. all layers of that step in one
                 event (one event per layer would be hundreds per second). special = an end
                 marker like <|end_of_text|>, sent with empty text
- request_end    {req, tokens, seconds, tokens_per_s, finish_reason, node_time_ms, summary}

The hub never blocks generation: each listener has a bounded queue, and a listener that does not
keep up loses events (counted in `dropped`) instead of slowing the model down.
"""

import json
import queue
import threading
import time

KEEPALIVE_S = 15  # an SSE comment line this often, so proxies and browsers keep the connection open


class EventHub:
    def __init__(self, max_queued: int = 1000):
        self.max_queued = max_queued
        self._listeners: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self.dropped = 0

    @property
    def active(self) -> bool:
        """True while someone is listening (the engine only collects routing details then)."""
        return bool(self._listeners)

    def publish(self, kind: str, **data):
        event = {"type": kind, "time": round(time.time(), 3), **data}
        with self._lock:
            listeners = list(self._listeners)
        for q in listeners:
            try:
                q.put_nowait(event)
            except queue.Full:
                self.dropped += 1

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=self.max_queued)
        with self._lock:
            self._listeners.add(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            self._listeners.discard(q)

    def stream(self, first_events: list[dict]):
        """SSE lines for one listener (a plain generator: Starlette runs it in a thread)."""
        q = self.subscribe()
        try:
            for event in first_events:
                yield f"data: {json.dumps(event)}\n\n"
            while True:
                try:
                    event = q.get(timeout=KEEPALIVE_S)
                except queue.Empty:
                    yield ": keep-alive\n\n"  # a comment line; EventSource ignores it
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            self.unsubscribe(q)
