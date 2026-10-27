"""The master's side of the network: talk to workers over HTTP.

`RemoteNode` looks exactly like a LocalNode to the dispatcher (`forward(bytes) -> bytes`), but
sends the message to a worker's `/internal/expert/forward`. Each RemoteNode keeps its connection
open between calls ("keep-alive"), so a call does not pay for setting up a new connection.

A failed call is tried again (`retries`) only when the worker did not answer or said "not ready"
(503). That is safe because workers are stateless: the same message always gives the same answer.
"""

import time

import httpx

from moe_dist.transport.codec import CONTENT_TYPE


class NodeError(RuntimeError):
    """A worker could not be reached, timed out, or answered with an error.

    `unreachable` is True when the worker did not answer (or answered "not ready"): a sign that it
    is gone, as opposed to a bad request, which is a bug and says nothing about its health.
    """

    def __init__(self, message: str, node_id: str = "", unreachable: bool = False):
        super().__init__(message)
        self.node_id = node_id
        self.unreachable = unreachable


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


class RemoteNode:
    def __init__(self, node_id: str, base_url: str, token: str = "", timeout_s: float = 5.0, retries: int = 0):
        self.node_id = node_id
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.calls = 0
        # trust_env=False: never send cluster traffic through a proxy from the laptop's settings (a company
        # proxy cannot reach other laptops on the local network).
        self._http = httpx.Client(base_url=self.base_url, headers=auth_headers(token), timeout=timeout_s, trust_env=False)

    def forward(self, message: bytes) -> bytes:
        for attempt in range(self.retries + 1):
            try:
                return self._forward_once(message)
            except NodeError as exc:
                if not exc.unreachable or attempt == self.retries:
                    raise

    def _forward_once(self, message: bytes) -> bytes:
        self.calls += 1
        try:
            reply = self._http.post("/internal/expert/forward", content=message, headers={"Content-Type": CONTENT_TYPE})
        except httpx.HTTPError as exc:
            raise NodeError(f"{self.node_id} ({self.base_url}) did not answer: {type(exc).__name__}: {exc}",
                            self.node_id, unreachable=True) from exc
        if reply.status_code != 200:
            raise NodeError(f"{self.node_id} answered {reply.status_code}: {reply.text[:300]}",
                            self.node_id, unreachable=reply.status_code == 503)
        return reply.content

    def ping(self, timeout_s: float) -> float:
        """One small round trip to the worker; returns the time it took in ms (raises httpx.HTTPError)."""
        start = time.perf_counter()
        self._http.post("/internal/ping", json={"sent_at": time.time()}, timeout=timeout_s).raise_for_status()
        return (time.perf_counter() - start) * 1000

    def health(self) -> dict:
        return self._http.get("/health").raise_for_status().json()

    def close(self):
        self._http.close()
