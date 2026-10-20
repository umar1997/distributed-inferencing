"""The master's side of the network: talk to workers over HTTP.

`RemoteNode` looks exactly like a LocalNode to the dispatcher (`forward(bytes) -> bytes`), but
sends the message to a worker's `/internal/expert/forward`. Each RemoteNode keeps its connection
open between calls ("keep-alive"), so a call does not pay for setting up a new connection.
"""

import time

import httpx

from moe_dist.transport.codec import CONTENT_TYPE


class NodeError(RuntimeError):
    """A worker could not be reached, timed out, or answered with an error."""


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


class RemoteNode:
    def __init__(self, node_id: str, base_url: str, token: str = "", timeout_s: float = 5.0):
        self.node_id = node_id
        self.base_url = base_url.rstrip("/")
        self.calls = 0
        self._http = httpx.Client(base_url=self.base_url, headers=_headers(token), timeout=timeout_s)

    def forward(self, message: bytes) -> bytes:
        self.calls += 1
        try:
            reply = self._http.post("/internal/expert/forward", content=message, headers={"Content-Type": CONTENT_TYPE})
        except httpx.HTTPError as exc:
            raise NodeError(f"{self.node_id} ({self.base_url}) did not answer: {type(exc).__name__}: {exc}") from exc
        if reply.status_code != 200:
            raise NodeError(f"{self.node_id} answered {reply.status_code}: {reply.text[:300]}")
        return reply.content

    def health(self) -> dict:
        return self._http.get("/health").raise_for_status().json()

    def close(self):
        self._http.close()


def wait_until_ready(nodes: list[RemoteNode], timeout_s: float, log=print) -> dict[str, dict]:
    """Ask every worker's /health until all say ready (they may still be loading). Returns their answers."""
    deadline = time.monotonic() + timeout_s
    ready, last_problem = {}, {}
    while True:
        for node in nodes:
            if node.node_id in ready:
                continue
            try:
                answer = node.health()
            except httpx.HTTPError as exc:
                last_problem[node.node_id] = f"{type(exc).__name__}"
                continue
            if answer.get("ready"):
                ready[node.node_id] = answer
                log(f"worker {node.node_id} is ready: experts {answer.get('experts')}")
            else:
                last_problem[node.node_id] = "still loading"
        if len(ready) == len(nodes):
            return ready
        if time.monotonic() > deadline:
            waiting = {n.node_id: last_problem.get(n.node_id) for n in nodes if n.node_id not in ready}
            raise NodeError(f"workers not ready after {timeout_s:.0f} s: {waiting}")
        time.sleep(1)
