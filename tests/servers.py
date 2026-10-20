"""Helpers for tests that run the real master and worker services inside the test process.

Each service runs with uvicorn in a background thread on a free port, so the tests use real HTTP
without Docker. Health timings are short (FAST) so the failure tests do not wait 15 s each time.
"""

import socket
import threading
import time

import uvicorn

from moe_dist.config import Settings
from moe_dist.master.app import create_app as create_master
from moe_dist.worker.app import create_app as create_worker

TOKEN = "test-token"
# Short health timings for tests (the real defaults are in moe_dist/config.py).
FAST = dict(heartbeat_interval_s=0.3, ping_interval_s=0.3, ping_timeout_s=1, dead_after_s=2, recover_after_ok=2,
            register_retry_max_s=0.5, expert_call_timeout_s=3)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Logs(list):
    """Collects a service's log lines; `.has(text)` checks whether any line contains `text`."""

    def __call__(self, line: str):
        self.append(line)

    def has(self, text: str) -> bool:
        return any(text in line for line in self)


class Server:
    """Run a FastAPI app with uvicorn in a background thread."""

    def __init__(self, app, port, logs=None):
        self.app = app
        self.port = port
        self.logs = logs if logs is not None else Logs()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.url = f"http://127.0.0.1:{port}"

    def __enter__(self):
        self.thread.start()
        while not self.server.started:
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self.stop()

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=15)


def start_worker(node_id, experts, dtype="bf16", master_url="", token=TOKEN, port=None, **settings):
    port = port or free_port()
    s = Settings(node_role="worker", node_id=node_id, listen_port=port, expert_ids=experts, dtype=dtype,
                 cluster_token=token, master_addr=master_url, advertise_addr=f"http://127.0.0.1:{port}",
                 **{**FAST, **settings})
    logs = Logs()
    server = Server(create_worker(s, log=logs), port, logs)
    server.app.state.on_leave = lambda: setattr(server.server, "should_exit", True)
    return server


def start_master(experts="0-9", dtype="bf16", token=TOKEN, port=None, **settings):
    port = port or free_port()
    s = Settings(node_role="master", node_id="master", listen_port=port, expert_ids=experts, dtype=dtype,
                 cluster_token=token, **{**FAST, **settings})
    logs = Logs()
    return Server(create_master(s, log=logs), port, logs)


def wait_for(condition, timeout=120, what="condition"):
    """Poll `condition()` until it returns something true; returns that value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(0.1)
    raise TimeoutError(f"{what} not reached after {timeout} s")


def status_of(master: Server, node_id: str) -> str | None:
    controller = master.app.state.engine.controller
    record = controller.nodes.get(node_id) if controller else None
    return record.status if record else None
