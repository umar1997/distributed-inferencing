"""All settings, read from environment variables in one place (docs/IMPLEMENTATION.md, section 3).

Example: NODE_ROLE=worker NODE_ID=worker-1 EXPERT_IDS=10-20 MASTER_ADDR=master:8000 python -m moe_dist.serve
"""

from pydantic_settings import BaseSettings

from moe_dist.models.catalog import DEFAULT_MODEL, pinned_revision


def http_url(addr: str) -> str:
    """'192.168.1.10:8000' or 'http://master:8000/' -> 'http://192.168.1.10:8000' / 'http://master:8000'."""
    addr = addr.strip().rstrip("/")
    return addr if addr.startswith(("http://", "https://")) else f"http://{addr}"


class Settings(BaseSettings):
    node_role: str = "worker"  # master / worker
    node_id: str = "node"
    listen_port: int = 8000
    expert_ids: str = "none"  # "0-10", "0,4,8,12" or "none"
    model_id: str = DEFAULT_MODEL
    model_revision: str = ""  # empty -> the pinned revision from models/catalog.py
    dtype: str = "bf16"  # fp32 / bf16
    # Where weights are stored: Hugging Face's own HF_HOME variable (Docker sets HF_HOME=/models).
    cluster_token: str = ""  # shared secret for internal calls; empty -> no check (local runs only)
    combine_mode: str = "exact"  # master: exact / fast
    api_key: str = ""  # master: if set, /v1/* needs "Authorization: Bearer <API_KEY>"
    max_queue: int = 16  # master: requests allowed to wait while one runs; more get 429 "busy"
    default_max_tokens: int = 512  # master: answer length when a request does not say
    torch_threads: int = 0  # 0 -> PyTorch's default (all cores)

    # Worker: where to register, and the address the master should use to reach this worker.
    master_addr: str = ""  # e.g. "192.168.1.10:8000" or "http://master:8000"
    advertise_addr: str = ""  # e.g. "192.168.1.21:8100"; empty -> http://<NODE_ID>:<LISTEN_PORT>
    register_retry_max_s: float = 10  # worker: longest wait between tries (1, 2, 4 ... this)

    # Master: health rules (docs/IMPLEMENTATION.md, section 5). The master tells each worker
    # its heartbeat interval when it registers, so this one value is used on both sides.
    heartbeat_interval_s: float = 2
    ping_interval_s: float = 3
    ping_timeout_s: float = 2
    slow_rtt_ms: float = 250  # typical (median) ping round trip above this -> "slow"
    dead_after_s: float = 15  # no successful contact for this long -> "dead"
    recover_after_ok: int = 3  # good contacts in a row needed to become "healthy"
    expert_call_timeout_s: float = 5
    expert_call_retries: int = 1  # retries on the same node before giving up

    def revision(self) -> str:
        return self.model_revision or pinned_revision(self.model_id)

    def master_url(self) -> str:
        return http_url(self.master_addr) if self.master_addr else ""

    def advertise_url(self) -> str:
        return http_url(self.advertise_addr or f"{self.node_id}:{self.listen_port}")
