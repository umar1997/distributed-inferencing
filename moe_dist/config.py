"""All settings, read from environment variables in one place (docs/IMPLEMENTATION.md, section 3).

Example: NODE_ROLE=worker NODE_ID=worker-1 EXPERT_IDS=10-20 python -m moe_dist.serve
"""

from pydantic_settings import BaseSettings

from moe_dist.models.catalog import DEFAULT_MODEL, pinned_revision


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
    # Phase 2.1: fixed worker list, "worker-1=http://worker-1:8100,worker-2=http://worker-2:8100".
    # Registration replaces this in Phase 2.2.
    workers: str = ""
    workers_ready_timeout_s: float = 600  # master: how long to wait for workers to finish loading
    expert_call_timeout_s: float = 5
    torch_threads: int = 0  # 0 -> PyTorch's default (all cores)

    def revision(self) -> str:
        return self.model_revision or pinned_revision(self.model_id)

    def worker_urls(self) -> dict[str, str]:
        urls = {}
        for item in filter(None, (x.strip() for x in self.workers.split(","))):
            node_id, sep, url = item.partition("=")
            if not sep or not node_id or not url.startswith("http"):
                raise ValueError(f"WORKERS entry {item!r} should look like 'worker-1=http://worker-1:8100'")
            urls[node_id.strip()] = url.strip().rstrip("/")
        return urls
