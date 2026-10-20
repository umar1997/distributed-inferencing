"""Start this node from environment variables: `python -m moe_dist.serve`.

NODE_ROLE=master or worker decides which service runs (settings: moe_dist/config.py).
The `moe-node serve` command (later) does the same.
"""

import torch
import uvicorn

from moe_dist.config import Settings


def main():
    settings = Settings()
    if settings.torch_threads:
        # Several containers on one laptop: each uses only its share of the CPU cores.
        torch.set_num_threads(settings.torch_threads)
    if settings.node_role == "master":
        from moe_dist.master.app import create_app
    elif settings.node_role == "worker":
        from moe_dist.worker.app import create_app
    else:
        raise SystemExit(f"NODE_ROLE must be 'master' or 'worker', not {settings.node_role!r}")
    print(f"{settings.node_id}: starting as {settings.node_role} on port {settings.listen_port}, "
          f"torch threads {torch.get_num_threads()}", flush=True)
    uvicorn.run(create_app(settings), host="0.0.0.0", port=settings.listen_port, log_level="warning")


if __name__ == "__main__":
    main()
