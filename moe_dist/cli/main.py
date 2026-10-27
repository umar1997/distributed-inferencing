"""`moe-node` (on a laptop: `./moe`): set up a node, start it, look at the cluster, or take a worker out.

    ./moe setup                                     questions -> node.env (docs/LAPTOP_SETUP.md)
    ./moe download                                  download the model now (setup does this too)
    ./moe serve                                     start this node (master or worker) from node.env
    ./moe join [--master 192.168.1.10:8000 --experts 0-15 --advertise auto]   start a worker
    ./moe status [--master localhost:8000]          table of nodes, health, round trips, experts
    ./moe leave                                     (on a worker) say goodbye, then stop

Settings come from environment variables and node.env (moe_dist/config.py); options win over both.
In the Docker image `moe-node` is installed as a command; elsewhere `python -m moe_dist.cli` works too.
"""

import os
from pathlib import Path

import httpx
import typer

from moe_dist.config import Settings, http_url
from moe_dist.experts.placement import format_expert_ids
from moe_dist.transport.client import auth_headers

app = typer.Typer(help="Distributed MoE inference: one command for every node.", no_args_is_help=True)


@app.command()
def serve():
    """Start this node (master or worker) from node.env and the environment variables."""
    from moe_dist.serve import main

    main()


@app.command()
def setup(
    role: str = typer.Option(None, help="master or worker"),
    master: str = typer.Option(None, help="Worker: the master's address, e.g. 192.168.1.10 or 192.168.1.10:8000"),
    token: str = typer.Option(None, help="The cluster token (shared password); the master makes one if empty"),
    experts: str = typer.Option(None, help="Experts this laptop holds, e.g. 0-9 or 0,4,8"),
    name: str = typer.Option(None, help="This laptop's name, e.g. laptop-b"),
    port: int = typer.Option(None, help="Port (default 8000 on the master, 8100 on a worker)"),
    ip: str = typer.Option(None, help="Worker: address the master uses to reach it (default 'auto')"),
    laptops: int = typer.Option(None, help="Master: how many laptops take part, itself included (for the suggested split)"),
    workers_after: int = typer.Option(None, help="Worker: how many more workers will be set up after this one"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Take the suggested answer for every question not given"),
    download: bool = typer.Option(True, help="Download the model now (--no-download: at the first start)"),
    env_file: Path = typer.Option(None, help="Where to save (default: node.env at the top of the repo)"),
):
    """Ask the few questions that set up this laptop, check the answers, and save them to node.env."""
    from moe_dist.cli.setup import run

    run(env_file, yes=yes, download=download, role=role, master=master, token=token, experts=experts, name=name,
        port=port, ip=ip, laptops=laptops, workers_after=workers_after)


@app.command()
def download():
    """Download the model in node.env (or MODEL_ID) now, so the first start does not wait for it."""
    from moe_dist.prepare import main

    main()


@app.command()
def join(
    master: str = typer.Option("", help="Where the master is, e.g. 192.168.1.10:8000 (default: MASTER_ADDR)"),
    experts: str = typer.Option("", help="Experts to hold in every layer, e.g. 0-15 (default: EXPERT_IDS)"),
    advertise: str = typer.Option("", help="Address the master uses to reach this worker, e.g. 192.168.1.21:8100 "
                                           "or auto (default: ADVERTISE_ADDR)"),
    node_id: str = typer.Option("", help="This worker's name (default: NODE_ID or the computer's name)"),
):
    """Start a worker. Without options it uses node.env (written by `./moe setup`); options win over it."""
    s = Settings()
    master, experts = master or s.master_addr, experts or (s.expert_ids if s.expert_ids != "none" else "")
    if not master or not experts:
        typer.echo("join needs the master's address and the experts: run ./moe setup first, or pass "
                   "--master 192.168.1.10:8000 --experts 0-15", err=True)
        raise typer.Exit(1)
    os.environ.update(NODE_ROLE="worker", MASTER_ADDR=master, EXPERT_IDS=experts)
    if advertise:
        os.environ["ADVERTISE_ADDR"] = advertise
    if node_id or ("NODE_ID" not in os.environ and s.node_id == "node"):
        os.environ["NODE_ID"] = node_id or os.uname().nodename.split(".")[0]
    if "LISTEN_PORT" not in os.environ and s.listen_port == 8000:  # 8000 is the master's
        os.environ["LISTEN_PORT"] = "8100"
    serve()


@app.command()
def status(master: str = typer.Option("", help="The master's address (default: MASTER_ADDR or localhost:8000)")):
    """Show every node, its health state, round trips and experts, and whether all experts are covered.

    RTT now = typical round trip of the last 5 pings (what "slow" is judged on); RTT p95 = 95 % of
    the last 50 pings were faster than this.
    """
    url = http_url(master or Settings().master_addr or "localhost:8000")
    try:
        answer = httpx.get(f"{url}/api/nodes", timeout=5, trust_env=False).raise_for_status().json()
    except httpx.HTTPError as exc:
        typer.echo(f"cannot reach the master at {url}: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1)
    rows = [("NODE", "STATUS", "ADDRESS", "RTT now", "RTT p95", "RAM free", "EXPERTS", "REASON")]
    for n in answer["nodes"]:
        rtt, p95 = (f"{n[k]:.1f} ms" if n.get(k) is not None else "-" for k in ("rtt_now_ms", "rtt_p95_ms"))
        ram = f"{n['ram_free_gb']} GB" if n.get("ram_free_gb") is not None else "-"
        rows.append((n["node_id"], n["status"], n.get("address") or "(master)", rtt, p95, ram,
                     format_expert_ids(n["experts"]["all_layers"]), n.get("reason", "")))
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]) - 1)]
    for row in rows:
        typer.echo("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)) + "  " + str(row[-1]))
    coverage = answer["coverage"]
    if coverage == "ok":
        typer.echo("\ncoverage: OK (every expert is held by a usable node)")
    elif coverage == "loading":
        typer.echo("\ncoverage: unknown (the master is still loading)")
    else:
        typer.echo(f"\ncoverage: MISSING experts {coverage['missing']} "
                   f"(held by unavailable nodes: {coverage['unavailable_nodes'] or 'none'})")
        raise typer.Exit(2)


@app.command()
def leave(port: int = typer.Option(0, help="This worker's port (default: LISTEN_PORT)")):
    """Run inside a worker: tell the master this worker is leaving, then stop the worker."""
    s = Settings()
    url = f"http://127.0.0.1:{port or s.listen_port}/internal/leave"
    try:
        answer = httpx.post(url, headers=auth_headers(s.cluster_token), timeout=10, trust_env=False).raise_for_status().json()
    except httpx.HTTPError as exc:
        typer.echo(f"could not ask the worker at {url} to leave: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"left the cluster (master confirmed: {answer['master_confirmed']}); the worker is stopping")


if __name__ == "__main__":
    app()
