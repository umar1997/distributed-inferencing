"""`./moe setup`: a few questions that turn a fresh laptop into the master or a worker.

The answers are checked (can I reach the master? is the cluster token right? do the experts fit in
memory?) and saved to `node.env` at the top of the repo, one NAME=value per line. Every later
`./moe serve` / `./moe join` reads that file, so the questions are asked only once. Running setup
again offers the saved answers as defaults. Guide: docs/LAPTOP_SETUP.md.

Every question can also be answered with an option (`./moe setup --role worker --master ...`),
and `--yes` takes the suggested answer for anything not given, so it can run without a person.
"""

import os
import secrets
from pathlib import Path

import httpx
import typer

from moe_dist.config import DEFAULT_ENV_FILE, http_url, lan_ips, local_ip
from moe_dist.experts.placement import format_expert_ids, parse_expert_ids
from moe_dist.models.catalog import DEFAULT_MODEL, pinned_revision
from moe_dist.transport.client import auth_headers

# What node.env holds, in this order (names as in moe_dist/config.py).
KEYS = ["NODE_ROLE", "NODE_ID", "LISTEN_PORT", "EXPERT_IDS", "MASTER_ADDR", "ADVERTISE_ADDR", "CLUSTER_TOKEN",
        "MODEL_ID", "DTYPE"]
WORKER_OVERHEAD_GB = 0.5  # Python + PyTorch + Transformers (measured ~0.35 GB, docs/AVAILABLE_MODELS.md 4.3)
MASTER_OVERHEAD_GB = 1.0  # the same, plus attention memory of a long conversation


def read_env(path: Path) -> dict[str, str]:
    """NAME=value lines -> {NAME: value}; blank lines and # comments are skipped."""
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def write_env(path: Path, values: dict[str, str]):
    lines = ["# This laptop's settings, written by `./moe setup` (docs/LAPTOP_SETUP.md).",
             "# Edit by hand or run `./moe setup` again. Never commit this file: it holds the cluster token.", ""]
    lines += [f"{k}={values[k]}" for k in KEYS if values.get(k) not in (None, "")]
    lines += [f"{k}={v}" for k, v in values.items() if k not in KEYS and v not in (None, "")]  # kept from before
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)  # only this user can read the token


def split_experts(ids: list[int], parts: int) -> list[list[int]]:
    """Neighbouring, near-equal shares; earlier shares get the extra ones. 32 in 3 -> 11, 11, 10."""
    parts = max(1, min(parts, len(ids))) if ids else 1
    size, extra = divmod(len(ids), parts)
    shares, start = [], 0
    for i in range(parts):
        end = start + size + (1 if i < extra else 0)
        shares.append(ids[start:end])
        start = end
    return shares


def free_experts(nodes: list[dict], num_experts: int, me: str) -> list[int]:
    """Experts nobody else holds (dead or departed nodes, and this laptop's own old entry, don't count)."""
    held = {e for n in nodes if n["node_id"] != me and n["status"] not in ("dead", "left")
            for e in n["experts"]["all_layers"]}
    return [e for e in range(num_experts) if e not in held]


def model_numbers(model_id: str, revision: str, dtype: str) -> dict:
    """Downloads the model if needed, then measures it: experts per layer, GB per expert (all layers
    together, in `dtype`) and GB of shared parts (everything that is not an expert; master only)."""
    from transformers import AutoConfig

    from moe_dist.experts.loader import WeightFiles, model_dir
    from moe_dist.models import adapter_for

    folder = model_dir(model_id, revision)
    adapter = adapter_for(AutoConfig.from_pretrained(folder))
    layers = adapter.moe_layers()
    gate_up, down = adapter.read_experts(WeightFiles(folder), layers[0], [0])  # one expert of one layer
    numbers = gate_up.numel() + down.numel()
    file_bytes = numbers * gate_up.element_size() * len(layers)
    files_total = sum(f.stat().st_size for f in folder.glob("*.safetensors"))
    scale = {"fp32": 4, "bf16": 2}[dtype] / gate_up.element_size()
    return {"folder": folder, "num_experts": adapter.num_experts, "layers": len(layers),
            "expert_gb": file_bytes * scale / 2**30,
            "shared_gb": (files_total - file_bytes * adapter.num_experts) * scale / 2**30}


class Wizard:
    def __init__(self, yes: bool):
        self.yes = yes

    def ask(self, question: str, default: str = "", check=None, secret: bool = False) -> str:
        """One question; Enter takes the default. `check(answer)` returns an error text or None."""
        while True:
            if self.yes:
                answer = default
                typer.echo(f"{question}: {'*' * 8 if secret and answer else answer}")
            else:
                answer = typer.prompt(question, default=default or None, hide_input=secret,
                                      show_default=not secret or not default).strip()
            problem = check(answer) if check else None
            if not problem:
                return answer
            typer.secho(f"  ✗ {problem}", fg="red")
            if self.yes:
                raise typer.Exit(1)

    @staticmethod
    def ok(text: str):
        typer.secho(f"  ✓ {text}", fg="green")

    @staticmethod
    def warn(text: str):
        typer.secho(f"  ! {text}", fg="yellow")


def run(env_path: Path | None = None, yes: bool = False, download: bool = True, **given):
    """The questions, in order. `given` = answers passed as options (None = ask)."""
    env_path = Path(env_path or os.environ.get("MOE_ENV_FILE") or DEFAULT_ENV_FILE)
    old = read_env(env_path)
    w = Wizard(yes)
    typer.secho("Setting up this laptop for the MoE cluster (docs/LAPTOP_SETUP.md).", bold=True)
    if old:
        typer.echo(f"Found earlier answers in {env_path}; Enter keeps each one.")
    typer.echo("")

    def pick(option: str, key: str, fallback: str) -> str:
        """Default for a question: the option given, else the saved answer, else `fallback`."""
        value = given.get(option)
        return str(value) if value is not None else old.get(key, fallback)

    role = w.ask("Is this laptop the master or a worker? (master / worker)", pick("role", "NODE_ROLE", "worker"),
                 lambda a: None if a in ("master", "worker") else "answer master or worker")
    master = role == "master"
    values = dict(old, NODE_ROLE=role)
    model_id, dtype = old.get("MODEL_ID", DEFAULT_MODEL), old.get("DTYPE", "bf16")

    if master:
        values.pop("MASTER_ADDR", None)
        values.pop("ADVERTISE_ADDR", None)
    else:
        def reach(answer: str) -> str | None:
            nonlocal health
            if not answer:
                return "type the master's IP address, e.g. 192.168.1.10"
            url = http_url(answer if ":" in answer.split("://")[-1] else f"{answer}:8000")
            try:
                health = httpx.get(f"{url}/health", timeout=5, trust_env=False).raise_for_status().json()
            except httpx.HTTPError as exc:
                return (f"cannot reach the master at {url} ({type(exc).__name__}: {exc}). Is it running (`./moe serve`)? "
                        "Are both laptops on the same Wi-Fi? See 'Problems and fixes' in docs/LAPTOP_SETUP.md")
            return None

        health = {}
        addr = w.ask("Master's address (its IP, e.g. 192.168.1.10; port 8000 unless you changed it)",
                     pick("master", "MASTER_ADDR", ""), reach)
        values["MASTER_ADDR"] = addr if ":" in addr.split("://")[-1] else f"{addr}:8000"
        model_id, dtype = health["model_id"], health["dtype"]
        w.ok(f"reached master '{health['node_id']}': model {model_id}, {dtype}, "
             f"{'loaded' if health['loaded'] else 'still loading'}")
        if health["model_revision"] != pinned_revision(model_id):
            w.warn(f"the master runs model version {health['model_revision'][:12]}, this code pins "
                   f"{pinned_revision(model_id)[:12]}: pull the same code on both laptops (`git pull`)")
    values.update(MODEL_ID=model_id, DTYPE=dtype)

    # --- cluster token: the shared password between master and workers ---
    if master:
        token = w.ask("Cluster token (shared password; every worker types the same one; Enter = this new random one)",
                      pick("token", "CLUSTER_TOKEN", "") or secrets.token_urlsafe(12))
    else:
        def token_ok(answer: str) -> str | None:
            if not answer:
                return "type the token shown when the master was set up (it is in the master's node.env)"
            try:
                r = httpx.post(f"{http_url(values['MASTER_ADDR'])}/internal/heartbeat", json={"node_id": "setup-check"},
                               headers=auth_headers(answer), timeout=5, trust_env=False)
            except httpx.HTTPError as exc:
                return f"could not check the token: {exc}"
            return "the master says this token is wrong" if r.status_code == 401 else None

        token = w.ask("Cluster token (the same one as on the master)", pick("token", "CLUSTER_TOKEN", ""), token_ok,
                      secret=True)
        w.ok("the master accepts this token")
    values["CLUSTER_TOKEN"] = token

    # --- name and port ---
    host = os.uname().nodename.split(".")[0].lower()
    values["NODE_ID"] = w.ask("Name for this laptop (shown in logs and on the web page; must be unique)",
                              pick("name", "NODE_ID", "master" if master else host),
                              lambda a: None if a and " " not in a else "one word, no spaces, e.g. laptop-b")
    values["LISTEN_PORT"] = w.ask("Port to listen on", pick("port", "LISTEN_PORT", "8000" if master else "8100"),
                                  lambda a: None if a.isdigit() and 1024 <= int(a) <= 65535 else "a number, 1024-65535")

    # --- this laptop's address ---
    cards = lan_ips()
    if master:
        here = local_ip()
        typer.echo(f"  This laptop's address{'es' if len(cards) > 1 else ''}: "
                   f"{', '.join(f'{ip} ({card})' for card, ip in cards) or here}. Workers will need it.")
    else:
        here = local_ip(values["MASTER_ADDR"])
        values["ADVERTISE_ADDR"] = w.ask(
            f"This laptop's address for the master to reach it (Enter = 'auto': found at every start, now {here})",
            pick("ip", "ADVERTISE_ADDR", "auto"))
        if values["ADVERTISE_ADDR"] != "auto" and ":" not in values["ADVERTISE_ADDR"]:
            values["ADVERTISE_ADDR"] += f":{values['LISTEN_PORT']}"

    # --- download the model and measure it ---
    if not download:
        numbers = None
        w.warn("model download skipped (--no-download): the first start will download it")
    else:
        typer.echo(f"\nGetting the model {model_id} (only the first time; it is several GB)...")
        try:
            numbers = model_numbers(model_id, pinned_revision(model_id), dtype)
        except Exception as exc:  # network, disk, certificates
            typer.secho(f"  ✗ could not download the model: {type(exc).__name__}: {exc}\n"
                        "    See 'Problems and fixes' in docs/LAPTOP_SETUP.md, then run ./moe setup again.", fg="red")
            raise typer.Exit(1)
        w.ok(f"model ready in {numbers['folder']}")
    num_experts = numbers["num_experts"] if numbers else 32

    # --- which experts ---
    every = list(range(num_experts))
    if master:
        laptops = w.ask("How many laptops will take part, this one included?", pick("laptops", "", "3"),
                        lambda a: None if a.isdigit() and int(a) >= 1 else "a number, 1 or more")
        shares = split_experts(every, int(laptops))
        suggestion = format_expert_ids(shares[0])
        if len(shares) > 1:
            typer.echo("  Suggested split (each worker's setup suggests its share again):")
            for i, share in enumerate(shares):
                typer.echo(f"    {'this master' if i == 0 else f'worker {i}':12} experts {format_expert_ids(share)}")
    else:
        try:
            nodes = httpx.get(f"{http_url(values['MASTER_ADDR'])}/api/nodes", timeout=5, trust_env=False).raise_for_status().json()
            free = free_experts(nodes["nodes"], num_experts, values["NODE_ID"])
        except httpx.HTTPError:
            free = every
            w.warn("could not ask the master which experts are free; suggesting all of them")
        if free:
            typer.echo(f"  Experts no other laptop holds yet: {format_expert_ids(free)}")
        else:
            typer.echo("  Every expert is already held by another laptop (this one would hold backup copies).")
        later = w.ask("How many more workers will you set up after this one? (they share the free experts)",
                      pick("workers_after", "", "0"), lambda a: None if a.isdigit() else "a number, 0 or more")
        suggestion = format_expert_ids(split_experts(free or every, int(later) + 1)[0])

    def experts_ok(answer: str) -> str | None:
        try:
            parse_expert_ids(answer, num_experts)
        except ValueError as exc:
            return str(exc)
        return None

    saved = old.get("EXPERT_IDS") if old.get("NODE_ROLE") == role else None
    values["EXPERT_IDS"] = w.ask(f"Experts this laptop holds (0-{num_experts - 1}, e.g. 0-9 or 0,4,8 or none)",
                                 str(given["experts"]) if given.get("experts") is not None else saved or suggestion,
                                 experts_ok)
    ids = parse_expert_ids(values["EXPERT_IDS"], num_experts)

    # --- memory check ---
    if numbers:
        from moe_dist.worker.membership import machine_info, ram_free_gb

        need = len(ids) * numbers["expert_gb"] + (numbers["shared_gb"] + MASTER_OVERHEAD_GB if master
                                                  else WORKER_OVERHEAD_GB)
        free_now, total = ram_free_gb(), machine_info().get("ram_total_gb")
        text = (f"{len(ids)} experts x {numbers['expert_gb'] * 2**30 / 1e6:.1f} MB"
                f"{' + shared parts' if master else ''} + program: needs about {need:.1f} GB")
        if free_now is not None and need > free_now:
            w.warn(f"{text}, but only {free_now:.1f} GB is free now (of {total} GB). Close other programs, "
                   "or give this laptop fewer experts.")
        else:
            w.ok(f"{text}; {f'{free_now:.1f} GB free now' if free_now is not None else 'free memory unknown'}"
                 f" (of {total} GB)")

    write_env(env_path, values)
    typer.secho(f"\nSaved to {env_path}.", bold=True)
    if master:
        typer.echo(f"Start the master:   ./moe serve\n"
                   f"Then on each worker run ./moe setup and answer:\n"
                   f"  master's address = {here}:{values['LISTEN_PORT']}\n  cluster token    = {token}\n"
                   f"Web page (once running): http://{here}:{values['LISTEN_PORT']}")
    else:
        typer.echo("Start this worker:  ./moe join\nCheck the cluster:  ./moe status")
