"""All settings, read from environment variables in one place (docs/IMPLEMENTATION.md, section 3).

Example: NODE_ROLE=worker NODE_ID=worker-1 EXPERT_IDS=10-20 MASTER_ADDR=master:8000 python -m moe_dist.serve

On a laptop the same names are kept in `node.env` at the top of the repo (one NAME=value per line,
written by `./moe setup`, see docs/LAPTOP_SETUP.md). Real environment variables win over the file.
MOE_ENV_FILE=<path> reads another file; MOE_ENV_FILE= (empty) reads none (the tests do this).
"""

import os
import socket
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from moe_dist.models.catalog import DEFAULT_MODEL, pinned_revision

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO / "node.env"


def env_file() -> Path | None:
    """The settings file to read: MOE_ENV_FILE if set (empty -> none), else <repo>/node.env."""
    path = os.environ.get("MOE_ENV_FILE", str(DEFAULT_ENV_FILE))
    return Path(path) if path else None


def http_url(addr: str) -> str:
    """'192.168.1.10:8000' or 'http://master:8000/' -> 'http://192.168.1.10:8000' / 'http://master:8000'."""
    addr = addr.strip().rstrip("/")
    return addr if addr.startswith(("http://", "https://")) else f"http://{addr}"


def lan_ips() -> list[tuple[str, str]]:
    """(network card, IP) of every Wi-Fi / Ethernet card that has an address, e.g. [("en0", "192.168.1.21")].

    Only cards with a broadcast address count: that leaves out the loopback card and VPN or
    company security tunnels (point-to-point cards), which other laptops cannot reach.
    """
    import re
    import subprocess
    import sys

    found = []
    try:
        if sys.platform == "darwin":
            card = ""
            for line in subprocess.run(["/sbin/ifconfig"], capture_output=True, text=True, timeout=5).stdout.splitlines():
                if line and not line[0].isspace():
                    card = line.split(":")[0]
                elif (m := re.search(r"inet (\d+\.\d+\.\d+\.\d+) .*broadcast", line)):
                    found.append((card, m.group(1)))
        else:  # Linux
            out = subprocess.run(["ip", "-4", "-o", "addr", "show", "scope", "global"], capture_output=True, text=True,
                                 timeout=5).stdout
            for line in out.splitlines():
                if " brd " in line and (m := re.search(r"^\d+: (\S+)\s+inet (\d+\.\d+\.\d+\.\d+)", line)):
                    found.append((m.group(1), m.group(2)))
    except (OSError, subprocess.SubprocessError):
        pass
    return found


def local_ip(toward: str = "") -> str:
    """This computer's IP address on the network that leads to `toward` (e.g. the master's address).

    Opening a UDP socket "towards" an address sends nothing; it only makes the operating system pick
    the network card it would use, and that card's address is what other laptops can reach. If that
    card is a VPN tunnel (not a Wi-Fi / Ethernet card) and there is exactly one Wi-Fi / Ethernet card,
    that card's address is used instead.
    """
    host = (toward.split("://")[-1].split("/")[0].rsplit(":", 1)[0] if toward else "") or "192.168.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((host, 9))
            ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    cards = lan_ips()
    if ip not in {addr for _, addr in cards} and len(cards) == 1:
        return cards[0][1]
    return ip


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    def __init__(self, **values):
        values.setdefault("_env_file", env_file())
        super().__init__(**values)

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
    ui_dir: str = ""  # master: the built web page (empty -> <repo>/ui/dist, if it was built)
    torch_threads: int = 0  # 0 -> PyTorch's default (all cores)
    log_level: str = "info"  # info / debug (debug = one line per token per layer)
    log_format: str = "pretty"  # pretty / json

    # Worker: where to register, and the address the master should use to reach this worker.
    master_addr: str = ""  # e.g. "192.168.1.10:8000" or "http://master:8000"
    # "auto" -> this laptop's current IP, found again at every start (Wi-Fi can hand out a new one)
    advertise_addr: str = ""  # e.g. "192.168.1.21:8100" or "auto"; empty -> http://<NODE_ID>:<LISTEN_PORT>
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
        if self.advertise_addr.strip().lower() == "auto":
            return http_url(f"{local_ip(self.master_addr)}:{self.listen_port}")
        return http_url(self.advertise_addr or f"{self.node_id}:{self.listen_port}")
