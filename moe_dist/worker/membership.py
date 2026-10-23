"""A worker's side of joining the cluster: register, send heartbeats, register again when needed.

One background thread:
- registers with the master (who I am, where to reach me, which experts, which model),
- then sends a heartbeat every few seconds (the master says how often),
- if the master answers "reregister" (it restarted, or it had marked this worker dead), registers
  again at once,
- if the master cannot be reached, keeps trying, waiting 1, 2, 4 ... up to REGISTER_RETRY_MAX_S
  seconds between tries.

Problems are logged once when they start (not every retry), and once when they are solved.
"""

import os
import platform
import re
import subprocess
import threading
from pathlib import Path

import httpx

from moe_dist import __version__
from moe_dist.logs import info_logger
from moe_dist.transport.client import auth_headers


def machine_info() -> dict:
    """RAM and CPU facts sent at registration (shown by `moe-node status` and the UI)."""
    info = {"cpu_cores": os.cpu_count(), "cpu_arch": platform.machine(), "platform": platform.system().lower(),
            "software_version": __version__}
    try:
        info["ram_total_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    limit = Path("/sys/fs/cgroup/memory.max")  # a container's own memory limit, if it has one
    if limit.exists() and limit.read_text().strip().isdigit():
        info["memory_limit_gb"] = round(int(limit.read_text()) / 2**30, 2)
    return info


def ram_free_gb() -> float | None:
    """Memory that programs can still get, in GB (Linux: MemAvailable; macOS: free + reclaimable pages)."""
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 2**20, 2)
        return None
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        page = int(re.search(r"page size of (\d+)", out).group(1)) if "page size of" in out else 4096
        pages = {m.group(1): int(m.group(2)) for m in re.finditer(r"Pages (\w+(?: \w+)?):\s+(\d+)", out)}
        # free, plus pages macOS hands out again as soon as a program asks (cached files, unused)
        reclaimable = sum(pages.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable"))
        return round(reclaimable * page / 2**30, 2)
    return None


class Membership:
    def __init__(self, settings, state, log=None):
        self.settings = settings
        self.state = state  # WorkerState: ready, expert ids, calls served
        self.log = log if log is not None else info_logger()
        self.master = settings.master_url()
        self.epoch: str | None = None  # set by a successful registration
        self.interval = settings.heartbeat_interval_s
        self.problem: str | None = None  # the current problem, logged once
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="membership", daemon=True)
        self._http = httpx.Client(base_url=self.master, headers=auth_headers(settings.cluster_token), timeout=5)

    @property
    def registered(self) -> bool:
        return self.epoch is not None

    def start(self):
        self._thread.start()

    def stop(self):
        """Stop quietly (no goodbye): to the master this looks like the worker died."""
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=10)

    def leave(self) -> bool:
        """Say goodbye to the master, then stop sending heartbeats."""
        self.stop()
        try:
            ok = self._http.post("/internal/leave", json={"node_id": self.settings.node_id}).is_success
        except httpx.HTTPError:
            ok = False
        self.log(f"left the cluster (master {'confirmed' if ok else 'did not answer'})")
        self.epoch = None
        return ok

    def _run(self):
        # Experts must be known before registering (they come from the model's config).
        while not self.state.experts_known.wait(0.5):
            if self._stop.is_set():
                return
        wait = 1.0
        while not self._stop.is_set():
            outcome = self._heartbeat() if self.registered else self._register()
            if outcome == "again":
                continue  # the master asked us to register again: do it right away
            if outcome == "ok":
                wait = 1.0
                self._stop.wait(self.interval)
            else:
                self._stop.wait(wait)
                wait = min(wait * 2, self.settings.register_retry_max_s)

    def _status(self) -> dict:
        s, st = self.settings, self.state
        return {"node_id": s.node_id, "ready": st.parts is not None, "error": st.error, "busy": st.busy > 0,
                "calls_served": st.calls, "ram_free_gb": ram_free_gb()}

    def _register(self) -> str:
        s = self.settings
        payload = {
            **self._status(),
            "advertise_addr": s.advertise_url(),
            "model_id": s.model_id,
            "model_revision": s.revision(),
            "dtype": s.dtype,
            "experts": {"all_layers": self.state.expert_ids},
            "info": machine_info(),
        }
        answer = self._post("/internal/register", payload, "registration")
        if answer is None:
            return "failed"
        if not answer.get("accepted"):
            self._report(f"registration refused by the master: {answer.get('reason')}")
            return "failed"
        self.epoch = answer["master_epoch"]
        self.interval = answer.get("heartbeat_interval_s", self.interval)
        self._solved()
        self.log(f"registered with master {self.master} as {s.advertise_url()} "
                 f"(master epoch {self.epoch}, heartbeat every {self.interval} s)")
        return "ok"

    def _heartbeat(self) -> str:
        answer = self._post("/internal/heartbeat", {**self._status(), "master_epoch": self.epoch}, "heartbeat")
        if answer is None:
            return "failed"
        self._solved()
        if answer.get("reregister"):
            self.log("master asked to register again (it restarted, or it had counted this worker out)")
            self.epoch = None
            return "again"
        return "ok"

    def _post(self, path: str, payload: dict, what: str) -> dict | None:
        try:
            reply = self._http.post(path, json=payload)
        except httpx.HTTPError as exc:
            self._report(f"{what} failed: master {self.master} did not answer ({type(exc).__name__})")
            return None
        if reply.status_code != 200:
            self._report(f"{what} failed: master answered {reply.status_code}: {reply.text[:200]}")
            return None
        return reply.json()

    def _report(self, problem: str):
        if problem != self.problem:
            self.log(f"{problem} (will keep trying)")
            self.problem = problem

    def _solved(self):
        if self.problem:
            self.log(f"master reachable again (was: {self.problem})")
            self.problem = None
