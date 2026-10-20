"""The master's list of workers and their health (docs/IMPLEMENTATION.md, section 5).

Workers register themselves and then send a heartbeat ("I'm alive") every few seconds. The master
also pings every worker on its own timer, so it notices a frozen worker even when that worker
cannot send anything. From these contacts each worker gets one of five states:

    joining  registered, still loading weights (or not yet enough good contacts)
    healthy  normal
    slow     answers, but late (typical round trip above SLOW_RTT_MS, or heartbeats overdue)
    dead     no contact for DEAD_AFTER_S, or an expert call failed after retries
    left     said goodbye

Only healthy and slow workers get work. A dead or departed worker is used again only after it
registers again (it does that by itself when the master answers its heartbeat with "reregister").

Every state change is logged: "worker-2 status: healthy → dead (no contact for 15.2 s)".
"""

import statistics
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

from moe_dist.experts.placement import format_expert_ids
from moe_dist.experts.slicing import check_expert_ids
from moe_dist.transport.client import RemoteNode

JOINING, HEALTHY, SLOW, DEAD, LEFT = "joining", "healthy", "slow", "dead", "left"
USABLE = (HEALTHY, SLOW)
SLOW_WINDOW = 5  # "slow" looks at the median of this many recent pings
HEARTBEAT_LATE_AFTER = 3  # heartbeat intervals without one (= late by more than 2) -> "slow"


@dataclass
class NodeRecord:
    node_id: str
    url: str
    experts: list[int]
    remote: RemoteNode
    info: dict = field(default_factory=dict)  # RAM, CPU, software version, as reported
    status: str = JOINING
    reason: str = "registered"
    ready: bool = False
    error: str | None = None
    busy: bool = False
    calls_served: int = 0
    registered_at: float = 0.0
    last_contact: float = 0.0  # last successful heartbeat or ping (monotonic clock)
    last_heartbeat: float = 0.0
    ok_streak: int = 0  # good contacts in a row
    rtts: deque = field(default_factory=lambda: deque(maxlen=50))  # recent ping round trips, ms

    def rtt_now(self) -> float | None:
        """The typical round trip of the last few pings: what the "slow" rule looks at."""
        recent = list(self.rtts)[-SLOW_WINDOW:]
        return round(statistics.median(recent), 2) if recent else None

    def rtt(self, q: float) -> float | None:
        """A percentile of the recent round trips (q=0.5 is the typical one)."""
        if not self.rtts:
            return None
        ordered = sorted(self.rtts)
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)


class Controller:
    def __init__(self, settings, num_experts: int, log=print, clock=time.monotonic):
        self.settings = settings
        self.num_experts = num_experts
        self.log = log
        self.clock = clock
        self.epoch = uuid.uuid4().hex[:12]  # new on every master start; workers notice and register again
        self.nodes: dict[str, NodeRecord] = {}
        self.version = 0  # goes up whenever the set of usable workers changes
        self.events: deque = deque(maxlen=200)  # (time, node_id, old, new, reason), newest last
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._pinging: set[str] = set()
        self._refused: dict[str, str] = {}  # node_id -> last refusal reason
        self._pinger = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ping")

    # --- what workers send -------------------------------------------------------------------

    def register(self, payload: dict) -> dict:
        reason = self._refusal(payload)
        node_id = payload.get("node_id", "?")
        if reason:
            if self._refused.get(node_id) != reason:  # the worker keeps retrying: log each reason once
                self.log(f"refused registration of {node_id}: {reason}")
                self._refused[node_id] = reason
            return {"accepted": False, "reason": reason}
        self._refused.pop(node_id, None)
        s = self.settings
        with self._lock:
            now = self.clock()
            url = payload["advertise_addr"].rstrip("/")
            old = self.nodes.get(node_id)
            # Same address: keep the open connection (the dispatcher may be using it right now).
            remote = old.remote if old and old.url == url else RemoteNode(
                node_id, url, s.cluster_token, s.expert_call_timeout_s, s.expert_call_retries)
            record = NodeRecord(node_id, url, sorted(payload["experts"]["all_layers"]), remote,
                                info=payload.get("info", {}), registered_at=now, last_contact=now, last_heartbeat=now)
            if old:
                record.status = old.status  # so the log shows e.g. "dead → joining"
                remote.calls = old.remote.calls if remote is old.remote else 0
            self.nodes[node_id] = record
            self.version += 1
            first = "registered again" if old else f"registered from {url}, experts {format_expert_ids(record.experts)}"
            self._set(record, JOINING, first, force_log=old is None)
            self._contact(record, payload, now, good=True)
        return {"accepted": True, "master_epoch": self.epoch, "heartbeat_interval_s": s.heartbeat_interval_s}

    def heartbeat(self, payload: dict) -> dict:
        with self._lock:
            record = self.nodes.get(payload.get("node_id"))
            if record is None or payload.get("master_epoch") != self.epoch or record.status in (DEAD, LEFT):
                return {"ok": False, "reregister": True}
            now = self.clock()
            record.last_heartbeat = now
            self._contact(record, payload, now, good=True)
        return {"ok": True}

    def leave(self, node_id: str) -> dict:
        with self._lock:
            record = self.nodes.get(node_id)
            if record is None:
                return {"ok": False, "reason": f"unknown node {node_id}"}
            self._set(record, LEFT, "said goodbye")
        return {"ok": True}

    # --- what the master sees itself ---------------------------------------------------------

    def record_ping(self, record: NodeRecord, rtt_ms: float | None, problem: str = ""):
        with self._lock:
            if self.nodes.get(record.node_id) is not record or record.status in (DEAD, LEFT):
                return  # registered again meanwhile, or no longer counted
            now = self.clock()
            if rtt_ms is None:
                record.ok_streak = 0
                record.reason = f"ping failed: {problem}"
            else:
                record.rtts.append(rtt_ms)
                record.last_contact = now
                record.ok_streak = record.ok_streak + 1 if rtt_ms <= self.settings.slow_rtt_ms else 0
            self._evaluate(record, now)

    def call_failed(self, node_id: str, problem: str):
        """An expert call failed even after the retries: the worker is treated as dead at once."""
        with self._lock:
            record = self.nodes.get(node_id)
            if record is not None and record.status not in (DEAD, LEFT):
                self._set(record, DEAD, f"expert call failed: {problem}")

    def tick(self):
        """Apply the time-based rules (dead after silence, slow when heartbeats are overdue)."""
        with self._lock:
            now = self.clock()
            for record in self.nodes.values():
                self._evaluate(record, now)

    # --- answers for the engine, /api/nodes and the logs -------------------------------------

    def usable(self) -> list[NodeRecord]:
        with self._lock:
            return sorted((r for r in self.nodes.values() if r.status in USABLE), key=lambda r: r.node_id)

    def records(self) -> list[NodeRecord]:
        with self._lock:
            return sorted(self.nodes.values(), key=lambda r: r.node_id)

    def describe(self) -> list[dict]:
        now = self.clock()
        with self._lock:
            return [
                {
                    "node_id": r.node_id,
                    "role": "worker",
                    "address": r.url,
                    "status": r.status,
                    "reason": r.reason,
                    "ready": r.ready,
                    "error": r.error,
                    "experts": {"all_layers": r.experts},
                    "last_contact_s_ago": round(now - r.last_contact, 1),
                    "rtt_now_ms": r.rtt_now(),  # median of the last 5 pings
                    "rtt_p50_ms": r.rtt(0.5),  # over the last 50 pings
                    "rtt_p95_ms": r.rtt(0.95),
                    "busy": r.busy,
                    "calls_served": r.calls_served,
                    **r.info,
                }
                for r in self.records()
            ]

    # --- background pings ------------------------------------------------------------------

    def start_monitor(self):
        threading.Thread(target=self._monitor, name="health-monitor", daemon=True).start()

    def stop(self):
        self._stop.set()
        self._pinger.shutdown(wait=False, cancel_futures=True)

    def _monitor(self):
        next_ping = 0.0
        while not self._stop.wait(0.2):
            self.tick()
            if time.monotonic() >= next_ping:
                next_ping = time.monotonic() + self.settings.ping_interval_s
                for record in self.records():
                    if record.status not in (DEAD, LEFT) and record.node_id not in self._pinging:
                        self._pinging.add(record.node_id)
                        self._pinger.submit(self._ping, record)

    def _ping(self, record: NodeRecord):
        try:
            self.record_ping(record, record.remote.ping(self.settings.ping_timeout_s))
        except httpx.HTTPError as exc:
            self.record_ping(record, None, type(exc).__name__)
        finally:
            self._pinging.discard(record.node_id)

    # --- the rules -------------------------------------------------------------------------

    def _refusal(self, payload: dict) -> str:
        s = self.settings
        node_id = payload.get("node_id") or ""
        for key, what, mine in (("model_id", "model", s.model_id), ("model_revision", "model revision", s.revision()),
                                ("dtype", "number format (DTYPE)", s.dtype)):
            if payload.get(key) != mine:
                return f"{what} differs: you have {payload.get(key)!r}, the cluster uses {mine!r}"
        if not node_id:
            return "no node_id"
        if node_id == s.node_id:
            return f"node id {node_id!r} is the master's own id; pick another NODE_ID"
        if not str(payload.get("advertise_addr", "")).startswith("http"):
            return f"bad advertise_addr {payload.get('advertise_addr')!r}"
        try:
            ids = payload["experts"]["all_layers"]
            if not ids:
                return "holds no experts (set EXPERT_IDS)"
            check_expert_ids(ids, self.num_experts)
        except (KeyError, TypeError, ValueError) as exc:
            return f"bad expert list: {exc}"
        with self._lock:
            old = self.nodes.get(node_id)
            if old and old.status in USABLE and old.url != payload["advertise_addr"].rstrip("/"):
                return (f"node id {node_id!r} is already used by a live node at {old.url} "
                        f"(it becomes free when that node is dead or has left)")
        return ""

    def _contact(self, record: NodeRecord, payload: dict, now: float, good: bool):
        record.last_contact = now
        record.ready = bool(payload.get("ready"))
        record.error = payload.get("error")
        record.busy = bool(payload.get("busy"))
        record.calls_served = payload.get("calls_served", record.calls_served)
        if "ram_free_gb" in payload:
            record.info["ram_free_gb"] = payload["ram_free_gb"]
        record.ok_streak = record.ok_streak + 1 if good else 0
        self._evaluate(record, now)

    def _evaluate(self, record: NodeRecord, now: float):
        if record.status in (DEAD, LEFT):
            return  # only a new registration brings it back
        silent = now - record.last_contact
        if silent > self.settings.dead_after_s:
            self._set(record, DEAD, f"no contact for {silent:.1f} s")
            return
        if not record.ready:
            self._set(record, JOINING, record.error or "still loading experts")
            return
        slow = self._slow_reason(record, now)
        if slow:
            self._set(record, SLOW, slow)
        elif record.status == HEALTHY or record.ok_streak >= self.settings.recover_after_ok:
            self._set(record, HEALTHY, f"{record.ok_streak} good contacts in a row")
        else:
            record.reason = f"{record.ok_streak} of {self.settings.recover_after_ok} good contacts in a row"

    def _slow_reason(self, record: NodeRecord, now: float) -> str:
        s = self.settings
        recent = list(record.rtts)[-SLOW_WINDOW:]
        if recent and statistics.median(recent) > s.slow_rtt_ms:
            return f"typical round trip {statistics.median(recent):.0f} ms > {s.slow_rtt_ms:.0f} ms"
        quiet = now - record.last_heartbeat
        if quiet > HEARTBEAT_LATE_AFTER * s.heartbeat_interval_s:
            return f"heartbeat late: last one {quiet:.1f} s ago"
        return ""

    def _set(self, record: NodeRecord, status: str, reason: str, force_log: bool = False):
        old = record.status
        record.reason = reason
        if old == status and not force_log:
            return
        record.status = status
        if (old in USABLE) != (status in USABLE):
            self.version += 1
        self.events.append((time.time(), record.node_id, old, status, reason))
        self.log(f"{record.node_id} status: {old} → {status} ({reason})" if old != status
                 else f"{record.node_id} status: {status} ({reason})")

