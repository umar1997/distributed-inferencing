"""One log format for every node, so a token can be followed across machines.

Every line has the time, the node, the level and, inside a request, `req=`, `token=` and `layer=`:

    12:04:01.512 master   DEBUG req=a1b2 token=14 layer=6 sending [3,7] → worker-1, [12] → local
    12:04:01.516 worker-1 DEBUG req=a1b2 token=14 layer=6 ran experts [3,7] on 1 token in 4.1 ms

- LOG_LEVEL=info: requests received / finished (with a summary), registration, health changes.
- LOG_LEVEL=debug: also one line per token per layer, on the master and the workers.
- LOG_FORMAT=json: the same fields as one JSON object per line, for later analysis.

`req` and `token` come from the request context (experts/dispatcher.py `current_request`) when the
line is written inside a request; code can also pass them, and `layer`, as `extra={...}`.
"""

import json
import logging
import sys
import time

LOGGER_NAME = "moe"
CONTEXT_FIELDS = ("req", "token", "layer")


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


class _RequestContext(logging.Filter):
    """Fills in req / token from the current request, unless the line already has them."""

    def filter(self, record):
        from moe_dist.experts.dispatcher import current_request  # here to avoid an import loop

        info = current_request.get()
        if info is not None:
            if getattr(record, "req", None) is None:
                record.req = info.request_id
            if getattr(record, "token", None) is None and info.token_index >= 0:
                record.token = info.token_index
        return True


def _clock(record) -> str:
    return time.strftime("%H:%M:%S", time.localtime(record.created)) + f".{int(record.msecs):03d}"


class PrettyFormatter(logging.Formatter):
    def __init__(self, node_id: str):
        super().__init__()
        self.node_id = node_id

    def format(self, record):
        context = " ".join(f"{k}={getattr(record, k)}" for k in CONTEXT_FIELDS if getattr(record, k, None) is not None)
        line = f"{_clock(record)} {self.node_id:<8s} {record.levelname:<5s} {context + ' ' if context else ''}" \
               f"{record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonFormatter(logging.Formatter):
    def __init__(self, node_id: str):
        super().__init__()
        self.node_id = node_id

    def format(self, record):
        data = {"time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)) + f".{int(record.msecs):03d}",
                "node": self.node_id, "level": record.levelname.lower(), "msg": record.getMessage()}
        for k in CONTEXT_FIELDS:
            if getattr(record, k, None) is not None:
                data[k] = getattr(record, k)
        data.update(getattr(record, "fields", None) or {})  # extra numbers, e.g. {"wait_ms": 9.8}
        if record.exc_info:
            data["error"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def setup_logging(node_id: str, level: str = "info", fmt: str = "pretty", stream=None) -> logging.Logger:
    """Send this node's log lines to stdout (what `docker logs` shows), in the chosen format."""
    if fmt not in ("pretty", "json"):
        raise ValueError(f"LOG_FORMAT must be 'pretty' or 'json', not {fmt!r}")
    if level.lower() not in ("debug", "info", "warning", "error"):
        raise ValueError(f"LOG_LEVEL must be debug, info, warning or error, not {level!r}")
    logger = get_logger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(PrettyFormatter(node_id) if fmt == "pretty" else JsonFormatter(node_id))
    handler.addFilter(_RequestContext())
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger


def info_logger():
    """The default `log(text)` function of the services: an info line."""
    logger = get_logger()
    return lambda text: logger.info(text)
