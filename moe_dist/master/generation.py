"""Generating one answer: settings, streaming of new text, stop words, the waiting line.

`GenerationSession.events()` runs Transformers' `generate()` in a background thread and hands out
new text as soon as each token is ready ("streaming"). With temperature 0 this is exactly the
same greedy decoding as reference.greedy_generate, so the tokens match the golden files.

Terms:
- temperature: how random the choice of the next token is. 0 = always the most likely token.
- top_p: only sample from the most likely tokens that together have probability top_p.
- seed: start value for the random numbers, so a sampled answer can be repeated exactly.
- stop: texts that end the answer when they appear (they are not included in the answer).
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from moe_dist.logs import get_logger
from moe_dist.reference import build_inputs

log = get_logger()


@dataclass
class GenerationRequest:
    messages: list[dict]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


class Busy(RuntimeError):
    """Too many requests are already waiting."""


class RequestQueue:
    """One request at a time; the others wait in arrival order (first come, first served)."""

    def __init__(self, max_waiting: int):
        self.max_waiting = max_waiting
        self._cond = threading.Condition()
        self._next_ticket = 0
        self._serving = 0  # the ticket whose turn it is
        self.running = False

    @property
    def waiting(self) -> int:
        with self._cond:
            return self._next_ticket - self._serving - (1 if self.running else 0)

    def enter(self):
        with self._cond:
            if self.waiting >= self.max_waiting and (self.running or self.waiting > 0):
                raise Busy(f"the master is busy: {self.waiting} requests are already waiting")
            ticket = self._next_ticket
            self._next_ticket += 1
            self._cond.wait_for(lambda: self._serving == ticket and not self.running)
            self.running = True

    def leave(self):
        with self._cond:
            self.running = False
            self._serving += 1
            self._cond.notify_all()


class _TokenStreamer(BaseStreamer):
    """Receives each new token from generate() and puts it on a queue (the first call is the prompt).

    `on_token` (if set) is called right here, in generate()'s own thread, before the next step
    starts: the moment the step that made this token is complete (used for the live UI events).
    """

    def __init__(self, events: queue.Queue, on_token=None):
        self.events = events
        self.on_token = on_token
        self.prompt_seen = False

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        for token in value.reshape(-1).tolist():
            if self.on_token is not None:
                self.on_token(token)
            self.events.put(("token", token))

    def end(self):
        pass


class _StopCheck(StoppingCriteria):
    """Ends generation when the client went away, or when a stop text appears in the answer."""

    def __init__(self, tokenizer, prompt_len: int, stop: list[str], cancelled: threading.Event):
        self.tokenizer, self.prompt_len, self.stop, self.cancelled = tokenizer, prompt_len, stop, cancelled

    def __call__(self, input_ids, scores, **kwargs):
        if self.cancelled.is_set():
            return torch.ones(input_ids.shape[0], dtype=torch.bool)
        if self.stop:
            text = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
            if any(s in text for s in self.stop):
                return torch.ones(input_ids.shape[0], dtype=torch.bool)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool)


def find_stop(text: str, stop: list[str]) -> int:
    """Position of the first stop text in `text`, or -1."""
    found = [i for i in (text.find(s) for s in stop) if i >= 0]
    return min(found) if found else -1


class GenerationSession:
    """One answer being generated. Use `events()` once; `close()` frees the master for the next request."""

    def __init__(self, model, tokenizer, request: GenerationRequest, request_id: str, on_close, run_in_scope):
        self.model, self.tokenizer, self.request = model, tokenizer, request
        self.request_id = request_id
        self.inputs = build_inputs(tokenizer, request.messages)
        self.prompt_tokens = self.inputs["input_ids"][0].tolist()
        self.output_tokens: list[int] = []
        self.text = ""
        self.finish_reason: str | None = None
        self._cancelled = threading.Event()
        self._on_close = on_close
        self._run_in_scope = run_in_scope  # runs generate() with this request's id set (for messages/logs)
        self._closed = False
        self._thread: threading.Thread | None = None
        self.started = 0.0  # set by the engine when it is this request's turn
        self.info = None  # RequestInfo: request id, token number, time per node (set by the engine)
        self.on_token = None  # called in generate()'s thread for each new token (set by the engine)

    def close(self):
        """Stop generating (if still running) and let the next request in. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._cancelled.set()
        if self._thread is not None:
            self._thread.join()
        self._on_close(self)

    def _generate_kwargs(self, events: queue.Queue) -> dict:
        r = self.request
        stop_check = _StopCheck(self.tokenizer, len(self.prompt_tokens), r.stop, self._cancelled)
        kwargs = dict(**self.inputs, max_new_tokens=r.max_new_tokens, streamer=_TokenStreamer(events, self.on_token),
                      stopping_criteria=StoppingCriteriaList([stop_check]))
        if r.temperature == 0:
            # The same settings as reference.greedy_generate: always the most likely token.
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            # top_k=0 turns off Transformers' own default "only the 50 most likely tokens" filter.
            kwargs.update(do_sample=True, temperature=r.temperature, top_p=r.top_p, top_k=0)
        return kwargs

    def events(self):
        """Yields ("text", new_text) while generating; at the end ("done", finish_reason).

        Errors from generate() (e.g. a worker stopped answering) are raised here, after the text so far.
        """
        events: queue.Queue = queue.Queue()
        kwargs = self._generate_kwargs(events)

        def run():
            try:
                if self.request.seed is not None:
                    torch.manual_seed(self.request.seed)  # safe: only one request runs at a time
                self._run_in_scope(lambda: self.model.generate(**kwargs))
                events.put(("end", None))
            except Exception as exc:  # handed to the reading side
                events.put(("error", exc))

        self._thread = threading.Thread(target=run, name=f"generate-{self.request_id}", daemon=True)
        self._thread.start()
        stop = self.request.stop
        hold_back = max((len(s) for s in stop), default=1) - 1  # text that might be the start of a stop text
        sent = 0
        try:
            while True:
                kind, value = events.get()
                if kind == "error":
                    raise value
                if kind == "token":
                    self.output_tokens.append(value)
                    text = self.tokenizer.decode(self.output_tokens, skip_special_tokens=True)
                    if log.isEnabledFor(logging.DEBUG):
                        self._log_token(value)
                    cut = find_stop(text, stop)
                    if cut >= 0:
                        self.text, self.finish_reason = text[:cut], "stop"
                        self._cancelled.set()
                    elif text.endswith("�"):
                        continue  # half of a character (e.g. an emoji split over two tokens): wait
                    else:
                        self.text = text
                        safe = len(text) - hold_back if stop else len(text)
                        if safe > sent:
                            yield "text", text[sent:safe]
                            sent = safe
                        continue
                # end of generation (or a stop text was found)
                if self.finish_reason is None:
                    self._thread.join()
                    self.text = self.tokenizer.decode(self.output_tokens, skip_special_tokens=True)
                    ended_by_eos = self.output_tokens and self.output_tokens[-1] in self._eos_ids()
                    self.finish_reason = "stop" if ended_by_eos or len(self.output_tokens) < self.request.max_new_tokens \
                        else "length"
                if len(self.text) > sent:
                    yield "text", self.text[sent:]
                yield "done", self.finish_reason
                return
        finally:
            self.close()

    def _log_token(self, token: int):
        seconds = time.perf_counter() - self.started
        n = len(self.output_tokens)
        log.debug(f"produced {self.tokenizer.decode([token])!r} (tok/s so far: {n / seconds if seconds else 0:.1f})",
                  # token=N is the forward pass that made it (0 = the pass over the prompt), so this line
                  # matches the layer lines with the same token number.
                  extra={"req": self.request_id, "token": n - 1, "fields": {"token_id": token}})

    def _eos_ids(self) -> set[int]:
        eos = self.model.generation_config.eos_token_id
        return set(eos if isinstance(eos, list) else [eos])
