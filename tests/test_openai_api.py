"""Phase 2.3: the OpenAI-style API (/v1/models, /v1/chat/completions with streaming).

Checked with the official `openai` Python client and with raw HTTP. One small cluster is shared
by all tests in this file (master with experts 0-15 + one worker with 16-31, in this process),
so the model is in memory only once.
"""

import json
import threading
import time
from pathlib import Path

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from moe_dist.config import Settings
from moe_dist.master.app import create_app
from moe_dist.master.generation import Busy, RequestQueue, find_stop
from moe_dist.master.openai_api import message_text
from moe_dist.models.catalog import DEFAULT_MODEL, short_name
from tests.servers import start_master, start_worker, status_of, wait_for

GOLDEN = json.loads((Path(__file__).resolve().parent / "golden" / short_name(DEFAULT_MODEL) / "bf16.json").read_text())


# --- no model needed -----------------------------------------------------------------------------

def test_queue_lets_requests_in_one_at_a_time_in_arrival_order():
    q, order = RequestQueue(max_waiting=5), []
    q.enter()  # the first request is running

    def later(name):
        q.enter()
        order.append(name)
        time.sleep(0.05)
        q.leave()

    threads = []
    for name in ("b", "c", "d"):
        threads.append(threading.Thread(target=later, args=(name,)))
        threads[-1].start()
        wait_for(lambda: q.waiting == len(threads), 5, "waiting")
    assert q.running and q.waiting == 3
    q.leave()
    for t in threads:
        t.join()
    assert order == ["b", "c", "d"] and not q.running and q.waiting == 0


def test_queue_refuses_when_full():
    q = RequestQueue(max_waiting=1)
    q.enter()
    t = threading.Thread(target=lambda: (q.enter(), q.leave()))
    t.start()
    wait_for(lambda: q.waiting == 1, 5, "one waiting")
    with pytest.raises(Busy, match="1 requests are already waiting"):
        q.enter()
    q.leave()
    t.join()


def test_small_helpers():
    assert find_stop("Paris is nice. Rome too.", [".", "Rome"]) == 13
    assert find_stop("no stop here", ["\n\n"]) == -1
    assert message_text([{"type": "text", "text": "Hello "}, {"type": "image_url"}, {"type": "text", "text": "you"}]) \
        == "Hello you"
    assert message_text("plain") == "plain"


def test_api_key_and_not_ready_errors_use_openai_format():
    app = create_app(Settings(node_role="master", node_id="master", api_key="secret"), log=lambda text: None)
    client = TestClient(app)  # without "with": the model is not loaded, so the master is "starting"
    assert client.get("/v1/models").status_code == 401
    answer = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert answer.status_code == 200 and answer.json()["data"][0]["id"] == DEFAULT_MODEL
    reply = client.post("/v1/chat/completions", headers={"Authorization": "Bearer secret"},
                        json={"messages": [{"role": "user", "content": "Hi"}]})
    assert reply.status_code == 503 and reply.json()["error"]["code"] == "not_ready"
    bad = client.post("/v1/chat/completions", headers={"Authorization": "Bearer secret"},
                      json={"messages": [{"role": "user", "content": "Hi"}], "temperature": 5})
    assert bad.status_code == 400 and "temperature" in bad.json()["error"]["message"]


# --- with the model ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cluster():
    with start_master("0-15") as m, start_worker("worker-1", "16-31", master_url=m.url) as w:
        wait_for(lambda: m.app.state.engine.ready and status_of(m, "worker-1") == "healthy", 120, "cluster ready")
        yield m, w


@pytest.fixture(scope="module")
def client(cluster):
    master, _ = cluster
    return openai.OpenAI(base_url=f"{master.url}/v1", api_key="not-needed", max_retries=0)


def ask(client, prompt, **options):
    return client.chat.completions.create(model=DEFAULT_MODEL, messages=prompt["messages"], **options)


@pytest.mark.model
def test_models_list(client):
    assert [m.id for m in client.models.list()] == [DEFAULT_MODEL]


@pytest.mark.model
def test_temperature_zero_gives_the_golden_text(client):
    for saved in GOLDEN["prompts"][:3]:
        answer = ask(client, saved, temperature=0, max_tokens=GOLDEN["max_new_tokens"])
        assert answer.choices[0].message.content == saved["output_text"], saved["id"]
        assert answer.usage.completion_tokens == len(saved["output_tokens"])
        assert answer.usage.prompt_tokens == len(saved["prompt_tokens"])
        assert answer.choices[0].finish_reason == ("length" if len(saved["output_tokens"]) == GOLDEN["max_new_tokens"]
                                                   else "stop")


@pytest.mark.model
def test_streaming_gives_the_same_text_piece_by_piece(client):
    saved = GOLDEN["prompts"][1]
    pieces, finish = [], None
    stream = ask(client, saved, temperature=0, max_tokens=GOLDEN["max_new_tokens"], stream=True,
                 stream_options={"include_usage": True})
    usage = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        for choice in chunk.choices:
            pieces.append(choice.delta.content or "")
            finish = choice.finish_reason or finish
    assert "".join(pieces) == saved["output_text"]
    assert len([p for p in pieces if p]) > 10  # many small pieces, not one block
    assert finish in ("stop", "length") and usage.completion_tokens == len(saved["output_tokens"])


@pytest.mark.model
def test_raw_stream_arrives_as_sse_lines_one_by_one_and_ends_with_done(cluster):
    master, _ = cluster
    body = {"messages": [{"role": "user", "content": "Write a long story about a lighthouse."}],
            "temperature": 0, "max_tokens": 30, "stream": True}
    arrivals, lines = [], []
    with httpx.stream("POST", f"{master.url}/v1/chat/completions", json=body, timeout=120) as reply:
        assert reply.headers["content-type"].startswith("text/event-stream")
        for line in reply.iter_lines():
            if line:
                lines.append(line)
                arrivals.append(time.perf_counter())
    assert all(line.startswith("data: ") for line in lines)
    assert lines[-1] == "data: [DONE]"
    assert len(lines) > 20  # role chunk + ~30 pieces + finish chunk + [DONE]
    assert arrivals[-2] - arrivals[1] > 0.1  # spread over time: sent while generating


@pytest.mark.model
def test_stop_text_ends_the_answer_and_is_left_out(client):
    saved = GOLDEN["prompts"][1]
    text = saved["output_text"]
    word = text.split()[5]  # a word from the middle of the golden answer
    expected = text[:text.find(word)]
    answer = ask(client, saved, temperature=0, max_tokens=GOLDEN["max_new_tokens"], stop=[word])
    assert answer.choices[0].message.content == expected and answer.choices[0].finish_reason == "stop"
    pieces = [c.choices[0].delta.content or "" for c in
              ask(client, saved, temperature=0, max_tokens=GOLDEN["max_new_tokens"], stop=word, stream=True)
              if c.choices]
    assert "".join(pieces) == expected


@pytest.mark.model
def test_max_tokens_and_finish_reason_length(client):
    answer = ask(client, GOLDEN["prompts"][0], temperature=0, max_tokens=5)
    assert answer.usage.completion_tokens == 5 and answer.choices[0].finish_reason == "length"


@pytest.mark.model
def test_sampling_with_a_seed_can_be_repeated(client):
    prompt = {"messages": [{"role": "user", "content": "Write one sentence about the sea."}]}
    first = ask(client, prompt, temperature=0.9, top_p=0.95, seed=7, max_tokens=25).choices[0].message.content
    again = ask(client, prompt, temperature=0.9, top_p=0.95, seed=7, max_tokens=25).choices[0].message.content
    other = ask(client, prompt, temperature=0.9, top_p=0.95, seed=8, max_tokens=25).choices[0].message.content
    assert first == again and first != other


@pytest.mark.model
def test_openai_style_errors(client):
    with pytest.raises(openai.NotFoundError, match="is not served here"):
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])
    with pytest.raises(openai.BadRequestError, match="only n=1"):
        ask(client, GOLDEN["prompts"][0], n=2)


@pytest.mark.model
def test_requests_wait_in_line_and_all_get_answers(cluster, client):
    master, _ = cluster
    prompts = GOLDEN["prompts"][:3]
    results, spans = {}, {}

    def run(saved):
        start = time.perf_counter()
        results[saved["id"]] = ask(client, saved, temperature=0, max_tokens=GOLDEN["max_new_tokens"])
        spans[saved["id"]] = (start, time.perf_counter())

    threads = [threading.Thread(target=run, args=(p,)) for p in prompts]
    for t in threads:
        t.start()
    wait_for(lambda: httpx.get(f"{master.url}/health").json()["queue"]["waiting"] >= 1, 30, "someone waiting")
    for t in threads:
        t.join()
    for saved in prompts:  # waiting in line does not change any answer
        assert results[saved["id"]].choices[0].message.content == saved["output_text"]
    assert httpx.get(f"{master.url}/health").json()["queue"] == {"running": False, "waiting": 0}


@pytest.mark.model
def test_a_client_that_goes_away_frees_the_master(cluster, client):
    master, _ = cluster
    body = {"messages": [{"role": "user", "content": "Write a long story about a lighthouse."}],
            "temperature": 0, "max_tokens": 400, "stream": True}
    with httpx.stream("POST", f"{master.url}/v1/chat/completions", json=body, timeout=120) as reply:
        for i, _ in enumerate(reply.iter_lines()):
            if i > 6:
                break  # leave in the middle of the answer
    wait_for(lambda: not master.app.state.engine.queue.running, 30, "master free again")
    assert master.logs.has("(stopped early)")
    answer = ask(client, GOLDEN["prompts"][0], temperature=0, max_tokens=3)
    assert answer.usage.completion_tokens == 3


@pytest.mark.model
def test_a_worker_failing_mid_stream_keeps_the_text_and_ends_with_an_error(cluster):
    """Runs last in this file: it stops the worker."""
    master, worker = cluster
    body = {"messages": [{"role": "user", "content": "Write a long story about a lighthouse."}],
            "temperature": 0, "max_tokens": 400, "stream": True}
    lines = []
    with httpx.stream("POST", f"{master.url}/v1/chat/completions", json=body, timeout=120) as reply:
        for line in reply.iter_lines():
            if line:
                lines.append(line)
            if len(lines) == 8:
                threading.Thread(target=worker.stop).start()
    texts = [json.loads(line[6:]) for line in lines[:-1]]
    assert lines[-1] == "data: [DONE]"
    assert "error" in texts[-1] and texts[-1]["error"]["code"] == "worker_failed"
    assert "worker-1 stopped responding" in texts[-1]["error"]["message"]
    assert sum(1 for t in texts if t.get("choices") and t["choices"][0]["delta"].get("content")) >= 6  # text kept
    reply = httpx.post(f"{master.url}/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi"}]})
    assert reply.status_code == 503 and reply.json()["error"]["details"]["unavailable_nodes"] == {"worker-1": "dead"}
