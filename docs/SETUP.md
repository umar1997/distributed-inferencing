# Setup

How to set up the Python environment for this project on a development machine (Steps 1–3 run on the 36 GB Mac). Docker setup comes later, in Phase 2.

> **Setting up a laptop to join the cluster (master or worker, no Docker)?** Use [LAPTOP_SETUP.md](LAPTOP_SETUP.md) instead: `./scripts/install.sh`, then `./moe setup`. To let an AI agent do it: [LAPTOP_HANDOFF.md](LAPTOP_HANDOFF.md).

The environment lives in a folder called **`.venv`** inside the repo. A "virtual environment" is a private copy of Python with its own packages, so this project's packages don't mix with anything else on the machine. `.venv` is listed in `.gitignore` and is never committed; anyone can rebuild it from `requirements.txt`.

---

## 1. What you need first

| Need | Why | Check with |
|---|---|---|
| macOS or Linux | Development machine | — |
| **Python 3.12** | The project needs 3.12. The Mac's built-in Python is 3.9, which is **too old** for these models. | `python3.12 --version` |
| `uv` (recommended) **or** `pip` | Installs the packages | `uv --version` |
| ~5 GB free disk | ~1 GB of packages + 2.7 GB for the Granite model | `df -h ~` |
| Internet | Downloads packages and model weights | — |

Install what's missing (macOS, with [Homebrew](https://brew.sh)):

```bash
brew install uv            # for Option A
brew install python@3.12   # for Option B (Option A downloads Python 3.12 by itself)
```

Run every command below **from the repo folder**:

```bash
cd ~/Desktop/distributed-inferencing
```

---

## 2. Create the environment and install packages

Pick **one** option.

### Option A — uv (recommended, faster)

```bash
# 1. Create .venv with Python 3.12 (uv downloads 3.12 if it isn't installed)
uv venv --python 3.12 .venv

# 2. Install the pinned packages into .venv
uv pip install --python .venv -r requirements.txt
```

### Option B — pip

```bash
# 1. Create .venv with Python 3.12
python3.12 -m venv .venv

# 2. Turn it on for this terminal
source .venv/bin/activate

# 3. Update pip, then install the pinned packages
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Turn the environment on

Every new terminal needs this once:

```bash
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. To turn it off: `deactivate`.

*With uv you can also skip activating and put `uv run` in front of commands, e.g. `uv run python scripts/make_golden.py`.*

---

## 4. Check that it works

```bash
python --version
# Python 3.12.x

python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
# torch 2.14.1 | transformers 5.20.1
```

If you see `Python 3.9` or `transformers 4.40`, the environment is not active: run `source .venv/bin/activate` again.

---

## 5. Download the first model (Granite)

The model is pinned to an exact version (commit), so every machine uses the same weights. See [AVAILABLE_MODELS.md](AVAILABLE_MODELS.md).

```bash
hf download ibm-granite/granite-3.0-1b-a400m-instruct \
  --revision ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445
```

- About 2.7 GB. It goes into the Hugging Face cache (`~/.cache/huggingface/`), **not** into the repo.
- Granite doesn't need a Hugging Face account. (Some other models do; then run `hf auth login` first.)
- To store models somewhere else, set `HF_HOME` before downloading, e.g. `export HF_HOME=/Volumes/External/hf`.

---

## 6. Run the tests

```bash
pytest -m "not cluster"               # everything on this machine (~6 min)
pytest -m "not model and not cluster" # only the quick tests that don't load the model (~15 s)
pytest -m cluster -s                  # against the running Docker containers (section 8)
```

The reference ("golden") outputs that tests compare against are made once with:

```bash
python scripts/make_golden.py --dtype fp32
python scripts/make_golden.py --dtype bf16
python scripts/make_golden.py --dtype fp32 --check   # re-run and confirm identical
```

They are committed in `tests/golden/`, so you only need to re-make them after changing the model, prompts or package versions. See [IMPLEMENTATION.md](IMPLEMENTATION.md), Phase 0.

---

## 7. Changing package versions

1. Edit the version in `requirements.txt`.
2. Re-install: `uv pip install --python .venv -r requirements.txt` (or `pip install -r requirements.txt`).
3. Re-run all tests. **A `transformers` upgrade can change the expert code we plug into**, so the "same tokens" tests must pass before the change is committed.

To start completely fresh: `rm -rf .venv`, then go back to step 2.

---

## 8. Run the cluster in Docker (Step 2)

One image for every node; 1 master + 2 workers on this laptop. Needs Docker Desktop running.

```bash
# Build the image and start everything (first time: also downloads the model into a Docker volume)
docker compose -f docker/docker-compose.yml up --build -d

# Watch it start (workers register with the master by themselves, in any order)
docker compose -f docker/docker-compose.yml logs -f

# Is it ready? Which nodes are alive?
curl localhost:8000/health
docker exec moe-master-1 moe-node status --master localhost:8000

# Ask something through the OpenAI-style API; -N shows the answer piece by piece as it is made
curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Name three colors."}], "max_tokens": 32, "stream": true}'

# Memory per container
docker stats --no-stream

# Stop (the downloaded model stays in the "models" volume)
docker compose -f docker/docker-compose.yml down
```

Settings you can change when starting: `DTYPE=fp32` (default `bf16`), `MASTER_THREADS=4`, `WORKER_THREADS=2`, `COMBINE_MODE=exact|fast`, `CLUSTER_TOKEN=...`, `LOG_LEVEL=info|debug`, `LOG_FORMAT=pretty|json`, and the health timings `HEARTBEAT_INTERVAL_S=2`, `PING_INTERVAL_S=3`, `SLOW_RTT_MS=250`, `DEAD_AFTER_S=15`, `RECOVER_AFTER_OK=3`, `EXPERT_CALL_TIMEOUT_S=5`, e.g. `DTYPE=fp32 docker compose -f docker/docker-compose.yml up -d`.

**Using it from other tools (Phase 2.3).** The master speaks the same API as OpenAI, Ollama and vLLM, at `http://localhost:8000/v1`. With the official Python client:

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")  # any key, unless API_KEY is set
for chunk in client.chat.completions.create(model="ibm-granite/granite-3.0-1b-a400m-instruct", stream=True,
                                            messages=[{"role": "user", "content": "Name three colors."}]):
    print(chunk.choices[0].delta.content or "", end="", flush=True) if chunk.choices else None
```

Supported: `stream`, `temperature` (0 = always the most likely word; default 1 like OpenAI), `top_p`, `max_tokens`, `stop`, `seed`. One answer is generated at a time; up to `MAX_QUEUE` (16) more wait in line, and beyond that requests get `429 busy`. Set `API_KEY=...` when starting to require a key.

**Open WebUI** (a chat website that works with OpenAI-style servers) can use the cluster. The image is a 5.8 GB download and it uses about 1 GB of memory (capped at 1.5 GB below), so start it only when you want it. The settings turn off login (only you use it, on this Mac) and the extra features that would download more models or send extra requests to the cluster (titles, tags, follow-up suggestions):

```bash
docker run -d --name open-webui --memory 1.5g -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 -e OPENAI_API_KEY=unused \
  -e ENABLE_OLLAMA_API=false -e WEBUI_AUTH=False -e OFFLINE_MODE=true -e HF_HUB_OFFLINE=1 \
  -e BYPASS_EMBEDDING_AND_RETRIEVAL=true -e ENABLE_TITLE_GENERATION=false -e ENABLE_TAGS_GENERATION=false \
  -e ENABLE_FOLLOW_UP_GENERATION=false -e ENABLE_AUTOCOMPLETE_GENERATION=false \
  -v open-webui:/app/backend/data ghcr.io/open-webui/open-webui:main
# then open http://localhost:3000, pick the Granite model and chat
docker rm -f open-webui    # stop it again (its settings stay in the "open-webui" volume)
```

**Small-laptop memory check (Phase 2.4).** The same cluster with a hard memory limit per container (master 2.5 GB, workers 2 GB each, bf16), then a check that runs all golden prompts and a long conversation and reports each container's peak memory:

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.8gb.yml up -d
.venv/bin/python scripts/memory_check.py     # ~15 minutes on a Mac; ends with "OK: ..." or "FAILED: ..."
docker compose -f docker/docker-compose.yml up -d   # back to no limits
```

**Following a token through the cluster (Phase 3.1).** Start with `LOG_LEVEL=debug` to get one line per token per layer on every node, then pick a request and a token:

```bash
LOG_LEVEL=debug docker compose -f docker/docker-compose.yml up -d
# ... send a prompt, then:
docker compose -f docker/docker-compose.yml logs --no-log-prefix | grep "req=cd116308 token=5 layer=6 " | sort
```

```
12:39:18.700 master   DEBUG req=cd116308 token=5 layer=6 router picked experts [5, 6, 10, 12, 15, 20, 21, 27]
12:39:18.700 master   DEBUG req=cd116308 token=5 layer=6 sending [5,6] → local, [10,12,15,20] → worker-1, [21,27] → worker-2
12:39:18.710 worker-2 DEBUG req=cd116308 token=5 layer=6 ran experts [21,27] on 1 token in 8.8 ms
12:39:18.715 master   DEBUG req=cd116308 token=5 layer=6 all results back in 15.0 ms (slowest: worker-1, net 1.2 ms + compute 13.7 ms)
12:39:18.715 worker-1 DEBUG req=cd116308 token=5 layer=6 ran experts [10,12,15,20] on 1 token in 13.7 ms
```

`token=0` is the pass over the whole prompt, `token=1` the first new token, and so on. At the normal `LOG_LEVEL=info`, each answer writes just two lines: `request received` and a `done:` summary with the speed and, per node, network time vs computing time. `LOG_FORMAT=json` writes the same as one JSON object per line.

**The web page (Phase 3.3).** Open **http://localhost:8000** once the cluster is ready: a chat window, and next to it a live view of the nodes and of which experts each layer uses for every token. Stop a worker (`docker stop moe-worker-2-1`) to watch its card turn red. The image builds the page itself; to work on the page with instant reloading, you need Node.js (`node --version` 20 or newer):

```bash
cd ui && npm install
npm run dev        # http://localhost:5173, passes API calls on to the master at localhost:8000
npm run build      # makes ui/dist, which a master started outside Docker serves at /
```

![The web page](images/ui-chat-and-grid.png)

**Live events (Phase 3.2).** The master also sends a live stream of what is happening (for the web page in Phase 3.3): node health, each request's start and end, and for every new token which experts each layer used and which node ran them. To watch it raw:

```bash
curl -N localhost:8000/api/events
```

**Workers coming and going (Phase 2.2).** Each worker registers itself with the master and then sends a heartbeat ("I'm alive") every 2 s; the master also pings each worker every 3 s. `moe-node status` shows each worker as `joining`, `healthy`, `slow`, `dead` or `left`. Things to try:

```bash
docker stop moe-worker-2-1                   # after ~15 s: worker-2 is "dead"; chat requests get 503 "missing experts"
docker start moe-worker-2-1                  # it registers again by itself and becomes "healthy"
docker restart moe-master-1                  # both workers register again with the new master by themselves
docker exec moe-worker-2-1 moe-node leave    # worker-2 says goodbye ("left") and its container stops
```

The failure tests do all of this automatically (they take a few minutes):

```bash
.venv/bin/python -m pytest -m cluster tests/test_cluster_failures.py -s
```

**Company network with a security filter (e.g. Zscaler)?** If the build or the model download fails with `CERTIFICATE_VERIFY_FAILED`, follow [docker/certs/README.md](../docker/certs/README.md) (one command to copy the filter's certificate into `docker/certs/`, then build again). Without such a filter, skip this; nothing changes.

**Golden files for Linux.** Containers run Linux, whose maths libraries round slightly differently from macOS; in bf16 that changes some tokens even for plain Transformers. So the cluster is compared with golden files made inside the container:

```bash
docker compose -f docker/docker-compose.yml stop master worker-1 worker-2   # frees memory for the full model
for d in fp32 bf16; do
  docker compose -f docker/docker-compose.yml run --rm --no-deps \
    -v "$PWD/tests/golden:/app/tests/golden" -v "$PWD/scripts:/app/scripts" \
    master python scripts/make_golden.py --dtype $d --platform linux-aarch64
done
```

---

## 9. Common problems

| Problem | Fix |
|---|---|
| `python3.12: command not found` | `brew install python@3.12`, or use Option A (uv gets Python for you). |
| Wrong Python / old transformers | The environment isn't active: `source .venv/bin/activate`. |
| `hf: command not found` | Same: activate the environment (`hf` is installed by `huggingface_hub`). |
| Download is very slow or stops | Run the same `hf download` command again; it continues where it stopped. |
| "No space left on device" | Free space, or set `HF_HOME` to a disk with more room (section 5). |
| Docker: `CERTIFICATE_VERIFY_FAILED` | A company security filter (e.g. Zscaler) is in the way: see [docker/certs/README.md](../docker/certs/README.md). |
| `moe-node status` shows a worker `joining` forever | It is still loading, or loading failed: `docker logs moe-worker-1-1` shows why (also in the worker's `/health` under `error`). |
| Worker log: `registration refused by the master: ...` | The worker runs another model, revision or `DTYPE` than the master; the reason says which. Start it with the master's settings. |
| Worker log: `registration failed: master answered 401` | `CLUSTER_TOKEN` differs between the worker and the master. |
| Docker: a container stops with no error | Probably out of memory: Docker Desktop's VM has one memory limit for all containers (Settings → Resources). The full fp32 model plus the running cluster does not fit in 8 GB; stop the cluster first. |
