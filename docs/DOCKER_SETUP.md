# Docker setup: the whole cluster on one computer

This guide runs the **whole cluster on one computer**, in Docker: one master and two workers, each in its own
container. It's the easiest way to try the project, and it's how the project is tested.

To run the cluster across **several laptops** (no Docker), use [LAPTOP_SETUP.md](LAPTOP_SETUP.md) instead.

**Words used here**

| Word | Meaning |
|---|---|
| **Docker image** | A packed program with everything it needs: Python, the packages, our code. One image is used for every node. |
| **Container** | A running copy of the image, like a small sealed Linux computer. The cluster is 3 containers: `master`, `worker-1`, `worker-2`. |
| **Master / worker** | The master runs the shared parts of the model and talks to you. The workers hold experts (small parts of the model) and compute them when the master asks. |
| **Volume** | A storage area Docker keeps between restarts. The model (2.7 GB) is downloaded once into a volume called `models`. |
| **venv (`.venv`)** | A private folder with its own Python and packages. Only needed here for running the tests. |

---

## 1. What you need

| Need | How to get it |
|---|---|
| **Docker Desktop**, running | [docker.com](https://www.docker.com/products/docker-desktop/). Check: `docker --version` |
| **~8 GB of memory for Docker** | Docker Desktop → Settings → Resources → Memory. The cluster uses about 4–6 GB. |
| **~10 GB free disk** | The image (~3 GB) plus the model (2.7 GB) |
| **git** | macOS: `xcode-select --install`. Linux: `sudo apt install git` |
| **Python 3.12** (only for the tests) | macOS: `brew install python@3.12`. Linux: `sudo apt install python3.12 python3.12-venv` |

**On a company network with a security filter (e.g. Zscaler)?** Docker downloads may fail with
`CERTIFICATE_VERIFY_FAILED`. Copy the filter's certificate into `docker/certs/` first, with one command, as
described in [docker/certs/README.md](../docker/certs/README.md). Without such a filter, skip this.

---

## 2. First time

**a) Get the code:**

```bash
git clone git@github.com:umar1997/distributed-inferencing.git
cd distributed-inferencing
```

**b) Build and start the cluster.** The first time takes a while: it builds the image and downloads the model.

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

**c) Watch it start.** The workers register with the master by themselves, in any order. Press `Ctrl+C` to stop
watching. The cluster keeps running.

```bash
docker compose -f docker/docker-compose.yml logs -f
```

**d) Check it's ready.** All three nodes should say `healthy`, and the last line should say `coverage: OK`.

```bash
docker exec moe-master-1 moe-node status --master localhost:8000
```

**e) Use it.** Open **http://localhost:8000** in a browser. On the left you chat, and on the right you watch
which experts each layer uses for every token, and which container ran them.

![The web page](images/ui-chat-and-grid.png)

**f) Optional: the Python environment, to run the tests** (section 6):

```bash
./scripts/install.sh
```

---

## 3. Every time after

```bash
cd distributed-inferencing
docker compose -f docker/docker-compose.yml up -d       # start (no --build needed unless the code changed)
# ... open http://localhost:8000 ...
docker compose -f docker/docker-compose.yml down        # stop (the model stays in the "models" volume)
```

After `git pull`, add `--build` once so the image gets the new code:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

---

## 4. Things to try

**Ask through the API.** The master speaks the same API as OpenAI, Ollama and vLLM, at
`http://localhost:8000/v1`. `-N` shows the answer piece by piece as it's made:

```bash
curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Name three colors."}], "max_tokens": 32, "stream": true}'
```

With the official Python client:

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")  # any key, unless API_KEY is set
for chunk in client.chat.completions.create(model="ibm-granite/granite-3.0-1b-a400m-instruct", stream=True,
                                            messages=[{"role": "user", "content": "Name three colors."}]):
    print(chunk.choices[0].delta.content or "", end="", flush=True) if chunk.choices else None
```

Supported: `stream`, `temperature` (0 = always the most likely word; default 1, like OpenAI), `top_p`,
`max_tokens`, `stop`, `seed`. One answer is made at a time. Up to 16 more requests wait in line, and beyond that
requests get `429 busy`.

**Stop a worker and watch what happens.** Keep the web page open while you do this:

```bash
docker stop moe-worker-2-1                   # after ~5 s "slow", after ~15 s "dead"; its experts turn red
docker start moe-worker-2-1                  # it registers again by itself and turns green
docker restart moe-master-1                  # both workers register again with the new master by themselves
docker exec moe-worker-2-1 moe-node leave    # worker-2 says goodbye ("left") and its container stops
```

**Follow one token through all three containers.** Start with detailed logs, send a prompt, then pick a request
ID and a token number from the logs (the web page shows the exact text to search for):

```bash
LOG_LEVEL=debug docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml logs --no-log-prefix | grep "req=cd116308 token=5 layer=6 " | sort
```

```
12:39:18.700 master   DEBUG req=cd116308 token=5 layer=6 router picked experts [5, 6, 10, 12, 15, 20, 21, 27]
12:39:18.700 master   DEBUG req=cd116308 token=5 layer=6 sending [5,6] → local, [10,12,15,20] → worker-1, [21,27] → worker-2
12:39:18.710 worker-2 DEBUG req=cd116308 token=5 layer=6 ran experts [21,27] on 1 token in 8.8 ms
12:39:18.715 master   DEBUG req=cd116308 token=5 layer=6 all results back in 15.0 ms (slowest: worker-1, net 1.2 ms + compute 13.7 ms)
12:39:18.715 worker-1 DEBUG req=cd116308 token=5 layer=6 ran experts [10,12,15,20] on 1 token in 13.7 ms
```

`token=0` is the pass over the whole prompt, `token=1` the first new token, and so on. At the normal level
(`info`), each answer writes just two lines: `request received` and a `done:` summary with the speed and, per
node, network time vs computing time.

**Watch the raw live events** (what the web page listens to):

```bash
curl -N localhost:8000/api/events
```

**Chat through Open WebUI** (a ChatGPT-like website). It's a 5.8 GB download and uses about 1 GB of memory,
so only start it when you want it:

```bash
docker run -d --name open-webui --memory 1.5g -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 -e OPENAI_API_KEY=unused \
  -e ENABLE_OLLAMA_API=false -e WEBUI_AUTH=False -e OFFLINE_MODE=true -e HF_HUB_OFFLINE=1 \
  -e BYPASS_EMBEDDING_AND_RETRIEVAL=true -e ENABLE_TITLE_GENERATION=false -e ENABLE_TAGS_GENERATION=false \
  -e ENABLE_FOLLOW_UP_GENERATION=false -e ENABLE_AUTOCOMPLETE_GENERATION=false \
  -v open-webui:/app/backend/data ghcr.io/open-webui/open-webui:main
# open http://localhost:3000, pick the Granite model and chat
docker rm -f open-webui    # stop it again (its settings stay in the "open-webui" volume)
```

**Check it fits a small (8 GB) laptop.** The same cluster with a hard memory limit per container (master 2.5 GB,
workers 2 GB each), then a check that runs all test prompts and a long conversation:

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.8gb.yml up -d
.venv/bin/python scripts/memory_check.py               # ~15 minutes; ends with "OK: ..." or "FAILED: ..."
docker compose -f docker/docker-compose.yml up -d      # back to no limits
```

---

## 5. Settings

Put them in front of the start command, e.g. `DTYPE=fp32 docker compose -f docker/docker-compose.yml up -d`.

| Setting | Default | What it does |
|---|---|---|
| `DTYPE` | `bf16` | Number format: `bf16` (2 bytes per number) or `fp32` (4 bytes, exact, twice the memory) |
| `LOG_LEVEL` | `info` | `debug` = one line per token per layer on every node |
| `LOG_FORMAT` | `pretty` | `json` = one JSON object per line |
| `CLUSTER_TOKEN` | a fixed test token | Shared password between master and workers |
| `API_KEY` | none | If set, chat requests need `Authorization: Bearer <key>` |
| `MASTER_THREADS` / `WORKER_THREADS` | `4` / `2` | CPU threads per container |
| `COMBINE_MODE` | `exact` | `fast` = workers add up their own results first (fewer bytes, may change bf16 answers) |
| `HEARTBEAT_INTERVAL_S`, `PING_INTERVAL_S`, `SLOW_RTT_MS`, `DEAD_AFTER_S`, `RECOVER_AFTER_OK`, `EXPERT_CALL_TIMEOUT_S` | `2`, `3`, `250`, `15`, `3`, `5` | Health timings: how often workers and master check on each other, and when a worker counts as slow or dead |

**Docker on a Mac is slow.** On a Mac, Docker runs everything inside a hidden Linux virtual machine, and there
the expert maths is 2–15× slower than on macOS itself: about 1.8 tokens per second, versus ~17 for one program
outside Docker. That's a property of Docker Desktop, not of the project. On the real laptops the code runs without
Docker ([LAPTOP_SETUP.md](LAPTOP_SETUP.md)).

---

## 6. Working on the code: tests and reference answers

These need the Python environment: run `./scripts/install.sh` once. It creates `.venv` with Python 3.12 and
installs the exact versions from `requirements.txt`.

```bash
.venv/bin/python -m pytest -m "not model and not cluster"   # quick tests, no model needed (~20 s)
.venv/bin/python -m pytest -m "not cluster"                 # everything on this computer, loads the model (~6 min)
.venv/bin/python -m pytest -m cluster -s                    # against the running containers
.venv/bin/python -m pytest -m cluster tests/test_cluster_failures.py -s   # stops/freezes/slows containers (a few minutes)
```

The model-loading tests need the model in this computer's own cache (outside Docker), once:
`./moe download`. It goes to `~/.cache/huggingface/`, not into the repo.

**Reference ("golden") answers.** The tests compare the cluster's answers with saved answers from plain,
unchanged Transformers, in `tests/golden/`. They're committed, so you only remake them after changing the
model, the prompts or package versions ([IMPLEMENTATION.md](IMPLEMENTATION.md), Phase 0):

```bash
.venv/bin/python scripts/make_golden.py --dtype fp32
.venv/bin/python scripts/make_golden.py --dtype bf16
.venv/bin/python scripts/make_golden.py --dtype fp32 --check   # run again and confirm identical
```

Containers run Linux, whose maths libraries round bf16 numbers slightly differently from macOS, so the
containers have their own golden files, made inside a container:

```bash
docker compose -f docker/docker-compose.yml stop master worker-1 worker-2   # frees memory for the full model
for d in fp32 bf16; do
  docker compose -f docker/docker-compose.yml run --rm --no-deps \
    -v "$PWD/tests/golden:/app/tests/golden" -v "$PWD/scripts:/app/scripts" \
    master python scripts/make_golden.py --dtype $d --platform linux-aarch64
done
```

**Changing a package version:** edit `requirements.txt`, run `./scripts/install.sh` again, and run all tests. A
`transformers` upgrade can change the expert code we plug into, so the "same tokens" tests must pass before the
change is committed. Rebuild the image afterwards (`up --build`).

**Working on the web page** with instant reloading needs Node.js 20 or newer:

```bash
cd ui && npm install
npm run dev        # http://localhost:5173, passes API calls on to the master at localhost:8000
npm run build      # makes ui/dist, which a master outside Docker serves at /
```

---

## 7. Common problems

| Problem | Fix |
|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop isn't running: start it and wait until it says "running". |
| Build or download fails with `CERTIFICATE_VERIFY_FAILED` | A company security filter (e.g. Zscaler) is in the way: see [docker/certs/README.md](../docker/certs/README.md), then build again. |
| `port is already allocated` (8000) | Something else uses port 8000, e.g. a master started outside Docker with `./moe serve`. Stop that first. |
| `moe-node status` shows a worker `joining` forever | It's still loading, or loading failed: `docker logs moe-worker-1-1` shows why. |
| Chat says `missing experts` | A worker is stopped or dead: `docker compose -f docker/docker-compose.yml ps`, then start it. |
| Worker log: `registration refused by the master: ...` | The worker runs another model, version or `DTYPE` than the master. The reason says which. Start everything with the same settings. |
| Worker log: `registration failed: master answered 401` | `CLUSTER_TOKEN` differs between the worker and the master. |
| A container stops with no error | Probably out of memory: Docker Desktop has one memory limit for all containers (Settings → Resources). Give it more, or stop other containers. |
| `python3.12: command not found` (for the tests) | macOS: `brew install python@3.12`. Linux: `sudo apt install python3.12 python3.12-venv`. |
| Download very slow or stopped | Run the same command again: it continues where it stopped. |
