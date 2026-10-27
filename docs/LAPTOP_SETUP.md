# Laptop setup: the cluster across several laptops

This guide runs the cluster across **several laptops on the same Wi-Fi**, straight on each laptop, with no
Docker. One laptop is the **master**: you chat with it, and it runs the shared parts of the model. The others
are **workers**: each holds some of the experts and computes them when the master asks.

- **The master laptop:** [section 2](#2-the-master-laptop)
- **A worker laptop:** [section 3](#3-a-worker-laptop)
- **Everything on one computer in Docker instead:** [DOCKER_SETUP.md](DOCKER_SETUP.md)

> **Want an AI agent to set up a laptop?** Open Claude Code in the cloned repo and say:
> **"Read docs/LAPTOP_HANDOFF.md and set up this laptop."** It follows this guide and asks you only for the
> answers it can't find by itself.

**Words used here**

| Word | Meaning |
|---|---|
| **Master** | The one laptop that receives chat requests, runs the shared parts of the model, and sends expert work to the workers. |
| **Worker** | A laptop that holds some experts and computes them for the master. One, two or many. |
| **Expert** | One of the 32 small parts in each layer of the model (numbered 0–31). Together, the laptops must hold all 32. |
| **IP address** | A laptop's number on the Wi-Fi, e.g. `192.168.1.10`. Other laptops use it to reach this one. |
| **Port** | Which "door" on the laptop the program listens at: 8000 for the master, 8100 for a worker. |
| **Cluster token** | A shared password. The master makes one, and every worker types the same one. |
| **`node.env`** | A small settings file at the top of the repo, written by `./moe setup` and read at every start. Never committed to git (it holds the token). |
| **venv (`.venv`)** | A private folder with its own Python and packages, so this project's versions don't clash with anything else. |

---

## 1. Before you start (every laptop)

| Need | macOS | Linux (Ubuntu/Debian) |
|---|---|---|
| **git** | `xcode-select --install` | `sudo apt install git` |
| **Python 3.12** | `brew install python@3.12` ([Homebrew](https://brew.sh)) | `sudo apt install python3.12 python3.12-venv` |
| **Node.js** (master only, for the web page) | `brew install node` | see [nodejs.org](https://nodejs.org) |
| **Disk** | ~5 GB free (packages + the 2.7 GB model) | same |
| **Memory** | ~1–2 GB free for a worker, ~2.5 GB for the master | same |

Check Python, which must say 3.12.x (3.11 is too old):

```bash
python3.12 --version
```

**Put all laptops on the same Wi-Fi.** On each laptop, find its IP address:

```bash
ipconfig getifaddr en0      # macOS
hostname -I                 # Linux
```

The first three numbers should match on all laptops, e.g. `10.189.119.60` and `10.189.119.201`. If they
don't, the laptops are on different networks and can't reach each other. Office and guest Wi-Fi sometimes
block laptops from talking to each other: if that happens, use a phone hotspot or home Wi-Fi for all of them.

**Windows** isn't supported directly. Use WSL2 (Ubuntu inside Windows) and follow the Linux column.

---

## 2. The master laptop

Set up the master **first**: a worker's setup asks the master which experts are still free.

### First time

**a) If the Docker cluster is running on this laptop, stop it.** Its master uses port 8000, the port the real
master needs:

```bash
docker compose -f docker/docker-compose.yml down
```

(Or keep it running and choose another port, like 8001, in step c.)

**b) Get the code and install:**

```bash
git clone git@github.com:umar1997/distributed-inferencing.git
cd distributed-inferencing
./scripts/install.sh        # creates .venv, installs the packages, builds the web page (a few minutes)
```

It must end with `Done. Next: ./moe setup`.

**c) Set up:**

```bash
./moe setup
```

Answer **`master`** to the first question, the **number of laptops** when asked, and press **Enter** for the rest.
The suggestion in `[brackets]` is what Enter picks:

```
Is this laptop the master or a worker? (master / worker) [worker]: master
Cluster token (shared password; every worker types the same one; Enter = this new random one) [k3J9xQ2mP7vR1tYb]:
Name for this laptop (shown in logs and on the web page; must be unique) [master]:
Port to listen on [8000]:
  This laptop's address: 192.168.1.10 (en0). Workers will need it.

Getting the model ibm-granite/granite-3.0-1b-a400m-instruct (only the first time; it is several GB)...
  ✓ model ready in ~/.cache/huggingface/hub/...
How many laptops will take part, this one included? [3]: 2
  Suggested split (each worker's setup suggests its share again):
    this master  experts 0-15
    worker 1     experts 16-31
Experts this laptop holds (0-31, e.g. 0-9 or 0,4,8 or none) [0-15]:
  ✓ 16 experts x 75.5 MB + shared parts + program: needs about 2.4 GB; 20.0 GB free now (of 36.0 GB)

Saved to node.env.
Start the master:   ./moe serve
Then on each worker run ./moe setup and answer:
  master's address = 192.168.1.10:8000
  cluster token    = k3J9xQ2mP7vR1tYb
```

**d) Write down the two lines at the end:** the master's address and the cluster token. Every worker needs
them. To look them up later: the token is the `CLUSTER_TOKEN=` line in `node.env`, and the address is this
laptop's IP (`ipconfig getifaddr en0`) plus the port.

**e) Start it, and leave this terminal open:**

```bash
./moe serve
```

Until the workers join, it says some experts are missing and refuses chat requests. That's expected.

If macOS asks *"Do you want the application python to accept incoming network connections?"*, click **Allow**.

### Every time after

```bash
cd distributed-inferencing
./moe serve
```

- **The IP may have changed** (Wi-Fi often hands out a new one after a restart). Check with
  `ipconfig getifaddr en0`. If it changed, the workers need the new address (section 3, "Every time after").
- **After a `git pull`** (new code), run `./scripts/install.sh` once before starting. Do it on every laptop:
  they must all run the same code.
- **To stop:** `Ctrl+C`. The workers keep trying to reach the master, and join again by themselves when it's back.

---

## 3. A worker laptop

### First time

**a) Get the code and install:**

```bash
git clone git@github.com:umar1997/distributed-inferencing.git
cd distributed-inferencing
./scripts/install.sh
```

**b) Check you can reach the master** (it must be running). Use the address you wrote down:

```bash
curl -m 5 http://192.168.1.10:8000/health
.venv/bin/python -c "import httpx; print(httpx.get('http://192.168.1.10:8000/health', timeout=5).text[:80])"
```

Both must print `{"status":"ok", ...}`.

- **`curl` fails:** the laptops can't reach each other. Check they're on the same Wi-Fi (section 1).
- **`curl` works, but the second line says `No route to host`** (macOS): macOS is blocking Python from the
  local network. Go to System Settings → Privacy & Security → **Local Network**, switch on your terminal app
  (Terminal, iTerm, VS Code, …), **quit it completely (⌘Q)** and reopen it.

**c) Set up:**

```bash
./moe setup
```

Answer **`worker`**, the **master's address** and the **token**, and press **Enter** for the rest:

```
Is this laptop the master or a worker? (master / worker) [worker]: worker
Master's address (its IP, e.g. 192.168.1.10; port 8000 unless you changed it): 192.168.1.10
  ✓ reached master 'master': model ibm-granite/granite-3.0-1b-a400m-instruct, bf16, loaded
Cluster token (the same one as on the master):
  ✓ the master accepts this token
Name for this laptop (shown in logs and on the web page; must be unique) [laptop-b]:
Port to listen on [8100]:
This laptop's address for the master to reach it (Enter = 'auto': found at every start, now 192.168.1.21) [auto]:

Getting the model ...
  ✓ model ready in ...
  Experts no other laptop holds yet: 16-31
How many more workers will you set up after this one? (they share the free experts) [0]:
Experts this laptop holds (0-31, e.g. 0-9 or 0,4,8 or none) [16-31]:
  ✓ 16 experts x 75.5 MB + program: needs about 1.6 GB; 5.2 GB free now (of 8.0 GB)

Saved to node.env.
Start this worker:  ./moe join
```

Things to know:

- **The master's address:** if the master doesn't use port 8000, type the port too, e.g. `192.168.1.10:8001`.
- **"Port to listen on" is this worker's own port.** Keep 8100: it has nothing to do with the master's port.
- **"How many more workers after this one?"** With one worker, answer 0 and it takes all the free experts. With
  more, the free experts are shared. For the first of two workers, answer 1.

**d) Start it, and leave this terminal open:**

```bash
./moe join
```

You should see `registered with master http://192.168.1.10:8000 as http://192.168.1.21:8100`. If macOS asks
about incoming network connections, click **Allow**.

### Every time after

The master must be running first.

```bash
cd distributed-inferencing
./moe join
```

- **If the master's IP changed:** open `node.env`, change the `MASTER_ADDR=` line to the new address (e.g.
  `MASTER_ADDR=192.168.1.15:8000`), and start again. Or, for one start only: `./moe join --master 192.168.1.15:8000`.
- **This worker's own IP** is found again at every start (`auto`), so nothing to do if it changes.
- **To stop:** `Ctrl+C` (the master notices within 15 seconds), or `./moe leave` from a second terminal on this
  laptop (it tells the master first).

---

## 4. Check it all works

From any laptop:

```bash
./moe status
```

```
NODE      STATUS   ADDRESS                    RTT now  RTT p95  RAM free  EXPERTS  REASON
master    healthy  (master)                   -        -        -         0-15     loaded
laptop-b  healthy  http://192.168.1.21:8100   21.2 ms  131.9 ms 15.98 GB  16-31    45 good contacts in a row

coverage: OK (every expert is held by a usable node)
```

- **healthy** on every line, and **coverage: OK** at the end: all 32 experts are held, so chat works.
- **RTT** (round-trip time) is how long a tiny message takes to the worker and back. Lower is faster.

Then open **`http://<master's IP>:<port>`** in a browser, e.g. `http://192.168.1.10:8000`, on any laptop or phone
on the same Wi-Fi. On the left you chat. On the right, the experts each token uses light up in the colour of the
laptop that computed them.

---

## 5. Example values

The IPs and the token below are made up; setup shows your own.

### Master + 1 worker (2 laptops)

This is the setup that was first run for real (with the master on port 8001, because the Docker master still had 8000).

| | Master | Worker |
|---|---|---|
| Role | `master` | `worker` |
| Name | `master` | `laptop-b` |
| IP (shown by setup) | 192.168.1.10 | 192.168.1.21 |
| Master's address | – | `192.168.1.10` |
| Cluster token | `k3J9xQ2mP7vR1tYb` (made by setup) | `k3J9xQ2mP7vR1tYb` |
| "How many laptops?" / "How many more workers after this one?" | `2` | `0` |
| Experts | `0-15` | `16-31` |
| Start with | `./moe serve` | `./moe join` |

### Master + 2 workers (3 laptops)

| | Master | Worker 1 | Worker 2 |
|---|---|---|---|
| Name | `master` | `laptop-b` | `laptop-c` |
| IP | 192.168.1.10 | 192.168.1.21 | 192.168.1.22 |
| Master's address | – | `192.168.1.10` | `192.168.1.10` |
| Cluster token | `k3J9xQ2mP7vR1tYb` | same | same |
| "How many laptops?" / "How many more workers after this one?" | `3` | `1` | `0` |
| Experts | `0-10` | `11-21` | `22-31` |

### Master + 3 workers (4 laptops)

| | Master | Worker 1 | Worker 2 | Worker 3 |
|---|---|---|---|---|
| Name | `master` | `laptop-b` | `laptop-c` | `laptop-d` |
| IP | 192.168.1.10 | 192.168.1.21 | 192.168.1.22 | 192.168.1.23 |
| Master's address | – | `192.168.1.10` | `192.168.1.10` | `192.168.1.10` |
| Cluster token | `k3J9xQ2mP7vR1tYb` | same | same | same |
| "How many laptops?" / "How many more workers after this one?" | `4` | `2` | `1` | `0` |
| Experts | `0-7` | `8-15` | `16-23` | `24-31` |

### What `node.env` looks like

On the master:

```
NODE_ROLE=master
NODE_ID=master
LISTEN_PORT=8000
EXPERT_IDS=0-10
CLUSTER_TOKEN=k3J9xQ2mP7vR1tYb
MODEL_ID=ibm-granite/granite-3.0-1b-a400m-instruct
DTYPE=bf16
```

On worker 1 (`laptop-b`, 3 laptops):

```
NODE_ROLE=worker
NODE_ID=laptop-b
LISTEN_PORT=8100
EXPERT_IDS=11-21
MASTER_ADDR=192.168.1.10:8000
ADVERTISE_ADDR=auto
CLUSTER_TOKEN=k3J9xQ2mP7vR1tYb
MODEL_ID=ibm-granite/granite-3.0-1b-a400m-instruct
DTYPE=bf16
```

### Uneven splits

The suggestions are even shares, but you can type any experts:

- **A small laptop takes fewer.** For example, with an 8 GB laptop: master `0-15`, laptop-c `16-25`, the 8 GB laptop `26-31`.
- **The master takes more.** Experts on the master need no Wi-Fi at all, so answers come faster.
- **The one rule:** together, the laptops hold every expert 0–31. Two laptops may hold the same experts (a
  backup); the master uses one of them.

More laptops means less memory per laptop, but **not** more speed. Every worker is another Wi-Fi round trip
in each of the 24 layers, and the slowest one decides how long a layer takes.

---

## 6. Changing things

| I want to… | Do this |
|---|---|
| Give a laptop other experts | Edit `EXPERT_IDS=` in its `node.env` (or run `./moe setup` again: it keeps your other answers), then restart it |
| Start a worker once with other settings, without changing `node.env` | `./moe join --experts 0-5` or `./moe join --master 192.168.1.15:8000` |
| Add another worker later | Set it up like any worker. Setup shows which experts are free. If none are free, take some away from another laptop first. |
| Get the newest code | `git pull && ./scripts/install.sh` on **every** laptop, then restart each one |
| See more detail in the logs | Add `LOG_LEVEL=debug` to `node.env`: one line per token per layer (lots of output) |
| Keep the laptop awake | On a Mac, `./moe serve` / `./moe join` already stop it from sleeping by itself. Closing the lid still sleeps it: keep it open and plugged in. |

---

## 7. Problems and fixes

| Problem | Fix |
|---|---|
| Setup: **"cannot reach the master"** | Is `./moe serve` running on the master? Are both laptops on the same Wi-Fi (section 1)? Is the IP right? If the master doesn't use 8000, type the port too (`192.168.1.10:8001`). The error shows the exact address it tried. |
| **`curl` works, but setup says "No route to host"** (macOS) | macOS's **Local Network** permission is blocking Python. System Settings → Privacy & Security → Local Network → switch on your terminal app, **quit it completely (⌘Q)** and reopen it. If the app isn't listed, run the command once from Apple's own Terminal app and click **Allow** on the pop-up, or restart the laptop. |
| Setup: **"the master says this token is wrong"** | Copy it exactly from the `CLUSTER_TOKEN=` line in the master's `node.env`. |
| The worker joins, but shows **dead** or **slow** | The master can't reach the worker back. In `./moe status`, check the worker's ADDRESS is its real Wi-Fi IP. If it isn't (e.g. a VPN address), set `ADVERTISE_ADDR=<its IP>:8100` in the worker's `node.env`. Click **Allow** if macOS asks about incoming connections. |
| **"registration refused by the master: ..."** | The reason says what differs: another model, version or number format (run `git pull` and `./scripts/install.sh` on both), or a name another laptop already uses (pick another `NODE_ID`). |
| Chat says **"missing experts"** | Some experts are held by no running laptop. `./moe status` lists them: start that worker, or give another laptop those experts. |
| **"port is already in use"** / "address already in use" | Something else uses that port, e.g. the Docker master on 8000. Stop it (section 2a), or run `./moe setup` again and pick another port. Workers then type the master's address with that port. |
| **Certificate errors** while installing or downloading (company networks, e.g. Zscaler) | Point Python at the company's certificate file and try again: `export SSL_CERT_FILE=/path/company-root.pem REQUESTS_CA_BUNDLE=/path/company-root.pem PIP_CERT=/path/company-root.pem` |
| Setup fails only when an AI agent runs it | Agents may block network access for the commands they run. Run `./moe setup` yourself in a terminal. |
| `./moe: No Python environment yet` | Run `./scripts/install.sh` first. |
| `python3.12: command not found` | Install it (section 1). |
| `python3.12 -m venv .venv` says "ensurepip is not available" (Linux) | `sudo apt install python3.12-venv`, then run `./scripts/install.sh` again. |

Each terminal shows its laptop's log. The web page shows the exact text to search for in the logs
(`req=… token=…`), so you can match what you see on the page with what each laptop did.

---

## 8. Doing it all by hand

If you'd rather not use the install script or the setup questions, from the repo folder:

```bash
# the Python environment (what install.sh does)
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
.venv/bin/python -c "import torch, transformers; print('OK', torch.__version__)"    # OK 2.14.1

# master only: the web page
cd ui && npm ci && npm run build && cd ..

# the settings (what ./moe setup does)
cp node.env.example node.env      # then edit it: section 5 shows what goes in it
./moe download                    # get the model once (~2.7 GB)

# start
./moe serve                       # master
./moe join                        # worker
```

The extra address in the install line makes Linux download the smaller PyTorch build for computers without a
graphics card; on a Mac it changes nothing. You never need to "activate" the venv: `./moe` always uses
`.venv/bin/python` by itself. Without `./moe`, `.venv/bin/python -m moe_dist.cli serve` (or `join`, `setup`,
`status`) does the same, run from the repo folder.
