# Setting up a laptop (no Docker)

This guide turns a fresh laptop into one node of the cluster: either the **master** (the laptop you chat with,
which runs the shared parts of the model) or a **worker** (a laptop that holds some of the experts and
computes them when the master asks). Every laptop runs the same code, straight on the laptop, with no Docker.

> **Want an AI agent to do it?** Open Claude Code in the cloned repo and say:
> **"Read docs/LAPTOP_HANDOFF.md and set up this laptop."** It follows this guide and asks you only for the
> few answers it can't find by itself.

**Words used here**

| Word | Meaning |
|---|---|
| **Node** | One laptop taking part in the cluster. |
| **Master** | The one node that receives chat requests, runs the shared parts of the model, and sends expert work to the workers. Exactly one. |
| **Worker** | A node that holds some experts and computes them for the master. Zero, one or many. |
| **Expert** | One of the 32 small sub-networks in each layer of the model (numbered 0–31). Each token uses 8 of them per layer. Together, the laptops must hold all 32. |
| **IP address** | A laptop's number on the Wi-Fi, e.g. `192.168.1.21`. Other laptops use it to reach this one. |
| **Port** | Which "door" on the laptop the program listens at: 8000 for the master, 8100 for a worker. |
| **Cluster token** | A shared password. The master makes one, and every worker types the same one. Without it, anyone on the Wi-Fi could send work to your laptops. |
| **`node.env`** | A small text file at the top of the repo with this laptop's settings, one `NAME=value` per line. `./moe setup` writes it, and every start reads it. Never committed to git (it holds the token). |
| **venv (`.venv`)** | A private folder with its own Python and packages, so this project's exact versions never clash with anything else on the laptop. |

---

## The short version

**First time, on each laptop (master first):**

```bash
git clone <this repo's URL> distributed-inferencing
cd distributed-inferencing
./scripts/install.sh        # Python environment + packages (+ the web page, if Node.js is installed)
./moe setup                 # a few questions -> node.env (also downloads the model, ~2.7 GB)
./moe serve                 # on the master
./moe join                  # on a worker
```

**Every time after that:**

```bash
cd distributed-inferencing
./moe serve                 # master (start it first)
./moe join                  # each worker
./moe status                # from any laptop: every node, its health and its experts
```

Then open **`http://<master's IP>:8000`** in a browser on any laptop (or phone) on the same Wi-Fi.

The rest of this guide explains each step, and how to do it all by hand.

---

## 1. What each laptop needs first

| Need | macOS | Linux (Ubuntu/Debian) |
|---|---|---|
| **git** | `xcode-select --install` (once) | `sudo apt install git` |
| **Python 3.12** | `brew install python@3.12` ([Homebrew](https://brew.sh)), or the installer from python.org | `sudo apt install python3.12 python3.12-venv` (Ubuntu 24.04 already has 3.12; `-venv` is the part that creates virtual environments) |
| **Node.js 24** (master only, for the web page) | `brew install node` | see [nodejs.org](https://nodejs.org) |
| **Disk** | ~5 GB free: ~2 GB of packages and 2.7 GB for the Granite model | same |
| **Memory** | ~1–2 GB free for a worker, ~2.5 GB for the master (Granite) | same |
| **Network** | All laptops on the **same Wi-Fi** (not a "guest" network, which usually stops laptops from seeing each other) | same |

**Windows** isn't supported directly. Use WSL2 (Ubuntu inside Windows) and follow the Linux column. WSL2 needs
extra port-forwarding before other laptops can reach it, so it's best avoided for workers.

Check Python:

```bash
python3.12 --version        # Python 3.12.x
python3.12 -m venv --help   # prints help = the venv command exists (if not, on Linux: sudo apt install python3.12-venv)
```

---

## 2. Get the code and install

```bash
git clone <this repo's URL> distributed-inferencing
cd distributed-inferencing
./scripts/install.sh
```

`install.sh` does four things (and is safe to run again, e.g. after `git pull`):

1. Finds Python 3.12. If it isn't there, it prints the install command for your system.
2. Creates `.venv` with `python3.12 -m venv .venv`.
3. Installs the exact package versions from `requirements.txt` into `.venv`. On Linux it takes PyTorch's CPU-only
   build, which avoids several GB of graphics-card libraries.
4. Builds the web page (`ui/` → `ui/dist`) if Node.js is installed. Only the master needs this; workers skip it.

You never need to "activate" the environment. The `./moe` command always uses `.venv` by itself.

### By hand (what install.sh runs)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
.venv/bin/python -c "import torch, transformers, moe_dist; print('OK', torch.__version__)"

# master only, for the web page:
cd ui && npm ci && npm run build && cd ..
```

---

## 3. Set up the master (do this laptop first)

```bash
./moe setup
```

Answer `master` to the first question. Enter accepts the suggestion in `[brackets]`:

```
Is this laptop the master or a worker? (master / worker) [worker]: master
Cluster token (shared password; every worker types the same one; Enter = this new random one) [k3J9xQ2mP7vR1tYb]:
Name for this laptop (shown in logs and on the web page; must be unique) [master]:
Port to listen on [8000]:
  This laptop's address: 192.168.1.10 (en0). Workers will need it.

Getting the model ibm-granite/granite-3.0-1b-a400m-instruct (only the first time; it is several GB)...
  ✓ model ready in ~/.cache/huggingface/hub/...
How many laptops will take part, this one included? [3]: 3
  Suggested split (each worker's setup suggests its share again):
    this master  experts 0-10
    worker 1     experts 11-21
    worker 2     experts 22-31
Experts this laptop holds (0-31, e.g. 0-9 or 0,4,8 or none) [0-10]:
  ✓ 11 experts x 75.5 MB + shared parts + program: needs about 2.0 GB; 20.0 GB free now (of 36.0 GB)

Saved to node.env.
Start the master:   ./moe serve
Then on each worker run ./moe setup and answer:
  master's address = 192.168.1.10:8000
  cluster token    = k3J9xQ2mP7vR1tYb
```

**Write down the two lines at the end** (the master's address and the cluster token). Every worker needs them.
You can look them up again at any time: `cat node.env` shows the token, and setup prints the address.

Then start it, and leave this terminal open:

```bash
./moe serve
```

Until the workers join, the master says some experts are missing and refuses chat requests. That's expected.

**macOS:** if the firewall is on, the first start pops up *"Do you want the application python to accept incoming network connections?"*.
Click **Allow**, or the workers can't reach the master.

---

## 4. Set up each worker

With the master running:

```bash
./moe setup
```

```
Is this laptop the master or a worker? (master / worker) [worker]:
Master's address (its IP, e.g. 192.168.1.10; port 8000 unless you changed it): 192.168.1.10
  ✓ reached master 'master': model ibm-granite/granite-3.0-1b-a400m-instruct, bf16, loaded
Cluster token (the same one as on the master):
  ✓ the master accepts this token
Name for this laptop (shown in logs and on the web page; must be unique) [laptop-b]:
Port to listen on [8100]:
This laptop's address for the master to reach it (Enter = 'auto': found at every start, now 192.168.1.21) [auto]:

Getting the model ...
  ✓ model ready in ...
  Experts no other laptop holds yet: 11-31
How many more workers will you set up after this one? (they share the free experts) [0]: 1
Experts this laptop holds (0-31, e.g. 0-9 or 0,4,8 or none) [11-21]:
  ✓ 11 experts x 75.5 MB + program: needs about 1.3 GB; 5.2 GB free now (of 8.0 GB)

Saved to node.env.
Start this worker:  ./moe join
```

What setup checks for you:
- **It can reach the master.** If not, it says so and explains what to look at (see Troubleshooting).
- **The token is right.** It asks the master, so a typo is caught now, not at the first start.
- **Which experts are free.** It asks the master which experts no running laptop holds, and suggests your share.
- **The model** is the one the master runs (same name, version and number format), downloaded now.
- **Memory:** whether your experts fit in this laptop's free memory.

Then start it, and leave this terminal open:

```bash
./moe join
```

You should see `registered with master http://192.168.1.10:8000 as http://192.168.1.21:8100`. On macOS, click
**Allow** on the network pop-up here too.

**Why "auto" for the address:** home Wi-Fi often gives a laptop a different IP after a restart. With `auto`, the
worker looks up its current IP at every start. It picks the network card that leads to the master, and skips VPN
or company security tunnels. Type a fixed IP only if `auto` picks the wrong one.

---

## 5. Check that it all works

From any laptop:

```bash
./moe status
```

```
NODE      STATUS   ADDRESS                    RTT now  RTT p95  RAM free  EXPERTS  REASON
master    healthy  (master)                   -        -        -         0-10     loaded
laptop-b  healthy  http://192.168.1.21:8100   6.1 ms   14.2 ms  5.02 GB   11-21    12 good contacts in a row
laptop-c  healthy  http://192.168.1.22:8100   5.4 ms   11.9 ms  9.40 GB   22-31    12 good contacts in a row

coverage: OK (every expert is held by a usable node)
```

- **RTT** (round-trip time) is how long a tiny message takes to the worker and back. Under ~20 ms is good on Wi-Fi.
- **coverage: OK** means the laptops hold all 32 experts between them, so chat works.

Then:
- **Web page:** open `http://192.168.1.10:8000`, chat, and watch which laptop computes each expert.
- **API:** `curl http://192.168.1.10:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Hello"}]}'`
- **Tests (optional, on any laptop):** `.venv/bin/python -m pytest -q -m "not model and not cluster"` runs the quick
  tests, which need no running cluster.

---

## 6. Example values

Two complete examples of what you would type and what ends up in each laptop's `node.env`. The IPs and token
are made up: use your own (setup shows them).

### Master + 2 workers (3 laptops)

| | Master | Worker 1 | Worker 2 |
|---|---|---|---|
| Role | `master` | `worker` | `worker` |
| Name | `master` | `laptop-b` | `laptop-c` |
| IP (shown by setup) | 192.168.1.10 | 192.168.1.21 | 192.168.1.22 |
| Master's address | – | `192.168.1.10` | `192.168.1.10` |
| Cluster token | `k3J9xQ2mP7vR1tYb` (made by setup) | `k3J9xQ2mP7vR1tYb` | `k3J9xQ2mP7vR1tYb` |
| "How many laptops?" / "more workers after this one?" | `3` | `1` | `0` |
| Experts | `0-10` | `11-21` | `22-31` |
| Start with | `./moe serve` | `./moe join` | `./moe join` |

`node.env` on worker 1 (`laptop-b`):

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

`node.env` on the master:

```
NODE_ROLE=master
NODE_ID=master
LISTEN_PORT=8000
EXPERT_IDS=0-10
CLUSTER_TOKEN=k3J9xQ2mP7vR1tYb
MODEL_ID=ibm-granite/granite-3.0-1b-a400m-instruct
DTYPE=bf16
```

### Master + 3 workers (4 laptops)

| | Master | Worker 1 | Worker 2 | Worker 3 |
|---|---|---|---|---|
| Name | `master` | `laptop-b` | `laptop-c` | `laptop-d` |
| IP | 192.168.1.10 | 192.168.1.21 | 192.168.1.22 | 192.168.1.23 |
| Master's address | – | `192.168.1.10` | `192.168.1.10` | `192.168.1.10` |
| Cluster token | `k3J9xQ2mP7vR1tYb` | same | same | same |
| "How many laptops?" / "more workers after this one?" | `4` | `2` | `1` | `0` |
| Experts | `0-7` | `8-15` | `16-23` | `24-31` |

### Uneven splits

The suggestions are even shares, but you can type anything:
- **A small laptop takes fewer experts.** For example, on an 8 GB laptop: master `0-15`, laptop-c `16-25`, laptop-a (8 GB) `26-31`.
- **The master takes more experts.** Experts on the master need no Wi-Fi at all, so answers come faster.
- **The rule:** together, the laptops hold every expert 0–31. Two laptops may hold the same experts (a backup);
  the master uses one of them.

Exactly what the master and workers do with these values is in `IMPLEMENTATION.md`, section 3.

---

## 7. Day to day

| I want to… | Do this |
|---|---|
| Start everything | master: `./moe serve`; then each worker: `./moe join` |
| Stop a worker | `Ctrl+C` in its terminal (the master notices within 15 s), or `./moe leave` from another terminal on that laptop (it tells the master first) |
| Stop the master | `Ctrl+C`. Workers keep trying to reach it, and join again by themselves when it's back. |
| Give a laptop other experts | Edit `EXPERT_IDS=` in its `node.env` (or run `./moe setup` again, which keeps your other answers), then restart it |
| Run a worker once with other settings, without changing `node.env` | `./moe join --experts 0-5` (options win over the file) |
| Get the newest code | `git pull && ./scripts/install.sh`, on **every** laptop (all laptops must run the same version) |
| Keep the laptop awake | On a Mac, `./moe serve` / `./moe join` already block idle sleep (`caffeinate`). Closing the lid still sleeps the laptop: keep it open, and plugged in. |

Without the `./moe` wrapper (e.g. on a system without bash): `.venv/bin/python -m moe_dist.cli join` (or `serve`,
`setup`, `status`) does the same, run from the repo folder.

Setting up by hand without the wizard: copy `node.env.example` to `node.env`, fill in the values (section 6
shows two examples), and run `./moe download` once to get the model.

---

## 8. Troubleshooting

| Problem | What to check |
|---|---|
| Setup: **"cannot reach the master"** | Is `./moe serve` running on the master? Same Wi-Fi (not a guest network)? Is the IP right (`./moe setup` on the master prints it; on macOS: System Settings → Wi-Fi → Details)? Did you click **Allow** on the master's network pop-up? Is the port right? Type it with the IP if the master doesn't use 8000, e.g. `192.168.1.10:8001` (the error shows the address it tried). Test by hand: `curl http://192.168.1.10:8000/health` |
| **`curl` works, but setup says "cannot reach the master … No route to host"** (macOS) | macOS's **Local Network permission** is blocking Python. Since macOS 15, apps need permission to talk to other devices on the network. Apple's own tools like `curl` don't, but Python does, and it's allowed or blocked per app you start it from. Fix: System Settings → Privacy & Security → **Local Network** → switch on the app you run the commands in (Terminal, iTerm, VS Code, …), **quit that app completely (⌘Q)** and reopen it. If the app isn't listed, run the command once from Apple's Terminal app and click **Allow** on the pop-up (or restart the laptop). Check: `.venv/bin/python -c "import httpx; print(httpx.get('http://192.168.1.10:8000/health', timeout=5).text[:80])"` must print `{"status":"ok",...}` |
| Setup or join fails only on a company laptop | A company proxy (a middleman server for web traffic) can't reach other laptops. Traffic between the laptops already skips proxy settings. If the model download fails, see the certificate row below. Run setup **yourself in a terminal**, not through an AI agent, because agents may block network access for the commands they run. |
| macOS: clicked "Deny" by mistake | System Settings → Network → Firewall → Options: set Python to "Allow incoming connections". |
| Setup: **"the master says this token is wrong"** | Copy it exactly from the master's `node.env` (`CLUSTER_TOKEN=` line). |
| Worker registers, but its status is **dead** or **slow** | The master can't reach the worker back. Run `./moe status` and check the worker's ADDRESS is its real Wi-Fi IP. If `auto` picked a VPN address, set `ADVERTISE_ADDR=<its IP>:8100` in `node.env`. Then check the macOS pop-up was allowed on the worker too. |
| **"registration refused by the master: ..."** | The reason says what differs: a different model, version or number format (`git pull` on both laptops), or a name already used by another live laptop (pick another `NODE_ID`). |
| **SSL / certificate errors** while installing or downloading (company networks with a security filter such as Zscaler) | Point Python at the company's certificate file, then run the step again: `export SSL_CERT_FILE=/path/company-root.pem REQUESTS_CA_BUNDLE=/path/company-root.pem PIP_CERT=/path/company-root.pem` |
| **Port already in use** | Something else uses 8000/8100: run `./moe setup` again and pick another port. Workers then type the master's address with the port, e.g. `192.168.1.10:8001`. |
| The master **refuses chat** with "missing experts" | Some expert numbers are held by no running laptop. `./moe status` lists what's missing; start that worker, or give another laptop those experts. |
| `./moe: No Python environment yet` | Run `./scripts/install.sh` first. |
| `python3.12: command not found` | Section 1. |

Logs: each terminal shows its node's log. For one line per expert call: add `LOG_LEVEL=debug` to `node.env`
(lots of output). The logs and the web page use the same request IDs, so you can match them.
