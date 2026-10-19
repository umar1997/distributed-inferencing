# Project: Distributed Mixture-of-Experts (MoE) Inference Across CPU-Only Laptops

## 1. What I want to build

I want to run one AI language model across several laptops at the same time, using only their CPUs (no GPUs).

The model will be a **Mixture-of-Experts (MoE)** model. In an MoE model, each layer has many small sub-networks called **experts**, plus a small **router** that picks a few experts for each word (token). Only the picked experts do work for that token.

My idea is to **put different experts on different laptops**:

- One laptop is the **master node**. It runs the chat UI, keeps track of all the other nodes, runs the shared parts of the model (the parts every token always uses), and also holds some experts.
- The other laptops are **worker nodes**. Each one holds only some of the experts.
- When I type a prompt, the master runs the model. Each time the router picks experts, the master sends the work to whichever node holds those experts, gets the results back, and continues. The final answer appears in the chat UI.

This is mainly a **learning and demo project**. Being able to clearly **see** the distributed inference happening (in logs and in the UI) is more important to me than speed.

## 2. Important background (please keep this in mind)

- Ollama (which uses llama.cpp underneath) and vLLM **do not support** putting specific experts on specific machines across CPU-only laptops.
  - llama.cpp has an RPC mode, but it splits a model by **layers**, not by **experts**.
  - vLLM's CPU version supports splitting a model across CPU groups inside **one** machine. Its multi-machine support is built for GPU clusters.
- So the expert-splitting part must be **custom code**, written in Python with PyTorch and Hugging Face Transformers.
- We will still use **ready-made MoE models from Hugging Face**. We are not training or creating a model. We only write the code that runs the model across machines.
- "Compatible with Ollama / vLLM" means only this: my system should offer the **same kind of API** (an OpenAI-style `/v1/chat/completions` endpoint, with streaming), so any tool that can talk to Ollama or vLLM can also talk to my system.
- llama.cpp is **not** part of this project. Do not plan for it, except as a possible optional comparison at the very end.

## 3. Hardware

| Machine | RAM | CPU / OS | Role |
|---|---|---|---|
| Laptop A | 8 GB | [fill in: Intel/AMD or Apple, and OS] | Worker (Step 4) |
| Laptop B (Mac) | 36 GB | Apple Silicon (ARM), macOS | Development machine for Steps 1–3, then master node |
| Laptop C (Mac) | More than 8 GB (exact amount TBD) | Apple Silicon (ARM), macOS | Worker (Step 4) |

- All laptops connect over **Wi-Fi** on the same home network. No Ethernet cables are available, so the design must work well over Wi-Fi.
- No GPUs are used. On the Macs, Docker runs Linux containers inside a virtual machine, so the Apple GPU is not available inside Docker anyway. Everything runs on CPU.
- Two machines are ARM (Apple Silicon). If Laptop A is Intel/AMD (x86), the Docker image must be built for **both** CPU types (a multi-architecture image).
- Development of Steps 1–3 happens on the 36 GB Mac, so memory is not a problem during development. The 8 GB laptop must still be able to run **one worker container** in Step 4. Plan its memory use carefully.

## 4. Models

**Starting model (must fit on the 8 GB laptop with all 3 containers running):**
- `ibm-granite/granite-3.0-1b-a400m-instruct`
  - About 1.3 billion parameters in total, about 400 million used per token.
  - 32 experts per layer, 8 picked per token.
  - About 2.7 GB of weights in bf16.

**Next step up:** `allenai/OLMoE-1B-7B` (or its instruct version). About 7 billion total, about 1 billion used per token, 64 experts per layer.

**Later, when the 36 GB Mac is the master:** `Qwen/Qwen3-30B-A3B` (or a similar model sized to the combined RAM of all laptops).

**Please verify before planning:**
- That these exact model names exist on Hugging Face and work with the current version of Transformers.
- The real memory use of each model in fp32, bf16 and int8 on CPU.
- How each model stores its experts in code. Some models keep each expert as a separate piece. Others (I believe Granite MoE does this) keep all experts of a layer packed together in one big tensor, so giving a worker "experts 8–15" means cutting those experts out of the packed tensor. Plan for both cases.

**Memory plan for the 8 GB laptop (3 containers):**
- The master container loads only the shared parts (embeddings, attention, router, output layer) plus its own experts.
- Each worker container loads only its own experts, never the full model.
- Docker Desktop has a memory limit setting. Include what this limit should be, and how much memory each container should be allowed.

## 5. How the system works (architecture)

### 5.1 Roles

**Master node:**
- **Controller:** keeps a list of all nodes, which experts each node holds, and whether each node is healthy.
- **Inference engine:** runs the shared parts of the model, runs the router, sends expert work to the right nodes, and puts the results together.
- **API server:** offers the OpenAI-style API to the chat UI and other tools.
- **React chat UI:** chat window plus a live view of the distributed system.
- **Also holds some experts**, like a worker.

**Worker node:**
- Loads only its assigned experts.
- Registers with the master when it starts.
- Sends regular heartbeats ("I am still alive") to the master.
- Receives expert work from the master, runs it, and sends results back.

### 5.2 What happens for one token (step by step)

1. The master turns the text into numbers (embeddings).
2. For each layer:
   - The master runs the attention part.
   - The master runs the router, which picks the top experts for this token (for example 8 out of 32).
   - The master groups the picked experts by which node holds them.
   - The master sends the data to each of those nodes **at the same time** (in parallel, not one after another).
   - Each node runs its experts and sends the results back.
   - The master combines the results using the router's weights.
3. After the last layer, the master picks the next token and streams it to the UI.

### 5.3 Expert assignment

- Simplest approach: a node holding "experts 0–10" holds experts 0–10 in **every** layer. Please confirm whether this is a good approach or suggest a better one.
- Before the master accepts any chat request, it must check that **every expert in every layer is held by at least one healthy node**. If not, it should refuse with a clear error saying which experts are missing.
- Optional: allow the same expert on more than one node (a backup copy), so the system keeps working if one node dies.

## 6. Build approach — build in small steps, test after each one

I want to build **only** the experts-across-workers system, but in steps. Each step adds one new thing, so if something breaks, I know where the problem is.

### Step 1: Everything in one program, no network (on the 36 GB Mac)
- Load the Hugging Face model normally.
- Replace each MoE layer with my own code that routes tokens to expert objects. At this step the experts still live in the same program.
- **Test:** with greedy decoding (temperature 0, always pick the most likely token), my version must produce **the same tokens** as the original, unchanged Transformers model on a set of test prompts. If tiny number differences cause a different token now and then, explain why and set an acceptable rule.
- Also test the "cutting experts out of the packed tensor" code on its own.

### Step 2: 3 Docker containers on one laptop (on the 36 GB Mac)
- Move experts into separate worker containers. Each loads only its own experts.
- The master sends expert work to them over the network.
- Add registration, heartbeats, pings and health status.
- **Test:** output must be the same as Step 1 for the same prompts. If it changed, the bug is in the network/data-sending code, because Step 1 already proved the model code works.
- **Test:** stop one worker container and check that the master notices and reports it correctly.
- **Extra check:** also run the 3 containers with memory limits that fit inside 8 GB total, to prove the starting model could run on the small laptop.

### Step 3: Clear logs and the React UI (on the 36 GB Mac)
- Add detailed logs (see section 9).
- Build the React UI (see section 10).
- **Test:** I send a prompt and can clearly follow every token's path across containers in `docker logs` and in the UI.

### Step 4: Real laptops over Wi-Fi
- Master: the 36 GB Mac (holds the chat UI, controller, shared model parts and some experts).
- Workers: Laptop A (8 GB) and Laptop C (Mac).
- Same Docker image on each laptop (built for both CPU types if needed).
- Workers join the master using the command-line tool, giving the master's IP and their experts.
- Give the 8 GB laptop **fewer experts** than the others, based on its free memory.
- **Before running the model**, include a network check tool that measures the round-trip time between the master and each worker over Wi-Fi, and shows it in the logs and UI.
- **Test:** the same prompts produce the same output as Step 2, now across real machines.
- After this works with the starting model, move to a bigger model sized to the combined RAM.

## 7. Configuration and command line

**Environment variables (examples, please improve):**
- `NODE_ROLE` = `master` or `worker`
- `NODE_ID` = a unique name, e.g. `worker-1`
- `MASTER_ADDR` = master's IP and port, e.g. `192.168.1.10:8000`
- `ADVERTISE_ADDR` = the address other nodes should use to reach **this** node. Inside Docker, a container's own IP is not reachable from other laptops, so each node must announce its laptop's LAN IP and the published port.
- `EXPERT_IDS` = which experts this node holds, e.g. `0-10` or `0,4,8,12`
- `MODEL_ID` = Hugging Face model name
- `DTYPE` = `fp32`, `bf16` or `int8`
- `LOG_FORMAT` = `pretty` or `json`
- `LOG_LEVEL` = `info` or `debug`

**Command-line tool (example):**
- `moe-node join --master 192.168.1.10:8000 --experts 0-15`
- `moe-node status` (shows all nodes, their health and their experts)
- `moe-node leave` (worker leaves cleanly)

All nodes must use the **same model and same model version**. The master should reject a worker that reports a different model.

## 8. Registration, heartbeats and health checks

I want this to work similar to FastChat's controller/worker design (https://github.com/lm-sys/fastchat), where workers register with a controller and send heartbeats, and the controller removes workers that stop responding. Please study it and reuse the good ideas.

**Registration:** when a worker starts, it sends the master:
- node ID, address, model ID, model version
- which experts it holds
- RAM, CPU cores, CPU type

**Two-way health checking:**
- Workers send a heartbeat every few seconds.
- The master also pings each worker on its own schedule.
- Each node has a status: `healthy`, `slow` (responding but late), or `dead` (no response after a timeout).
- All times should be settings, not hard-coded.

**When things go wrong:**
- A worker dies while idle: mark it dead and show it in the UI and logs. Refuse new requests if its experts are not available elsewhere.
- A worker dies **during** a response: decide what happens. Options: stop with a clear error, retry on a backup copy of the experts, or something else. Please recommend one.
- A dead worker comes back: it should re-register automatically.
- The master restarts: workers should notice and re-register automatically.

**Internal endpoints (examples, please improve):**
- `POST /register`, `POST /heartbeat`, `GET /health`, `POST /leave` on the master
- `POST /expert/forward` on each worker (runs experts on the data sent to it)
- `GET /nodes` on the master (current status of all nodes, used by the UI and CLI)

**Security:** this is for a home network only. Add at least a simple shared secret (a token all nodes must send) so random devices on the network cannot join or send work.

## 9. Logs (very important)

When I send a prompt, I want to watch the work move across nodes in `docker logs`. Logs should be easy to read and consistent on every node.

**Every log line about a request should include:** time, node ID, request ID, token number, layer number.

**Examples of what I want to see (format is only an example):**
- Master: request received, prompt length, model
- Master: `req=a1b2 token=14 layer=6 router picked experts [3, 7, 19, 22, ...]`
- Master: `req=a1b2 token=14 layer=6 sending experts [3,7] → worker-1, [19,22] → worker-2`
- Worker: `req=a1b2 token=14 layer=6 ran experts [3,7] in 4.1 ms`
- Master: `req=a1b2 token=14 layer=6 all results back in 9.8 ms (slowest: worker-2)`
- Master: token produced, tokens per second so far
- Master, end of request: a summary with total tokens, total time, tokens per second, and how much of the time was spent on network vs. computing on each node

**Also:**
- Per-token logs at `debug` level, per-request summaries at `info` level, so normal logs are not flooded.
- Optional JSON log format for later analysis.
- Registration, heartbeat and health status changes are always logged clearly (e.g. `worker-2 status: healthy → dead`).

## 10. React UI

**Chat page:**
- Normal chat window with streaming responses.
- Uses the same OpenAI-style API as any other tool would.

**Live system view (on the same page or a second tab):**
- **Node map:** every node, its status (healthy / slow / dead), its IP, RAM and CPU.
- **Expert placement grid:** a grid of layers × experts, colored by which node holds each expert.
- **Live routing view:** while a response is being generated, show which experts were picked for each token and which node ran them (for example, cells in the grid lighting up).
- **Stats:** tokens per second, time per layer, and time per node.
- The UI should get live updates from the master using WebSockets or Server-Sent Events. Please recommend which one.

## 11. Docker

- **One Docker image** for all nodes. The role comes from environment variables.
- Built for both x86 and ARM (multi-architecture).
- CPU-only PyTorch (no CUDA), to keep the image small.
- Model weights stored in a Docker volume, so they are downloaded once and not every time a container starts.
- A `docker-compose.yml` for Step 2 (1 master + 2 workers on one laptop).
- Simple run commands for Step 4 (one container per laptop).
- Explain the networking setup for Step 4: which ports to publish, whether host networking works on Docker Desktop for Mac, and how each node announces its real LAN IP.
- Docker health checks on each container.

## 12. Networking across laptops

- Explain how to give laptops fixed IP addresses (for example, a DHCP reservation in the router).
- List which ports need to be open in each laptop's firewall (including the macOS firewall).
- Explain how to test that laptops can reach each other before running the system.

## 13. Performance expectations

## 13. Performance expectations

- I understand this will probably be **slower** than running the whole model on one laptop, because every MoE layer needs a network round trip for every token.
- **All laptops use Wi-Fi.** Wi-Fi adds delay to every round trip, and that delay changes a lot from moment to moment (sudden spikes). Please design with this in mind.
- Please estimate realistic tokens per second for: one laptop alone, 3 containers on one laptop, and 3 laptops over Wi-Fi.
- Suggest ways to reduce network cost, for example:
  - sending all tokens that need the same expert in one message (batching)
  - sending data in a compact binary format instead of JSON
  - keeping network connections open instead of opening a new one for every call
  - placing the most-used experts on the master, so fewer calls go over Wi-Fi
  - measuring which experts are picked most often, and using that to decide placement
- Give practical Wi-Fi tips: for example, keeping all laptops on the same Wi-Fi band (5 GHz), staying close to the router, and turning off power-saving settings that can slow Wi-Fi on laptops.
- Health check timeouts must handle Wi-Fi delay spikes, so a node is not marked "dead" just because of one slow reply.

## 14. Testing

- Correctness test: distributed output matches the original single-machine Transformers output (greedy decoding).
- Unit tests for expert cutting/slicing and for combining results.
- Failure tests: kill a worker while idle and during a response, restart the master, and add a worker with the wrong model.
- A simple benchmark script that runs a fixed set of prompts and records tokens per second and network time.

## 15. Technology preferences

- Python, PyTorch (CPU), Hugging Face Transformers
- FastAPI for the HTTP services (or suggest something better)
- React with Vite for the UI
- Docker and docker-compose
- If you recommend gRPC or another way for nodes to talk to each other instead of HTTP, explain the trade-offs in simple terms.

## 16. Out of scope for now

- Training or fine-tuning models
- GPUs
- Running over the internet (home network only)
- Many users at the same time (please tell me whether supporting a few users at once changes the design much)

## 17. Questions — answer these or state your assumption before planning

1. Should nodes talk using HTTP + JSON (easiest to debug), HTTP + binary data, or gRPC (faster)? Which do you recommend for a learning project, and why?
2. Should each expert live on exactly one node, or should important experts have backup copies?
3. Should the master send expert work per token, or group work per layer? How much does this matter on a home network?
4. Which number format should I use on CPU (fp32, bf16 or int8)? Which works reliably on both x86 and ARM in PyTorch?
5. How should the system handle one laptop being much slower than the others, since the master has to wait for the slowest node at every layer?
6. How should the KV cache (the saved attention data that speeds up generating each next token) work? I assume it stays on the master, since only the master runs attention. Please confirm.
7. Is my plan to hold "the same expert numbers in every layer" on a node a good idea?
8. Anything in this design you think is a mistake? Tell me directly, with the reason and a better alternative.

## 18. What I want from you

**Do not write code yet.**

Create a detailed **phase-by-phase implementation plan in Markdown format**, following the 4 build steps in section 6 (you can split them into smaller phases if that helps). For each phase include:

- **Goal:** what this phase achieves, in one or two sentences
- **What gets built:** components, files and folders
- **Key decisions:** what needs to be decided in this phase, with your recommendation
- **How to test it:** exact tests to run
- **Done when:** a clear, checkable finish line
- **Risks:** what is most likely to go wrong, and how to spot it early

Also include:
- A proposed folder/repository structure
- A list of the internal APIs between master and workers, with what each one sends and receives
- Your answers to the questions in section 17

Please write in plain, simple English and explain any technical term the first time you use it.