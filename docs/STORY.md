# The Story So Far

This is the story of this project, from the first idea to where we are now (the start of Step 4, real laptops, October 2026). It explains what we built and why, what went wrong and how we fixed it, in plain English. Technical words are explained the first time they appear. The detailed plan and all measurements are in [IMPLEMENTATION.md](IMPLEMENTATION.md); this document is the "why" behind it.

---

## 1. The idea

We wanted to run one AI language model across three laptops at the same time, using only their normal processors (CPUs), no graphics cards (GPUs). The laptops are ordinary: a Mac with 36 GB of memory, a laptop with only 8 GB, and a second Mac. On its own, the small laptop could never hold a big model. Together, the three have much more memory than any one of them.

The trick that makes this possible is a special kind of model called a **Mixture-of-Experts** model, or **MoE** for short. A normal language model is built from layers, and in every layer every word goes through the same big block of maths. An MoE model replaces that big block with many smaller blocks called **experts**, plus a tiny part called the **router**. For every word, the router looks at it and picks a few experts (for example 8 out of 32) that should work on it. The other experts do nothing for that word.

(Strictly speaking, models work on **tokens**, not words. A token is a word or a piece of a word, like "light" + "house". We will say "token" from here on.)

This gave us the idea: **if each token only needs a few experts, the experts do not all have to live on the same computer.** We could put experts 0–9 on one laptop, 10–20 on the second and 21–31 on the third. Every time the router picks experts, the token's numbers are sent to the laptops that hold them, they do their part, and send the answer back.

One laptop is the boss, called the **master**. It runs everything that is *not* an expert, decides where work goes, collects the answers, and talks to the user. The other laptops are **workers**: they only hold experts and do what the master asks.

This is mainly a learning and demo project. Seeing the work move between machines matters more to us than raw speed.

## 2. Why we did not just use Ollama, llama.cpp or vLLM

The first question was: does a tool for this already exist? We looked at the three best-known ways of running models yourself:

- **llama.cpp** is a fast program, written in C++, for running models on normal computers. **Ollama** is a friendly app built on top of llama.cpp. llama.cpp can spread a model over several computers ("RPC mode"), but it splits the model **by layers**: computer 1 gets layers 1–12, computer 2 gets layers 13–24. Every token then travels through all computers one after the other. It cannot put **specific experts** on specific computers, which is exactly our idea.
- **vLLM** is a fast server for running models, mostly used with GPUs in data centres. Its CPU version can split work inside **one** machine, and its multi-machine support is built for clusters of GPUs. It also does not do "these experts live on that laptop" for CPU laptops.

So the expert-splitting had to be **our own code**. We chose Python with **PyTorch** (the library that does the maths of neural networks) and **Hugging Face Transformers** (the library that knows how to build and run thousands of published models). We do not train or invent a model; we take a ready-made model from Hugging Face and only write the code that runs it across machines.

We still wanted our system to feel like those tools from the outside. Ollama, vLLM and OpenAI all speak the same "language" for chat programs: an **API** (a fixed way for programs to talk to each other) called the OpenAI-style chat API, at an address ending in `/v1/chat/completions`. If our master speaks that language too, every chat app made for Ollama or vLLM can talk to our cluster without changes. That is what "compatible with Ollama and vLLM" means in this project: same language at the door, our own engine inside. llama.cpp may come back at the very end only as a speed comparison ("how fast is one Mac on its own?").

## 3. Choosing the models

Before writing any code we checked the real models on Hugging Face, with the exact versions of the libraries we would use, instead of trusting numbers from memory. We built each model "empty" (only the shapes, no memory used) to measure it exactly.

Some words first. A **parameter** is one number inside the model; a "1 billion parameter" model has a billion of them. How much memory a parameter needs depends on the **number format**:

- **fp32**: 4 bytes per number, the most exact.
- **bf16**: 2 bytes per number, slightly rounded. Models are published in this format.
- **int8**: 1 byte per number, needs extra work and loses a little quality.

| Model | Role | Total size (bf16) | Experts per layer | Picked per token | Layers | Shared parts (stay on the master) |
|---|---|---|---|---|---|---|
| IBM Granite 3.0 1B-A400M | **first model** (now) | 2.7 GB | 32 | 8 | 24 | 0.25 GB |
| Qwen1.5-MoE-A2.7B | second model (real laptops) | 28.6 GB | 60 | 4 | 24 | 3.7 GB |
| Qwen3-30B-A3B | stretch goal | 61 GB (~32 GB in int8) | 128 | 8 | 48 | 3.1 GB |

**Granite** is our test model: small (2.7 GB), quick to download, and small enough that all three parts would fit on the 8 GB laptop. Its answers are not very clever, but it is a real MoE model with the same structure as the big ones. **Qwen1.5-MoE** is the "real" target: much better answers, and at 28.6 GB too big for any single laptop except the 36 GB Mac, which makes it a good showcase for spreading experts. **Qwen3-30B** is a modern, strong model, but even split three ways it needs the smaller int8 format, so it is a stretch goal. The original plan named OLMoE as the second model; we replaced it with Qwen1.5-MoE. We also ruled out the newest Qwen3.5/3.6 models: they are much bigger and add new kinds of attention and an image part, too complex for this project.

The most important number in that table is the last column. **Around 90 % of each model is experts** (Granite 91 %, Qwen1.5 87 %, Qwen3 95 %). The shared parts that every token needs are small. So if we spread the experts, we spread almost all of the memory. That is why the whole idea is worth doing.

## 4. Why experts can move, and why the rest cannot

It is worth understanding why experts are so easy to move, because not every part of a model is.

An expert is a small, self-contained calculation: numbers go in, numbers come out. It **remembers nothing** from one token to the next. It also only works for the tokens the router sends to it. So a computer holding an expert just needs to receive "here are some numbers, run your expert on them" and send back the result. If the same message is sent twice, the answer is the same. We call such workers **stateless**. That makes them easy to put anywhere, easy to restart, and safe to ask again if a message gets lost.

The other parts of the model are different, and they all stay on the master:

- **Attention** is the part where a token "looks back" at all earlier tokens to understand the context. To do this quickly, the model keeps a memory of every earlier token, called the **KV cache** (the "keys" and "values" of attention). This memory grows with every token and is needed by every layer, every time. Moving attention to another computer would mean moving this growing memory around all the time. Because only the master runs attention, the KV cache lives only on the master, and the workers stay simple.
- **The router** must run before anyone knows which experts are needed, so it must sit where the token is: on the master.
- **Embeddings** (turning token numbers into vectors, long lists of numbers the model works with), **norms** (small "keep the numbers in a good range" steps) and the **output layer** (turning the final vector back into "which token comes next") are used by every token in every step. Sending them away would add network trips without saving much memory.

Some models also have a **shared expert**: an expert that *every* token uses, next to the routed ones. Qwen1.5-MoE has one. It looks like an expert, but it behaves like a shared part, because it is always needed. Putting it on a worker would add a network trip for every token in every layer and save nothing. So the shared expert stays on the master too.

And this is also why our idea only works for MoE models. In a normal ("dense") model, every token goes through the whole feed-forward block of every layer. There is nothing to choose, so there is nothing that can usefully live elsewhere: splitting it would mean every computer works on every token, with a network trip in between, every time.

## 5. How the model is cut in two: "one line" in Transformers

The design decision we are most happy with is how little of the model we touch.

When we read the code inside Transformers, we found that in every MoE layer of all three models, the expert work happens in a single line:

```python
output = self.experts(hidden_states, top_k_index, top_k_weights)
```

In plain words: "here are the tokens, here are the experts the router picked for each, here is how much each pick counts; give me the combined result." We replace that one `experts` object with our own object, which we call the **dispatcher**. It takes the same inputs and gives the same output, but instead of doing the work itself, it sends each piece of work to whichever computer holds the expert, and puts the answers back together. Everything else (attention, the router, the KV cache, the chat formatting, choosing the next token) stays exactly as Transformers wrote and tested it. We did not rewrite the model.

There was one more thing to understand: **how experts are stored**. In Transformers' memory, all 32 experts of a layer are **packed** into one big block of numbers, like a stack of 32 sheets of paper, where expert number *i* is sheet *i*. To give a worker "experts 10–20", we cut sheets 10 to 20 out of the stack. On disk, Granite stores them packed the same way; the Qwen models store each expert as a separate piece, which we will glue into the same stack. Each model family gets one small **adapter**, a piece of code that knows only that model's file names. Everything else works on the common stacked layout.

## 6. Phase 0: the reference answers

Before building anything, we needed a way to know whether our version was right. We wrote ten fixed test prompts (a short question, a long one, a list, some code, a maths question, a German one, and so on) and ran them through plain, unchanged Transformers. We used **greedy decoding**: always pick the single most likely next token, no randomness. We saved every token of every answer. These are our **golden files**: the answers our distributed system must reproduce.

We ran it twice to make sure plain Transformers itself always gives the same answer. It did. Plain Transformers on the 36 GB Mac makes about **17–18 tokens per second**; that is our "one laptop" speed to compare against. We also noticed that fp32 and bf16 give somewhat different answers to the same prompt, and that bf16 sometimes has exact ties between the two best tokens. Both are normal rounding effects, so each number format got its own golden file.

## 7. Step 1: everything in one program

We first built the whole idea **inside one program on one Mac, with no network at all**, with pretend "nodes" that were just objects in memory. If the answers were wrong here, the problem could only be in our maths, not in networking.

**Cutting and loading experts (Phase 1.1).** We wrote the code that cuts experts out of the stack, and a loader that reads **only** a node's own experts straight from the model file, without loading the rest. The file format, **safetensors**, allows reading part of a big block without reading all of it. We checked that experts read this way are **bit-for-bit identical** to the ones Transformers builds.

The first surprise was about memory. Our memory test gave nonsense numbers. The reason: safetensors uses **memory mapping**, where the operating system pretends the whole file is in memory and only loads pages when they are touched. A "slice" of the file is not a real copy; it is a window onto the mapped file. If we kept those windows, a node's memory would be tied to the whole file and impossible to predict. So the loader now always makes a real copy of the experts it keeps. After that, loading 4 of 32 experts kept about 4/32 of the memory, as it should. We also learned that while loading, a node briefly needs about **twice** its expert size (the file pages plus our copy). This mattered again later.

**Running experts ourselves (Phase 1.2).** We wrote our own version of the expert maths and compared it with Transformers' version. Our goal was **bit-identical** numbers, not just "close", because then any later difference is clearly a bug. Two surprises showed how picky computers are with numbers:

- **Order of addition matters.** Adding the same experts' results in a different order gives slightly different bits, because computers round after every addition. So the master always adds results in the same order Transformers does: expert 0, 1, 2, and so on.
- **The batch matters.** Multiplying one row of numbers alone gives slightly different bits than multiplying the same row inside a group of rows, because the maths library uses a different method for a single row. So every expert must receive exactly the same group of rows, in the same order, as it would inside Transformers.

With both rules followed, our expert maths was bit-identical to Transformers, for Granite and for the shapes of both Qwen models. We also found that Granite, Qwen1.5 and Qwen3 use the very same expert loop inside Transformers, so one piece of our code serves all three.

**The dispatcher with pretend nodes (Phase 1.3).** Then we swapped the dispatcher into every layer and split the experts across three pretend nodes. Every message between the master and a pretend node was already packed into the exact **binary format** we would later send over the network: a small readable header (which request, which token, which layer) followed by the raw numbers. Raw numbers instead of text (JSON) keep them exact, small and fast.

The result: **identical tokens to the golden files**, in fp32 and bf16, for all ten prompts and for several different ways of splitting the experts (three ranges, every third expert, all on one node, one expert per node). If an expert is missing from the plan, the system refuses with a clear message ("missing experts: [17] in all layers") instead of quietly giving worse text. Our extra code cost about 11–15 % of the speed.

We also tried a faster way of combining results, called **fast mode**, where each node adds up its own experts first and sends back one result per token instead of one per expert. Less data, but a different order of addition. In fp32 the change was invisible. In bf16 it was big enough to change some answers. So the exact mode is the default, and fast mode stays an option we will only consider later, carefully.

## 8. Step 2: real network, real containers

**Docker.** For the next step we used **Docker**. Docker packs a program with everything it needs (Python, libraries, our code) into an **image**, and runs it as a **container**: a small, sealed Linux computer. The same image runs on every laptop, so all nodes run exactly the same software. We built one image for every node; a setting (`NODE_ROLE=master` or `worker`) decides what it does. A `docker-compose` file starts one master and two workers on one laptop, and the model file is downloaded once into a shared storage area (a "volume") that all three use.

**The workers and the master as real services (Phase 2.1).** Each worker is now a small web service that loads only its own experts and waits for work. The master talks to the workers over HTTP (the protocol the web uses), keeping each connection open between calls so it does not pay for opening a new one every time, and calling all workers at the same time.

The master itself must never load the full model, because with bigger models that alone would not fit. We solved this with a small trick: while Transformers builds the model, we put an **empty placeholder** where each layer's experts would go. Transformers then loads everything else normally (with its own conversions) and simply skips the expert weights. The master's model is **242 MB instead of 2,546 MB**, and everything in it is bit-for-bit identical to a normal load. Our first try at this crashed, because Transformers checks the type of the expert objects while setting up; replacing them one level higher, at the MoE block, solved it.

**A company firewall in the way.** Building the image failed with "certificate verify failed". This Mac's internet goes through **Zscaler**, a company security filter that inspects encrypted traffic and re-signs it with its own certificate (a digital ID card). The Mac trusts that certificate; a fresh container does not, so downloads looked like an attack. We added an optional folder where such a certificate can be placed; the image then trusts it. On laptops without such a filter the folder is empty and nothing changes. We checked both ways.

**A surprise: Mac and Linux disagree.** The containers gave different answers from our golden files in bf16. We first suspected our network code, but then ran **plain, unchanged Transformers inside a container**: it also gave different answers in bf16 (5 of 10 prompts), while in fp32 all 10 were the same. The reason: containers run Linux, and Linux and macOS use different low-level maths libraries that round bf16 numbers slightly differently. Not our bug. So we made separate golden files for Linux, and the containers matched those **exactly**, in fp32 and bf16. This finding shapes the plan for the real laptops (see section 13).

**Why Docker on a Mac is slow.** Speed was the biggest surprise. In Docker, the cluster made only about **1.8 tokens per second**, compared with 16.5 for our code in one program on macOS. We measured where the time goes:

| Measurement | Result |
|---|---|
| One call from the master to a worker (1 token, 4 experts) | **2 ms** in total, of which only **0.6 ms** is network, HTTP and packing |
| Expert work for one token, macOS directly | 0.23–0.40 ms |
| The same expert work inside Docker | 2–5.5 ms (2–15× slower) |
| Reading memory with one thread, macOS / inside Docker | ~55 GB/s / ~4 GB/s |
| Plain Transformers: macOS / inside one container | ~18 / ~5–6 tokens per second |

So the network was not the problem. On a Mac, Docker cannot run Linux containers directly; it runs a hidden **Linux virtual machine** (a pretend computer inside the Mac), and the containers run inside that. For one token, expert work is mostly **reading weights from memory**, and inside this virtual machine memory reading is far slower than on macOS itself. Giving containers more or fewer CPU threads made almost no difference. This is a property of Docker Desktop on a Mac, not of our code; on a Linux laptop, Docker runs without a virtual machine.

**Workers that come and go (Phase 2.2).** Next, the cluster had to handle real life: computers crash, freeze, restart. We copied good ideas from **FastChat**, an open-source project with a similar "controller and workers" design. Each worker now **registers** with the master by itself ("here I am, this is my address, these are my experts, this is my model and number format"), and then sends a **heartbeat**, a short "I'm still alive" message, every 2 seconds. The master also **pings** each worker every 3 seconds, so it notices a frozen worker even if that worker cannot send anything. Each worker gets one of five states:

| State | Meaning |
|---|---|
| joining | registered, still loading its experts |
| healthy | normal (after 3 good contacts in a row) |
| slow | answers, but late (typical ping above 250 ms, or heartbeats late) |
| dead | nothing heard for 15 seconds, or a call failed even after one retry |
| left | said goodbye |

Only healthy and slow workers get work. A worker with a different model, version or number format is turned away, with the reason in both logs. If a worker dies **in the middle of an answer**, the master tries once more and then stops with a clear error ("worker-2 stopped responding; experts [22, 23, 26, 27, 31] in layer 0 are unavailable"). It never skips experts silently, because that would quietly produce worse text. Each time the master starts, it picks a random ID (its "epoch"), so workers notice a restarted master and register again by themselves. We tested all of this by really stopping, freezing and restarting containers, and by slowing a worker's network on purpose by 350 ms:

| What we did | What happened |
|---|---|
| stopped worker-2 | counted dead after 14.9 s; requests got a clear "experts missing" answer |
| started it again | it registered by itself and became healthy; same answers as before |
| froze worker-2 during an answer | the answer ended with a clear error after 10.1 s (2 tries of 5 s); the master kept working |
| restarted the master | both workers registered again by themselves |
| added 350 ms network delay | marked slow, never dead, still used |
| worker with another number format, or a wrong password | turned away, reason in both logs |

**Talking like OpenAI (Phase 2.3).** The master now offers the OpenAI-style chat API, including **streaming**: the answer is sent piece by piece while it is being made, like ChatGPT typing, instead of all at once at the end. It supports the usual settings people expect: **temperature** (how random the answer is; 0 means always the most likely token, and then we get exactly the golden answer), **top_p** (only pick from the most likely tokens), **max_tokens**, **stop** words, and a **seed** (a starting number that makes a random answer repeatable). Requests are answered one at a time, and others wait in line. If someone closes their chat in the middle, the master stops and moves on. We tested it with the official OpenAI Python library, and then with **Open WebUI**, a free ChatGPT-like website you run yourself: it found our model and chatted with the cluster without any changes.

**Will it fit on the small laptop? (Phase 2.4).** Finally, we gave each container a hard memory limit, as a small laptop would: master 2.5 GB, workers 2 GB each, 6.5 GB in total, and no "swap" (using the disk as extra memory), so that a container that needs too much is really stopped instead of quietly getting slow. We ran all ten test prompts and a long conversation (over 1,200 tokens of context, which makes the master's attention memory grow).

| Container | Limit | Highest use | What the program itself needs |
|---|---|---|---|
| master | 2,560 MB | 2,407 MB | ~1,330 MB |
| worker-1 | 2,048 MB | 1,916 MB (while loading) | ~1,175 MB |
| worker-2 | 2,048 MB | 1,919 MB (while loading) | ~1,180 MB |

All answers were identical to the golden files and **nothing ran out of memory**. The "highest use" looks close to the limits, but a big part of it is copies of files kept in memory, which Linux drops whenever memory gets tight; what the programs really need is the last column. The workers' highest point is while loading, the "twice the size while loading" effect we found in Phase 1.1.

## 9. Step 3: watching it work

At the end of Step 2 the cluster worked, but you couldn't *see* it work. You typed a question, an answer came back, and everything in between was hidden. Since the point of the project is to watch distributed inference happen, Step 3 was about making it visible: first in the logs, then live on a web page.

**Logs that follow one token (Phase 3.1).** A **log** is the running diary a program writes about what it is doing. We gave every node the same log format: time, node name, and, inside a request, three labels: `req=` (which request), `token=` (which pass through the model; 0 is the whole prompt at once, 1 the first new token, and so on) and `layer=`. With the detailed setting (`LOG_LEVEL=debug`), every token and every layer gets a few lines:

- the master: "router picked experts [3, 6, 9, 12, 18, 20, 21, 28]" and "sending [6, 9] → local, [12, 18, 20] → worker-1, …"
- each worker: "ran experts [12, 18, 20] on 1 token in 8.8 ms"
- the master again: "all results back in 15 ms (slowest: worker-1)"

Because every node uses the same labels, a search for `req=cd116308 token=5 layer=6` across all three containers shows one complete step from three angles: the master's choice, and each worker doing exactly the experts it was sent. At the normal setting, a request prints just two lines: "received" and a summary like "9 tokens in 5.9 s = 1.5 tok/s | worker-1 net 0.4 s / compute 2.9 s". We split each worker's time into **network time** and **compute time**: the master knows how long it waited, and the worker reports how long it computed, so the difference is the network. This confirmed what Step 2 had measured: in Docker on the Mac, the computing, not the network, took most of the time.

**Live events (Phase 3.2).** A web page can't read log files, so the master got a new address, `/api/events`, that **streams** what happens as it happens. It uses **Server-Sent Events**: the browser opens one connection and the server keeps sending small messages through it. When you connect, you get a "hello" with the model and which node holds which expert. After that you get a message whenever a node's health changes, when a request starts and ends, and one message **per new token** with all 24 layers inside: per layer, the 8 experts the router picked, which computer ran each, how much each counted, and how long the layer waited. One such message is about 11 KB. Collecting this costs a little time, so the master only does it while someone is actually watching.

**The web page (Phase 3.3).** Finally, a web page, built with **React** (a popular library for building interactive web pages). The master serves it itself, so you just open `http://localhost:8000`.

![The web page: chat on the left, the live cluster view on the right](images/ui-chat-and-grid.png)

- **Left:** a normal chat window. The answer appears word by word.
- **Right, top:** one card per computer, coloured green, yellow or red by health, with its address, experts, round-trip time and free memory.
- **Right, middle:** a grid with the 24 layers as rows and the 32 experts as columns. Each cell has the colour of the computer that holds it. While the answer is being written, the 8 experts the router picked in each layer **light up**, token by token, so you can literally watch the work jump between computers. Clicking an earlier token, or dragging a slider, replays it.
- **Right, bottom:** speed, and for each computer a bar showing compute time versus network time.
- On a phone, chat and cluster view become two tabs.

We checked the page against the logs: for one token in layer 6, the page lit experts 3, 6, 9, 12, 18, 20, 21 and 28, and the master's log for the same step said exactly "router picked experts [3, 6, 9, 12, 18, 20, 21, 28]". The page even shows the label to search for in the logs. When we stopped worker-2, its card turned yellow after about 5 seconds and red at 15, a red banner listed the missing experts, and its cells in the grid turned red. After restarting it, everything went green again by itself.

Testing the page in a real browser found three bugs that the code's own tests could not: the page crashed the first time you sent a message (a small React rule we had broken), the grid was cut off on phones, and the grid kept showing a dead worker's colour, because the page only learned who holds which expert once, when it connected. All three are fixed.

## 10. Getting ready for real laptops: no Docker

Step 4 is the real thing: three laptops on Wi-Fi. The original plan was to run the same Docker image on every laptop. We changed that, for the reason Step 2 had shown: on a Mac, Docker runs everything inside a hidden virtual machine, and there the expert work was **2–15× slower**. Two of the three laptops are Macs. So the laptops now run the code **directly** ("natively"): clone the repo, create a Python environment, start the node. Docker stays for the three-containers-on-one-Mac test cluster.

That meant a new laptop had to be easy to set up without Docker. Each laptop really has only a few things of its own: whether it is the master or a worker, its name, which experts it holds, its IP address, and the master's address. We discussed two ways to give it these: a **wizard** (a program that asks questions) or a **settings file**. We chose both, working together. The wizard asks once, checks the answers, and writes them into a small settings file, `node.env`. Every later start just reads the file, so no questions. To change something, you edit the file or run the wizard again.

Three design choices came out of that discussion:

- **The laptop's own IP is found automatically, at every start** (`auto`). Home Wi-Fi often hands out a different IP after a restart, and an IP saved once would then be wrong.
- **The wizard asks the master which experts are still free**, and suggests a share. The master already knew this; it's the same information the web page uses to colour its grid.
- **A cluster token** (a shared password) is now part of setup. In Docker it hardly mattered, but on real Wi-Fi anyone on the network could otherwise send work to the laptops.

Any number of workers works: one, two or ten. There is no fixed list of workers anywhere, since they register themselves (Phase 2.2). The only rule is that together the laptops hold all 32 experts. More laptops means less memory per laptop, but not more speed. Every extra laptop is another Wi-Fi round trip to wait for in each layer, and the slowest one decides how long that layer takes.

What was built:

| File | What it is |
|---|---|
| `scripts/install.sh` | One-time install: finds Python 3.12, creates `.venv` with `python3.12 -m venv .venv`, installs the exact versions from `requirements.txt`, and builds the web page if Node.js is installed |
| `./moe` | The one command on every laptop: `setup`, `serve`, `join`, `status`, `leave`, `download`. It uses `.venv` by itself, so there's nothing to "activate". On a Mac it also keeps the laptop from going to sleep while running |
| `./moe setup` | The wizard. It checks the answers as you go: can it reach the master, is the token right, which experts are still free, do they fit in memory. It also downloads the model, then saves everything to `node.env` |
| `node.env.example` | A sample settings file with explanations, for setting up by hand without the wizard |
| `docs/LAPTOP_SETUP.md` | The step-by-step guide. It includes the by-hand commands, example values for master + 2 workers and master + 3 workers, day-to-day commands and a troubleshooting table |
| `docs/LAPTOP_HANDOFF.md` | Instructions for an AI agent. On the new laptop you just say: "Read docs/LAPTOP_HANDOFF.md and set up this laptop." |
| `pipeline-parallel-prototype/` | The renamed `distributed-inference-project/`. It's an older, separate experiment that split a model by layers across 2 GPU machines. A note at the top of its README says so |

So a new laptop needs only this:

```bash
git clone <repo> && cd distributed-inferencing
./scripts/install.sh     # once
./moe setup              # once: the questions, saved to node.env
./moe serve              # the master (start it first)
./moe join               # each worker
```

**How we tested it.** We made a fresh copy of the repo, with only the files git would give a new laptop, and set it up from nothing. `install.sh` built the environment and all 177 quick tests passed inside it. Then the fresh copy, running natively on the Mac, joined the Docker cluster as an extra worker: the wizard checked the master and the token, the worker found its own IP, showed up as healthy, and left cleanly when asked.

**Two things we learned on the way.**

- **The company security filter fooled the IP detection.** The standard trick for "what's my IP?" asks the computer which network card it would use to reach the master. On this Mac, the Zscaler security filter (the same one that broke Docker's downloads in Step 2) has its own pretend network card that catches all traffic to unknown addresses. So the answer was that card's address, which no other laptop can reach. Now only real Wi-Fi and Ethernet cards count.
- **Free memory on a Mac** wasn't measured before: the code only knew Linux's way. Now Mac workers report their free memory too, which the wizard uses to warn if the chosen experts won't fit.

## 11. Could the bigger Qwen models run?

With everything working for Granite, the natural question was: if every laptop were like the 36 GB Mac, could the big Qwen models run across two or three of them? Using the measured sizes from section 3:

| Model | Size (bf16) | This Mac + 1 other | This Mac + 2 others |
|---|---|---|---|
| Qwen1.5-MoE (60 experts × 415 MB) | 28.6 GB | ✅ easy: ~16 GB on the master, ~13 GB on the worker | ✅ ~10 GB each |
| Qwen3-30B (128 experts × 453 MB) | 61 GB | ❌ too tight: ~31 GB each leaves macOS almost nothing | ✅ memory OK, ~21 GB each |

So: yes for Qwen1.5-MoE on two or three laptops, and Qwen3-30B only on three. Three catches remain:

- **The code can't run Qwen yet.** Each model family stores its experts differently in its files. So far only Granite's adapter (the small piece of code that knows a model's file layout, section 5) is written. The Qwen adapters are the plan's Phases 4.5 and 4.6.
- **Disk space.** Today every laptop downloads the whole model file, even though it only loads its own experts. For Qwen3 that's 61 GB per laptop, and the Mac had 39 GB free. The planned fix is to cut each laptop's experts into a smaller file of its own, or to use the half-size FP8 version.
- **Speed.** Qwen3 has 48 layers instead of 24: twice as many Wi-Fi round trips per token.

## 12. Where we are

At the start of Step 4 (October 2026), on one Mac:

- The model is split: the master holds the shared parts and some experts, the workers hold the rest, and nobody holds the full model.
- The answers are **exactly** the same as plain Transformers on the same platform, checked token by token.
- Workers join, leave, die and come back by themselves, and the master always knows who is alive.
- Any OpenAI-style chat tool can use it, with streaming.
- It fits inside an 8 GB laptop's budget.
- You can **watch it work**: logs that follow one token across all nodes, and a web page where the experts light up as the answer is written.
- A new laptop can join with four commands, without Docker, and an AI agent can do the setup from a single instruction.

What's not done yet: running on the real laptops over Wi-Fi, and speed.

## 13. What comes next, and what we expect on real laptops

**Step 4: real laptops over Wi-Fi.** The repo gets cloned onto the other laptops, each one is set up with `./moe setup`, and the cluster runs across them. Here is what we expect, and why.

*The network will decide the speed.* Every token needs one round trip to the workers for **every MoE layer**: 24 for Granite and Qwen1.5, 48 for Qwen3. On Wi-Fi a round trip typically takes 3–10 ms, plus sudden spikes. 24 layers × 5 ms is already 120 ms per token, before any computing. Running natively instead of in Docker removes the slow computing we measured in Step 2, so we expect roughly **3–7 tokens per second** for Granite across the laptops. That's still slower than one laptop alone, which is the price of the idea, and we accept it for a learning project. A small network-check tool (the rest of Phase 4.1) will measure the real round-trip times first and print the best speed that network allows. The main ways to lose less time are already planned: sending all work for a node in one message per layer (already done), putting the most-used experts on the master (no network at all for those), and good Wi-Fi habits (5 GHz band, close to the router, power saving off).

*Different computers round differently.* We learned that macOS and Linux already give different answers in bf16. Across three real laptops, each expert runs on a different machine, so one answer mixes their rounding. We therefore decided on two checks. In **fp32**, the answers must be **exactly** the golden ones (macOS and Linux agreed perfectly in fp32). In **bf16**, the normal setting, a different token is only allowed where the model was almost undecided between its two best tokens (a "near-tie"), and we also read some answers by hand to make sure they make sense.

*Memory will be fine for Granite, and planned for Qwen.* Granite fits easily. For Qwen1.5-MoE, most experts go on the 36 GB Mac, which also means fewer Wi-Fi trips, and the 8 GB laptop gets only a few experts. The exact split depends on Laptop C's memory, which we still need to find out.

*Then the bigger models.* Once Granite works across the laptops, we switch to Qwen1.5-MoE with its own adapter and its own golden files. Then, if memory and disk allow (section 11), we try Qwen3-30B.

## Glossary

| Word | Meaning |
|---|---|
| Token | A word or piece of a word; models read and write tokens |
| MoE (Mixture-of-Experts) | A model where each layer has many small "experts" and a router picks a few for each token |
| Expert | A small, self-contained block of maths inside an MoE layer |
| Router | The tiny part that picks which experts each token uses |
| Shared parts | Everything that is not a routed expert (attention, router, embeddings, output); always on the master |
| Shared expert | An expert every token uses (Qwen1.5 has one); kept on the master |
| Attention / KV cache | The part that looks back at earlier tokens, and its memory of them; only on the master |
| Master / worker | The boss computer that runs the model and talks to users / the helpers that only hold experts |
| Dispatcher | Our replacement for a layer's experts: sends work to the right computer and combines the answers |
| fp32 / bf16 / int8 | Number formats: 4, 2 or 1 byte per number; smaller is less exact |
| Golden files | Saved answers from plain Transformers that our system must reproduce |
| Greedy decoding | Always picking the most likely next token (no randomness) |
| Docker image / container | A packed program with everything it needs / a running copy of it, like a small sealed computer |
| Virtual machine | A pretend computer running inside a real one (how Docker runs on a Mac) |
| Heartbeat / ping | "I'm alive" messages from workers / the master's own check on each worker |
| Streaming | Sending the answer piece by piece while it is being made |
| API | A fixed way for programs to talk to each other (here: the OpenAI-style chat API) |
| Log | The running diary a program writes about what it's doing |
| Server-Sent Events | One open connection over which a server keeps sending small messages to a web page |
| React | A popular library for building interactive web pages |
| Native | Running directly on the laptop's own operating system, not inside Docker |
| IP address / port | A computer's number on the network / which "door" on it a program listens at |
| venv (`.venv`) | A private folder with its own Python and packages, so the project's versions don't clash with anything else |
| `node.env` | A laptop's small settings file (role, name, experts, master's address, token), written by the wizard |
| Wizard | A program that asks a few questions, checks the answers and saves them |
| Cluster token | A shared password between the master and its workers |
