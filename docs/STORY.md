# The Story So Far

This is the story of this project, from the first idea to where we are now (end of Step 2, October 2026). It explains what we built and why, what went wrong and how we fixed it, in plain English. Technical words are explained the first time they appear. The detailed plan and all measurements are in [IMPLEMENTATION.md](IMPLEMENTATION.md); this document is the "why" behind it.

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

**A surprise: Mac and Linux disagree.** The containers gave different answers from our golden files in bf16. We first suspected our network code, but then ran **plain, unchanged Transformers inside a container**: it also gave different answers in bf16 (5 of 10 prompts), while in fp32 all 10 were the same. The reason: containers run Linux, and Linux and macOS use different low-level maths libraries that round bf16 numbers slightly differently. Not our bug. So we made separate golden files for Linux, and the containers matched those **exactly**, in fp32 and bf16. This finding shapes the plan for the real laptops (see section 10).

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

## 9. Where we are

At the end of Step 2, the system does everything the plan asked for on one laptop:

- The model is split: the master holds the shared parts plus experts 0–9, two workers hold 10–20 and 21–31, and nobody holds the full model.
- The answers are **exactly** the same as plain Transformers on the same platform, checked token by token.
- Workers join, leave, die and come back by themselves, and the master always knows who is alive.
- Any OpenAI-style chat tool can use it, with streaming.
- It fits inside an 8 GB laptop's budget.

What it is not yet: easy to watch, and fast.

## 10. What comes next, and what we expect on real laptops

**Step 3: seeing it work.** The point of the project is to *see* distributed inference happen. Next we add clear logs on every node (for each token and layer: which experts the router picked, which worker got which, how long each took) and a web page with a live view: the nodes and their health, a grid of layers × experts coloured by which laptop holds them, and cells lighting up as tokens are processed.

**Step 4: real laptops over Wi-Fi.** Then the same image runs on three real laptops. Here is what we expect, and why.

*The network will decide the speed.* Every token needs one round trip to the workers for **every MoE layer**: 24 for Granite and Qwen1.5, 48 for Qwen3. On Wi-Fi a round trip typically takes 3–10 ms, plus sudden spikes. 24 layers × 5 ms is already 120 ms per token, before any computing. So we expect roughly **2–6 tokens per second** for Granite across the laptops, slower than one laptop alone. That is the price of the idea, and we accept it for a learning project. Before running the model, a small network-check tool will measure the real round-trip times and print the best speed that network allows. The main ways to lose less time are already planned: sending all work for a node in one message per layer (already done), putting the most-used experts on the master (no network at all for those), and good Wi-Fi habits (5 GHz band, close to the router, power saving off).

*Docker on the Macs will still be slow.* Two of our three laptops are Macs, so Docker there again runs inside the hidden Linux virtual machine, with the same slow memory reading we measured. Because our code is plain Python, the Mac nodes could also run **without Docker**, directly on macOS, where the expert work is 2–15× faster. We will measure both. If Laptop A is a Linux or Windows machine, its Docker runs differently, and we will measure that too.

*Different computers round differently.* We learned that macOS and Linux already give different answers in bf16. Across three real laptops, each expert runs on a different machine, so one answer mixes their rounding. We therefore decided on two checks: in **fp32**, the answers must be **exactly** the golden ones (macOS and Linux agreed perfectly in fp32); in **bf16**, the normal setting, a different token is only allowed where the model was almost undecided between its two best tokens (a "near-tie"), and we also read some answers by hand to make sure they make sense.

*Memory will be fine for Granite, and planned for Qwen.* Granite fits easily. For Qwen1.5-MoE (28.6 GB), most experts go on the 36 GB Mac, which also means fewer Wi-Fi trips; the 8 GB laptop gets only a few experts. The exact split depends on Laptop C's memory, which we still need to find out, as well as Laptop A's processor type (if it is an Intel/AMD processor, the Docker image must be built for both processor types).

*Then the bigger models.* Once Granite works across the laptops, we switch to Qwen1.5-MoE with its own adapter and its own golden files, and, if there is enough memory, to Qwen3-30B with experts stored in the smaller int8 format.

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
