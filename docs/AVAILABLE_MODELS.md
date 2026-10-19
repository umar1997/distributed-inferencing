# Available Models

This document lists the models we will use, how big they are, how they store their experts, and the order in which we will move from one model to the next.

All numbers below were **checked on 2026-10-19** against Hugging Face and **Transformers 5.20.1 / PyTorch 2.14.1**. They come from the real model configs and from building each model "empty" (on PyTorch's `meta` device, which creates the shapes without allocating memory). No guesses.

---

## 1. The short version

| Order | Model | Why | Where it runs |
|---|---|---|---|
| **1st (now)** | `ibm-granite/granite-3.0-1b-a400m-instruct` | Small (2.7 GB). Fast to download and test. Fits on the 8 GB laptop. | Steps 1, 2, 3 and first run of Step 4 |
| **2nd** | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | A real "Qwen" MoE with 60 experts. Much better answers. Fits across the 3 laptops in bf16. | Step 4 (real laptops) |
| **Stretch goal** | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Modern, strong model. Too big for bf16 on our laptops, so it needs int8. | Step 4, only after the Qwen1.5 run works |

**Not used:** `allenai/OLMoE-1B-7B` (from the original plan) — replaced by Qwen1.5-MoE as the "next step up". Qwen3.5 / Qwen3.6 MoE models (35B+, 256 experts, a new kind of attention, plus an image part) — too big and too complex for this project.

---

## 2. Words used in this document

- **Parameter:** one number inside the model. A "1.3B" model has 1.3 billion of them.
- **bf16 / fp32 / int8:** how many bytes each parameter takes in memory. **fp32 = 4 bytes** (most exact), **bf16 = 2 bytes** (the format these models are shipped in), **int8 = 1 byte** (smallest, slightly less exact; needs extra work, see section 7).
- **Expert:** a small feed-forward network inside an MoE layer. The **router** picks a few experts for each token.
- **Shared parts:** everything that is *not* a routed expert: embeddings (turn tokens into vectors), attention, layer norms, the router itself, and the output layer. Every token uses these, so they live on the **master**.
- **Expert slot:** in our plan, a node holding "expert 5" holds expert 5 **in every layer**. One *slot* = the same expert number across all layers. Memory is counted per slot.
- **Active parameters:** how many parameters are actually used for one token (shared parts + only the picked experts). This decides speed, not the total size.
- **KV cache:** saved attention data from earlier tokens, so the model doesn't redo work for every new token. It lives on the master (only the master runs attention).

---

## 3. Model configurations

### 3.1 Granite 3.0 1B-A400M Instruct (first model)

| Setting | Value |
|---|---|
| Hugging Face ID | `ibm-granite/granite-3.0-1b-a400m-instruct` |
| Pinned version (commit) | `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445` |
| License | Apache 2.0 |
| Transformers model type | `granitemoe` (`GraniteMoeForCausalLM`) |
| Layers | 24 |
| Hidden size (length of each token's vector) | 1024 |
| Experts per layer | **32** |
| Experts picked per token | **8** |
| Expert inner size | 512 |
| Shared expert | None |
| Attention heads / KV heads | 16 / 8 |
| Vocabulary | 49,155 tokens |
| Max context | 4,096 tokens |
| Input and output embeddings tied (shared) | Yes |
| Special scaling numbers | `embedding_multiplier=12`, `residual_multiplier=0.22`, `attention_multiplier=0.015625`, `logits_scaling=6` — handled by Transformers' own code, we don't touch them |
| Router rule | Pick the top 8 router scores, then softmax over **only those 8** (weights add up to 1) |
| Files on disk | 1 file, `model.safetensors`, 2.67 GB |

### 3.2 Qwen1.5-MoE-A2.7B Chat (second model)

| Setting | Value |
|---|---|
| Hugging Face ID | `Qwen/Qwen1.5-MoE-A2.7B-Chat` |
| Pinned version (commit) | `ec052fda178e241c7c443468d2fa1db6618996be` |
| License | Tongyi Qianwen license (free for personal / research use — fine for this project) |
| Transformers model type | `qwen2_moe` (`Qwen2MoeForCausalLM`) |
| Layers | 24 |
| Hidden size | 2048 |
| Experts per layer | **60** |
| Experts picked per token | **4** |
| Expert inner size | 1408 |
| Shared expert | **Yes** — 1 big always-on expert per layer (inner size 5632) with its own small gate. Every token uses it, so it is a **shared part and stays on the master**. |
| Attention heads / KV heads | 16 / 16 (attention has bias terms) |
| Vocabulary | 151,936 tokens |
| Max context | 32,768 tokens |
| Embeddings tied | No (separate output layer) |
| Router rule | Softmax over **all 60** scores, then keep the top 4. The 4 weights are **not** re-scaled to add up to 1 (`norm_topk_prob=False`). |
| Files on disk | 8 files, 28.6 GB total |

### 3.3 Qwen3-30B-A3B Instruct 2507 (stretch goal)

| Setting | Value |
|---|---|
| Hugging Face ID | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| Pinned version (commit) | `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` |
| License | Apache 2.0 |
| Transformers model type | `qwen3_moe` (`Qwen3MoeForCausalLM`) |
| Layers | **48** (twice as many network round trips per token as the others) |
| Hidden size | 2048 |
| Experts per layer | **128** |
| Experts picked per token | **8** |
| Expert inner size | 768 |
| Shared expert | None |
| Attention heads / KV heads | 32 / 4, head size 128, with extra Q/K norms |
| Vocabulary | 151,936 tokens |
| Max context | 262,144 tokens (we will use far less) |
| Embeddings tied | No |
| Router rule | Softmax over all 128, keep top 8, **then** re-scale the 8 weights to add up to 1 (`norm_topk_prob=True`) |
| Files on disk | 61.1 GB in bf16 (an official FP8 version also exists, about half the size) |

---

## 4. How much memory each model needs

### 4.1 Whole model (weights only)

| Model | Total params | Active per token | fp32 | **bf16** | int8 (approx.) |
|---|---|---|---|---|---|
| Granite 1B-A400M | 1.33 B | ~0.43 B | 5.3 GB | **2.7 GB** | ~1.4 GB |
| Qwen1.5-MoE-A2.7B | 14.32 B | ~2.7 B | 57.3 GB | **28.6 GB** | ~14.5 GB |
| Qwen3-30B-A3B | 30.53 B | ~3.3 B | 122 GB | **61.1 GB** | ~31 GB |

### 4.2 Split into "shared parts" (master) and "experts" (spread across nodes)

| Model | Shared parts (always on master), bf16 | All experts, bf16 | Experts as % of model |
|---|---|---|---|
| Granite | **0.25 GB** | 2.42 GB | 91 % |
| Qwen1.5-MoE | **3.72 GB** (includes the shared expert) | 24.9 GB | 87 % |
| Qwen3-30B | **3.08 GB** | 58.0 GB | 95 % |

This is why the idea works: almost all of the memory is in the experts, and the experts are what we spread out.

### 4.3 Cost of one expert slot (one expert number, across all layers)

| Model | Slots | One expert in one layer (bf16) | **One slot, all layers** — fp32 / **bf16** / int8 |
|---|---|---|---|
| Granite | 32 | 3.1 MB | 151 MB / **75.5 MB** / ~38 MB |
| Qwen1.5-MoE | 60 | 17.3 MB | 830 MB / **415 MB** / ~208 MB |
| Qwen3-30B | 128 | 9.4 MB | 906 MB / **453 MB** / ~227 MB |

### 4.4 Extra memory on top of the weights

| Item | Granite | Qwen1.5-MoE | Qwen3-30B |
|---|---|---|---|
| KV cache per token (bf16, master only) | 48 KB | 192 KB | 96 KB |
| KV cache for 4,096 tokens | ~0.2 GB | ~0.8 GB | ~0.4 GB |
| Python + PyTorch + Transformers per process | ~0.6–1.0 GB (to be measured in Phase 2) | same | same |
| Data sent per token per layer (bf16) | 2 KB | 4 KB | 4 KB |

The data sent over the network is tiny. **The cost is the number of round trips** (one per layer per token), not the size.

### 4.5 Example placements (to be finalised from real measurements)

**Granite — Step 2, all 3 containers inside an 8 GB budget (bf16):**

| Container | Holds | Weights | + overhead | Container memory limit |
|---|---|---|---|---|
| master | shared parts + experts 0–9 (10 slots) | 0.25 + 0.76 = 1.0 GB | ~1.0 GB | 2.5 GB |
| worker-1 | experts 10–20 (11 slots) | 0.83 GB | ~0.8 GB | 2 GB |
| worker-2 | experts 21–31 (11 slots) | 0.83 GB | ~0.8 GB | 2 GB |
| **Total** | | **2.7 GB** | | **6.5 GB → Docker Desktop limit: 7 GB** |

**Qwen1.5-MoE — Step 4, real laptops (bf16).** Laptop C's RAM is still unknown, so this assumes 16 GB:

| Node | Holds | Weights | Notes |
|---|---|---|---|
| Master (36 GB Mac) | shared parts + 36 slots | 3.7 + 14.9 = **18.6 GB** | Most experts here = fewest Wi-Fi calls |
| Laptop A (8 GB) | 8 slots | **3.3 GB** | + ~1 GB overhead, Docker limit ~5 GB |
| Laptop C (Mac, assume 16 GB) | 16 slots | **6.6 GB** | Docker limit ~9 GB |
| **Total** | 60 slots | **28.6 GB** | |

**Qwen3-30B — stretch goal (int8 experts, bf16 shared parts):** about 3.1 GB shared + 29 GB of experts = **~32 GB across all laptops**. Possible only if Laptop C has 16 GB or more. Example: master 64 slots (~17.6 GB), Laptop A 12 slots (~2.7 GB), Laptop C 52 slots (~11.8 GB).

**Disk space:** the 36 GB Mac has about 73 GB free. Granite (2.7 GB) and Qwen1.5 (28.6 GB) are fine. Qwen3-30B in bf16 (61 GB) would almost fill the disk, so for that model we should use the FP8 files (~31 GB) or prepare smaller per-node files (see section 6.4).

---

## 5. What "the same packed-expert layout" means

### 5.1 Two ways to store experts

Think of a layer with 32 experts. Each expert has a few weight matrices: `gate`, `up` and `down`.

**Way 1 — separate:** each expert is its own object with its own matrices.

```
layer.experts[0].gate_proj   (512 × 1024)
layer.experts[0].up_proj     (512 × 1024)
layer.experts[0].down_proj   (1024 × 512)
layer.experts[1].gate_proj   ...
... 32 times
```

**Way 2 — packed:** all 32 experts are stacked into one big 3-D tensor, like a stack of 32 sheets of paper. Expert number *i* is sheet *i*.

```
layer.experts.gate_up_proj   (32 × 1024 × 1024)   ← gate and up glued together
layer.experts.down_proj      (32 × 1024 × 512)
```

To give a worker "experts 8–15", we **cut sheets 8 to 15 out of the stack** (`gate_up_proj[8:16]`, `down_proj[8:16]`). That is the "cutting experts out of the packed tensor" code.

### 5.2 What we found

**In memory (inside Transformers 5.x), all three models use Way 2 — the exact same layout:**

| Model | `experts.gate_up_proj` shape | `experts.down_proj` shape |
|---|---|---|
| Granite | (32, 1024, 1024) | (32, 1024, 512) |
| Qwen1.5-MoE | (60, 2816, 2048) | (60, 2048, 1408) |
| Qwen3-30B | (128, 1536, 2048) | (128, 2048, 768) |

The pattern is always `(number of experts, 2 × expert inner size, hidden size)` and `(number of experts, hidden size, expert inner size)`. The first half of each `gate_up_proj` sheet is `gate`, the second half is `up`.

All three models also **call their experts in the same way**. Inside each MoE layer, Transformers does:

```python
top_k_index, top_k_weights = router(hidden_states)        # which experts, how much each counts
output = self.experts(hidden_states, top_k_index, top_k_weights)
```

That one line, `self.experts(...)`, is **the only place we plug in**. We replace `self.experts` with our own object that has the same inputs and outputs, but sends the work to whichever node holds each expert. Everything else (attention, router, scaling numbers, shared expert, KV cache) stays as Transformers' own, tested code. This is what makes switching models cheap.

**On disk (in the downloaded files), they differ:**

| Model | How experts are stored in the files | Names in the file |
|---|---|---|
| Granite | Packed (Way 2) | `layers.N.block_sparse_moe.input_linear.weight` (32, 1024, 1024) and `output_linear.weight` (32, 1024, 512) |
| Qwen1.5-MoE | Separate (Way 1) | `layers.N.mlp.experts.E.gate_proj.weight`, `.up_proj.weight`, `.down_proj.weight` |
| Qwen3-30B | Separate (Way 1) | same as Qwen1.5 |

When you call `from_pretrained`, Transformers renames and glues these together into the in-memory layout above. **Our workers will not call `from_pretrained`** (that would load the whole model into memory). Instead each worker reads **only its own experts** directly from the files:

- **Granite:** open the packed tensor in the file and read only the sheets it needs (safetensors can read a slice without loading the rest).
- **Qwen:** read only the `experts.E.*` tensors for its expert numbers, and glue `gate` + `up` together into the same layout.

So there is **one small "model adapter" per model family**. It only knows the file names and the router rule. The rest of the system works on the common in-memory layout and does not care which model it is.

### 5.3 Small differences each adapter must handle

| | Granite | Qwen1.5-MoE | Qwen3-30B |
|---|---|---|---|
| Expert storage on disk | packed | separate | separate |
| Shared expert | none | yes → master only | none |
| Router weights | softmax over top-k | softmax over all, no re-scaling | softmax over all, re-scaled |
| Experts per token | 8 of 32 | 4 of 60 | 8 of 128 |
| Layers (= round trips per token) | 24 | 24 | 48 |

The router always runs on the master, using Transformers' own code, so the router rule doesn't change any of *our* code. It is listed because it affects how much each expert counts and is useful when reading logs.

---

## 6. How we move from one model to the next

### 6.1 The rule

**A new model only starts once the previous model has passed its finish line.** Each model change is a separate, small phase. If something breaks, it must be the new model's adapter, because the rest of the system was already proven.

### 6.2 Granite (first model) — goes through every step

| Step | Granite must pass |
|---|---|
| Step 1 (one program) | Same tokens as plain Transformers, on all test prompts, in fp32 and bf16 |
| Step 2 (3 containers on one Mac) | Same tokens as Step 1. Worker-death detection. **Runs inside the 7 GB Docker limit.** |
| Step 3 (logs + UI) | A token's path is visible in `docker logs` and in the UI |
| Step 4 (real laptops) | Same tokens as Step 2, over Wi-Fi |

### 6.3 Qwen1.5-MoE (second model) — added once Granite passes Step 4

| Phase | What happens | How we test |
|---|---|---|
| Q1. Adapter | Write the Qwen adapter: read separate expert files, glue gate+up, keep the shared expert on the master. | Unit test: experts loaded by our adapter are **identical** to what `from_pretrained` builds, for a few layers. |
| Q2. One program on the 36 GB Mac | Same as Step 1, with Qwen1.5 in bf16. The full model (28.6 GB) fits on the 36 GB Mac in one process. Our code reuses the same weight memory (no second copy). | Same tokens as plain Transformers on the test prompts. |
| Q3. Skip "3 containers on one Mac" | Not possible in bf16: ~31 GB inside Docker leaves macOS almost nothing. The network code was already proven with Granite, and Q2 proved the Qwen adapter, so this step adds little. *Optional:* run it with int8 experts once int8 exists. | — |
| Q4. Real laptops | Placement from section 4.5 (most experts on the master). Same image, `MODEL_ID` changed on every node. | Same tokens as Q2. The master refuses to start until all 60 slots are covered. |

### 6.4 Qwen3-30B (stretch goal) — only after Qwen1.5 works on the laptops

1. **Add int8 expert weights.** Store each expert's weights as int8 plus one scale number per row, and turn them back into bf16 just before use. This is our own small code, tested on Granite and Qwen1.5 first (check how much output quality changes).
2. **Prepare per-node files.** A small tool that cuts the downloaded files into "only the experts for node X" files, so the 8 GB laptop doesn't need 61 GB of disk.
3. **Run the Qwen3 adapter tests**, same as Q1–Q2, then real laptops.
4. **Expect it to be slower:** 48 layers = 48 Wi-Fi round trips per token.

### 6.5 How to switch models in practice

- Set `MODEL_ID` (and `MODEL_REVISION`, the pinned commit above) on **every** node. The master rejects any worker that reports a different model or version.
- Set each node's `EXPERT_IDS` for the new model (e.g. Granite `0-31`, Qwen1.5 `0-59`).
- Model weights are stored in a Docker volume, one folder per model, so switching back and forth doesn't download again.
- Each model gets its own set of "golden outputs" (the reference tokens from plain Transformers) saved in the repo, so the correctness test runs for every model.

---

## 7. Open points to check while building

- **Exact process overhead** (Python + PyTorch + Transformers) per container: measured in Phase 2. The 8 GB numbers above assume about 1 GB.
- **bf16 speed on each CPU:** bf16 works on both ARM and x86 in PyTorch, but its speed depends on the CPU (newer Apple chips and newer Intel/AMD chips have special bf16 instructions; older ones are slower). Measured in Phase 1.
- **int8:** PyTorch's old built-in int8 tools don't support packed 3-D expert tensors, so int8 means our own small "int8 weight + scale" code. Not needed for Granite or Qwen1.5.
- **Laptop A's CPU type and Laptop C's RAM:** still to fill in. They decide the Docker image types (ARM only, or ARM + x86) and the Qwen placements.
- **Python version:** the Mac's built-in Python is 3.9 with Transformers 4.40, which is too old for these models. The project will use its own Python 3.12 environment (and the Docker image has its own Python anyway).
