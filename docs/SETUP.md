# Setup

How to set up the Python environment for this project on a development machine (Steps 1–3 run on the 36 GB Mac). Docker setup comes later, in Phase 2.

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
pytest                  # all tests
pytest -m "not model"   # only the quick tests that don't load the model
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

## 8. Common problems

| Problem | Fix |
|---|---|
| `python3.12: command not found` | `brew install python@3.12`, or use Option A (uv gets Python for you). |
| Wrong Python / old transformers | The environment isn't active: `source .venv/bin/activate`. |
| `hf: command not found` | Same: activate the environment (`hf` is installed by `huggingface_hub`). |
| Download is very slow or stops | Run the same `hf download` command again; it continues where it stopped. |
| "No space left on device" | Free space, or set `HF_HOME` to a disk with more room (section 5). |
