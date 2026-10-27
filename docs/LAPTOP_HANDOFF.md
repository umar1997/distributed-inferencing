# Laptop handoff: instructions for an AI agent

**For the person:** on the new laptop, clone the repo, open Claude Code (or another coding agent) in the repo
folder, and say:

> Read docs/LAPTOP_HANDOFF.md and set up this laptop.

Everything below is written to the agent. You can read it too: it's the same steps as
[LAPTOP_SETUP.md](LAPTOP_SETUP.md), which you can follow by hand instead.

---

## Your task (agent)

You're setting up **this laptop** as one node of a distributed Mixture-of-Experts inference cluster. The code
is already here and complete. **Don't change any code, and don't commit anything.** Your job is to install,
configure, start and check, following [docs/LAPTOP_SETUP.md](LAPTOP_SETUP.md) (read it fully first). Background,
only if you need it: `docs/STORY.md` (what the project is) and `docs/IMPLEMENTATION.md` section 3 (every setting).

### Rules

- **Talk in plain, simple English.** Define any technical term the first time you use it. The person may not
  know what a port, venv or IP address is.
- **Ask the person** for the things only they know (below). Never invent a cluster token or a master address.
- **Install system software only with the person's OK** (Homebrew, apt, Python, Node.js). Say what you'll run
  and why first.
- Keep memory use light: don't run the slow test suites (`-m model`, `-m cluster`) unless asked.
- Never print the cluster token in a summary or share it outside this laptop. It's a password.
- If something fails, look it up in the guide's "Problems and fixes" table (section 7) before improvising.

### Steps

1. **Check what's installed:** `git --version`, `python3.12 --version`, `python3.12 -m venv --help`,
   `node --version` (only needed on the master), the operating system (`uname -a`), memory, and free disk
   (~5 GB needed). Report anything missing, with the install command from the guide's section 1.

2. **Ask the person:**
   - Is this laptop the **master** or a **worker**? (There is exactly one master. It's usually set up first.)
   - Worker only: the **master's address** (e.g. `192.168.1.10`) and the **cluster token**. Both were printed when
     the master was set up, and the token is in the master's `node.env`.
   - Worker only: **how many more workers** will be set up after this one, so the free experts can be shared.
     Or which **experts** this laptop should hold, if they already know (e.g. `22-31`).
   - Master only: **how many laptops** will take part in total, this one included.
   - Optional: a **name** for this laptop (default: the computer's name, or `master` on the master).

3. **Install:** `./scripts/install.sh` (it takes a few minutes; PyTorch is large). It must end with
   `packages OK` and `Done`.

4. **Configure,** without interactive prompts, using the answers from step 2. The wizard checks everything
   (master reachable, token right, experts valid, memory) and writes `node.env`:

   ```bash
   # master
   ./moe setup --yes --role master --laptops 3
   # worker (the master must already be running)
   ./moe setup --yes --role worker --master 192.168.1.10 --token '<token>' --workers-after 1
   # add --experts 22-31 if the person chose them; --name laptop-b for a name
   ```

   If it stops with a ✗ line, fix that cause ("Problems and fixes") and run it again. Show the person the
   setup output, except the token. On the master, tell the person to write down the master's address
   and token from the last lines, because the workers need them.

5. **Start the node.** It runs until stopped, so it's best started by the person in their own terminal window,
   where it can stay open: `./moe serve` (master) or `./moe join` (worker). If you start it yourself, run it in the
   background and keep its output in a log file. On macOS, tell the person to click **Allow** if a pop-up asks
   whether Python may accept incoming network connections.

6. **Check:**
   - Worker: its log shows `registered with master ... as http://<this laptop's IP>:8100`.
   - `./moe status`: this laptop is `healthy`, and its ADDRESS is its real Wi-Fi IP, not a VPN address.
   - Once all laptops have joined, the last line says `coverage: OK`.
   - Optional quick tests (no cluster needed): `.venv/bin/python -m pytest -q -m "not model and not cluster"`.

7. **Report to the person,** in plain English: what was installed, this laptop's role, name, IP and experts,
   and the `./moe status` table. Then tell them the two commands for next time:
   `cd distributed-inferencing` and then `./moe serve` / `./moe join`. On the master, also give the web page
   address, `http://<master IP>:8000`.
