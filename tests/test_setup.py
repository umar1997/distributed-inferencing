"""The laptop setup (docs/LAPTOP_SETUP.md): node.env, the wizard's suggestions, and auto addresses.

No model and no servers: the master's answers are faked, and the download is skipped.
"""

import httpx
import pytest
from typer.testing import CliRunner

from moe_dist.cli import setup as wizard
from moe_dist.cli.main import app
from moe_dist.config import Settings, local_ip
from moe_dist.models.catalog import DEFAULT_MODEL, pinned_revision


def test_split_experts_gives_neighbouring_near_equal_shares():
    every = list(range(32))
    assert [len(s) for s in wizard.split_experts(every, 3)] == [11, 11, 10]
    assert [(s[0], s[-1]) for s in wizard.split_experts(every, 4)] == [(0, 7), (8, 15), (16, 23), (24, 31)]
    assert wizard.split_experts(every, 1) == [every]
    assert wizard.split_experts([22, 23, 24, 25], 2) == [[22, 23], [24, 25]]
    assert len(wizard.split_experts([1, 2], 5)) == 2  # never an empty share


def test_free_experts_ignores_dead_nodes_and_this_laptops_old_entry():
    nodes = [
        {"node_id": "master", "status": "healthy", "experts": {"all_layers": list(range(0, 11))}},
        {"node_id": "laptop-b", "status": "healthy", "experts": {"all_layers": list(range(11, 22))}},
        {"node_id": "laptop-c", "status": "dead", "experts": {"all_layers": list(range(22, 32))}},
    ]
    assert wizard.free_experts(nodes, 32, "laptop-d") == list(range(22, 32))
    assert wizard.free_experts(nodes, 32, "laptop-b") == list(range(11, 32))  # setting laptop-b up again


def test_env_file_round_trip_and_settings_read_it(tmp_path, monkeypatch):
    path = tmp_path / "node.env"
    wizard.write_env(path, {"NODE_ROLE": "worker", "NODE_ID": "laptop-b", "EXPERT_IDS": "11-21",
                            "MASTER_ADDR": "192.168.1.10:8000", "ADVERTISE_ADDR": "auto", "CLUSTER_TOKEN": "abc",
                            "LISTEN_PORT": "8100"})
    assert oct(path.stat().st_mode & 0o777) == "0o600"  # only this user can read the token
    assert wizard.read_env(path)["EXPERT_IDS"] == "11-21"

    monkeypatch.setenv("MOE_ENV_FILE", str(path))
    s = Settings()
    assert (s.node_role, s.node_id, s.expert_ids, s.listen_port, s.cluster_token) == \
        ("worker", "laptop-b", "11-21", 8100, "abc")
    monkeypatch.setenv("EXPERT_IDS", "0-3")  # a real environment variable wins over the file
    assert Settings().expert_ids == "0-3"
    monkeypatch.setenv("MOE_ENV_FILE", "")  # no file at all
    assert Settings().node_id == "node"


def test_auto_advertise_address_is_this_laptops_ip():
    s = Settings(advertise_addr="auto", listen_port=8100, master_addr="192.168.1.10:8000")
    assert s.advertise_url() == f"http://{local_ip('192.168.1.10:8000')}:8100"
    assert not s.advertise_url().startswith("http://127.")


class FakeMaster:
    """Answers /health, /api/nodes and the token check like a master holding experts 0-10."""

    def __init__(self, token="right-token", nodes=None):
        self.token = token
        self.nodes = nodes or [{"node_id": "master", "status": "healthy", "experts": {"all_layers": list(range(11))}}]

    def get(self, url, timeout=None, trust_env=True):
        if url.endswith("/health"):
            body = {"node_id": "master", "model_id": DEFAULT_MODEL, "model_revision": pinned_revision(DEFAULT_MODEL),
                    "dtype": "bf16", "loaded": True}
        else:
            body = {"nodes": self.nodes}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    def post(self, url, json=None, headers=None, timeout=None, trust_env=True):
        ok = headers.get("Authorization") == f"Bearer {self.token}"
        return httpx.Response(200 if ok else 401, json={}, request=httpx.Request("POST", url))


@pytest.fixture
def fake_master(monkeypatch):
    master = FakeMaster()
    monkeypatch.setattr(wizard.httpx, "get", master.get)
    monkeypatch.setattr(wizard.httpx, "post", master.post)
    return master


def run_setup(path, *options):
    return CliRunner().invoke(app, ["setup", "--yes", "--no-download", "--env-file", str(path), *options])


def test_worker_setup_suggests_a_share_of_the_free_experts(tmp_path, fake_master):
    path = tmp_path / "node.env"
    result = run_setup(path, "--role", "worker", "--master", "192.168.1.10", "--token", "right-token",
                       "--name", "laptop-b", "--workers-after", "1")
    assert result.exit_code == 0, result.output
    assert "Experts no other laptop holds yet: 11-31" in result.output
    saved = wizard.read_env(path)
    assert saved["EXPERT_IDS"] == "11-21"  # first of 2 shares of 11-31
    assert saved["MASTER_ADDR"] == "192.168.1.10:8000"  # port added
    assert (saved["NODE_ID"], saved["LISTEN_PORT"], saved["ADVERTISE_ADDR"]) == ("laptop-b", "8100", "auto")
    assert (saved["MODEL_ID"], saved["DTYPE"], saved["CLUSTER_TOKEN"]) == (DEFAULT_MODEL, "bf16", "right-token")

    # Run again: the saved answers are the defaults, so nothing changes.
    assert run_setup(path).exit_code == 0
    assert wizard.read_env(path) == saved


def test_worker_setup_refuses_a_wrong_token(tmp_path, fake_master):
    result = run_setup(tmp_path / "node.env", "--role", "worker", "--master", "192.168.1.10", "--token", "wrong")
    assert result.exit_code == 1
    assert "the master says this token is wrong" in result.output
    assert not (tmp_path / "node.env").exists()


def test_worker_setup_says_clearly_when_the_master_is_unreachable(tmp_path, monkeypatch):
    def unreachable(url, timeout=None, trust_env=True):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(wizard.httpx, "get", unreachable)
    result = run_setup(tmp_path / "node.env", "--role", "worker", "--master", "192.168.1.10")
    assert result.exit_code == 1
    assert "cannot reach the master at http://192.168.1.10:8000" in result.output


def test_master_setup_makes_a_token_and_suggests_a_split(tmp_path):
    path = tmp_path / "node.env"
    result = run_setup(path, "--role", "master", "--laptops", "3")
    assert result.exit_code == 0, result.output
    saved = wizard.read_env(path)
    assert (saved["NODE_ROLE"], saved["NODE_ID"], saved["LISTEN_PORT"], saved["EXPERT_IDS"]) == \
        ("master", "master", "8000", "0-10")
    assert len(saved["CLUSTER_TOKEN"]) >= 16 and "MASTER_ADDR" not in saved
    assert "worker 1     experts 11-21" in result.output and "worker 2     experts 22-31" in result.output
    assert saved["CLUSTER_TOKEN"] in result.output  # shown, so it can be typed on the workers


def test_status_on_the_master_uses_its_own_port(tmp_path, monkeypatch):
    path = tmp_path / "node.env"
    wizard.write_env(path, {"NODE_ROLE": "master", "LISTEN_PORT": "8001"})
    monkeypatch.setenv("MOE_ENV_FILE", str(path))
    asked = []

    def nodes(url, timeout=None, trust_env=True):
        asked.append(url)
        return httpx.Response(200, json={"nodes": [], "coverage": "ok"}, request=httpx.Request("GET", url))

    monkeypatch.setattr("moe_dist.cli.main.httpx.get", nodes)
    assert CliRunner().invoke(app, ["status"]).exit_code == 0
    assert asked == ["http://localhost:8001/api/nodes"]


def test_join_without_settings_explains_what_to_do(monkeypatch):
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("EXPERT_IDS", raising=False)
    result = CliRunner().invoke(app, ["join"])
    assert result.exit_code == 1
    assert "run ./moe setup first" in result.output
