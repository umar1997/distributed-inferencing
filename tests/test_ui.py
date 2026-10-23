"""Phase 3.3: the master serves the web page at / (built from ui/ with `npm run build`).

Skipped when the page has not been built (it needs Node.js; the Docker image always builds it).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from moe_dist.config import Settings
from moe_dist.master.app import create_app

DIST = Path(__file__).resolve().parent.parent / "ui" / "dist"


@pytest.fixture
def client():
    if not (DIST / "index.html").exists():
        pytest.skip("the web page is not built (cd ui && npm install && npm run build)")
    # Without "with": the model is not loaded; serving the page does not need it.
    return TestClient(create_app(Settings(node_role="master", node_id="master"), log=lambda text: None))


def test_the_page_and_its_files_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert "<title>MoE Cluster</title>" in page.text
    for asset in [part for part in page.text.split('"') if part.startswith("/assets/")]:
        assert client.get(asset).status_code == 200, asset


def test_the_api_still_wins_over_the_page(client):
    assert client.get("/health").json()["role"] == "master"
    assert client.get("/v1/models").json()["object"] == "list"
    assert client.get("/no-such-page").status_code == 404


def test_a_missing_page_is_only_a_log_line(tmp_path):
    lines = []
    app = create_app(Settings(node_role="master", node_id="master", ui_dir=str(tmp_path)), log=lines.append)
    assert any("no web page at" in line for line in lines)
    assert TestClient(app).get("/").status_code == 404
