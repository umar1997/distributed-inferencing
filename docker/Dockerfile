# One image for every node (master and workers); NODE_ROLE decides what it runs.
# Single architecture for now (built for the machine it is built on). Multi-architecture: Phase 4.
FROM python:3.12.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Optional extra trusted certificates, e.g. a company security filter like Zscaler
# (docker/certs/README.md). Without .crt files there, nothing changes.
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Dependencies first (this layer is cached while only our code changes).
# The PyTorch CPU index gives CPU-only wheels (no multi-GB GPU libraries).
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
    # Python's own certificate list (certifi, used by the Hugging Face downloader) gets the extras too.
    && for f in /usr/local/share/ca-certificates/extra/*.crt; do \
         [ -e "$f" ] && cat "$f" >> "$(python -c 'import certifi; print(certifi.where())')"; \
       done; true

COPY pytest.ini ./
COPY moe_dist ./moe_dist
COPY scripts ./scripts
COPY tests ./tests

CMD ["python", "-m", "moe_dist.serve"]
