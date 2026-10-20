"""Download the pinned model once (Docker's one-time "prepare" step): `python -m moe_dist.prepare`.

Runs before the master and workers start, so 3 containers never download the same files at once.
"""

from moe_dist.config import Settings
from moe_dist.experts.loader import model_dir


def main():
    s = Settings()
    print(f"model {s.model_id} @ {s.revision()[:12]} is in {model_dir(s.model_id, s.revision())}", flush=True)


if __name__ == "__main__":
    main()
