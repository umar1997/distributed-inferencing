"""Shared test setup."""

import os

# A laptop's node.env (docs/LAPTOP_SETUP.md) must not change the settings the tests start with,
# also in the master/worker processes the tests start (they inherit this).
os.environ["MOE_ENV_FILE"] = ""
