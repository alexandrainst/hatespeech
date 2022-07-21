"""Session-wide fixtures for tests."""

import pytest
from hydra import compose, initialize

# Initialise Hydra
initialize(config_path="../config", version_base=None)


@pytest.fixture(scope="session")
def config():
    cfg = compose(config_name="config")
    cfg.testing = True
    return cfg
