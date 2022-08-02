"""Session-wide fixtures for tests."""

import pytest
from hydra import compose, initialize

# Initialise Hydra
initialize(config_path="../config", version_base=None)


@pytest.fixture(scope="session")
def config():
    return compose(
        config_name="config", overrides=["testing=True", "data=test_offensive"]
    )
