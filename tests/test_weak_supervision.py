"""Unit tests related to the weak_supervision module."""

from pathlib import Path

import pandas as pd
import pytest

ws = pytest.importorskip("src.dr_hatespeech.weak_supervision")


def test_weak_supervision(config):
    # Set up path to weakly supervised test dataset
    path = Path("data") / "final" / "test_data_train.parquet"

    # Remove the weakly supervised test dataset if it exists
    path.unlink(missing_ok=True)

    # Apply weak supervision, generating the weakly supervised test dataset
    ws.apply_weak_supervision(config)

    # Load the weakly supervised test dataset
    df = pd.read_parquet(path)

    # Check that the weakly supervised test dataset has a "label" column
    assert "label" in df.columns
