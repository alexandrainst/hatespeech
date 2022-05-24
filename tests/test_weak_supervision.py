"""Unit tests related to the weak_supervision module."""

import pandas as pd
from pathlib import Path
from src.weak_supervision import main as apply_weak_supervision


def test_weak_supervision():
    # Set up path to weakly supervised test dataset
    path = Path("data") / "processed" / "test_data_weakly_supervised.parquet"

    # Remove the weakly supervised test dataset if it exists
    path.unlink(missing_ok=True)

    # Apply weak supervision, generating the weakly supervised test dataset
    apply_weak_supervision(test=True)

    # Load the weakly supervised test dataset
    df = pd.read_parquet("data/processed/test_data_weakly_supervised.parquet")

    # Check that the weakly supervised test dataset has a "label" column
    assert "label" in df.columns
