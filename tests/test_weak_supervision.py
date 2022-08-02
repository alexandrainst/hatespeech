"""Unit tests related to the weak_supervision module."""

import pytest

from src.dr_hatespeech.load_data import load_weakly_supervised_data
from src.dr_hatespeech.weak_supervision import apply_weak_supervision

ws = pytest.importorskip("src.dr_hatespeech.weak_supervision")


def test_weak_supervision(config):
    # Apply weak supervision, generating the weakly supervised test dataset
    apply_weak_supervision(config)

    # Load the weakly supervised test dataset
    df = load_weakly_supervised_data(config)

    # Check that the weakly supervised test dataset has a "label" column
    assert "label" in df.columns
