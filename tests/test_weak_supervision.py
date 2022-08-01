"""Unit tests related to the weak_supervision module."""

from src.dr_hatespeech.load_data import load_weakly_supervised_data
from src.dr_hatespeech.weak_supervision import apply_weak_supervision


def test_weak_supervision(config):
    apply_weak_supervision(config)
    df = load_weakly_supervised_data(config)
    assert "label" in df.columns
