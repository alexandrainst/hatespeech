"""Unit tests related to the weak_supervision module."""

import pandas as pd
from src.weak_supervision import main as apply_weak_supervision


def test_weak_supervision():
    apply_weak_supervision(test=True)
    df = pd.read_csv('data/processed/test_data_weakly_supervised.parquet')
    assert 'label' in df.columns
