"""Unit tests related to the data module."""

import pytest
import pandas as pd
from src.data import load_data


class TestData:
    @pytest.fixture(scope="class")
    def data(self):
        return load_data(test=True)

    def test_data_is_dataframe(self, data):
        assert isinstance(data, pd.DataFrame)

    def test_data_columns(self, data):
        cols = ["account", "text", "date", "action"]
        assert data.columns.tolist() == cols

    def test_data_dtypes(self, data):
        dtypes = ["category", "object", "datetime64[ns]", "category"]
        assert data.dtypes.map(str).tolist() == dtypes

    def test_accounts(self, data):
        assert data.account.unique().tolist() == ["dr nyheder"]
