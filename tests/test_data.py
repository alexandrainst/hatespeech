"""Unit tests related to the data module."""

import pandas as pd
import pytest

from src.dr_hatespeech.data import load_data


class TestData:
    @pytest.fixture(scope="class")
    def data(self):
        yield load_data(test=True)

    def test_data_is_dataframe(self, data):
        assert isinstance(data, pd.DataFrame)

    def test_data_columns(self, data):
        cols = [
            "account",
            "text",
            "url",
            "date",
            "action",
            "post_id",
            "comment_id",
            "reply_comment_id",
        ]
        assert data.columns.tolist() == cols

    def test_data_dtypes(self, data):
        dtypes = [
            "category",
            "object",
            "object",
            "datetime64[ns]",
            "category",
            "Int64",
            "Int64",
            "Int64",
        ]
        assert data.dtypes.map(str).tolist() == dtypes

    def test_accounts(self, data):
        assert data.account.unique().tolist() == ["dr nyheder"]
