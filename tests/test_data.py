"""Unit tests related to the data module."""

import pandas as pd
import pytest

from src.dr_hatespeech.load_data import (
    load_cleaned_data,
    load_final_data,
    load_raw_data,
    load_weakly_supervised_data,
)


class TestRawData:
    @pytest.fixture(scope="class")
    def data(self):
        yield load_raw_data(test=True)

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


class TestCleanedData:
    @pytest.fixture(scope="class")
    def data(self):
        yield load_cleaned_data(test=True)

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
