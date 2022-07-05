"""Unit tests related to the data module."""

import csv
from pathlib import Path

import pandas as pd
import pytest

from src.dr_hatespeech.clean_data import clean_data
from src.dr_hatespeech.load_data import (
    load_cleaned_data,
    load_final_data,
    load_raw_data,
    load_weakly_supervised_data,
)


class TestRawData:
    def create_test_data(self):
        base_record = dict(
            platform="facebook",
            account="page: dr nyheder",
            text="",
            url="https://www.facebook.com/115104055206794/posts/12345/?comment_id=123",
            date="2022-01-21 00:00:00",
            action="deleted",
        )
        records = [
            {**base_record, "text": "Dette er noget tekst."},
            {**base_record, "text": "Dette er noget andet tekst."},
            {**base_record, "text": "Dette er også noget tekst."},
            {**base_record, "text": "Dette er noget helt andet tekst."},
            {**base_record, "text": "Du er da et kæmpe røvhul!"},
            {**base_record, "text": "Sikke et møgsvin!"},
        ]
        df = pd.DataFrame.from_records(records)
        df.to_csv(
            Path("data") / "raw" / "test_data.csv",
            encoding="windows-1252",
            index=False,
            quoting=csv.QUOTE_ALL,
        )

    @pytest.fixture(scope="class")
    def data(self, config):
        self.create_test_data()
        yield load_raw_data(config)

    @pytest.fixture(scope="class")
    def df(self, data):
        yield data["df"]

    def test_data_is_dict(self, data):
        assert isinstance(data, dict)

    def test_data_keys(self, data):
        assert list(data.keys()) == ["df", "path"]

    def test_data_path_is_path(self, data):
        assert isinstance(data["path"], Path)

    def test_data_path_exists(self, data):
        assert data["path"].exists()

    def test_df_is_dataframe(self, df):
        assert isinstance(df, pd.DataFrame)

    def test_df_has_records(self, df):
        assert len(df) > 0

    def test_df_columns(self, df):
        assert df.columns.tolist() == ["account", "text", "url", "date", "action"]

    def test_df_dtypes(self, df):
        assert df.dtypes.map(str).tolist() == ["object"] * 5

    def test_df_accounts(self, df):
        assert df.account.unique().tolist() == ["page: dr nyheder"]


class TestCleanedData:
    @pytest.fixture(scope="class")
    def data(self, config):
        clean_data(config)
        yield load_cleaned_data(config)

    @pytest.fixture(scope="class")
    def df(self, data):
        yield data["df"]

    def test_data_is_dict(self, data):
        assert isinstance(data, dict)

    def test_data_keys(self, data):
        assert list(data.keys()) == ["df", "path"]

    def test_data_path_is_path(self, data):
        assert isinstance(data["path"], Path)

    def test_data_path_exists(self, data):
        assert data["path"].exists()

    def test_df_is_dataframe(self, df):
        assert isinstance(df, pd.DataFrame)

    def test_df_has_records(self, df):
        assert len(df) > 0

    def test_df_columns(self, df):
        assert df.columns.tolist() == [
            "account",
            "text",
            "url",
            "date",
            "action",
            "post_id",
            "comment_id",
            "reply_comment_id",
        ]

    def test_df_dtypes(self, df):
        assert df.dtypes.map(str).tolist() == [
            "category",
            "object",
            "object",
            "datetime64[ns]",
            "category",
            "Int64",
            "Int64",
            "Int64",
        ]

    def test_df_accounts(self, df):
        assert df.account.unique().tolist() == ["dr nyheder"]

    def test_df_comment_id(self, df):
        assert list(df.comment_id.unique()) == [123]
