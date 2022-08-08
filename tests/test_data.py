"""Unit tests related to the data module."""

import csv
from pathlib import Path

import pandas as pd
import pytest

from src.hatespeech.clean_data import clean_data
from src.hatespeech.create_train_split import create_train_split
from src.hatespeech.load_data import (
    load_annotated_data,
    load_cleaned_data,
    load_raw_data,
    load_splits,
    load_weakly_supervised_data,
)
from src.hatespeech.prepare_data_for_annotation import prepare_data_for_annotation
from src.hatespeech.weak_supervision import apply_weak_supervision


class TestRawData:
    @pytest.fixture(scope="class")
    def data(self, config):
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
            Path(config.data.raw.dir) / config.data.raw.fname,
            encoding="windows-1252",
            index=False,
            quoting=csv.QUOTE_ALL,
        )
        yield load_raw_data(config)

    def test_is_dataframe(self, data):
        assert isinstance(data, pd.DataFrame)

    def test_has_records(self, data):
        assert len(data) > 0

    def test_columns(self, data):
        assert data.columns.tolist() == ["account", "text", "url", "date", "action"]

    def test_dtypes(self, data):
        assert data.dtypes.map(str).tolist() == ["object"] * 5

    def test_accounts(self, data):
        assert data.account.unique().tolist() == ["page: dr nyheder"]


class TestCleanedData:
    @pytest.fixture(scope="class")
    def data(self, config):
        clean_data(config)
        yield load_cleaned_data(config)

    def test_data_is_dataframe(self, data):
        assert isinstance(data, pd.DataFrame)

    def test_has_records(self, data):
        assert len(data) > 0

    def test_columns(self, data):
        assert data.columns.tolist() == [
            "account",
            "text",
            "url",
            "date",
            "action",
            "post_id",
            "comment_id",
            "reply_comment_id",
        ]

    def test_dtypes(self, data):
        assert data.dtypes.map(str).tolist() == [
            "category",
            "object",
            "object",
            "datetime64[ns]",
            "category",
            "Int64",
            "Int64",
            "Int64",
        ]

    def test_accounts(self, data):
        assert data.account.unique().tolist() == ["dr nyheder"]

    def test_comment_id(self, data):
        assert list(data.comment_id.unique()) == [123]


class TestWeaklySupervisedData:
    @pytest.fixture(scope="class")
    def data(self, config):
        apply_weak_supervision(config)
        yield load_weakly_supervised_data(config)

    def test_data_is_dataframe(self, data):
        assert isinstance(data, pd.DataFrame)

    def test_has_records(self, data):
        assert len(data) > 0

    def test_columns(self, data):
        assert data.columns.tolist() == [
            "account",
            "text",
            "url",
            "date",
            "action",
            "post_id",
            "comment_id",
            "reply_comment_id",
            "label",
        ]

    def test_dtypes(self, data):
        assert data.dtypes.map(str).tolist() == [
            "category",
            "object",
            "object",
            "datetime64[ns]",
            "category",
            "Int64",
            "Int64",
            "Int64",
            "int64",
        ]

    def test_accounts(self, data):
        assert data.account.unique().tolist() == ["dr nyheder"]

    def test_comment_id(self, data):
        assert list(data.comment_id.unique()) == [123]


class TestForAnnotationData:
    @pytest.fixture(scope="class")
    def data(self, config):
        yield prepare_data_for_annotation(config)

    def test_data_is_list_of_str(self, data):
        assert isinstance(data, list)
        for line in data:
            assert isinstance(line, str)

    def test_has_records(self, data):
        assert len(data) > 0


class TestAnnotatedData:
    @pytest.fixture(scope="class")
    def data(self, config):
        prepare_data_for_annotation(config)
        yield load_annotated_data(config)

    def test_is_dataframe(self, data):
        assert isinstance(data, pd.DataFrame)

    def test_has_records(self, data):
        assert len(data) > 0

    def test_columns(self, data):
        assert sorted(data.columns) == ["label", "text"]


class TestSplits:
    @pytest.fixture(scope="class")
    def data(self, config):
        create_train_split(config)
        yield load_splits(config)

    def test_data_is_dict(self, data):
        assert isinstance(data, dict)

    def test_keys(self, data):
        assert list(data.keys()) == ["train", "val", "test"]

    def test_values_are_dataframes(self, data):
        for value in data.values():
            assert isinstance(value, pd.DataFrame)

    def test_columns(self, data):
        for df in data.values():
            assert "text" in df.columns
            assert "label" in df.columns

    def test_no_overlap_between_splits(self, data):
        assert data["train"].text.isin(data["val"].text).sum() == 0
        assert data["train"].text.isin(data["test"].text).sum() == 0
        assert data["val"].text.isin(data["test"].text).sum() == 0
