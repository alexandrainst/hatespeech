"""Unit tests related to the labeling_functions module."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.dr_hatespeech.labeling_functions import (
    ABSTAIN,
    NOT_OFFENSIVE,
    OFFENSIVE,
    contains_offensive_word,
    has_been_moderated,
    is_dr_answer,
    is_mention,
    sentiment,
    use_tfidf_model,
    use_transformer_ensemble,
)


@pytest.mark.skip(reason="Snorkel is not supporting Mac M1's yet")
class TestLabelingFunctions:
    @pytest.fixture(scope="class")
    def fatsvag_record(self):
        data_dict = dict(
            account="dr nyheder",
            url="https://www.facebook.com/123",
            text="Din fuck lort! Er du fatsvag eller hvad?",
            date=dt.datetime(2020, 1, 1, 12, 0, 0),
            action="deleted",
            post_id=123,
            comment_id=np.nan,
            reply_comment_id=np.nan,
        )
        yield pd.Series(data_dict)

    @pytest.fixture(scope="class")
    def mention_record(self):
        data_dict = dict(
            account="dr nyheder",
            url="https://www.facebook.com/123",
            text="Hans Christian",
            date=dt.datetime(2020, 1, 1, 12, 0, 0),
            action="none",
            post_id=123,
            comment_id=np.nan,
            reply_comment_id=np.nan,
        )
        yield pd.Series(data_dict)

    @pytest.fixture(scope="class")
    def dr_answer_record(self):
        data_dict = dict(
            account="dr nyheder",
            url="https://www.facebook.com/123",
            text="Hej. Vi har slettet din kommentar [fatsvage kælling] // Michael, DR",
            date=dt.datetime(2020, 1, 1, 12, 0, 0),
            action="deleted",
            post_id=123,
            comment_id=np.nan,
            reply_comment_id=np.nan,
        )
        yield pd.Series(data_dict)

    def test_contains_offensive_word(self, fatsvag_record, dr_answer_record):
        assert contains_offensive_word(fatsvag_record) == OFFENSIVE
        assert contains_offensive_word(dr_answer_record) == ABSTAIN

    def test_is_mention(self, fatsvag_record, mention_record):
        assert is_mention(mention_record) == NOT_OFFENSIVE
        assert is_mention(fatsvag_record) == ABSTAIN

    def test_is_dr_answer(self, fatsvag_record, dr_answer_record):
        assert is_dr_answer(dr_answer_record) == NOT_OFFENSIVE
        assert is_dr_answer(fatsvag_record) == ABSTAIN

    def test_use_transformer_ensemble(self, fatsvag_record, mention_record):
        assert use_transformer_ensemble(fatsvag_record) == OFFENSIVE
        assert use_transformer_ensemble(mention_record) == ABSTAIN

    def test_use_tfidf_model(self, fatsvag_record, mention_record):
        assert use_tfidf_model(fatsvag_record) == OFFENSIVE
        assert use_tfidf_model(mention_record) == ABSTAIN

    def test_has_been_moderated(self, fatsvag_record, mention_record):
        assert has_been_moderated(fatsvag_record) == OFFENSIVE
        assert has_been_moderated(mention_record) == ABSTAIN

    def test_sentiment(self, fatsvag_record, mention_record):
        assert sentiment(fatsvag_record) == ABSTAIN
        assert sentiment(mention_record) == NOT_OFFENSIVE
